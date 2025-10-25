#!/usr/bin/env python3
# parse_captured_content.py
# Walks vendor/cloudflare/**, detects Cloudflare features, extracts wrangler bindings,
# produces per-repo markdown + JSON artifacts, and a global index. Fail-soft throughout.

import os, re, json, hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

# Config
ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "vendor" / "cloudflare"
DOCS_ROOT = ROOT / "repo_root" / "docs"
FILES_OF_INTEREST = (".ts", ".tsx", ".js", ".mjs", ".cjs", ".toml", ".json", ".md", ".yaml", ".yml")
MAX_READ = 200_000

PROXY_URL = os.environ.get("OPENAI_PROXY_URL", "").rstrip("/")
PROXY_TOKEN = os.environ.get("OPENAI_PROXY_TOKEN", "")

# Heuristics
CF_TS_HINTS = {
    "durable_objects": [
        r"\bclass\s+\w+\s+extends\s+DurableObject\b",
        r"\bDurableObjectNamespace\b",
        r"\bstate:\s*DurableObjectState\b",
        r"\[\[durable_objects\.bindings\]\]"
    ],
    "d1": [r"\bD1Database\b", r"\[\[d1_databases\]\]"],
    "kv": [r"\bKVNamespace\b", r"\[\[kv_namespaces\]\]"],
    "queues": [r"\[\[queues\]\]", r"\bqueue\s*=\s*\""],
    "workflows": [r"\bWorkflows?Client\b", r"\bworkflows?\.register\b"],
    "actors": [r"@cloudflare/actors", r"\bActors?Namespace\b"],
    "agents_sdk": [r"@cloudflare/agents", r"\bagent\("],
    "containers": [r"\bsandbox\b", r"\bcontainers?\b", r"@cloudflare/workers-types/experimental"],
    "websockets": [r'headers\.get\("Upgrade"\)\s*===\s*"websocket"'],
    "assets": [r"\bassets\s*=\s*\{[^}]*directory[^}]*\}"],
    "wrangler_toml": [r"\bcompatibility_date\b", r"\bmain\s*=\s*"],
    "ui_shadcn": [r"\bshadcn\b", r"from\s+['\"]/?.*components/ui"],
    "react_frontend": [r"\.tsx?\b", r"\breact\b", r"\bnext\b", r"\btailwindcss\b", r"\bvite\b"]
}

# ---- Safe network helper ----
import time, requests
def safe_request_json(method, url, *, headers=None, json_body=None, timeout=20, retries=2, backoff=1.6):
    for attempt in range(retries + 1):
        try:
            resp = requests.request(method, url, headers=headers, json=json_body, timeout=timeout)
            code = resp.status_code
            if 200 <= code < 300:
                try:
                    return True, resp.json()
                except Exception as e:
                    return False, f"JSON decode error: {e}"
            if code in (429,) or 500 <= code < 600:
                if attempt < retries:
                    time.sleep(backoff ** attempt)
                    continue
                return False, f"HTTP {code}: {resp.text[:500]}"
            return False, f"HTTP {code}: {resp.text[:500]}"
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(backoff ** attempt)
                continue
            return False, f"Request error: {e}"

def read_text_safe(p: Path) -> str:
    try:
        data = p.read_text(encoding="utf-8", errors="ignore")
        return data[:MAX_READ]
    except Exception:
        return ""

def extract_wrangler_bindings(toml_text: str) -> Dict[str, Any]:
    out = {"durable_objects": [], "d1": [], "kv": [], "queues": [], "assets": None}
    for m in re.finditer(r"\[\[durable_objects\.bindings\]\]\s*name\s*=\s*\"([^\"]+)\".*?class_name\s*=\s*\"([^\"]+)\"", toml_text, re.S):
        out["durable_objects"].append({"name": m.group(1), "class": m.group(2)})
    for m in re.finditer(r"\[\[d1_databases\]\]\s*binding\s*=\s*\"([^\"]+)\".*?database_name\s*=\s*\"([^\"]+)\"", toml_text, re.S):
        out["d1"].append({"binding": m.group(1), "database": m.group(2)})
    for m in re.finditer(r"\[\[kv_namespaces\]\]\s*binding\s*=\s*\"([^\"]+)\"", toml_text, re.S):
        out["kv"].append({"binding": m.group(1)})
    for m in re.finditer(r"\[\[queues\]\]\s*binding\s*=\s*\"([^\"]+)\".*?queue\s*=\s*\"([^\"]+)\"", toml_text, re.S):
        out["queues"].append({"binding": m.group(1), "queue": m.group(2)})
    m = re.search(r"\bassets\s*=\s*\{([^}]+)\}", toml_text, re.S)
    if m: out["assets"] = m.group(1).strip()
    return out

def sniff_features(text: str) -> Dict[str, bool]:
    hits = {}
    for key, patterns in CF_TS_HINTS.items():
        hits[key] = any(re.search(p, text, re.I|re.S) for p in patterns)
    return hits

def summarize_package(pkg_json: Dict[str, Any]) -> Dict[str, Any]:
    deps = {}
    deps.update(pkg_json.get("dependencies") or {})
    deps.update(pkg_json.get("devDependencies") or {})
    flags = {
        "react": "react" in deps,
        "vite": "vite" in deps,
        "next": "next" in deps,
        "tailwind": "tailwindcss" in deps,
        "shadcn": any("shadcn" in k or "shadcn" in v for k, v in deps.items())
    }
    scripts = pkg_json.get("scripts") or {}
    return {"name": pkg_json.get("name"), "flags": flags, "scripts": scripts, "deps_sample": list(deps.keys())[:30]}

def collect_repo(repo_dir: Path) -> Dict[str, Any]:
    meta = {"repo_name": repo_dir.name, "wrangler": {}, "packages": [], "features": {k: False for k in CF_TS_HINTS.keys()}, "files_scanned": 0, "notable_paths": []}
    wrangler = next((p for p in repo_dir.rglob("wrangler.toml")), None)
    if wrangler:
        wt = read_text_safe(wrangler)
        meta["wrangler"] = extract_wrangler_bindings(wt)
        meta["notable_paths"].append(str(wrangler.relative_to(repo_dir)))
        meta["features"]["wrangler_toml"] = True

    for pkg in repo_dir.rglob("package.json"):
        try:
            pkg_obj = json.loads(read_text_safe(pkg) or "{}")
            meta["packages"].append(summarize_package(pkg_obj))
            meta["notable_paths"].append(str(pkg.relative_to(repo_dir)))
        except Exception:
            pass

    aggregate_text_sample = []
    for p in repo_dir.rglob("*"):
        if not p.is_file(): continue
        if p.suffix.lower() not in FILES_OF_INTEREST: continue
        text = read_text_safe(p)
        meta["files_scanned"] += 1
        feats = sniff_features(text)
        for k, v in feats.items():
            meta["features"][k] = meta["features"][k] or v
        if p.suffix in (".ts", ".tsx", ".toml") and len(aggregate_text_sample) < 30:
            for pat in (r"class\s+\w+\s+extends\s+DurableObject", r"\[\[.*?\]\]", r"binding\s*=", r"class_name\s*=", r"from\s+['\"][^'\"]+['\"]"):
                m = re.search(pat, text, re.I)
                if m:
                    aggregate_text_sample.append(f"### {p.relative_to(repo_dir)}\n```snippet\n{text[m.start():m.end()+200]}\n```")
                    break
    meta["sample_snippets"] = aggregate_text_sample
    return meta

def render_md(repo_name: str, meta: Dict[str, Any]) -> str:
    f = meta["features"]
    wr = meta.get("wrangler") or {}
    lines = [
        f"# {repo_name} overview",
        "",
        "## Detected features",
        "- Durable Objects: " + ("yes" if f.get("durable_objects") else "no"),
        "- D1: " + ("yes" if f.get("d1") else "no"),
        "- KV: " + ("yes" if f.get("kv") else "no"),
        "- Queues: " + ("yes" if f.get("queues") else "no"),
        "- Workflows: " + ("yes" if f.get("workflows") else "no"),
        "- Actors: " + ("yes" if f.get("actors") else "no"),
        "- Agents SDK: " + ("yes" if f.get("agents_sdk") else "no"),
        "- Containers/Sandbox: " + ("yes" if f.get("containers") else "no"),
        "- WebSockets: " + ("yes" if f.get("websockets") else "no"),
        "- Assets: " + ("yes" if f.get("assets") else "no"),
        "- UI (shadcn): " + ("yes" if f.get("ui_shadcn") else "no"),
        "- React Frontend: " + ("yes" if f.get("react_frontend") else "no"),
        "",
        "## Wrangler bindings (naive parse)",
        f"- DOs: {wr.get('durable_objects')}",
        f"- D1: {wr.get('d1')}",
        f"- KV: {wr.get('kv')}",
        f"- Queues: {wr.get('queues')}",
        f"- Assets: {wr.get('assets')}",
        "",
        "## Packages detected",
    ]
    for pkg in meta.get("packages", []):
        lines.append(f"- `{pkg.get('name')}` flags={pkg.get('flags')} scripts={list((pkg.get('scripts') or {}).keys())[:8]}")
    lines += ["", "## Notable paths"] + [f"- `{p}`" for p in meta.get("notable_paths", [])]
    if meta.get("sample_snippets"):
        lines += ["", "## Sample snippets (heuristic)"] + meta["sample_snippets"]
    return "\n".join(lines)

def write_docs(repo_name: str, meta: Dict[str, Any]) -> None:
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    (DOCS_ROOT / f"{repo_name}_overview.md").write_text(render_md(repo_name, meta), encoding="utf-8")
    (DOCS_ROOT / f"{repo_name}_artifacts.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def send_to_proxy(repo_name: str, meta: Dict[str, Any]) -> Optional[str]:
    if not (PROXY_URL and PROXY_TOKEN):
        return None
    payload = {
        "title": f"{repo_name} Cloudflare signals",
        "sections": [
            {"h2": "Detected features", "list": [k for k, v in meta["features"].items() if v]},
            {"h2": "Wrangler bindings", "text": json.dumps(meta.get("wrangler") or {}, indent=2)},
            {"h2": "Packages", "list": [p.get("name") for p in meta.get("packages", []) if p.get("name")]},
        ],
        "snippets": meta.get("sample_snippets", [])[:10],
    }
    ok, data = safe_request_json(
        "POST",
        f"{PROXY_URL}/summarize",
        headers={"Authorization": f"Bearer {PROXY_TOKEN}", "Content-Type": "application/json"},
        json_body=payload,
        timeout=25, retries=2, backoff=1.7
    )
    if ok and isinstance(data, dict) and data.get("markdown"):
        return data["markdown"]
    return f"_Proxy summarize unavailable (non-blocking). Details: {data if isinstance(data, str) else 'unknown error'}_"

def main():
    repos = [p for p in VENDOR_ROOT.iterdir() if p.is_dir()] if VENDOR_ROOT.exists() else []
    index = []
    for repo_dir in repos:
        try:
            meta = collect_repo(repo_dir)
            write_docs(repo_dir.name, meta)
            md = send_to_proxy(repo_dir.name, meta)
            if md:
                (DOCS_ROOT / f"{repo_dir.name}_proxy.md").write_text(md, encoding="utf-8")
            index.append({
                "repo": repo_dir.name,
                "features": {k: v for k, v in meta["features"].items() if v},
                "wrangler": meta.get("wrangler"),
                "packages": [p.get("name") for p in meta.get("packages", []) if p.get("name")],
            })
        except Exception as e:
            stub = f"# {repo_dir.name} overview\n\n_Parse failed (non-blocking): {type(e).__name__}: {e}\n"
            DOCS_ROOT.mkdir(parents=True, exist_ok=True)
            (DOCS_ROOT / f"{repo_dir.name}_overview.md").write_text(stub, encoding="utf-8")
            index.append({"repo": repo_dir.name, "error": f"{type(e).__name__}: {e}"})

    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    (DOCS_ROOT / "_cf_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps({"parsed_repos": len(index)}, indent=2))

if __name__ == "__main__":
    main()
