#!/usr/bin/env python3

# import_cloudflare.py
# End-to-end importer for Cloudflare org repositories with fail-soft network and proxy summarize.
# Modes:
#   - vendor: copy selected files into vendor/cloudflare/<repo>
#   - subtree: git subtree add (preserve history)
#   - submodule: git submodule add
#
# Outputs:
#   - repo_root/docs/<repo>_overview.md (via proxy or fallback)
#   - IMPORT_SUMMARY.json
# Env:
#   - GH_TOKEN or GITHUB_TOKEN
#   - OPENAI_PROXY_URL (optional), OPENAI_PROXY_TOKEN (optional)
#
# Usage example:
#   python scripts/import_cloudflare.py \
#     --org cloudflare \
#     --keywords "agents,durable-objects,actors,sandbox,containers,workflows,queues,shadcn,react,ai,sdk" \
#     --allow-topics "cloudflare,workers,durable-objects,actors,agents,queues,workflows" \
#     --vendor-dir vendor/cloudflare \
#     --docs-dir repo_root/docs \
#     --mode vendor \
#     --dry-run false

import argparse, base64, json, os, re, sys, time, subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import requests

# ---------- Fail-soft JSON HTTP helper ----------
def safe_request_json(method: str, url: str, *, headers=None, json_body=None, timeout=20, retries=2, backoff=1.6):
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

def gh_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

# ---------- Search & fetch helpers (REST; no PyGithub dependency) ----------
def gh_search_repos(token: str, org: str, keywords: List[str], allow_topics: set, per_page: int = 50, pages: int = 2):
    seen = {}
    headers = gh_headers(token)
    for kw in keywords:
        q = f"org:{org} {kw} in:name,description,readme"
        for page in range(1, pages + 1):
            url = f"https://api.github.com/search/repositories?q={requests.utils.quote(q)}&per_page={per_page}&page={page}"
            ok, data = safe_request_json("GET", url, headers=headers, timeout=20, retries=2)
            if not ok or "items" not in (data or {}):
                continue
            for item in data["items"]:
                full = item["full_name"]
                if full in seen:
                    continue
                # Topics require a separate call to /topics
                turl = f"https://api.github.com/repos/{item['owner']['login']}/{item['name']}/topics"
                th = dict(headers)
                th["Accept"] = "application/vnd.github+json"
                tok, tdata = safe_request_json("GET", turl, headers=th, timeout=15, retries=1)
                topics = (tdata or {}).get("names", []) if tok else []
                if allow_topics and not (set(topics) & allow_topics):
                    # keep it if description or name matches strongly
                    pass  # We'll still allow; keyword match already applied
                seen[full] = {
                    "owner": item["owner"]["login"],
                    "name": item["name"],
                    "default_branch": item.get("default_branch") or "main",
                    "description": item.get("description") or "",
                    "stargazers_count": item.get("stargazers_count", 0),
                    "topics": topics,
                }
    # Return sorted by stars then name
    return [seen[k] for k in sorted(seen.keys(), key=lambda x: (seen[x]["stargazers_count"], x), reverse=True)]

def gh_get_tree(token: str, owner: str, repo: str, ref: str) -> List[Dict[str, Any]]:
    headers = gh_headers(token)
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
    ok, data = safe_request_json("GET", url, headers=headers, timeout=25, retries=2)
    if ok and isinstance(data, dict):
        return data.get("tree", []) or []
    return []

def gh_get_file_b64(token: str, owner: str, repo: str, path: str, ref: str) -> Optional[bytes]:
    headers = gh_headers(token)
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    ok, data = safe_request_json("GET", url, headers=headers, timeout=25, retries=2)
    if not ok or not isinstance(data, dict):
        return None
    if data.get("encoding") == "base64" and data.get("content"):
        try:
            return base64.b64decode(data["content"])
        except Exception:
            return None
    return None

# ---------- Selection heuristics ----------
DEFAULT_PATHS = [
    "examples","example","demo","demos","templates","template",
    "src","workflows","queues","actors","agents","durable-objects"
]
README_FILES = ["README.md","readme.md"]
def want_path(path: str) -> bool:
    low = path.lower()
    parts = low.split("/")
    return any(p in parts for p in DEFAULT_PATHS) or (low in [x.lower() for x in README_FILES])

# ---------- Proxy summarize ----------
def call_proxy(openai_url: str, token: str, repo_name: str, metadata: Dict[str, Any], dest_md: Path):
    dest_md.parent.mkdir(parents=True, exist_ok=True)
    if not openai_url or not token:
        dest_md.write_text(
            f"# {repo_name} overview\n\n(No proxy configured)\n\nMetadata:\n```json\n{json.dumps(metadata, indent=2)}\n```",
            encoding="utf-8"
        )
        return
    payload = {
        "title": f"{repo_name} overview",
        "sections": [
            {"h2": "Summary", "text": metadata.get("description","")},
            {"h2": "Key topics", "list": metadata.get("topics",[])},
            {"h2": "Notable paths", "list": metadata.get("paths",[])},
            {"h2": "Why imported", "text": metadata.get("why","") or "Heuristic import"}
        ]
    }
    ok, data = safe_request_json(
        "POST",
        openai_url.rstrip("/") + "/summarize",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json_body=payload,
        timeout=25, retries=2, backoff=1.7
    )
    if ok and isinstance(data, dict) and data.get("markdown"):
        md = data["markdown"]
    else:
        md = (
            f"# {repo_name} overview\n\n"
            f"**Proxy summarize unavailable** — proceeding with metadata only.\n\n"
            f"Error: {data if isinstance(data, str) else 'unknown'}\n\n"
            f"Metadata:\n```json\n{json.dumps(metadata, indent=2)}\n```"
        )
    dest_md.write_text(md, encoding="utf-8")

# ---------- Vendor/subtree/submodule ops ----------
def vendor_repo(token: str, owner: str, repo: str, ref: str, vendor_dir: Path, allow_paths=None, dry_run=True) -> Tuple[List[str], List[str]]:
    tree = gh_get_tree(token, owner, repo, ref)
    planned = []
    for node in tree:
        p = node.get("path","")
        if node.get("type") == "blob" and (want_path(p) if allow_paths is None else p in allow_paths):
            planned.append(p)
    copied = []
    for p in planned:
        blob = gh_get_file_b64(token, owner, repo, p, ref)
        dst = vendor_dir / repo / p
        if dry_run:
            print(f"[DRY] copy {owner}/{repo}:{p} -> {dst}")
        else:
            if blob is None:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f:
                f.write(blob)
        copied.append(str(dst))
    return planned, copied

def add_subtree(owner: str, repo: str, ref: str, vendor_dir: Path, dry_run=True):
    dst = vendor_dir / repo
    if dry_run:
        print(f"[DRY] git subtree add --prefix {dst} https://github.com/{owner}/{repo}.git {ref} --squash")
        return
    subprocess.check_call(["git","remote","add",f"tmp-{repo}",f"https://github.com/{owner}/{repo}.git"])
    try:
        subprocess.check_call(["git","fetch",f"tmp-{repo}",ref,"--depth","1"])
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git","subtree","add","--prefix",str(dst),f"tmp-{repo}",ref,"--squash"])
    finally:
        subprocess.call(["git","remote","remove",f"tmp-{repo}"])

def add_submodule(owner: str, repo: str, ref: str, vendor_dir: Path, dry_run=True):
    dst = vendor_dir / repo
    url = f"https://github.com/{owner}/{repo}.git"
    if dry_run:
        print(f"[DRY] git submodule add -b {ref} {url} {dst}")
        return
    subprocess.check_call(["git","submodule","add","-b",ref,url,str(dst)])

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--keywords", required=True, help="comma-separated")
    ap.add_argument("--allow-topics", default="", help="comma-separated")
    ap.add_argument("--vendor-dir", required=True)
    ap.add_argument("--docs-dir", required=True)
    ap.add_argument("--mode", choices=["vendor","subtree","submodule"], default="vendor")
    ap.add_argument("--dry-run", choices=["true","false"], default="true")
    args = ap.parse_args()

    dry_run = args.dry_run.lower() == "true"
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GH_TOKEN or GITHUB_TOKEN required", file=sys.stderr)
        sys.exit(1)

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    allow_topics = set([t.strip() for t in args.allow_topics.split(",") if t.strip()])
    vendor_dir = Path(args.vendor_dir)
    docs_dir = Path(args.docs_dir)
    vendor_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    repos = gh_search_repos(token, args.org, keywords, allow_topics)
    print(f"Found {len(repos)} candidate repos")

    openai_url = os.environ.get("OPENAI_PROXY_URL","").strip()
    openai_token = os.environ.get("OPENAI_PROXY_TOKEN","").strip()

    imported = []
    for r in repos:
        try:
            owner = r["owner"]
            name = r["name"]
            default_branch = r.get("default_branch","main")
            desc = r.get("description","")
            topics = r.get("topics",[])

            if args.mode == "vendor":
                planned, _ = vendor_repo(token, owner, name, default_branch, vendor_dir, allow_paths=None, dry_run=dry_run)
            elif args.mode == "subtree":
                add_subtree(owner, name, default_branch, vendor_dir, dry_run=dry_run)
                planned = ["(subtree root)"]
            else:
                add_submodule(owner, name, default_branch, vendor_dir, dry_run=dry_run)
                planned = ["(submodule root)"]

            # per-repo doc via proxy (fail-soft)
            call_proxy(
                openai_url,
                openai_token,
                name,
                {"description": desc, "topics": topics, "paths": planned, "why": "Keyword/topic match"},
                dest_md=docs_dir / f"{name}_overview.md"
            )

            imported.append({
                "repo": f"{owner}/{name}",
                "branch": default_branch,
                "mode": args.mode,
                "paths": planned[:20]
            })
        except Exception as e:
            imported.append({
                "repo": f"{r.get('owner','?')}/{r.get('name','?')}",
                "branch": r.get("default_branch","?"),
                "mode": args.mode,
                "paths": [],
                "error": f"{type(e).__name__}: {e}"
            })
            print(f"[WARN] Skipped {r.get('owner')}/{r.get('name')}: {e}", file=sys.stderr)
            continue

    summary = {"total": len(imported), "mode": args.mode, "dry_run": dry_run, "imported": imported}
    Path("IMPORT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
