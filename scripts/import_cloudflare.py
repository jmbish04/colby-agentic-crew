# scripts/import_cloudflare.py
import argparse, os, sys, json, base64, tempfile, shutil, subprocess, re
from pathlib import Path
import requests
from github import Github

DEFAULT_PATHS = [
    "examples", "example", "demo", "demos", "templates", "template",
    "src", "workflows", "queues", "actors", "agents", "durable-objects"
]
README_FILES = ["README.md", "readme.md"]

def gh_headers(token): return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

def search_repos(gh, org, keywords, allow_topics):
    q_terms = [f"org:{org}"]
    for k in keywords: q_terms.append(k)
    q = " ".join(q_terms)
    repos = set()
    for k in keywords:
        for r in gh.search_repositories(query=f'org:{org} {k} in:name,description,readme'):
            if allow_topics and not set(r.get_topics() or []).intersection(allow_topics):
                continue
            repos.add(r)
    return sorted(repos, key=lambda r: (r.stargazers_count, r.name), reverse=True)

def want_path(path):
    low = path.lower()
    return any(p in low.split("/") for p in DEFAULT_PATHS) or low in README_FILES

def fetch_tree(owner, repo, ref, token):
    # Get full repo tree (recursive)
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
    r = requests.get(url, headers=gh_headers(token))
    r.raise_for_status()
    return r.json().get("tree", [])

def fetch_file(owner, repo, path, ref, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    r = requests.get(url, headers=gh_headers(token))
    if r.status_code == 404: return None, None
    r.raise_for_status()
    data = r.json()
    if data.get("encoding") == "base64":
        return base64.b64decode(data["content"]), data.get("path")
    if isinstance(data, list):
        return None, None
    return data.get("content"), data.get("path")

def write_bytes(dst, content):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f: f.write(content)

def vendor_repo(owner, repo, ref, token, vendor_dir, allow_paths=None, dry_run=True):
    tree = fetch_tree(owner, repo, ref, token)
    planned = []
    for node in tree:
        p = node.get("path","")
        if node["type"] == "blob" and (want_path(p) if allow_paths is None else p in allow_paths):
            planned.append(p)

    copied = []
    for p in planned:
        blob, _ = fetch_file(owner, repo, p, ref, token)
        if not blob: continue
        dst = Path(vendor_dir) / repo / p
        if dry_run:
            print(f"[DRY] copy {owner}/{repo}:{p} -> {dst}")
        else:
            write_bytes(dst, blob)
        copied.append(str(dst))
    return planned, copied

def add_subtree(owner, repo, ref, vendor_dir, dry_run=True):
    dst = Path(vendor_dir)/repo
    if dry_run:
        print(f"[DRY] git subtree add --prefix {dst} https://github.com/{owner}/{repo}.git {ref} --squash")
        return
    subprocess.check_call(["git","remote","add",f"tmp-{repo}",f"https://github.com/{owner}/{repo}.git"], stderr=subprocess.DEVNULL)
    try:
        subprocess.check_call(["git","fetch",f"tmp-{repo}",ref,"--depth","1"])
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git","subtree","add","--prefix",str(dst),f"tmp-{repo}",ref,"--squash"])
    finally:
        subprocess.call(["git","remote","remove",f"tmp-{repo}"])

def add_submodule(owner, repo, ref, vendor_dir, dry_run=True):
    dst = Path(vendor_dir)/repo
    url = f"https://github.com/{owner}/{repo}.git"
    if dry_run:
        print(f"[DRY] git submodule add -b {ref} {url} {dst}")
        return
    subprocess.check_call(["git","submodule","add","-b",ref,url,str(dst)])

def call_proxy(openai_url, token, repo_name, metadata, dest_md):
    if not openai_url or not token: return
    payload = {
        "title": f"{repo_name} overview",
        "sections": [
            {"h2": "Summary", "text": metadata.get("description","")},
            {"h2": "Key topics", "list": metadata.get("topics",[])},
            {"h2": "Notable paths", "list": metadata.get("paths",[])},
            {"h2": "Why imported", "text": metadata.get("why","")}
        ]
    }
    try:
        r = requests.post(openai_url.rstrip("/") + "/summarize",
                          headers={"Authorization": f"Bearer {token}",
                                   "Content-Type":"application/json"},
                          data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        out = r.json().get("markdown") or "# Overview\n\n(No content returned)"
    except Exception as e:
        out = f"# {repo_name} overview\n\nProxy call failed: {e}\n\nMetadata:\n```json\n{json.dumps(metadata, indent=2)}\n```"
    dest_md.parent.mkdir(parents=True, exist_ok=True)
    dest_md.write_text(out, encoding="utf-8")

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
        print("GH_TOKEN or GITHUB_TOKEN required", file=sys.stderr); sys.exit(1)

    gh = Github(token)
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    allow_topics = set([t.strip() for t in args.allow_topics.split(",") if t.strip()])

    repos = search_repos(gh, args.org, keywords, allow_topics)
    print(f"Found {len(repos)} candidate repos")

    imported = []
    for r in repos:
        owner = r.owner.login
        name = r.name
        default_branch = r.default_branch or "main"

        why = []
        topics = r.get_topics() or []
        desc = r.description or ""
        low = (desc or "").lower() + " " + " ".join(topics)

        if any(k in low for k in keywords):
            why.append("Keyword/topic match")

        vendor_dir = Path(args.vendor_dir)
        docs_dir = Path(args.docs_dir)

        if args.mode == "vendor":
            planned, _ = vendor_repo(owner, name, default_branch, token, vendor_dir, allow_paths=None, dry_run=dry_run)
        elif args.mode == "subtree":
            add_subtree(owner, name, default_branch, vendor_dir, dry_run=dry_run)
            planned = ["(subtree root)"]
        else:
            add_submodule(owner, name, default_branch, vendor_dir, dry_run=dry_run)
            planned = ["(submodule root)"]

        # Per-repo overview via your proxy
        call_proxy(
            os.environ.get("OPENAI_PROXY_URL"),
            os.environ.get("OPENAI_PROXY_TOKEN"),
            name,
            {"description": desc, "topics": topics, "paths": planned, "why": ", ".join(why) or "Heuristic import"},
            dest_md=docs_dir / f"{name}_overview.md"
        )

        imported.append({
            "repo": f"{owner}/{name}",
            "branch": default_branch,
            "mode": args.mode,
            "paths": planned[:20]
        })

    # Write summary
    summary = {
        "total": len(imported),
        "mode": args.mode,
        "dry_run": dry_run,
        "imported": imported
    }
    Path("IMPORT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
