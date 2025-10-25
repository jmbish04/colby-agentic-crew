# In a Colab cell, run this first to install dependencies:
# !pip -q install google-genai PyGithub

import os
import sys
import json
import time
import base64
import requests
from dataclasses import dataclass
from typing import Optional, List, Tuple
from io import BytesIO

# --- Google/Colab Modules ---
try:
    from google.colab import drive
    from google.colab import userdata
    COLAB_ENV = True
except ImportError:
    COLAB_ENV = False

# --- Google GenAI SDK ---
try:
    from google import genai
    from google.genai import types
    from google.generativeai.types import GenerationConfig
except ImportError:
    print("Error: 'google-genai' library not found. Please run: !pip install google-genai")
    sys.exit(1)

# --- GitHub SDK ---
try:
    from github import Github, GithubException
except ImportError:
    print("Error: 'PyGithub' library not found. Please run: !pip install PyGithub")
    sys.exit(1)

# --- Images ---
# (Image imports removed)


# --------------------------------------
# Configuration
# --------------------------------------

# --- Model Configuration ---
# Using latest preview models as of Oct 2025
GEN_MODEL_TEXT = "gemini-2.5-flash-preview-09-2025"
# (Image model removed)

# --- Cost Control Configuration ---
# Prices per 1,000,000 tokens (as of late 2025 previews)
# $0.35 / 1M input tokens, $0.70 / 1M output tokens for flash
PRICE_FLASH_INPUT_PER_TOKEN = 0.35 / 1_000_000
PRICE_FLASH_OUTPUT_PER_TOKEN = 0.70 / 1_000_000

# (Image cost removed)

COST_LIMIT_USD = 25.00
TOTAL_ESTIMATED_COST = 0.0  # Global cost tracker
COST_LEDGER = [] # New: Itemized list of costs


# --- Google Drive Configuration ---
GDRIVE_MOUNT_POINT = '/content/drive'
OUTPUT_FOLDER_NAME = 'colby_agentic_crew'
OUTPUT_FOLDER_PATH = os.path.join(GDRIVE_MOUNT_POINT, 'MyDrive', OUTPUT_FOLDER_NAME)

# --- GitHub Configuration ---
# Updated to your specified repo
GITHUB_REPO = "jmbish04/colby-agentic-crew"
GITHUB_SAVE_PATH = "agent_generated_files" # Subfolder in the repo
GITHUB_TOKEN = "" # Will be loaded by init_clients


# --------------------------------------
# Utilities
# --------------------------------------

def read_secret(name: str, prompt: Optional[str] = None) -> str:
    """
    Read a secret from Colab userdata if available, else environment, else prompt stdin.
    """
    # 1) Colab userdata (preferred)
    if COLAB_ENV:
        try:
            val = userdata.get(name) # Raises if not set
            if val:
                return val
        except Exception:
            pass

    # 2) Environment variable
    val = os.getenv(name)
    if val:
        return val
    
    # 3) Stdin prompt (last resort, for local testing)
    if 'getpass' not in sys.modules:
        import getpass
    
    if prompt is None:
        prompt = f"Enter value for {name}: "
    try:
        return getpass.getpass(prompt)
    except Exception:
        # If getpass fails (non-tty), fall back to input()
        return input(prompt)


def backoff_sleep(retry: int):
    # simple exponential backoff with jitter
    delay = min(2 ** retry, 30) + (0.1 * retry)
    print(f"  ...backing off for {delay:.1f}s")
    time.sleep(delay)

# --------------------------------------
# Cost Control
# --------------------------------------

def check_and_update_cost(
    task_name: str,
    prompt_tokens: int = 0,
    output_tokens: int = 0,
    # images_generated (removed)
    model: str = GEN_MODEL_TEXT
) -> bool:
    """
    Checks if a new call would exceed the limit, logs the itemized cost,
    and updates the global cost.
    Returns True if the call can proceed, False if it's blocked.
    """
    global TOTAL_ESTIMATED_COST, COST_LEDGER
    
    call_cost = 0.0
    if model == GEN_MODEL_TEXT:
        call_cost += prompt_tokens * PRICE_FLASH_INPUT_PER_TOKEN
        call_cost += output_tokens * PRICE_FLASH_OUTPUT_PER_TOKEN
    # (Image cost logic removed)

    if (TOTAL_ESTIMATED_COST + call_cost) > COST_LIMIT_USD:
        print("="*50)
        print(f"COST LIMIT BREACHED: Task '{task_name}' cannot proceed.")
        print(f"  Current Cost: ${TOTAL_ESTIMATED_COST:.4f}")
        print(f"  Attempted Call Cost: ${call_cost:.4f}")
        print(f"  Limit: ${COST_LIMIT_USD:.4f}")
        print("="*50)
        # Log the breach attempt
        COST_LEDGER.append({
            "task": f"FAILED: {task_name} (Cost Limit Breach)",
            "cost": call_cost,
            "details": f"Attempted {prompt_tokens} in, {output_tokens} out"
        })
        return False
        
    # --- Log and Update ---
    TOTAL_ESTIMATED_COST += call_cost
    item_details = f"{prompt_tokens} in, {output_tokens} out"
    COST_LEDGER.append({
        "task": task_name,
        "cost": call_cost,
        "details": item_details
    })
    print(f"[CostTracker] Task '{task_name}' cost: ${call_cost:.6f}")
    print(f"[CostTracker] New total estimated cost: ${TOTAL_ESTIMATED_COST:.4f}")
    return True

# --------------------------------------
# Clients
# --------------------------------------
@dataclass
class Clients:
    genai_client: genai.Client
    gh: Github


def init_clients() -> Optional[Clients]:
    """Initializes and returns all API clients."""
    global GITHUB_TOKEN # Load the global token
    try:
        print("Initializing clients...")
        gemini_key = read_secret("GEMINI_API_KEY", "GEMINI_API_KEY (Gemini API key): ")
        GITHUB_TOKEN = read_secret("GITHUB_TOKEN", "GITHUB_TOKEN (GitHub PAT): ")
        
        if not gemini_key:
            print("ERROR: GEMINI_API_KEY is not set in Colab secrets (View > Show secrets).")
            return None
        
        if not GITHUB_TOKEN:
            print("ERROR: GITHUB_TOKEN is not set in Colab secrets (View > Show secrets).")
            return None

        genai_client = genai.Client(api_key=gemini_key)
        gh = Github(GITHUB_TOKEN, per_page=50)
        
        # Test GitHub connection
        _ = gh.get_user().login
        print("GitHub client authenticated.")
        
        print("All clients initialized successfully.")
        return Clients(genai_client, gh)

    except Exception as e:
        print(f"Error initializing clients: {e}")
        if 'Bad credentials' in str(e):
            print("Hint: Check your GITHUB_TOKEN.")
        return None


# --------------------------------------
# Google GenAI: text generation
# --------------------------------------

def gen_text(clients: Clients, task_name: str, prompt: str, max_retries: int = 3) -> Optional[str]:
    """
    Generate text content from a prompt using the google-genai SDK.
    Requires a task_name for itemized cost logging.
    Returns None if cost limit is breached or generation fails.
    """
    global TOTAL_ESTIMATED_COST
    print(f"\n--- Starting Task: {task_name} ---")
    print(f"Generating text for prompt: '{prompt[:50]}...'")

    # --- Pre-call Cost Check ---
    # Check if the *prompt alone* will break the bank.
    try:
        prompt_tokens = clients.genai_client.models.count_tokens(
            model=GEN_MODEL_TEXT,
            contents=[prompt]
        ).total_tokens
        print(f"[CostTracker] Estimated prompt tokens: {prompt_tokens}")
    except Exception as e:
        print(f"Warning: Could not pre-count tokens: {e}. Proceeding with caution.")
        prompt_tokens = len(prompt) // 4 # Rough fallback

    pre_check_cost = prompt_tokens * PRICE_FLASH_INPUT_PER_TOKEN
    if (TOTAL_ESTIMATED_COST + pre_check_cost) > COST_LIMIT_USD:
        print("="*50)
        print(f"COST LIMIT BREACH: Pre-check for task '{task_name}' failed.")
        print(f"  Current Cost: ${TOTAL_ESTIMATED_COST:.4f}")
        print(f"  Est. Prompt Cost: ${pre_check_cost:.4f}")
        print(f"  Limit: ${COST_LIMIT_USD:.4f}")
        print("="*50)
        COST_LEDGER.append({
            "task": f"SKIPPED: {task_name} (Pre-check failed cost limit)",
            "cost": 0.0,
            "details": f"Est. prompt tokens: {prompt_tokens}"
        })
        return None # Cost limit would be breached

    # --- API Call ---
    for attempt in range(max_retries + 1):
        try:
            resp = clients.genai_client.models.generate_content(
                model=GEN_MODEL_TEXT,
                contents=[prompt],
                system_instruction="You are an expert AI assistant specializing in Cloudflare Workers, agentic systems, and CrewAI. Provide complete, production-ready code and configurations."
            )
            
            # --- Post-call Cost Calculation ---
            usage = getattr(resp, "usage_metadata", None)
            if usage:
                in_tokens = usage.prompt_token_count
                out_tokens = usage.candidates_token_count
                
                # --- Post-call Cost Calculation & Logging ---
                if not check_and_update_cost(
                    task_name=task_name,
                    prompt_tokens=in_tokens, 
                    output_tokens=out_tokens, 
                    model=GEN_MODEL_TEXT
                ):
                    # This should rarely happen if pre-check is good, but it's a safety stop
                    return "(Cost limit reached during generation)"
            else:
                print("Warning: Could not get usage_metadata. Cost tracking may be inaccurate.")
                # Fallback to prompt_tokens only (log 0 output tokens)
                check_and_update_cost(
                    task_name=f"{task_name} (Usage data missing)",
                    prompt_tokens=prompt_tokens, 
                    output_tokens=0,
                    model=GEN_MODEL_TEXT
                )

            # Gather all text parts
            out = []
            for cand in getattr(resp, "candidates", []):
                for part in cand.content.parts:
                    if getattr(part, "text", None):
                        out.append(part.text)
            
            generated_text = "\n".join(out).strip()
            if not generated_text:
                raise RuntimeError("No text output received from model.")
            
            print("Text generation successful.")
            return generated_text

        except Exception as e:
            print(f"GenAI text error (Attempt {attempt+1}): {e}")
            if attempt >= max_retries:
                raise
            backoff_sleep(attempt)
    return None # Failed after retries


# (gen_image_png function removed)


# --------------------------------------
# GitHub operations
# --------------------------------------

def gh_get_repo(gh: Github, full_name: str):
    """
    Get a repo like 'owner/name'. Raises if not found or no access.
    """
    try:
        return gh.get_repo(full_name)
    except GithubException as e:
        print(f"GitHub repo error for {full_name}: {e.data if hasattr(e, 'data') else e}")
        raise


def gh_list_issues(gh: Github, repo_full_name: str, state: str = "open", limit: int = 5):
    print(f"Listing last {limit} '{state}' issues for {repo_full_name}...")
    repo = gh_get_repo(gh, repo_full_name)
    query = {"state": state}
    issues = repo.get_issues(**query)
    out = []
    for i in issues[:limit]: # Simple limit
        out.append({
            "number": i.number,
            "title": i.title,
            "state": i.state,
            "url": i.html_url
        })
    print(f"Found {len(out)} issues.")
    return out


def gh_create_issue(gh: Github, repo_full_name: str, title: str, body: str = "", labels: Optional[List[str]] = None) -> int:
    print(f"Creating issue in {repo_full_name}: '{title}'")
    repo = gh_get_repo(gh, repo_full_name)
    issue = repo.create_issue(title=title, body=body, labels=labels or [])
    print(f"Issue #{issue.number} created successfully.")
    return issue.number


def gh_comment_issue(gh: Github, repo_full_name: str, number: int, body: str) -> str:
    print(f"Commenting on issue #{number} in {repo_full_name}...")
    repo = gh_get_repo(gh, repo_full_name)
    issue = repo.get_issue(number=number)
    comment = issue.create_comment(body)
    print(f"Comment posted: {comment.html_url}")
    return comment.html_url


# --------------------------------------
# Google Drive & GitHub File Ops (NEW)
# --------------------------------------

# ========== Google Drive (mounted) ==========
def ensure_drive_ready() -> bool:
    """Mounts Drive (if in Colab) and ensures the output folder exists."""
    if not COLAB_ENV:
        print("[drive] Not in Colab; skipping mount.")
        return False
    try:
        print("[drive] Mounting…")
        drive.mount(GDRIVE_MOUNT_POINT)
        os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
        print(f"[drive] Ready: {OUTPUT_FOLDER_PATH}")
        return True
    except Exception as e:
        print(f"[drive] ERROR mounting Drive: {e}")
        return False


def save_to_drive(rel_path: str, content: str | bytes, binary: bool = False) -> str:
    """
    Saves a file under your Drive output folder.
    rel_path: path relative to OUTPUT_FOLDER_PATH (e.g., 'plans/wrangler.toml').
    content:  str or bytes. Set binary=True if bytes.
    """
    if not COLAB_ENV:
        print("[drive] Google Drive save requires Colab (mounted Drive). Skipping.")
        return ""

    abs_path = os.path.join(OUTPUT_FOLDER_PATH, rel_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    try:
        mode = "wb" if binary else "w"
        encoding = None if binary else "utf-8"
        with open(abs_path, mode, encoding=encoding) as f:
            f.write(content)
        print(f"[drive] Saved → {abs_path}")
        return abs_path
    except Exception as e:
        print(f"[drive] ERROR saving to Drive: {e}")
        return ""


# ========== GitHub (Contents API) ==========
_GH_API = "https://api.github.com"

def _gh_headers() -> dict:
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is missing (set in Colab secrets or env).")
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "genai-colab-uploader"
    }

def _gh_get_default_branch(repo_full: str) -> str:
    r = requests.get(f"{_GH_API}/repos/{repo_full}", headers=_gh_headers(), timeout=30)
    r.raise_for_status()
    return r.json().get("default_branch", "main")

def _gh_get_file_sha(repo_full: str, path: str, ref: Optional[str] = None) -> Optional[str]:
    """
    Return SHA if file exists on branch/ref, else None.
    """
    params = {"ref": ref} if ref else {}
    r = requests.get(f"{_GH_API}/repos/{repo_full}/contents/{path}", headers=_gh_headers(), params=params, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("sha")

def _gh_put_contents(repo_full: str, path: str, message: str, b64content: str, branch: Optional[str], sha: Optional[str]):
    payload = {"message": message, "content": b64content}
    if branch:
        payload["branch"] = branch
    if sha:
        payload["sha"] = sha
    r = requests.put(f"{_GH_API}/repos/{repo_full}/contents/{path}", headers=_gh_headers(), json=payload, timeout=60)
    # Helpful error surfacing
    if r.status_code >= 400:
        try:
            print("[github] Error payload:", r.json())
        except Exception:
            print("[github] Error text:", r.text)
    r.raise_for_status()
    return r.json()

def save_to_github(
    path: str,
    content: str | bytes,
    commit_message: str,
    repo_full: str = GITHUB_REPO,
    branch: Optional[str] = None,
    max_retries: int = 3
) -> dict:
    """
    Create or update a file in GitHub using the Contents API.
    - path: repo-relative path (e.g., 'src/index.ts')
    - content: str or bytes
    - commit_message: commit title/message
    - branch: target branch; if None, uses default branch
    Returns the GitHub API response JSON (includes content download URL, commit SHA, etc.)
    """
    if branch is None:
        branch = _gh_get_default_branch(repo_full)

    # Normalize to bytes + base64
    content_bytes = content if isinstance(content, (bytes, bytearray)) else content.encode("utf-8")
    b64 = base64.b64encode(content_bytes).decode("ascii")

    # If file exists, we must send its current SHA
    sha = _gh_get_file_sha(repo_full, path, ref=branch)

    for attempt in range(max_retries):
        try:
            print(f"[github] PUT {repo_full}:{branch}/{path} (attempt {attempt+1})")
            return _gh_put_contents(repo_full, path, commit_message, b64, branch, sha)
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (429, 502, 503, 504) and attempt < max_retries - 1:
                delay = min(2 ** attempt, 30) + 0.2 * (attempt + 1)
                print(f"[github] {code} retrying in {delay:.1f}s …")
                time.sleep(delay)
                continue
            raise
    # This line is theoretically unreachable due to `raise` in the loop,
    # but added for clarity that retries were exhausted.
    raise RuntimeError(f"Failed to upload to GitHub after {max_retries} attempts.")


# --------------------------------------
# Cost Report Generation
# --------------------------------------

def generate_cost_report() -> str:
    """
    Generates a formatted Markdown string from the COST_LEDGER.
    """
    print("Generating cost report...")
    global COST_LEDGER, TOTAL_ESTIMATED_COST, COST_LIMIT_USD
    
    report_lines = [
        "# Agent Run Cost Report",
        f"**Run completed:** {time.ctime()}",
        f"**Cost Limit:** ${COST_LIMIT_USD:.2f}",
        f"**Total Estimated Cost:** ${TOTAL_ESTIMATED_COST:.6f}",
        "\n## Itemized Costs\n",
        "| Task | Details (Tokens) | Cost (USD) |",
        "| :--- | :--- | :--- |"
    ]
    
    for item in COST_LEDGER:
        task = item.get("task", "Unknown Task")
        details = item.get("details", "N/A")
        cost = item.get("cost", 0.0)
        report_lines.append(f"| {task} | {details} | ${cost:.6f} |")
        
    report_lines.append("\n---\n*End of Report*")
    return "\n".join(report_lines)


# --------------------------------------
# Main Execution
# --------------------------------------

if __name__ == "__main__":
    print("="*50)
    print("Colby Agentic Crew - GenAI + GitHub Ops")
    print("="*50)
    print(f"Cost Limit set to: ${COST_LIMIT_USD:.2f}")
    print(f"GitHub Repo set to: {GITHUB_REPO}")
    print("NOTE: Make sure 'GEMINI_API_KEY' and 'GITHUB_TOKEN' are set in Colab secrets (View > Show secrets).")

    # Mount drive first
    drive_ready = ensure_drive_ready()
    
    if not drive_ready and COLAB_ENV:
        print("Could not mount Google Drive. Aborting to prevent lost work.")
    else:
        # Initialize clients
        clients = init_clients()
        
        if clients:
            # --- 1. Generate Text Artifact ---
            print("\n--- 1. Generating Text (wrangler.toml) ---")
            text_prompt = """
            Generate a complete 'wrangler.toml' file for a Cloudflare agentic crew project named 'colby-agentic-crew'.
            The project must include:
            1. A main worker entrypoint 'src/index.ts'.
            2. A Durable Object binding named 'AGENT_SUPERVISOR' for the class 'AgentSupervisor'.
            3. A KV namespace binding named 'CREW_MEMORY_KV'.
            4. A D1 database binding named 'CREW_STATE_DB'.
            5. An R2 bucket binding named 'AGENT_ARTIFACTS_R2'.
            """
            generated_text = gen_text(clients, text_prompt)
            
            if generated_text:
                print("Generated Text:\n", generated_text)
                timestamp = int(time.time())

                # --- 2. Save Text to Drive ---
                print("\n--- 2. Saving Text to Google Drive ---")
                output_filename = f"wrangler_config_{timestamp}.toml"
                save_to_drive(output_filename, generated_text)
                
                # --- 3. Save Text to GitHub ---
                print("\n--- 3. Saving Text to GitHub ---")
                try:
                    gh_path = f"{GITHUB_SAVE_PATH}/{output_filename}"
                    gh_message = f"Agent: Add wrangler.toml config ({timestamp})"
                    save_to_github(
                        path=gh_path,
                        content=generated_text,
                        commit_message=gh_message
                    )
                except Exception as e:
                    print(f"Error saving text to GitHub: {e}")
            else:
                print("Skipping text saving steps as generation failed or was cost-blocked.")

            # --- 4. Generate Image Artifact (REMOVED) ---
            
            # --- 5. Save Image to GitHub (REMOVED) ---

            # --- 6. Perform GitHub Issue Operations ---
            print("\n--- 6. Performing GitHub Issue Operations ---")
            if GITHUB_REPO == "jmbish04/colby-agentic-crew": # Check if it's the real repo
                try:
                    # List existing issues
                    issues = gh_list_issues(clients.gh, GITHUB_REPO, state="open", limit=5)
                    print("Existing open issues:", json.dumps(issues, indent=2))
                    
                    if generated_text:
                        # Create a new issue only if text was generated
                        issue_title = "Bot Task: Add new `wrangler.toml` configuration"
                        issue_body = f"Generated by agent.\n\nSee file in repo: `{GITHUB_SAVE_PATH}/{output_filename}`\n\n**Configuration:**\n```toml\n{generated_text}\n```"
                        new_issue_num = gh_create_issue(clients.gh, GITHUB_REPO, issue_title, issue_body, labels=["bot", "config"])
                        
    generated_text_toml = None
    generated_text_readme = None
    timestamp = int(time.time())

    try:
        if not drive_ready and COLAB_ENV:
            print("Could not mount Google Drive. Aborting to prevent lost work.")
            sys.exit(1) # Use sys.exit to trigger finally
        
        # Initialize clients
        clients = init_clients()
        
        if clients:
            # --- 1. Generate Text Artifact (wrangler.toml) ---
            text_prompt_toml = """
            Generate a complete 'wrangler.toml' file for a Cloudflare agentic crew project named 'colby-agentic-crew'.
            5. An R2 bucket binding named 'AGENT_ARTIFACTS_R2'.
            """
            generated_text_toml = gen_text(
                clients, 
                task_name="Generate wrangler.toml",
                prompt=text_prompt_toml
            )
            
            if generated_text_toml:
                print("Generated wrangler.toml:\n", generated_text_toml)

                # --- 2. Save Text to Drive ---
                print("\n--- Saving wrangler.toml to Google Drive ---")
                output_filename_toml = f"wrangler_config_{timestamp}.toml"
                save_to_drive(output_filename_toml, generated_text_toml)
                COST_LEDGER.append({"task": "Save wrangler.toml to Drive", "cost": 0.0, "details": "File I/O"})
                
                # --- 3. Save Text to GitHub ---
                print("\n--- Saving wrangler.toml to GitHub ---")
                try:
                    gh_path_toml = f"{GITHUB_SAVE_PATH}/{output_filename_toml}"
                    gh_message_toml = f"Agent: Add wrangler.toml config ({timestamp})"
                    save_to_github(
                        path=gh_path_toml,
                        content=generated_text_toml,
                        commit_message=gh_message_toml
                    )
                    COST_LEDGER.append({"task": "Save wrangler.toml to GitHub", "cost": 0.0, "details": "GitHub API call"})
                except Exception as e:
                    print(f"Error saving text to GitHub: {e}")
                    COST_LEDGER.append({"task": "Save wrangler.toml to GitHub (FAILED)", "cost": 0.0, "details": str(e)})
            else:
                print("Skipping wrangler.toml saving steps as generation failed or was cost-blocked.")

            # --- 4. Generate Text Artifact (README.md) ---
            text_prompt_readme = """
            Generate a professional README.md for the 'colby-agentic-crew' project. 
            It should describe a system using Cloudflare Workers, Durable Objects, D1, R2, and KV for running agentic crews. 
            Mention that the configuration is managed in 'wrangler.toml'.
            """
            generated_text_readme = gen_text(
                clients,
                task_name="Generate README.md",
                prompt=text_prompt_readme
            )

            if generated_text_readme:
                print("Generated README.md:\n", generated_text_readme)
                
                # --- 5. Save README to Drive & GitHub ---
                print("\n--- Saving README.md to Google Drive ---")
                output_filename_readme = f"README_{timestamp}.md"
                save_to_drive(output_filename_readme, generated_text_readme)
                COST_LEDGER.append({"task": "Save README.md to Drive", "cost": 0.0, "details": "File I/O"})

                print("\n--- Saving README.md to GitHub ---")
                try:
                    gh_path_readme = f"{GITHUB_SAVE_PATH}/{output_filename_readme}"
                    gh_message_readme = f"Agent: Add README.md documentation ({timestamp})"
                    save_to_github(
                        path=gh_path_readme,
                        content=generated_text_readme,
                        commit_message=gh_message_readme
                    )
                    COST_LEDGER.append({"task": "Save README.md to GitHub", "cost": 0.0, "details": "GitHub API call"})
                except Exception as e:
                    print(f"Error saving README to GitHub: {e}")
                    COST_LEDGER.append({"task": "Save README.md to GitHub (FAILED)", "cost": 0.0, "details": str(e)})
            else:
                 print("Skipping README.md saving steps as generation failed or was cost-blocked.")

            # --- 6. Perform GitHub Issue Operations ---
            print("\n--- 6. Performing GitHub Issue Operations ---")
            if GITHUB_REPO == "jmbish04/colby-agentic-crew": # Check if it's the real repo
                try:
                    # List existing issues
                    issues = gh_list_issues(clients.gh, GITHUB_REPO, state="open", limit=5)
                    print("Existing open issues:", json.dumps(issues, indent=2))
                    COST_LEDGER.append({"task": "List GitHub Issues", "cost": 0.0, "details": "GitHub API call"})
                    
                    if generated_text_toml:
                        # Create a new issue only if text was generated
                        issue_title = f"Bot Task: Review new `wrangler.toml` ({timestamp})"
                        issue_body = f"Generated by agent.\n\nSee file in repo: `{gh_path_toml}`\n\n**Configuration:**\n```toml\n{generated_text_toml}\n```"
                        new_issue_num = gh_create_issue(clients.gh, GITHUB_REPO, issue_title, issue_body, labels=["bot", "config"])
                        COST_LEDGER.append({"task": "Create GitHub Issue (wrangler.toml)", "cost": 0.0, "details": f"Created #{new_issue_num}"})
                        
                        # Comment on the new issue
                        comment_body = f"Task #{new_issue_num} created."
                        if generated_text_readme:
                            comment_body += f"\nAssociated README also generated: `{gh_path_readme}`"
                        
                        gh_comment_issue(clients.gh, GITHUB_REPO, new_issue_num, comment_body)
                        COST_LEDGER.append({"task": "Comment on GitHub Issue", "cost": 0.0, "details": f"Commented on #{new_issue_num}"})
                    else:
                        print("Skipping issue creation as text generation failed or was cost-blocked.")

                except GithubException as e:
                    print(f"Error during GitHub operations: {e}")
                    print(f"Please ensure the token has 'repo' scope and access to '{GITHUB_REPO}'.")
                    COST_LEDGER.append({"task": "GitHub Issue Ops (FAILED)", "cost": 0.0, "details": str(e)})
                except Exception as e:
                    print(f"An unexpected error occurred during GitHub ops: {e}")
                    COST_LEDGER.append({"task": "GitHub Issue Ops (FAILED)", "cost": 0.0, "details": str(e)})
        else:
            print("Failed to initialize clients. Check secrets.")
            COST_LEDGER.append({"task": "Client Initialization", "cost": 0.0, "details": "FAILED"})

    except Exception as e:
        print(f"An uncaught error occurred in main execution: {e}")
        COST_LEDGER.append({"task": "Main Execution (CRITICAL FAIL)", "cost": 0.0, "details": str(e)})
    
    finally:
        # --- 7. Generate and Save Final Cost Report ---
        print("\n" + "="*50)
        print("--- 7. Generating Final Cost Report ---")
        report_content = generate_cost_report()
        print(report_content)
        
        print("\n--- 8. Saving Final Cost Report ---")
        report_filename = f"cost_report_{timestamp}.md"
        
        # Save to Drive
        save_to_drive(report_filename, report_content)
        
        # Save to GitHub
        try:
            save_to_github(
                path=f"{GITHUB_SAVE_PATH}/{report_filename}",
                content=report_content,
                commit_message=f"Agent: Final cost report ({timestamp})"
            )
        except Exception as e:
            print(f"Error saving final cost report to GitHub: {e}")

        print("\n" + "="*50)
        print("Script finished.")
        print(f"FINAL ESTIMATED COST: ${TOTAL_ESTIMATED_COST:.6f}")
        print("="*50)

