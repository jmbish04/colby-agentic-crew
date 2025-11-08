CREATE TABLE IF NOT EXISTS code_artifacts (
    id TEXT PRIMARY KEY,               -- UUID, links to Vectorize
    repo_full_name TEXT NOT NULL,      -- e.g., "cloudflare/workers-sdk"
    file_path TEXT NOT NULL,           -- e.g., "src/index.ts"
    repo_url TEXT,                     -- Full URL to the file
    content_hash TEXT NOT NULL,        -- SHA-256 of the content to prevent re-processing
    content_snippet TEXT,              -- First few lines
    ai_summary TEXT,                   -- AI-generated summary
    ai_tags TEXT,                      -- JSON array: ["d1", "best-practice", "deprecation"]
    ai_use_case TEXT,                  -- AI-generated use case
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    -- Ensure we don't process the exact same file version twice
    UNIQUE(repo_full_name, file_path, content_hash)
);
