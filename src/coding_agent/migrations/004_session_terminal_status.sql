PRAGMA foreign_keys = OFF;

BEGIN IMMEDIATE;

CREATE TABLE session_new (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    model TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'active', 'completed', 'partial', 'blocked', 'interrupted', 'failed'
        )
    ),
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

INSERT INTO session_new(
    id, workspace, model, system_prompt, status, error, created_at, updated_at
)
SELECT
    id,
    workspace,
    model,
    system_prompt,
    CASE status WHEN 'error' THEN 'failed' ELSE status END,
    error,
    created_at,
    updated_at
FROM session;

DROP TABLE session;
ALTER TABLE session_new RENAME TO session;

INSERT OR IGNORE INTO schema_migration(version, applied_at)
VALUES (4, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));

COMMIT;

PRAGMA foreign_keys = ON;
