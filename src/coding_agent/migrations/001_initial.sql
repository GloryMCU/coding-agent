BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS schema_migration (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL,
    model TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'completed', 'error')),
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS message (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    provider_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, seq)
);

CREATE TABLE IF NOT EXISTS part (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL REFERENCES message(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('text', 'reasoning', 'tool')),
    call_id TEXT,
    tool_name TEXT,
    status TEXT CHECK (
        status IS NULL OR status IN (
            'pending', 'running', 'completed', 'error', 'interrupted'
        )
    ),
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (message_id, seq),
    UNIQUE (session_id, call_id)
);

CREATE INDEX IF NOT EXISTS idx_message_session_seq
    ON message(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_part_message_seq
    ON part(message_id, seq);
CREATE INDEX IF NOT EXISTS idx_part_session_status
    ON part(session_id, status);

INSERT OR IGNORE INTO schema_migration(version, applied_at)
VALUES (1, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));

COMMIT;
