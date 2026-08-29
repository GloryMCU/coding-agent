BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS context_summary (
    session_id TEXT PRIMARY KEY REFERENCES session(id) ON DELETE CASCADE,
    through_seq INTEGER NOT NULL CHECK (through_seq >= 1),
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

INSERT OR IGNORE INTO schema_migration(version, applied_at)
VALUES (2, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));

COMMIT;
