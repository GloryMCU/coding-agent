BEGIN IMMEDIATE;

CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5(
    session_id UNINDEXED,
    message_id UNINDEXED,
    part_id UNINDEXED,
    message_seq UNINDEXED,
    role UNINDEXED,
    part_type UNINDEXED,
    content,
    tokenize = 'trigram'
);

DELETE FROM history_fts;

INSERT INTO history_fts(
    session_id, message_id, part_id, message_seq, role, part_type, content
)
SELECT
    part.session_id,
    part.message_id,
    part.id,
    message.seq,
    message.role,
    part.type,
    CASE part.type
        WHEN 'text' THEN COALESCE(json_extract(part.data_json, '$.text'), '')
        WHEN 'reasoning' THEN COALESCE(json_extract(part.data_json, '$.text'), '')
        ELSE COALESCE(part.tool_name, '') || ' ' || part.data_json
    END
FROM part
JOIN message ON message.id = part.message_id
WHERE trim(
    CASE part.type
        WHEN 'text' THEN COALESCE(json_extract(part.data_json, '$.text'), '')
        WHEN 'reasoning' THEN COALESCE(json_extract(part.data_json, '$.text'), '')
        ELSE COALESCE(part.tool_name, '') || ' ' || part.data_json
    END
) <> '';

CREATE TRIGGER IF NOT EXISTS history_fts_part_insert
AFTER INSERT ON part
BEGIN
    INSERT INTO history_fts(
        session_id, message_id, part_id, message_seq, role, part_type, content
    )
    SELECT
        NEW.session_id,
        NEW.message_id,
        NEW.id,
        message.seq,
        message.role,
        NEW.type,
        CASE NEW.type
            WHEN 'text' THEN COALESCE(json_extract(NEW.data_json, '$.text'), '')
            WHEN 'reasoning' THEN COALESCE(json_extract(NEW.data_json, '$.text'), '')
            ELSE COALESCE(NEW.tool_name, '') || ' ' || NEW.data_json
        END
    FROM message
    WHERE message.id = NEW.message_id
      AND trim(
        CASE NEW.type
            WHEN 'text' THEN COALESCE(json_extract(NEW.data_json, '$.text'), '')
            WHEN 'reasoning' THEN COALESCE(json_extract(NEW.data_json, '$.text'), '')
            ELSE COALESCE(NEW.tool_name, '') || ' ' || NEW.data_json
        END
      ) <> '';
END;

CREATE TRIGGER IF NOT EXISTS history_fts_part_update
AFTER UPDATE OF type, tool_name, data_json ON part
BEGIN
    DELETE FROM history_fts WHERE part_id = OLD.id;
    INSERT INTO history_fts(
        session_id, message_id, part_id, message_seq, role, part_type, content
    )
    SELECT
        NEW.session_id,
        NEW.message_id,
        NEW.id,
        message.seq,
        message.role,
        NEW.type,
        CASE NEW.type
            WHEN 'text' THEN COALESCE(json_extract(NEW.data_json, '$.text'), '')
            WHEN 'reasoning' THEN COALESCE(json_extract(NEW.data_json, '$.text'), '')
            ELSE COALESCE(NEW.tool_name, '') || ' ' || NEW.data_json
        END
    FROM message
    WHERE message.id = NEW.message_id
      AND trim(
        CASE NEW.type
            WHEN 'text' THEN COALESCE(json_extract(NEW.data_json, '$.text'), '')
            WHEN 'reasoning' THEN COALESCE(json_extract(NEW.data_json, '$.text'), '')
            ELSE COALESCE(NEW.tool_name, '') || ' ' || NEW.data_json
        END
      ) <> '';
END;

CREATE TRIGGER IF NOT EXISTS history_fts_part_delete
AFTER DELETE ON part
BEGIN
    DELETE FROM history_fts WHERE part_id = OLD.id;
END;

INSERT OR IGNORE INTO schema_migration(version, applied_at)
VALUES (3, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));

COMMIT;
