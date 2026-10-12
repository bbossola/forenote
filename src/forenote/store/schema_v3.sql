-- Schema v3: notes enhancement (sub-project B, spec B section 4.3).
-- Bullets can have no time and remember a user edit. Notes outputs are stored
-- step by step, so summary and action items can be NULL. A full-text index
-- over the transcript lines serves the keyword search of the notes job.

CREATE TABLE bullets_v3 (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id      INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    position        INTEGER NOT NULL,
    text            TEXT NOT NULL,
    t_written_ms    INTEGER,
    enhanced_text   TEXT,
    enhanced_edited INTEGER NOT NULL DEFAULT 0
);
INSERT INTO bullets_v3(id, meeting_id, position, text, t_written_ms, enhanced_text)
    SELECT id, meeting_id, position, text, t_written_ms, enhanced_text FROM bullets;
DROP TABLE bullets;
ALTER TABLE bullets_v3 RENAME TO bullets;
CREATE INDEX bullets_meeting ON bullets(meeting_id, position);

CREATE TABLE meeting_outputs_v3 (
    meeting_id   INTEGER PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    summary      TEXT,
    action_items TEXT,                 -- JSON list
    model        TEXT NOT NULL,
    mode         TEXT NOT NULL CHECK (mode IN ('prefix', 'map_reduce')),
    created_at   TEXT NOT NULL
);
INSERT INTO meeting_outputs_v3(meeting_id, summary, action_items, model, mode, created_at)
    SELECT meeting_id, summary, action_items, model, 'map_reduce', created_at FROM meeting_outputs;
DROP TABLE meeting_outputs;
ALTER TABLE meeting_outputs_v3 RENAME TO meeting_outputs;

CREATE VIRTUAL TABLE transcript_fts USING fts5(text, content='transcript_lines', content_rowid='id');
INSERT INTO transcript_fts(rowid, text) SELECT id, text FROM transcript_lines;

-- Keep the full-text index in step with the transcript lines.
-- Foreign key cascades also fire these triggers.
CREATE TRIGGER transcript_lines_after_insert AFTER INSERT ON transcript_lines BEGIN
    INSERT INTO transcript_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER transcript_lines_after_delete AFTER DELETE ON transcript_lines BEGIN
    INSERT INTO transcript_fts(transcript_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER transcript_lines_after_update AFTER UPDATE OF text ON transcript_lines BEGIN
    INSERT INTO transcript_fts(transcript_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO transcript_fts(rowid, text) VALUES (new.id, new.text);
END;
