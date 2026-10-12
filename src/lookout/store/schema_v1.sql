-- Lookout schema v1 (spec section 7).
-- Timestamps are ISO 8601 strings in UTC.

CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL                -- JSON
);

CREATE TABLE meetings (
    id           INTEGER PRIMARY KEY,
    title        TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    ended_at     TEXT,
    status       TEXT NOT NULL
                 CHECK (status IN ('recording', 'processing', 'done', 'interrupted')),
    start_reason TEXT NOT NULL DEFAULT 'manual'
                 CHECK (start_reason IN ('manual', 'detected')),
    stop_reason  TEXT
                 CHECK (stop_reason IN ('manual', 'silence', 'error'))
);

CREATE TABLE transcript_lines (
    id         INTEGER PRIMARY KEY,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    speaker    TEXT NOT NULL CHECK (speaker IN ('me', 'others')),
    t_start_ms INTEGER NOT NULL,
    t_end_ms   INTEGER NOT NULL,
    text       TEXT NOT NULL
);
CREATE INDEX transcript_lines_meeting ON transcript_lines(meeting_id, t_start_ms);

CREATE TABLE bullets (
    id            INTEGER PRIMARY KEY,
    meeting_id    INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    text          TEXT NOT NULL,
    t_written_ms  INTEGER NOT NULL,
    enhanced_text TEXT
);
CREATE INDEX bullets_meeting ON bullets(meeting_id, position);

CREATE TABLE meeting_outputs (
    meeting_id   INTEGER PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    summary      TEXT NOT NULL,
    action_items TEXT NOT NULL,        -- JSON list
    model        TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE folders (
    id     INTEGER PRIMARY KEY,
    path   TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'missing'))
);

CREATE TABLE documents (
    id        INTEGER PRIMARY KEY,
    folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
    path      TEXT NOT NULL UNIQUE,
    mtime     REAL NOT NULL,
    sha256    TEXT,
    status    TEXT NOT NULL CHECK (status IN ('indexed', 'pending', 'error', 'skipped')),
    reason    TEXT
);

CREATE INDEX documents_folder ON documents(folder_id);

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    meeting_id  INTEGER REFERENCES meetings(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('file', 'transcript')),
    text        TEXT NOT NULL,
    locator     TEXT NOT NULL,         -- JSON: page, heading, or time range
    CHECK ((document_id IS NULL) <> (meeting_id IS NULL)),
    CHECK ((kind = 'file') = (document_id IS NOT NULL))
);
CREATE INDEX chunks_document ON chunks(document_id);
CREATE INDEX chunks_meeting ON chunks(meeting_id);

-- vec0 has no foreign key. Insert a chunk and its vector in one transaction.
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    chunk_id  INTEGER PRIMARY KEY,
    embedding float[768] distance_metric=cosine
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id');

-- Keep the full-text and vector indexes in step with the chunks table.
-- Foreign key cascades also fire these triggers.
CREATE TRIGGER chunks_after_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER chunks_after_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    DELETE FROM chunks_vec WHERE chunk_id = old.id;
END;

-- After a text update, the kb module must write the embedding again.
CREATE TRIGGER chunks_after_update AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE chat_sessions (
    id         INTEGER PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE chat_messages (
    id         INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    text       TEXT NOT NULL,
    citations  TEXT NOT NULL DEFAULT '[]',   -- JSON list
    created_at TEXT NOT NULL
);
CREATE INDEX chat_messages_session ON chat_messages(session_id, id);

CREATE TABLE jobs (
    id          INTEGER PRIMARY KEY,
    type        TEXT NOT NULL,
    priority    INTEGER NOT NULL,
    status      TEXT NOT NULL
                CHECK (status IN ('pending', 'running', 'done', 'failed', 'cancelled')),
    payload     TEXT NOT NULL DEFAULT '{}',  -- JSON
    error       TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT
);
CREATE INDEX jobs_status ON jobs(status, priority, id);
