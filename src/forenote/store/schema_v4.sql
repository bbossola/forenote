-- Forenote schema v4: the knowledge base (spec C section 4.4).
-- No code used the v1 KB tables, so they are empty. Drop them and make them again.

DROP TRIGGER chunks_after_insert;
DROP TRIGGER chunks_after_delete;
DROP TRIGGER chunks_after_update;
DROP TABLE chunks_fts;
DROP TABLE chunks_vec;
DROP TABLE chunks;

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    meeting_id  INTEGER REFERENCES meetings(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('file', 'transcript', 'notes')),
    position    INTEGER NOT NULL,      -- the order of the chunk in its source
    text        TEXT NOT NULL,
    locator     TEXT NOT NULL,         -- JSON: page, slide, sheet and rows, heading, time range, or notes part
    embed_model TEXT NOT NULL,         -- the manifest id of the model that made the vector
    CHECK ((document_id IS NULL) <> (meeting_id IS NULL)),
    CHECK ((kind = 'file') = (document_id IS NOT NULL))
);
CREATE INDEX chunks_document ON chunks(document_id, position);
CREATE INDEX chunks_meeting ON chunks(meeting_id, kind, position);

-- vec0 has no foreign key. Insert a chunk and its vector in one transaction.
-- The metadata column kind lets the search filter inside the KNN query.
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    chunk_id  INTEGER PRIMARY KEY,
    embedding float[768] distance_metric=cosine,
    kind      text
);

-- The porter tokenizer stems words, the same as transcript_fts.
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id',
    tokenize='porter unicode61');

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

ALTER TABLE documents ADD COLUMN size INTEGER NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN indexed_at TEXT;

CREATE TABLE meeting_index (
    meeting_id  INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    part        TEXT NOT NULL CHECK (part IN ('transcript', 'notes')),
    source_hash TEXT NOT NULL,
    indexed_at  TEXT NOT NULL,
    PRIMARY KEY (meeting_id, part)
);

ALTER TABLE chat_messages ADD COLUMN status TEXT NOT NULL DEFAULT 'done'
    CHECK (status IN ('done', 'cancelled', 'failed'));
ALTER TABLE chat_messages ADD COLUMN scope TEXT CHECK (scope IN ('all', 'meetings', 'files'));
ALTER TABLE chat_messages ADD COLUMN model TEXT;
