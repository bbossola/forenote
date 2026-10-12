-- Forenote schema v2: markers in a meeting transcript (audio spec 4.3).

CREATE TABLE meeting_markers (
    id         INTEGER PRIMARY KEY,
    meeting_id INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    t_ms       INTEGER NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN
               ('source_changed', 'source_lost', 'transcription_failed')),
    speaker    TEXT CHECK (speaker IN ('me', 'others'))
);
CREATE INDEX meeting_markers_meeting ON meeting_markers(meeting_id, t_ms);
