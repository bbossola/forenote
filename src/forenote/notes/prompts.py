"""The fixed prompt texts of the notes job (spec B sections 5.3 and 5.4).

The texts never contain user content. The job puts the transcript and the
bullets between them.
"""

from __future__ import annotations

from typing import Any

HEADER = (
    "You help a user with the notes of a meeting. The transcript comes from speech recognition, "
    "so it can contain errors. Each line has the time from the start of the meeting "
    '(minutes:seconds), the speaker, and the text. "Me" is the user. "Others" are the other '
    "people in the meeting.\n\n"
)
TRANSCRIPT_TITLE = "Transcript:\n"
PARTS_TITLE = "Parts of the transcript:\n"
NOTES_TITLE = "The notes of the user:\n"
NO_NOTES = "The user wrote no notes.\n"
REDUCE_TITLE = "Points about the parts of the meeting:\n"

SUMMARY_TASK = (
    "Task: write a summary of the meeting in 3 to 6 sentences. Then write the main points as a list, "
    'one point on each line, each line starting with "- ". Use only facts from the transcript.\n'
)
ACTION_ITEMS_TASK = (
    "Task: list the action items of the meeting as a JSON array. Each item has \"owner\" (the name of the "
    'person from the transcript who must do it, "Me" when the user must do it, or null) and "text" '
    "(what the person must do). Use only facts from the transcript. "
    "If the meeting has no action items, write [].\n"
)
GROUP_TASK = (
    "Task: combine these points into fewer points. Keep the topics, the decisions, the numbers, "
    'and the names. Write at most 10 lines, each line starting with "- ".\n'
)

ACTION_ITEMS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"owner": {"type": ["string", "null"]}, "text": {"type": "string"}},
        "required": ["owner", "text"],
    },
}


def bullet_task(
    number: int, text: str, written: str | None, window: tuple[str, str] | None, matches: list[str]
) -> str:
    """The instruction for one bullet. written and window are mm:ss texts, or None."""
    if written is not None and window is not None:
        parts = [
            f"Task: Expand note {number}, «{text}», written at {written}.",
            f"The relevant part of the transcript is probably between {window[0]} and {window[1]}.",
        ]
    else:
        parts = [f"Task: Expand note {number}, «{text}». The note has no time."]
    if matches:
        parts.append(f"Transcript lines with words of the note start at: {', '.join(matches)}.")
    else:
        parts.append("No transcript line has the words of the note.")
    parts.append(
        "Write 1 to 3 sentences that explain the note. Use only facts from the transcript. "
        "Write only the sentences."
    )
    return " ".join(parts) + "\n"


def map_task(index: int, count: int) -> str:
    return (
        f"Task: this is part {index} of {count} of the transcript. Write short points about this part: "
        'the topics, the decisions, the numbers, and the names. Write at most 10 lines, each line starting with "- ". '
        "Use only facts from this part.\n"
    )
