"""The name and number check of spec B 10.4: each name and number in the output occurs in the transcript."""

from __future__ import annotations

import re

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
# Capitalized words that are not names, anywhere in a sentence.
_COMMON = {"I", "Me", "Others", "OK", "Okay", "Mr", "Mrs", "Ms", "TV", "LCD", "AI"}
# Words that often start a sentence or a list line and are not names.
_START_WORDS = {
    "The", "This", "These", "Those", "They", "We", "It", "He", "She", "You", "A", "An", "In", "On", "At",
    "For", "And", "But", "So", "If", "When", "After", "Before", "Then", "There", "That", "Each", "All",
    "Some", "No", "Also", "Both", "Next", "First", "Finally", "Overall", "Team", "Meeting", "Summary",
    "Action", "Owner", "Decision", "Decisions", "Topic", "Notes", "Note", "Other", "One", "Two",
}


def number_words(n: int) -> str:
    """English words for 0 to 999,999, without "and" (AMI style: "twenty five")."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_words(rest))
    rest = n % 1000
    return number_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + number_words(rest))


def _norm(text: str) -> str:
    """Lower-case the text and keep only words, digits, ":" in times, and "." in decimals."""
    cleaned = re.sub(r"[^a-z0-9:.]+", " ", text.lower().replace("-", " "))
    return " " + re.sub(r"(?<!\d)\.|\.(?!\d)", " ", cleaned) + " "


def unsupported_terms(output: str, transcript: str, notes: str = "") -> list[str]:
    """Return the names and numbers of the output that the transcript and the notes do not hold.

    A name is a capitalized word. A word at the start of a sentence or a list line
    passes when it is a common start word, when the transcript or the notes hold it,
    or when the output also holds it in lower case. A number passes when its digits or
    its words occur. A decimal ending in .5 or .50 also passes for "<n> and a half" or
    "<n> point five". A time (mm:ss) passes when it occurs. AMI writes all numbers as
    words, so an invented number from 1 to 9 can pass: its word form is common.
    """
    output = re.sub(r"[*_#`]", "", output)
    source = _norm(transcript + "\n" + notes)
    found: list[str] = []
    for m in re.finditer(r"\d+(?::\d+)?(?:[.,]\d+)?", output):
        token = m.group(0)
        if f" {token.lower()} " in source:
            continue
        digits = token.replace(",", "")
        if digits.isdigit() and int(digits) < 1_000_000 and f" {number_words(int(digits))} " in source:
            continue
        whole, _, fraction = token.partition(".")
        if whole.isdigit() and fraction in ("5", "50") and int(whole) < 1_000_000:
            words = number_words(int(whole))
            if f" {words} and a half " in source or f" {words} point five " in source:
                continue
        found.append(token)
    for m in re.finditer(r"\b[A-Z][a-zA-Z]+\b", output):
        word = m.group(0)
        if word in _COMMON:
            continue
        before = output[: m.start()].rstrip(" \"'«(")
        if f" {word.lower()} " in source:
            continue
        if not before or before[-1] in ".!?\n-":
            if word in _START_WORDS or re.search(rf"\b{word.lower()}\b", output):
                continue
        found.append(word)
    return found
