"""The name and number check of spec B 10.4: each name and number in the output occurs in the transcript."""

from __future__ import annotations

import re

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
# Capitalized words that are not names.
_COMMON = {"I", "Me", "Others", "OK", "Okay", "Mr", "Mrs", "Ms", "TV", "LCD", "AI"}


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
    return " " + re.sub(r"[^a-z0-9:]+", " ", text.lower().replace("-", " ")) + " "


def unsupported_terms(output: str, transcript: str, notes: str = "") -> list[str]:
    """Return the names and numbers of the output that the transcript and the notes do not hold.

    A name is a capitalized word that does not start a sentence or a list line. A
    number passes when its digits or its words occur. Times (mm:ss) pass when they occur.
    """
    source = _norm(transcript + "\n" + notes)
    found: list[str] = []
    for m in re.finditer(r"\d+(?::\d+)?(?:[.,]\d+)?", output):
        token = m.group(0)
        if f" {token.lower()} " in source:
            continue
        digits = token.replace(",", "")
        if digits.isdigit() and int(digits) < 1_000_000 and f" {number_words(int(digits))} " in source:
            continue
        found.append(token)
    for m in re.finditer(r"\b[A-Z][a-zA-Z]+\b", output):
        word = m.group(0)
        before = output[: m.start()].rstrip(" \"'«(")
        if not before or before[-1] in ".!?:\n-*" or word in _COMMON:
            continue
        if f" {word.lower()} " not in source:
            found.append(word)
    return found
