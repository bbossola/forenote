"""A token estimate for the chunk size (spec C 5.3).

The index job does not load the LLM, and LiteRT-LM has no tokenizer API.
The estimate counts each digit as one token (Gemma splits the digits), one
token for each 8 letters of a word, and one token for each 2 other
characters. Then it adds a margin. A models test checks the estimate
against the Gemma tokenizer.
"""

from __future__ import annotations

import math
import re

MARGIN = 1.15
_WORD = re.compile(r"\S+")


def word_cost(word: str) -> float:
    digits = letters = 0
    for c in word:
        if c.isdigit():
            digits += 1
        elif c.isalpha():
            letters += 1
    other = len(word) - digits - letters
    return MARGIN * (digits + math.ceil(letters / 8) + math.ceil(other / 2))


def estimate(text: str) -> int:
    return math.ceil(sum(word_cost(w) for w in _WORD.findall(text)))
