"""A token estimate for the chunk size (spec C 5.3).

The index job does not load the LLM, and LiteRT-LM has no tokenizer API.
The estimate counts each digit as one token (Gemma splits the digits), one
token for each 8 ASCII letters of a word, and one token for each 2 other
characters. Non-ASCII letters cost more: see the two constants. Then it adds a margin. A models test checks the estimate
against the Gemma tokenizer.
"""

from __future__ import annotations

import math
import re

MARGIN = 1.15
CJK_COST = 0.7  # Gemma uses 0.46 to 0.59 tokens for each kana, hangul or ideograph in prose (up to 1.0 for rare ones)
NON_ASCII_COST = 0.4  # Gemma uses 0.2 to 0.36 tokens for each other non-ASCII letter in prose (Russian, Greek)
_WORD = re.compile(r"\S+")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff66-\uff9f\U00020000-\U0002ffff]")


def word_cost(word: str) -> float:
    digits = letters = cjk = wide = 0
    for c in word:
        if c.isdigit():
            digits += 1
        elif c.isascii() and c.isalpha():
            letters += 1
        elif c.isalpha():
            if _CJK.match(c):
                cjk += 1
            else:
                wide += 1
    other = len(word) - digits - letters - cjk - wide
    cost = digits + math.ceil(letters / 8) + math.ceil(other / 2) + CJK_COST * cjk + NON_ASCII_COST * wide
    return MARGIN * cost


def estimate(text: str) -> int:
    return math.ceil(sum(word_cost(w) for w in _WORD.findall(text)))
