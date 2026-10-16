"""A section of a parsed file: text and where it comes from (spec C 4.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Section:
    text: str
    locator: dict[str, Any] = field(default_factory=dict)  # for example {"page": 3}
