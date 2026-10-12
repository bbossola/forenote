"""Logging helpers. Logs must never contain user content (spec 10.1)."""

from __future__ import annotations

import logging


class UserContentFilter(logging.Filter):
    """Drop log records that carry user content.

    Code must not log user content at all. This filter is a guard: a record
    created with extra={"user_content": True} never reaches a handler.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "user_content", False)
