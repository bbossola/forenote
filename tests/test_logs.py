import logging

from forenote.engine.main import setup_logging
from forenote.logs import UserContentFilter
from forenote.paths import Paths


def test_filter_drops_records_marked_as_user_content():
    f = UserContentFilter()
    plain = logging.LogRecord("forenote", logging.INFO, __file__, 1, "count %d", (3,), None)
    marked = logging.LogRecord("forenote", logging.INFO, __file__, 1, "text %s", ("secret",), None)
    marked.user_content = True
    assert f.filter(plain) is True
    assert f.filter(marked) is False


def test_setup_logging_installs_the_filter_on_every_handler(tmp_path):
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    try:
        setup_logging(Paths(tmp_path / "d", tmp_path / "s", tmp_path / "r"), "INFO")
        assert root.handlers
        for handler in root.handlers:
            assert any(isinstance(f, UserContentFilter) for f in handler.filters)
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:], level = saved
        root.setLevel(level)
