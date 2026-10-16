# Knowledge base format fixtures

`make_fixtures.py` builds every file in this directory. We wrote all the
text. There is no third-party content. Run it again only when a test needs a
new fixture:

    uv run python tests/fixtures/kb/make_fixtures.py
