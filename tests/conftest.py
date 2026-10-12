import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def short_tmp():
    d = Path(tempfile.mkdtemp(prefix="lk", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)
