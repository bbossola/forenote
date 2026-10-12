import subprocess
import sys


def test_the_engine_import_loads_no_heavy_module():
    code = (
        "import sys, forenote.engine.main; "
        "bad=[m for m in ('faster_whisper','onnxruntime','ctranslate2','av','pulsectl','pasimple') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
