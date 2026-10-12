import subprocess

import pytest

from lookout.hw import (
    HardwareReport,
    detect,
    detect_audio,
    detect_gpus,
    parse_cpuinfo,
    parse_meminfo,
    select_tier,
)

CPUINFO = """processor\t: 0
vendor_id\t: AuthenticAMD
physical id\t: 0
core id\t\t: 0
flags\t\t: fpu sse2 avx avx2 fma

processor\t: 1
physical id\t: 0
core id\t\t: 0
flags\t\t: fpu sse2 avx avx2 fma

processor\t: 2
physical id\t: 0
core id\t\t: 1
flags\t\t: fpu sse2 avx avx2 fma
"""

PACTL = "Server String: /run/user/1000/pulse/native\nServer Name: PulseAudio (on PipeWire 1.0.5)\n"

VULKAN = """Devices:
========
GPU0:
\tapiVersion         = 1.3.289
\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tdeviceName         = NVIDIA GeForce GTX 1650 SUPER
GPU1:
\tdeviceType         = PHYSICAL_DEVICE_TYPE_CPU
\tdeviceName         = llvmpipe (LLVM 17.0.6, 256 bits)
"""


def fake_run(outputs):
    """outputs: command name -> stdout, or an exception instance to raise."""

    def run(cmd, **kwargs):
        out = outputs.get(cmd[0])
        if isinstance(out, BaseException):
            raise out
        if out is None:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error")
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    return run


def test_parse_cpuinfo_counts_physical_cores():
    cpu = parse_cpuinfo(CPUINFO)
    assert cpu.avx2 is True
    assert cpu.avx512 is False
    assert cpu.physical_cores == 2


def test_parse_meminfo():
    assert parse_meminfo("MemTotal:       16303004 kB\nMemFree: 1 kB\n") == pytest.approx(15.548, abs=0.01)


@pytest.mark.parametrize("ram,tier", [(60.0, "default"), (15.5, "default"), (14.9, "low"), (8.0, "low"), (7.5, None)])
def test_select_tier(ram, tier):
    assert select_tier(ram) == tier


def test_detect_audio():
    assert detect_audio(fake_run({"pactl": PACTL})) == "PulseAudio (on PipeWire 1.0.5)"
    assert detect_audio(fake_run({"pactl": FileNotFoundError()})) is None
    assert detect_audio(fake_run({})) is None


def test_detect_gpus_skips_cpu_devices():
    assert detect_gpus(fake_run({"vulkaninfo": VULKAN})) == ["NVIDIA GeForce GTX 1650 SUPER"]
    assert detect_gpus(fake_run({"vulkaninfo": FileNotFoundError()})) == []


def test_report_problems_and_warnings():
    ok = HardwareReport(True, False, 8, 31.0, "default", "PulseAudio", ())
    assert ok.problems == [] and ok.warnings == [] and not ok.fatal
    assert ok.threads == 7

    weak = HardwareReport(False, False, 4, 6.0, None, None, ())
    assert weak.fatal
    assert len(weak.problems) == 2
    assert any("audio" in w for w in weak.warnings)

    low = HardwareReport(True, False, 4, 12.0, "low", "PulseAudio", ())
    assert any("low model tier" in w for w in low.warnings)


def test_report_to_dict_is_json_ready():
    d = HardwareReport(True, False, 8, 31.0, "default", "PulseAudio", ("GPU",)).to_dict()
    assert d["threads"] == 7
    assert d["gpus"] == ["GPU"]
    assert d["problems"] == []


def test_detect_reads_proc_files(tmp_path):
    cpu = tmp_path / "cpuinfo"
    mem = tmp_path / "meminfo"
    cpu.write_text(CPUINFO)
    mem.write_text("MemTotal:       32000000 kB\n")
    r = detect(cpu, mem, fake_run({"pactl": PACTL, "vulkaninfo": VULKAN}))
    assert r.tier == "default"
    assert r.physical_cores == 2
    assert r.gpus == ("NVIDIA GeForce GTX 1650 SUPER",)
