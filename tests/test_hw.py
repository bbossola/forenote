import subprocess

import pytest

from forenote.hw import (
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


def test_detect_audio_uses_the_probe():
    assert detect_audio(lambda: "PulseAudio (on PipeWire 1.0.5)") == "PulseAudio (on PipeWire 1.0.5)"
    assert detect_audio(lambda: None) is None
    assert detect_audio(lambda: "") is None


def test_detect_audio_treats_probe_errors_as_no_server():
    def broken():
        raise OSError("libpulse.so.0: cannot open shared object file")

    assert detect_audio(broken) is None


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
    r = detect(cpu, mem, fake_run({"vulkaninfo": VULKAN}), audio_probe=lambda: "PulseAudio (on PipeWire 1.0.5)")
    assert r.tier == "default"
    assert r.audio_server == "PulseAudio (on PipeWire 1.0.5)"
    assert r.physical_cores == 2
    assert r.gpus == ("NVIDIA GeForce GTX 1650 SUPER",)


def test_child_processes_use_c_locale():
    seen = {}

    def run(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout=PACTL, stderr="")

    detect_gpus(run)
    assert seen["env"]["LC_ALL"] == "C"
    assert seen["env"]["LANGUAGE"] == "C"


def test_detect_gpus_removes_duplicates():
    text = VULKAN + (
        "GPU2:\n"
        "\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n"
        "\tdeviceName         = NVIDIA GeForce GTX 1650 SUPER\n"
    )
    assert detect_gpus(fake_run({"vulkaninfo": text})) == ["NVIDIA GeForce GTX 1650 SUPER"]


def test_vulkaninfo_timeout_is_short():
    seen = {}

    def run(cmd, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    assert detect_gpus(run) == []
    assert seen["timeout"] <= 3


def test_asr_threads_leave_two_cores_free():
    assert HardwareReport(True, False, 8, 31.0, "default", "x", ()).asr_threads == 6
    assert HardwareReport(True, False, 2, 31.0, "default", "x", ()).asr_threads == 1
    assert HardwareReport(True, False, 8, 31.0, "default", "x", ()).to_dict()["asr_threads"] == 6


def test_live_threads_are_half_the_cores():
    base = dict(avx2=True, avx512=False, ram_gib=31.0, tier="default", audio_server=None, gpus=())
    assert HardwareReport(physical_cores=8, **base).live_threads == 4
    assert HardwareReport(physical_cores=1, **base).live_threads == 1
    assert "live_threads" not in HardwareReport(physical_cores=8, **base).to_dict()
