"""Hardware detection and model tier selection (spec section 9.3)."""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A 16 GB machine reports about 15.5 GiB, so the limit is 15.
DEFAULT_TIER_MIN_GIB = 15.0
LOW_TIER_MIN_GIB = 8.0

Run = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class CpuInfo:
    avx2: bool
    avx512: bool
    physical_cores: int


def parse_cpuinfo(text: str) -> CpuInfo:
    flags: set[str] = set()
    cores: set[tuple[str, str]] = set()
    phys = core = None
    for line in text.splitlines() + [""]:
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "flags" and not flags:
            flags = set(value.split())
        elif key == "physical id":
            phys = value
        elif key == "core id":
            core = value
        elif not line.strip():
            if phys is not None and core is not None:
                cores.add((phys, core))
            phys = core = None
    return CpuInfo(
        avx2="avx2" in flags,
        avx512="avx512f" in flags,
        physical_cores=len(cores) or (os.cpu_count() or 1),
    )


def parse_meminfo(text: str) -> float:
    m = re.search(r"^MemTotal:\s+(\d+) kB", text, re.M)
    if not m:
        raise ValueError("MemTotal not found in meminfo")
    return int(m.group(1)) / (1024 * 1024)


def select_tier(ram_gib: float) -> str | None:
    if ram_gib >= DEFAULT_TIER_MIN_GIB:
        return "default"
    if ram_gib >= LOW_TIER_MIN_GIB:
        return "low"
    return None


def _run(run: Run, cmd: list[str], timeout: float) -> str | None:
    try:
        out = run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def detect_audio(run: Run = subprocess.run) -> str | None:
    """Return the name of the PulseAudio-compatible server, or None.

    The audio plan replaces this check with libpulse.
    """
    out = _run(run, ["pactl", "info"], timeout=3)
    if out is None:
        return None
    m = re.search(r"^Server Name:\s*(.+)$", out, re.M)
    return m.group(1).strip() if m else None


def detect_gpus(run: Run = subprocess.run) -> list[str]:
    """Return the names of Vulkan devices that are not CPU devices."""
    out = _run(run, ["vulkaninfo", "--summary"], timeout=10)
    if out is None:
        return []
    gpus: list[str] = []
    device_type = ""
    for line in out.splitlines():
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key == "deviceType":
            device_type = value
        elif key == "deviceName" and device_type != "PHYSICAL_DEVICE_TYPE_CPU":
            gpus.append(value)
    return gpus


@dataclass(frozen=True)
class HardwareReport:
    avx2: bool
    avx512: bool
    physical_cores: int
    ram_gib: float
    tier: str | None
    audio_server: str | None
    gpus: tuple[str, ...]

    @property
    def threads(self) -> int:
        return max(1, self.physical_cores - 1)

    @property
    def problems(self) -> list[str]:
        out = []
        if not self.avx2:
            out.append("This CPU does not support AVX2. Lookout needs a CPU with AVX2.")
        if self.tier is None:
            out.append(f"This computer has {self.ram_gib:.1f} GiB of RAM. Lookout needs 8 GiB or more.")
        return out

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.tier == "low":
            out.append(
                f"This computer has {self.ram_gib:.1f} GiB of RAM. Lookout uses the low model tier. "
                "Quality and speed are lower."
            )
        if self.audio_server is None:
            out.append("No PulseAudio-compatible audio server found. Recording is not available.")
        return out

    @property
    def fatal(self) -> bool:
        return bool(self.problems)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["gpus"] = list(self.gpus)
        d["threads"] = self.threads
        d["problems"] = self.problems
        d["warnings"] = self.warnings
        return d


def detect(
    cpuinfo: Path = Path("/proc/cpuinfo"),
    meminfo: Path = Path("/proc/meminfo"),
    run: Run = subprocess.run,
) -> HardwareReport:
    cpu = parse_cpuinfo(cpuinfo.read_text())
    ram = parse_meminfo(meminfo.read_text())
    return HardwareReport(
        avx2=cpu.avx2,
        avx512=cpu.avx512,
        physical_cores=cpu.physical_cores,
        ram_gib=round(ram, 1),
        tier=select_tier(ram),
        audio_server=detect_audio(run),
        gpus=tuple(detect_gpus(run)),
    )
