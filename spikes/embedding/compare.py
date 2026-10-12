"""Print the comparison table of all spike results.

Usage: uv run --with numpy python compare.py
"""

import json
from pathlib import Path

import numpy as np

from spike_common import RESULTS, agreement

COLUMNS = [
    ("name", "Result"),
    ("recall_at_1", "R@1"),
    ("recall_at_3", "R@3"),
    ("mrr", "MRR"),
    ("agree_mean", "Cos mean"),
    ("agree_min", "Cos min"),
    ("long_chunks_per_s", "Chunks/s"),
    ("load_s", "Load s"),
    ("peak_rss_mb", "RSS MB"),
    ("image", "Images"),
    ("error", "Error"),
]


def rows(results_dir: Path = RESULTS) -> list[dict]:
    ref_path = results_dir / "reference.npy"
    ref = np.load(ref_path) if ref_path.exists() else None
    out = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        row = {"name": path.stem, **data}
        npy = path.with_suffix(".npy")
        if ref is not None and npy.exists():
            row["agree_mean"], row["agree_min"] = agreement(np.load(npy), ref)
        image = data.get("image")
        if isinstance(image, dict):
            row["image"] = f"{image['correct']}/{image['total']}" if image["supported"] else "no"
        out.append(row)
    return out


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main():
    table = rows()
    print("| " + " | ".join(h for _, h in COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in table:
        print("| " + " | ".join(fmt(r.get(k)) for k, _ in COLUMNS) + " |")


if __name__ == "__main__":
    main()
