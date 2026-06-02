#!/usr/bin/env python3
"""Generate comparison visualizations for the pipeline benchmark."""

import json
import os
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib not available, skipping visualization")
    sys.exit(0)

ROOT = Path(__file__).parent.resolve()
OUT_DIR = ROOT / "output" / "pipeline_comparison"


def load_results():
    path = OUT_DIR / "benchmark_results.json"
    if not path.exists():
        print(f"No results at {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def plot_comparison(results: list):
    scenes = {}
    for r in results:
        s = r["scenario"]
        if s not in scenes:
            scenes[s] = []
        scenes[s].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (scene, runs) in zip(axes, sorted(scenes.items())):
        valid = [r for r in runs if r.get("wall_ms") is not None and r.get("wall_ms") > 0]
        if not valid:
            continue

        # Separate B=32 and B=75 experiments
        b32 = [r for r in valid if r.get("avg_batch_size", 0) < 60]
        b75 = [r for r in valid if r.get("avg_batch_size", 0) >= 60]

        all_walls = [r["wall_ms"] for r in valid]
        all_labels = [r["mode"] for r in valid]

        # Use first run's speedup as baseline
        serial_wall = next((r["wall_ms"] for r in valid if "serial" in r["mode"]), None)
        serial_b75 = next((r["wall_ms"] for r in b75 if "serial" in r["mode"]), None)

        if not all_walls:
            continue

        colors = {"serial": "#e74c3c", "async_plain": "#f39c12", "async_v2": "#27ae60", "async_v2_fixed": "#9b59b6"}

        # Plot B=32 group
        ax2 = ax
        if b32:
            labels32 = [r["mode"] for r in b32]
            walls32 = [r["wall_ms"] for r in b32]
            speeds32 = [serial_wall / w if serial_wall else 1.0 for w in walls32]
            x32 = range(len(b32))
            bars32 = ax2.bar(
                [x - 0.2 for x in x32], walls32,
                width=0.35, color=[colors.get(m, "#95a5a6") for m in labels32],
                alpha=0.8, label="B=32" if ax == axes[0] else None
            )
            for bar, spd in zip(bars32, speeds32):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                        f"{spd:.2f}x", ha="center", va="bottom", fontsize=8)

        # Plot B=75 group
        if b75:
            labels75 = [r["mode"] for r in b75]
            walls75 = [r["wall_ms"] for r in b75]
            speeds75 = [serial_wall / w if serial_wall else 1.0 for w in walls75]
            x75 = range(len(b75))
            bars75 = ax2.bar(
                [x + 0.2 for x in x75], walls75,
                width=0.35, color=[colors.get(m, "#95a5a6") for m in labels75],
                alpha=0.8, hatch="//"
            )
            for bar, spd in zip(bars75, speeds75):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                        f"{spd:.2f}x", ha="center", va="bottom", fontsize=8)

        all_labels_clean = [l.replace("_fixed", " (fixed)").replace("async_", "async-\n") for l in all_labels]
        ax2.set_xticks(range(len(all_labels)))
        ax2.set_xticklabels(all_labels_clean, fontsize=9)
        ax2.set_ylabel("Wall Time (ms)")
        ax2.set_title(f"Scene {scene}")
        ax2.set_ylim(0, max(all_walls) * 1.2)

        # Legend
        import matplotlib.patches as mpatches
        patch32 = mpatches.Patch(facecolor="#95a5a6", alpha=0.8, label="B=32")
        patch75 = mpatches.Patch(facecolor="#95a5a6", alpha=0.8, hatch="//", label="B=75")
        serial_patch = mpatches.Patch(facecolor=colors["serial"], alpha=0.8, label="serial")
        plain_patch = mpatches.Patch(facecolor=colors["async_plain"], alpha=0.8, label="async_plain")
        v2_patch = mpatches.Patch(facecolor=colors["async_v2"], alpha=0.8, label="async_v2")
        ax2.legend(handles=[patch32, patch75, serial_patch, plain_patch, v2_patch],
                  loc="upper right", fontsize=8)

    fig.suptitle("Async RAG Pipeline — Mode Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "pipeline_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


def plot_batch_size_impact(results: list):
    """Show how wall time decreases with batch size."""
    fig, ax = plt.subplots(figsize=(7, 4))

    # Data from benchmark
    # serial B=32 S1, serial B=75 S1, serial B=32 S5, serial B=75 S5
    data = {
        "S1 serial": [(32, 19179.9), (75, 11870.3)],
        "S5 serial": [(32, 17329.2), (75, 11870.3)],
    }

    markers = ["o", "s"]
    for (label, pts), marker in zip(data.items(), markers):
        bs = [p[0] for p in pts]
        walls = [p[1] for p in pts]
        ax.plot(bs, walls, marker=marker, label=label, linewidth=2, markersize=8)

    # Theory curve: wall ≈ gen_base/B + gen_per_token * avg_out + const
    gen_base = 1109
    gen_pt = 25.62
    const = 0.5  # emb + ret overhead + queue
    theory_B = [1, 8, 16, 32, 64, 75, 128]
    theory_walls = [gen_base/b + gen_pt * 120 + const for b in theory_B]
    ax.plot(theory_B, theory_walls, "k--", alpha=0.5, label="Theory: 1109/B + 3074")

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Wall Time (ms)")
    ax.set_title("Prefill Amortization: Larger Batch → Lower Per-Query Cost")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 8, 16, 32, 64, 75, 128])
    ax.set_xticklabels(["1", "8", "16", "32", "64", "75", "128"])
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = OUT_DIR / "batch_size_impact.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


def main():
    results = load_results()
    plot_comparison(results)
    plot_batch_size_impact(results)


if __name__ == "__main__":
    main()
