#!/usr/bin/env python3
"""
Generate roofline point data (arithmetic intensity, achieved GFLOP/s) for the
octonion matmul Criterion benchmarks, with optional f32 baseline comparison.

This intentionally has *no* third-party Python dependencies so it can run in
minimal environments.  Matplotlib is imported lazily and only when --plot-png
is requested.

Input:
  target/criterion/octonion_matmul/matmul_{N}x{N}/new/estimates.json
  target/criterion/f32_matmul_baseline/matmul_{N}x{N}/new/estimates.json  (optional)

Output CSV columns:
  type,n,time_ns,gflops,intensity_conservative,intensity_ideal,flops,bytes_conservative,bytes_ideal

Arithmetic model assumptions — octonion:
  - Octonion multiply: 120 FLOPs (as documented in benches/compiler/octonion_benchmark.rs)
  - Octonion add: 8 FLOPs
  - Naive matmul implementation does, per inner-loop iteration:
      sum = sum + (a * b)
    i.e. 1 mul + 1 add = 128 FLOPs per k
  - Total FLOPs ~= 128 * N^3

Memory traffic bounds — octonion (bytes moved):
  - Conservative (naive / no reuse): for each multiply, load 2 octonions (64 B),
    plus write the output matrix once (32 B per output).
      bytes = 64*N^3 + 32*N^2
  - Ideal (perfect reuse): read A once, read B once, write C once.
      bytes = 96*N^2

Arithmetic model assumptions — f32:
  - 1 fmul + 1 fadd per inner iteration = 2 FLOPs per k
  - Total FLOPs = 2 * N^3

Memory traffic bounds — f32 (bytes moved):
  - Conservative (naive / no reuse): load 2 floats per inner iteration
    (8 B per k), plus write output once (4 B per output).
      bytes = 8*N^3 + 4*N^2
  - Ideal (perfect reuse): read A once, read B once, write C once.
    Each matrix is N^2 elements * 4 bytes.
      bytes = 12*N^2

These are estimates to position points on a roofline plot; the manuscript should
state the assumptions (or replace them with profiler-derived traffic).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


MATMUL_DIR_RE = re.compile(r"^matmul_(\d+)x(\d+)$")

# Default hardware ceilings (from paper TeX / Reviewer #2 discussion).
DEFAULT_PEAK_COMPUTE_GFLOPS = 134.0
DEFAULT_PEAK_BANDWIDTH_GBS = 25.0


@dataclass(frozen=True)
class MatmulPoint:
    kind: str  # "octonion" or "f32"
    n: int
    time_ns: float
    flops: int
    bytes_conservative: int
    bytes_ideal: int

    @property
    def gflops(self) -> float:
        # FLOPs / ns == GFLOP/s
        return float(self.flops) / float(self.time_ns)

    @property
    def intensity_conservative(self) -> float:
        return float(self.flops) / float(self.bytes_conservative)

    @property
    def intensity_ideal(self) -> float:
        return float(self.flops) / float(self.bytes_ideal)


def iter_estimates(criterion_dir: Path) -> Iterable[MatmulPoint]:
    """Yield roofline points for *octonion* matmul Criterion output."""
    for child in sorted(criterion_dir.iterdir()):
        if not child.is_dir():
            continue
        m = MATMUL_DIR_RE.match(child.name)
        if not m:
            continue
        n1 = int(m.group(1))
        n2 = int(m.group(2))
        if n1 != n2:
            # We currently only handle square matmul ids.
            continue

        estimates_path = child / "new" / "estimates.json"
        if not estimates_path.exists():
            continue

        with estimates_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        time_ns = float(data["mean"]["point_estimate"])

        # FLOPs = (120 mul + 8 add) * N^3
        flops = 128 * (n1**3)

        bytes_conservative = (64 * (n1**3)) + (32 * (n1**2))
        bytes_ideal = 96 * (n1**2)

        yield MatmulPoint(
            kind="octonion",
            n=n1,
            time_ns=time_ns,
            flops=flops,
            bytes_conservative=bytes_conservative,
            bytes_ideal=bytes_ideal,
        )


def iter_f32_estimates(criterion_dir: Path) -> Iterable[MatmulPoint]:
    """Yield roofline points for *f32* matmul Criterion output.

    FLOP model:  2 FLOPs per inner iteration (1 fmul + 1 fadd).
                 Total = 2 * N^3.

    Memory traffic:
      Conservative (naive): 8*N^3 + 4*N^2 bytes.
      Ideal (perfect reuse): 12*N^2 bytes.
    """
    for child in sorted(criterion_dir.iterdir()):
        if not child.is_dir():
            continue
        m = MATMUL_DIR_RE.match(child.name)
        if not m:
            continue
        n1 = int(m.group(1))
        n2 = int(m.group(2))
        if n1 != n2:
            continue

        estimates_path = child / "new" / "estimates.json"
        if not estimates_path.exists():
            continue

        with estimates_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        time_ns = float(data["mean"]["point_estimate"])

        flops = 2 * (n1**3)

        bytes_conservative = (8 * (n1**3)) + (4 * (n1**2))
        bytes_ideal = 12 * (n1**2)

        yield MatmulPoint(
            kind="f32",
            n=n1,
            time_ns=time_ns,
            flops=flops,
            bytes_conservative=bytes_conservative,
            bytes_ideal=bytes_ideal,
        )


def write_csv(
    points: list[MatmulPoint],
    out_csv: Path,
    peak_compute: float,
    peak_bandwidth: float,
) -> None:
    """Write roofline data to CSV with header comments for machine ceilings."""
    ridge_point = peak_compute / peak_bandwidth

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        # Header comments with machine parameters.
        f.write(f"# peak_compute_gflops={peak_compute}\n")
        f.write(f"# peak_bandwidth_gbs={peak_bandwidth}\n")
        f.write(f"# ridge_point={ridge_point:.2f}\n")

        w = csv.writer(f)
        w.writerow(
            [
                "type",
                "n",
                "time_ns",
                "gflops",
                "intensity_conservative",
                "intensity_ideal",
                "flops",
                "bytes_conservative",
                "bytes_ideal",
            ]
        )
        for p in points:
            w.writerow(
                [
                    p.kind,
                    p.n,
                    f"{p.time_ns:.6f}",
                    f"{p.gflops:.6f}",
                    f"{p.intensity_conservative:.6f}",
                    f"{p.intensity_ideal:.6f}",
                    p.flops,
                    p.bytes_conservative,
                    p.bytes_ideal,
                ]
            )


def plot_roofline(
    points: list[MatmulPoint],
    out_png: Path,
    peak_compute: float,
    peak_bandwidth: float,
) -> bool:
    """Generate a log-log roofline plot.  Returns True on success.

    Matplotlib is imported lazily so the script stays zero-dep when
    --plot-png is not used.
    """
    try:
        import matplotlib  # noqa: F811

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "WARNING: matplotlib is not installed; skipping roofline plot.",
            file=sys.stderr,
        )
        return False

    ridge_point = peak_compute / peak_bandwidth

    # Separate by kind.
    octonion = [p for p in points if p.kind == "octonion"]
    f32 = [p for p in points if p.kind == "f32"]

    # Determine x-axis range from data (with some padding).
    all_intensities = (
        [p.intensity_conservative for p in points]
        + [p.intensity_ideal for p in points]
    )
    if not all_intensities:
        print("WARNING: no data points to plot.", file=sys.stderr)
        return False

    x_min = min(all_intensities) * 0.5
    x_max = max(max(all_intensities) * 2.0, ridge_point * 2.0)

    fig, ax = plt.subplots(figsize=(10, 7))

    # --- Roofline ceilings ---------------------------------------------------
    # Bandwidth ceiling: perf = bandwidth * intensity (diagonal line).
    # Extends from x_min to the ridge point.
    import numpy as np

    xs_bw = np.logspace(np.log10(x_min), np.log10(ridge_point), 200)
    ax.plot(
        xs_bw,
        peak_bandwidth * xs_bw,
        color="black",
        linewidth=2,
        label=f"BW ceiling ({peak_bandwidth} GB/s)",
    )

    # Compute ceiling: horizontal line from ridge point onward.
    xs_compute = np.logspace(np.log10(ridge_point), np.log10(x_max), 200)
    ax.plot(
        xs_compute,
        np.full_like(xs_compute, peak_compute),
        color="black",
        linewidth=2,
        label=f"Compute ceiling ({peak_compute} GFLOP/s)",
    )

    # Ridge point marker.
    ax.plot(
        ridge_point,
        peak_compute,
        marker="D",
        color="black",
        markersize=8,
        zorder=5,
    )
    ax.annotate(
        f"ridge = {ridge_point:.2f}",
        xy=(ridge_point, peak_compute),
        xytext=(ridge_point * 1.3, peak_compute * 1.15),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="grey"),
    )

    # --- Data points ----------------------------------------------------------
    # We plot conservative-intensity points (the more pessimistic bound) as the
    # primary markers; ideal-intensity as lighter hollow markers for context.

    def _scatter(
        subset: list[MatmulPoint],
        marker: str,
        color: str,
        label: str,
    ) -> None:
        if not subset:
            return
        # Conservative intensity.
        xs_cons = [p.intensity_conservative for p in subset]
        ys = [p.gflops for p in subset]
        ax.scatter(
            xs_cons, ys, marker=marker, color=color, s=80, zorder=4,
            label=f"{label} (conservative)",
        )
        # Label each point with N.
        for p in subset:
            ax.annotate(
                f"N={p.n}",
                xy=(p.intensity_conservative, p.gflops),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )

        # Ideal intensity (hollow markers, smaller).
        xs_ideal = [p.intensity_ideal for p in subset]
        ax.scatter(
            xs_ideal, ys, marker=marker, facecolors="none", edgecolors=color,
            s=50, zorder=3, alpha=0.6,
            label=f"{label} (ideal)",
        )

    _scatter(octonion, "o", "tab:blue", "Octonion")
    _scatter(f32, "s", "tab:red", "Float32")

    # --- Formatting -----------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("Roofline Analysis: Octonion vs Float32 MatMul")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate roofline data for octonion (and optionally f32) matmul benchmarks.",
    )
    ap.add_argument(
        "--criterion-dir",
        required=True,
        type=Path,
        help="Path to Criterion group dir, e.g. target/criterion/octonion_matmul",
    )
    ap.add_argument(
        "--f32-criterion-dir",
        required=False,
        type=Path,
        default=None,
        help="Path to f32 baseline Criterion dir, e.g. target/criterion/f32_matmul_baseline",
    )
    ap.add_argument(
        "--out-csv",
        required=True,
        type=Path,
        help="Output CSV path, e.g. docs/compiler/figures/octonion_matmul_points.csv",
    )
    ap.add_argument(
        "--peak-compute",
        type=float,
        default=DEFAULT_PEAK_COMPUTE_GFLOPS,
        help=f"Peak compute in GFLOP/s (default: {DEFAULT_PEAK_COMPUTE_GFLOPS})",
    )
    ap.add_argument(
        "--peak-bandwidth",
        type=float,
        default=DEFAULT_PEAK_BANDWIDTH_GBS,
        help=f"Peak memory bandwidth in GB/s (default: {DEFAULT_PEAK_BANDWIDTH_GBS})",
    )
    ap.add_argument(
        "--plot-png",
        type=Path,
        default=None,
        help="If given, generate a roofline PNG plot at this path (requires matplotlib)",
    )
    args = ap.parse_args()

    criterion_dir: Path = args.criterion_dir
    f32_criterion_dir: Optional[Path] = args.f32_criterion_dir
    out_csv: Path = args.out_csv
    peak_compute: float = args.peak_compute
    peak_bandwidth: float = args.peak_bandwidth
    plot_png: Optional[Path] = args.plot_png

    if not criterion_dir.exists():
        raise SystemExit(f"criterion dir not found: {criterion_dir}")

    points: list[MatmulPoint] = sorted(
        iter_estimates(criterion_dir), key=lambda p: p.n,
    )
    if not points:
        raise SystemExit(
            f"no matmul estimates found under {criterion_dir} "
            f"(expected matmul_*x*/new/estimates.json)"
        )

    # Optionally collect f32 baselines.
    if f32_criterion_dir is not None:
        if not f32_criterion_dir.exists():
            raise SystemExit(f"f32 criterion dir not found: {f32_criterion_dir}")
        f32_points = sorted(
            iter_f32_estimates(f32_criterion_dir), key=lambda p: p.n,
        )
        if not f32_points:
            print(
                f"WARNING: no f32 estimates found under {f32_criterion_dir}",
                file=sys.stderr,
            )
        else:
            points.extend(f32_points)

    write_csv(points, out_csv, peak_compute, peak_bandwidth)
    print(f"Wrote {len(points)} points -> {out_csv}")

    if plot_png is not None:
        if plot_roofline(points, plot_png, peak_compute, peak_bandwidth):
            print(f"Wrote roofline plot -> {plot_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
