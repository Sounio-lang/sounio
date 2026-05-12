#!/usr/bin/env python3
"""
Sub-PTX B2: cohort-scale ORC sweep across the entire local ABIDE-I CC200
collection (1035 .1D subjects). Each subject is processed by the same
pipeline shipped in `scripts/research/abide_orc_sweep.py`:

  read .1D -> 200x200 Pearson correlation -> directed k-NN connectome
  (k=15 -> 3000 edges) -> 16-element neighborhoods -> per-edge K matrix
  -> chunked GPU launches (12 chunks x 256 threads on the Sounio
  Sinkhorn-LSE N=16 PTX kernel) -> transport-plan reconstruction ->
  Wasserstein-1 -> ORC matrix -> save artifacts/research/abide_orc/<subject>_orc.npy

Resume capability: subjects whose `<subject>_orc.npy` already exists
are skipped, so re-runs after interruption pick up where they left off.

After the sweep, the script joins per-subject summary statistics with
the cohort phenotype manifest (subject_id / label / site) and prints
per-site and per-label distributional summaries. The per-subject
summaries are also written to a TSV for downstream analysis.

Usage:
    python3 scripts/research/abide_cohort_orc_sweep.py \\
        [--limit 50] \\
        [--k 15] \\
        [--lambda 1.0] \\
        [--ptx /tmp/sinkhorn16iter.ptx] \\
        [--runner /tmp/kaxi_runner] \\
        [--report artifacts/research/abide_orc/cohort_summary.tsv]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
ABIDE_DIR = ROOT / "artifacts" / "research" / "abide"
OUT_DIR = ROOT / "artifacts" / "research" / "abide_orc"
MANIFEST = ROOT / "artifacts" / "research" / "brain_ossm_full_cohort" / "abide_roi_manifest.tsv"
N = 16
EDGE_SLOTS = 320
CHUNK = 256


def subject_from_path(path: Path) -> str:
    """CMU_a_0050642_rois_cc200.1D -> CMU_a_0050642."""
    name = path.name
    if name.endswith("_rois_cc200.1D"):
        return name[: -len("_rois_cc200.1D")]
    return path.stem


def subject_to_manifest_id(subject: str) -> int | None:
    """CMU_a_0050642 -> 50642 (drop site prefix, strip leading zeros).

    Manifest uses bare numeric subject_id (e.g. 50952). The .1D filename
    is `<SITE>_<NUMERIC>_rois_cc200.1D` where NUMERIC is zero-padded.
    """
    m = re.search(r"_(\d{5,})$", subject)
    if not m:
        # CMU_a/CMU_b have an extra letter (CMU_a_0050642). Try a more
        # general split.
        tail = subject.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail)
        return None
    return int(m.group(1))


def read_subject_ts(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def build_correlation(ts: np.ndarray) -> np.ndarray:
    std = ts.std(axis=0)
    zero_cols = std < 1e-12
    if zero_cols.any():
        rng = np.random.default_rng(42)
        ts = ts.copy()
        ts[:, zero_cols] = rng.standard_normal(
            (ts.shape[0], int(zero_cols.sum()))
        ) * 1e-3
    return np.corrcoef(ts, rowvar=False)


def build_neighborhood(corr: np.ndarray, node: int, k: int = N - 1) -> np.ndarray:
    row = corr[node].copy()
    row[node] = -np.inf
    partners = np.argpartition(row, -k)[-k:]
    partners = partners[np.argsort(-row[partners])]
    return np.concatenate(([node], partners)).astype(np.int64)


def compute_subject_orc(ts: np.ndarray, *, k: int, lam: float,
                        ptx: str, runner: str) -> tuple[np.ndarray, float]:
    """Compute the full directed kNN ORC matrix for one subject.

    Returns (orc_matrix, gpu_seconds). orc_matrix is (R, R) float64 with
    NaN at non-edge positions.
    """
    R = ts.shape[1]
    corr = build_correlation(ts)

    # Build kNN graph.
    neighbors = np.zeros((R, k), dtype=np.int64)
    for u in range(R):
        row = corr[u].copy()
        row[u] = -np.inf
        top = np.argpartition(row, -k)[-k:]
        top = top[np.argsort(-row[top])]
        neighbors[u] = top
    n_edges = R * k

    # Per-node 16-element neighborhoods (reused across all edges from u).
    neigh = np.zeros((R, N), dtype=np.int64)
    for u in range(R):
        neigh[u] = build_neighborhood(corr, u, k=N - 1)

    la = np.full(N, np.log2(1.0 / N), dtype=np.float32)
    lb = la.copy()

    inv_lam_ln2 = 1.0 / (lam * np.log(2.0))
    batch_input = np.zeros((n_edges, EDGE_SLOTS), dtype=np.float32)
    cost_store = np.zeros((n_edges, N, N), dtype=np.float64)
    edge_index = np.zeros((n_edges, 2), dtype=np.int64)

    eid = 0
    for u in range(R):
        nu = neigh[u]
        for vi in range(k):
            v = neighbors[u, vi]
            nv = neigh[v]
            C = 1.0 - corr[np.ix_(nu, nv)]
            K_log = (-C * inv_lam_ln2).astype(np.float32)
            batch_input[eid, 0:N] = la
            batch_input[eid, N:2 * N] = lb
            batch_input[eid, 2 * N:2 * N + N * N] = K_log.flatten()
            cost_store[eid] = C
            edge_index[eid] = (u, v)
            eid += 1
    assert eid == n_edges

    # Chunked launch: 12 x 256 threads on sm_89 (255 regs/thread cap).
    n_chunks = (n_edges + CHUNK - 1) // CHUNK
    values = np.zeros(n_edges * EDGE_SLOTS, dtype=np.float64)
    t_gpu = time.time()
    for c in range(n_chunks):
        lo, hi = c * CHUNK, min((c + 1) * CHUNK, n_edges)
        n = hi - lo
        chunk_buf = np.zeros((CHUNK, EDGE_SLOTS), dtype=np.float32)
        chunk_buf[:n] = batch_input[lo:hi]
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(chunk_buf.tobytes())
            bin_path = f.name
        try:
            proc = subprocess.run(
                [runner, ptx,
                 "--threads", str(CHUNK),
                 "--mem-words", str(CHUNK * EDGE_SLOTS),
                 "--init-file", bin_path,
                 "--type", "f32"],
                capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"runner failed at chunk {c} rc={proc.returncode}\n"
                    f"stdout: {proc.stdout[:500]}\nstderr: {proc.stderr[:500]}")
            mem_line = next((ln for ln in proc.stdout.splitlines()
                             if ln.startswith("MEM:")), None)
            if mem_line is None:
                raise RuntimeError(f"no MEM: line at chunk {c}\n{proc.stdout[:500]}")
            chunk_vals = np.array([float(x) for x in mem_line.split()[1:]],
                                  dtype=np.float64)
            expected = CHUNK * EDGE_SLOTS
            if chunk_vals.size != expected:
                raise RuntimeError(
                    f"chunk {c}: got {chunk_vals.size} values, expected {expected}")
            values[lo * EDGE_SLOTS:hi * EDGE_SLOTS] = chunk_vals[:n * EDGE_SLOTS]
        finally:
            os.unlink(bin_path)
    gpu_seconds = time.time() - t_gpu

    # Reconstruct transport plans and ORC.
    values = values.reshape(n_edges, EDGE_SLOTS)
    u_out = values[:, 288:304]
    v_out = values[:, 304:320]
    K_log_all = (-cost_store * inv_lam_ln2).astype(np.float64)
    arg = u_out[:, :, None] + K_log_all + v_out[:, None, :]
    P = np.exp2(arg)
    W = (P * cost_store).sum(axis=(1, 2))
    kappa = 1.0 - W

    orc_mat = np.full((R, R), np.nan, dtype=np.float64)
    for e in range(n_edges):
        u, v = edge_index[e]
        orc_mat[u, v] = kappa[e]
    return orc_mat, gpu_seconds


def load_manifest() -> dict[int, tuple[str, str]]:
    """Return {numeric_subject_id: (label, site)} from the phenotype manifest."""
    out: dict[int, tuple[str, str]] = {}
    if not MANIFEST.is_file():
        return out
    with open(MANIFEST) as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if header is None:
                header = fields
                continue
            try:
                sid = int(fields[0])
            except ValueError:
                continue
            label = fields[1]
            site = fields[2]
            out[sid] = (label, site)
    return out


def summary_stats(orc_mat: np.ndarray) -> dict[str, float]:
    """Edge-level summary statistics from a (200x200, NaN-padded) ORC matrix."""
    finite = orc_mat[np.isfinite(orc_mat)]
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "median": float("nan"), "frac_pos": float("nan"),
                "frac_neg": float("nan")}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "median": float(np.median(finite)),
        "frac_pos": float((finite > 0).sum() / finite.size),
        "frac_neg": float((finite < 0).sum() / finite.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--ptx", default="/tmp/sinkhorn16iter.ptx")
    ap.add_argument("--runner", default="/tmp/kaxi_runner")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most this many subjects (0 = all)")
    ap.add_argument("--report",
                    default=str(OUT_DIR / "cohort_summary.tsv"))
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    if not os.path.isfile(args.ptx):
        sys.exit(f"PTX not found: {args.ptx}")
    if not os.path.isfile(args.runner):
        sys.exit(f"runner not found: {args.runner}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(ABIDE_DIR.glob("*_rois_cc200.1D"))
    if args.limit > 0:
        files = files[:args.limit]
    print(f"== ABIDE-I cohort ORC sweep ==")
    print(f"subjects available: {len(files)}")
    print(f"output dir:         {OUT_DIR}")
    print(f"kernel:             {args.ptx}")
    print()

    manifest = load_manifest()
    print(f"phenotype manifest: {len(manifest)} labelled subjects "
          f"({MANIFEST.name if MANIFEST.exists() else 'MISSING'})")
    print()

    # Plan: classify each input as DONE / TODO.
    todo = []
    done_existing = 0
    for path in files:
        subject = subject_from_path(path)
        out_npy = OUT_DIR / f"{subject}_orc.npy"
        if out_npy.is_file():
            done_existing += 1
        else:
            todo.append(path)
    print(f"already computed:   {done_existing}")
    print(f"to compute:         {len(todo)}")
    print()

    t_cohort = time.time()
    failures: list[tuple[str, str]] = []
    completed = 0
    for i, path in enumerate(todo, 1):
        subject = subject_from_path(path)
        out_npy = OUT_DIR / f"{subject}_orc.npy"
        t_subj = time.time()
        try:
            ts = read_subject_ts(path)
            if ts.ndim != 2 or ts.shape[1] != 200:
                raise RuntimeError(f"unexpected shape {ts.shape}")
            orc_mat, gpu_s = compute_subject_orc(
                ts, k=args.k, lam=args.lam,
                ptx=args.ptx, runner=args.runner)
            np.save(out_npy, orc_mat)
            completed += 1
            dt = time.time() - t_subj
            if i <= 3 or i % args.progress_every == 0 or i == len(todo):
                print(f"[{i:4d}/{len(todo)}] {subject:30s} {dt:5.1f}s "
                      f"(gpu {gpu_s:.1f}s)  total {time.time() - t_cohort:.0f}s")
        except Exception as e:
            failures.append((subject, str(e)[:200]))
            print(f"[{i:4d}/{len(todo)}] {subject:30s} FAIL: {e}",
                  file=sys.stderr)

    print()
    print(f"sweep wall:  {time.time() - t_cohort:.1f}s  "
          f"({completed} new, {done_existing} cached, {len(failures)} failed)")
    print()

    # Per-subject summary statistics across all saved subjects.
    summary_rows: list[dict] = []
    for path in files:
        subject = subject_from_path(path)
        out_npy = OUT_DIR / f"{subject}_orc.npy"
        if not out_npy.is_file():
            continue
        try:
            orc = np.load(out_npy)
        except Exception:
            continue
        s = summary_stats(orc)
        sid = subject_to_manifest_id(subject)
        label, site = ("UNKNOWN", "UNKNOWN")
        if sid is not None and sid in manifest:
            label, site = manifest[sid]
        elif sid is not None:
            # Fall back: derive site from subject filename prefix.
            site_guess = subject.rsplit("_", 1)[0]
            site = site_guess
        row = {"subject": subject, "manifest_id": sid or -1,
               "label": label, "site": site, **s}
        summary_rows.append(row)

    cols = ["subject", "manifest_id", "label", "site",
            "n", "mean", "std", "min", "max", "median",
            "frac_pos", "frac_neg"]
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for row in summary_rows:
            f.write("\t".join(str(row[c]) for c in cols) + "\n")
    print(f"wrote per-subject summary: {report_path}  ({len(summary_rows)} rows)")
    print()

    # Per-site distributional ORC across labelled subjects.
    def fmt_group(rows: list[dict], name: str) -> str:
        if not rows:
            return f"  {name:18s}  (no subjects)"
        means = np.array([r["mean"] for r in rows], dtype=np.float64)
        stds = np.array([r["std"] for r in rows], dtype=np.float64)
        return (f"  {name:18s}  n={len(rows):4d}  "
                f"mean(mean)={means.mean():+.4f}  "
                f"sd(mean)={means.std():.4f}  "
                f"mean(std)={stds.mean():.4f}")

    labelled = [r for r in summary_rows if r["label"] != "UNKNOWN"]
    print(f"labelled subjects (joined with manifest): {len(labelled)}")
    print()
    sites = sorted({r["site"] for r in labelled})
    print("per-site (labelled only):")
    for site in sites:
        rows = [r for r in labelled if r["site"] == site]
        print(fmt_group(rows, site))
    print()
    print("per-label:")
    for label in sorted({r["label"] for r in labelled}):
        rows = [r for r in labelled if r["label"] == label]
        print(fmt_group(rows, label))
    print()

    # ASD vs TD on per-subject mean ORC (Cohen's d, simple).
    asd = [r for r in labelled if r["label"] == "ASD"]
    ctrl = [r for r in labelled if r["label"] in ("CONTROL", "TC", "TD")]
    if asd and ctrl:
        m_asd = np.array([r["mean"] for r in asd], dtype=np.float64)
        m_ctrl = np.array([r["mean"] for r in ctrl], dtype=np.float64)
        pooled = np.sqrt(((m_asd.std() ** 2) + (m_ctrl.std() ** 2)) / 2.0)
        d = (m_asd.mean() - m_ctrl.mean()) / pooled if pooled > 0 else float("nan")
        print(f"ASD vs CONTROL on per-subject mean ORC:")
        print(f"  ASD     n={len(asd):4d}  mean(mean)={m_asd.mean():+.4f}  sd(mean)={m_asd.std():.4f}")
        print(f"  CONTROL n={len(ctrl):4d}  mean(mean)={m_ctrl.mean():+.4f}  sd(mean)={m_ctrl.std():.4f}")
        print(f"  Cohen's d (per-subject mean ORC): {d:+.4f}")
    print()

    if failures:
        print(f"failures ({len(failures)}):")
        for s, err in failures[:20]:
            print(f"  {s}: {err}")
        sys.exit(2)


if __name__ == "__main__":
    main()
