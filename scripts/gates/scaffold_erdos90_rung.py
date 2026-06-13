#!/usr/bin/env python3
"""Scaffold grid/subset export, cluster, gates, and Lean generators for one ladder rung."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

RUNGS = [
    (441, 21, 1896),
    (484, 22, 2116),
    (529, 23, 2348),
    (576, 24, 2592),
    (625, 25, 2848),
    (676, 26, 3116),
    (729, 27, 3396),
    (784, 28, 3688),
    (841, 29, 3992),
    (900, 30, 4308),
    (961, 31, 4636),
]


def replace_n(text: str, n: int, w: int, grid: int) -> str:
    """Token-safe renames from the n=400 template (no bare 400/1688 global replace)."""
    wxw = f"{w}x{w}"
    pairs = [
        ("BEST_N400", f"BEST_N{n}"),
        ("CLUSTER400", f"CLUSTER{n}"),
        ("BEAT-GRID1688", f"BEAT-GRID{grid}"),
        ("grid20x20=1688", f"grid{wxw}={grid}"),
        ("grid20x20", f"grid{wxw}"),
        ("20×20", wxw),
        ("20x20", wxw),
        ("build_grid(20)", f"build_grid({w})"),
        ("if GN < 400", f"if GN < {n}"),
        ("if WN < 400", f"if WN < {n}"),
        ("WN < 400", f"WN < {n}"),
        ("[i64; 400]", f"[i64; {n}]"),
        ("[0; 400]", f"[0; {n}]"),
        ("track_best_n400", f"track_best_n{n}"),
        ("erdos90-400-runs", f"erdos90-{n}-runs"),
        ("erdos90-400-", f"erdos90-{n}-"),
        ("subset400", f"subset{n}"),
        ("grid400", f"grid{n}"),
        ("e90_400", f"e90_{n}"),
        ("witness400", f"witness{n}"),
        ("grid_n400", f"grid_n{n}"),
        ("subset_n400", f"subset_n{n}"),
        ("u400_", f"u{n}_"),
        ("Grid400", f"Grid{n}"),
        ("Subset400", f"Subset{n}"),
        ("at n=400", f"at n={n}"),
        ("n=400", f"n={n}"),
        ("u(400)", f"u({n})"),
        ("harb 400", f"harb {n}"),
        (": 1688 <", f": {grid} <"),
        ("TARGET_N: i64 = 400", f"TARGET_N: i64 = {n}"),
        ("expected 400 points", f"expected {n} points"),
        ("len(pts) != 400", f"len(pts) != {n}"),
        ("len(set(pts)) != 400", f"len(set(pts)) != {n}"),
    ]
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def scaffold(n: int, w: int, grid: int) -> None:
    wxw = f"{w}x{w}"
    # grid export already created for batch; ensure correct
    grid_src = ROOT / "stdlib/research/erdos90_grid400_export.sio"
    grid_out = ROOT / f"stdlib/research/erdos90_grid{n}_export.sio"
    grid_out.write_text(replace_n(grid_src.read_text(), n, w, grid))

    for src_name, dst_name in [
        ("erdos90_subset400_cluster.sio", f"erdos90_subset{n}_cluster.sio"),
        ("erdos90_subset400_export.sio", f"erdos90_subset{n}_export.sio"),
        ("submit_subset400_array.sh", f"submit_subset{n}_array.sh"),
        ("gen_lean_grid400_witness.py", f"gen_lean_grid{n}_witness.py"),
        ("gen_lean_subset400_witness.py", f"gen_lean_subset{n}_witness.py"),
    ]:
        src = ROOT / ("slurm-jobs/erdos90" if src_name.endswith(".sh") else "scripts/gates" if src_name.endswith(".py") else "stdlib/research") / src_name
        if src_name.endswith(".sio"):
            src = ROOT / "stdlib/research" / src_name
        dst_dir = src.parent if dst_name == src_name else (
            ROOT / "slurm-jobs/erdos90" if dst_name.endswith(".sh") else
            ROOT / "scripts/gates" if dst_name.endswith(".py") else
            ROOT / "stdlib/research"
        )
        dst = dst_dir / dst_name
        text = replace_n(src.read_text(), n, w, grid)
        if dst_name.endswith(".py"):
            hb = max(1_000_000, n * 2200)
            text = text.replace("maxHeartbeats 1000000", f"maxHeartbeats {hb}")
        dst.write_text(text)
        if dst_name.endswith(".sh") or dst_name.endswith(".py"):
            dst.chmod(0o755)

    grid_gate = ROOT / f"scripts/gates/erdos90_grid{n}_witness_gate.sh"
    subset_gate = ROOT / f"scripts/gates/erdos90_subset{n}_witness_gate.sh"
    grid_gate.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${{SOUC:-$ROOT/bin/souc}}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
MIN_EDGES="${{MIN_EDGES:-{grid}}}"

"$SOUC" stdlib/research/erdos90_grid{n}_export.sio "$WORK/export.elf" >/dev/null
chmod +x "$WORK/export.elf"
"$WORK/export.elf" >"$WORK/witness.log" 2>&1
grep -q "EXPORT_DONE" "$WORK/witness.log"
edges="$(grep 'edges=' "$WORK/witness.log" | head -1 | sed 's/.*edges=//')"
[ "$edges" -ge "$MIN_EDGES" ]
python3 scripts/gates/gen_lean_grid{n}_witness.py "$WORK/witness.log" formal/lean4/SounioErdos90Grid{n}Witness.lean
(cd formal/lean4 && lake env lean SounioErdos90Grid{n}Witness.lean) >/dev/null
echo "[erdos90-grid{n}-witness] PASS (countGridUnit25 = $edges)"
""")
    subset_gate.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${{SOUC:-$ROOT/bin/souc}}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
MIN_EDGES="${{MIN_EDGES:-{grid + 1}}}"

"$SOUC" stdlib/research/erdos90_subset{n}_export.sio "$WORK/export.elf" >/dev/null
chmod +x "$WORK/export.elf"
"$WORK/export.elf" >"$WORK/witness.log" 2>&1
grep -q "EXPORT_DONE" "$WORK/witness.log"
edges="$(grep 'edges=' "$WORK/witness.log" | head -1 | sed 's/.*edges=//')"
[ "$edges" -ge "$MIN_EDGES" ]
python3 scripts/gates/gen_lean_subset{n}_witness.py "$WORK/witness.log" formal/lean4/SounioErdos90Subset{n}Witness.lean
(cd formal/lean4 && lake env lean SounioErdos90Subset{n}Witness.lean) >/dev/null
echo "[erdos90-subset{n}-witness] PASS (countGridUnit25 = $edges)"
""")
    grid_gate.chmod(0o755)
    subset_gate.chmod(0o755)
    print(f"scaffolded n={n} ({wxw}, grid={grid})")


def main() -> int:
    rungs = RUNGS
    if len(sys.argv) > 1:
        rungs = [r for r in RUNGS if str(r[0]) in sys.argv[1:]]
    for n, w, grid in rungs:
        scaffold(n, w, grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())