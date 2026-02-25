#!/usr/bin/env bash
# Reproduce artifacts cited in the IEEE CiSE manuscript.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/paper"
STAMP="$(date +%Y-%m-%d_%H%M%S)"
ARTIFACT_DIR="$PAPER_DIR/artifacts/$STAMP"

mkdir -p "$ARTIFACT_DIR"

echo "[1/6] Build IEEE paper PDF"
(cd "$PAPER_DIR" && make ieee)
cp "$PAPER_DIR/sounio-ieee-cise.pdf" "$ARTIFACT_DIR/"

echo "[2/6] Run Python baseline benchmarks"
bash "$ROOT_DIR/benchmarks/comparison/run_all.sh" --python-only \
  >"$ARTIFACT_DIR/benchmarks_python.log" 2>&1 || true

echo "[3/6] Run Julia baseline benchmarks"
bash "$ROOT_DIR/benchmarks/comparison/run_all.sh" --julia-only \
  >"$ARTIFACT_DIR/benchmarks_julia.log" 2>&1 || true

echo "[4/6] Attempt Sounio benchmark run (captured for transparency)"
bash "$ROOT_DIR/benchmarks/comparison/run_all.sh" --sounio-only \
  >"$ARTIFACT_DIR/benchmarks_sounio.log" 2>&1 || true

echo "[5/6] Generate Bateman convergence CSV"
python3 "$PAPER_DIR/scripts/bateman_convergence.py" \
  --output "$ARTIFACT_DIR/bateman_convergence.csv"

echo "[6/6] Check bibliography URL/DOI links"
python3 "$PAPER_DIR/scripts/check_references.py" "$PAPER_DIR/references.bib" \
  >"$ARTIFACT_DIR/reference_check.txt" 2>&1 || true

cat <<EOF
Reproducibility artifacts generated:
  $ARTIFACT_DIR
EOF
