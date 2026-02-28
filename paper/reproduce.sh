#!/usr/bin/env bash
# Reproduce artifacts cited in the IEEE CiSE manuscript.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/paper"
STAMP="$(date +%Y-%m-%d_%H%M%S)"
ARTIFACT_DIR="$PAPER_DIR/artifacts/$STAMP"

mkdir -p "$ARTIFACT_DIR"

run_logged_step() {
  local logfile="$1"
  shift
  if "$@" >"$logfile" 2>&1; then
    echo 0
  else
    echo $?
  fi
}

status_from_rc() {
  local rc="$1"
  if [[ "$rc" -eq 0 ]]; then
    echo "pass"
  else
    echo "fail"
  fi
}

echo "[1/6] Build IEEE paper PDF"
(cd "$PAPER_DIR" && make ieee)
cp "$PAPER_DIR/sounio-ieee-cise.pdf" "$ARTIFACT_DIR/"
PDF_RC=0

echo "[2/6] Run Python baseline benchmarks"
PYTHON_RC="$(run_logged_step "$ARTIFACT_DIR/benchmarks_python.log" \
  bash "$ROOT_DIR/benchmarks/comparison/run_all.sh" --python-only)"

echo "[3/6] Run Julia baseline benchmarks"
JULIA_RC="$(run_logged_step "$ARTIFACT_DIR/benchmarks_julia.log" \
  bash "$ROOT_DIR/benchmarks/comparison/run_all.sh" --julia-only)"

echo "[4/6] Attempt Sounio benchmark run (captured for transparency)"
SOUNIO_RC="$(run_logged_step "$ARTIFACT_DIR/benchmarks_sounio.log" \
  bash "$ROOT_DIR/benchmarks/comparison/run_all.sh" --sounio-only)"

echo "[5/6] Generate Bateman convergence CSV"
python3 "$PAPER_DIR/scripts/bateman_convergence.py" \
  --output "$ARTIFACT_DIR/bateman_convergence.csv"
BATEMAN_RC=0

echo "[6/6] Check bibliography URL/DOI links"
REFERENCE_RC="$(run_logged_step "$ARTIFACT_DIR/reference_check.txt" \
  python3 "$PAPER_DIR/scripts/check_references.py" "$PAPER_DIR/references.bib")"

cat >"$ARTIFACT_DIR/status_summary.v1.json" <<EOF
{
  "schema": "sounio.paper.reproduce.status.v1",
  "generated_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "artifact_dir": "$ARTIFACT_DIR",
  "steps": {
    "pdf_build": {
      "status": "$(status_from_rc "$PDF_RC")",
      "rc": $PDF_RC,
      "output": "sounio-ieee-cise.pdf"
    },
    "python_benchmark": {
      "status": "$(status_from_rc "$PYTHON_RC")",
      "rc": $PYTHON_RC,
      "output": "benchmarks_python.log"
    },
    "julia_benchmark": {
      "status": "$(status_from_rc "$JULIA_RC")",
      "rc": $JULIA_RC,
      "output": "benchmarks_julia.log"
    },
    "sounio_benchmark": {
      "status": "$(status_from_rc "$SOUNIO_RC")",
      "rc": $SOUNIO_RC,
      "output": "benchmarks_sounio.log"
    },
    "bateman_convergence": {
      "status": "$(status_from_rc "$BATEMAN_RC")",
      "rc": $BATEMAN_RC,
      "output": "bateman_convergence.csv"
    },
    "reference_check": {
      "status": "$(status_from_rc "$REFERENCE_RC")",
      "rc": $REFERENCE_RC,
      "output": "reference_check.txt"
    }
  }
}
EOF

cat <<EOF
Reproducibility artifacts generated:
  $ARTIFACT_DIR
Status summary:
  $ARTIFACT_DIR/status_summary.v1.json
EOF
