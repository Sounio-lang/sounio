#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUNIO_BIN="${ROOT_DIR}/target/release/souc"
SOUNIO_PIPELINE="${ROOT_DIR}/examples/real_world/06_darwin_atlas_pipeline.sio"

DARWIN_ATLAS_REPO="/tmp/darwin-atlas-ag"
MAX_GENOMES="32"
SEED="42"
ASSEMBLY_SUMMARY=""
WORK_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --darwin-atlas-repo)
      DARWIN_ATLAS_REPO="$2"
      shift 2
      ;;
    --max-genomes)
      MAX_GENOMES="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --assembly-summary)
      ASSEMBLY_SUMMARY="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: run_darwin_atlas_parity.sh [options]

Options:
  --darwin-atlas-repo <path>  Darwin Atlas Julia repo path (default: /tmp/darwin-atlas-ag)
  --max-genomes <n>           Max genomes (default: 32)
  --seed <n>                  Seed (default: 42)
  --assembly-summary <path>   Optional local assembly_summary.txt for real metadata mode
  --work-dir <path>           Optional output root (default: /tmp/darwin_atlas_parity_<timestamp>)
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "${SOUNIO_BIN}" ]]; then
  echo "Missing souc binary: ${SOUNIO_BIN}" >&2
  exit 1
fi
if [[ ! -f "${SOUNIO_PIPELINE}" ]]; then
  echo "Missing Sounio pipeline: ${SOUNIO_PIPELINE}" >&2
  exit 1
fi
if [[ ! -f "${DARWIN_ATLAS_REPO}/julia/scripts/run_pipeline.jl" ]]; then
  echo "Missing Darwin Atlas Julia pipeline at ${DARWIN_ATLAS_REPO}" >&2
  exit 1
fi

if [[ -z "${WORK_DIR}" ]]; then
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
  WORK_DIR="/tmp/darwin_atlas_parity_${RUN_ID}"
fi

SOUNIO_DIR="${WORK_DIR}/sounio_data"
JULIA_DIR="${WORK_DIR}/julia_data"
mkdir -p "${SOUNIO_DIR}"/{raw,manifest,tables}
mkdir -p "${JULIA_DIR}"/{raw,manifest,tables}

echo "[1/5] Running Sounio pipeline (full) ..."
SOUNIO_CMD=(
  "${SOUNIO_BIN}" run "${SOUNIO_PIPELINE}" --
  --data-dir "${SOUNIO_DIR}"
  --max-genomes "${MAX_GENOMES}"
  --seed "${SEED}"
)
if [[ -n "${ASSEMBLY_SUMMARY}" ]]; then
  SOUNIO_CMD+=(--assembly-summary "${ASSEMBLY_SUMMARY}")
fi
"${SOUNIO_CMD[@]}"

echo "[2/5] Mirroring Sounio manifest/raw into Julia baseline input ..."
cp -f "${SOUNIO_DIR}/manifest/manifest.jsonl" "${JULIA_DIR}/manifest/manifest.jsonl"
cp -f "${SOUNIO_DIR}/manifest/checksums.sha256" "${JULIA_DIR}/manifest/checksums.sha256"
cp -f "${SOUNIO_DIR}/raw/"*.fna "${JULIA_DIR}/raw/"

echo "[3/5] Ensuring Julia dependencies are instantiated ..."
(
  cd "${DARWIN_ATLAS_REPO}"
  if ! julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; then
    echo "Primary instantiate failed, trying MbedTLS_jll repair..."
    julia --project=julia -e 'using Pkg; Pkg.add("MbedTLS_jll"); Pkg.instantiate(); Pkg.precompile()'
  fi
)

echo "[4/5] Running Julia pipeline (skip-download) ..."
(
  cd "${DARWIN_ATLAS_REPO}"
  julia --project=julia julia/scripts/run_pipeline.jl \
    --data-dir "${JULIA_DIR}" \
    --max-genomes "${MAX_GENOMES}" \
    --seed "${SEED}" \
    --skip-download
)

echo "[5/5] Running strict parity check in Sounio ..."
PARITY_REPORT_PATH="${SOUNIO_DIR}/manifest/parity_report.json"
"${SOUNIO_BIN}" run "${SOUNIO_PIPELINE}" -- \
  --data-dir "${SOUNIO_DIR}" \
  --skip-download \
  --max-genomes "${MAX_GENOMES}" \
  --seed "${SEED}" \
  --parity-with "${JULIA_DIR}" \
  --parity-report "${PARITY_REPORT_PATH}"

if [[ ! -f "${PARITY_REPORT_PATH}" ]]; then
  echo "Parity report not found: ${PARITY_REPORT_PATH}" >&2
  exit 1
fi
if ! grep -q '"all_passed": true' "${PARITY_REPORT_PATH}"; then
  echo "Strict parity failed; inspect ${PARITY_REPORT_PATH}" >&2
  exit 1
fi

echo
echo "Parity run complete."
echo "Work dir: ${WORK_DIR}"
echo "Parity report: ${PARITY_REPORT_PATH}"
