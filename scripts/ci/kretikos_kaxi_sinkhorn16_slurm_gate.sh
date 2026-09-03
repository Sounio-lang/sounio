#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${SOUNIO_KAXI_SINKHORN16_SLURM_DIR:-$(mktemp -d /tmp/kaxi-sinkhorn16-slurm.XXXXXX)}"
exec "$ROOT_DIR/scripts/ci/moonshot_a_sinkhorn16_slurm_gate_common.sh" scalar "$OUT_DIR"
