#!/usr/bin/env bash
# Source-built runtime coverage. No scientific preservation/non-vacuity theorem claim.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
MADAROS="${SCO_MADAROS:-$ROOT/artifacts/self-hosted/madaros-sco}"
OUT="${SCO_OUTPUT_DIR:-$ROOT/artifacts/sco}"
mkdir -p "$OUT"
source_digest() {
  find self-hosted stdlib -type f -name '*.sio' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
}
receipt="$MADAROS.sco-source.sha256"
before="$(source_digest)"
if [[ "${SCO_SKIP_REBUILD:-0}" != 1 ]]; then
  # This script is executed on CI/Foundry workers, never as a workspace compile.
  # build_modular_madaros.sh owns its build lock; do not wrap that lock twice.
  bash scripts/ci/build_modular_madaros.sh "$MADAROS" >"$OUT/build.log" 2>&1
  test -x "$MADAROS"
  [[ "$(source_digest)" == "$before" ]] || { echo 'FAIL: sources changed during build'; exit 1; }
  printf 'source_sha256=%s\nelf_sha256=%s\n' "$before" \
    "$(sha256sum "$MADAROS" | cut -d' ' -f1)" >"$receipt"
fi
test -x "$MADAROS" && test -f "$receipt" || { echo 'FAIL: missing compiler/source receipt'; exit 1; }
grep -qx "source_sha256=$before" "$receipt" || { echo 'FAIL: source mismatch; rebuild required'; exit 1; }
grep -qx "elf_sha256=$(sha256sum "$MADAROS" | cut -d' ' -f1)" "$receipt" || { echo 'FAIL: ELF mismatch'; exit 1; }
SOUC="$MADAROS" bash scripts/ci/sco_corpus_gate.sh
SCO_MADAROS="$MADAROS" bash scripts/ci/sco_strategy_probe_gate.sh
echo 'SCO_STEP4_RUNTIME_GATE_OK'
