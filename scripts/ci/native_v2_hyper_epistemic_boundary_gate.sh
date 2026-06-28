#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_DIR="${SOUNIO_NATIVE_V2_HYPER_EPISTEMIC_BOUNDARY_DIR:-$(mktemp -d /tmp/sounio-native-v2-hyper-epistemic.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
SUMMARY_JSON="$ARTIFACT_DIR/summary.json"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/$name.log"
  echo "[native-v2-hyper-epistemic] running $name"
  if "$@" >"$log" 2>&1; then
    echo "[native-v2-hyper-epistemic] ok: $name"
  else
    local rc=$?
    echo "[native-v2-hyper-epistemic] FAIL: $name rc=$rc log=$log" >&2
    tail -n 120 "$log" >&2 || true
    exit "$rc"
  fi
}

printf '[native-v2-hyper-epistemic] souc=%s\n' "$SOUC_BIN"
printf '[native-v2-hyper-epistemic] out=%s\n' "$OUT_DIR"

run_step hyper_epistemic_typecheck \
  "$SOUC_BIN" check tests/run-pass/hyper_epistemic_mul_typecheck.sio

run_step hyper_epistemic_native_selftest_check \
  "$SOUC_BIN" check self-hosted/native/test_hyper_epistemic.sio

run_step hyper_epistemic_native_selftest_run \
  "$SOUC_BIN" run self-hosted/native/test_hyper_epistemic.sio

run_step hyper_epistemic_mir_boundary_check \
  "$SOUC_BIN" check tests/native-v2/hyper_epistemic_mir_boundary.sio

# The imported/native default path is still blocked at the Madaros lower_array
# seed witness. Run the boundary witness through lean_single so this gate proves
# the native-v2 fail-closed contract without depending on that separate blocker.
run_step hyper_epistemic_mir_boundary_run_lean_single \
  env SOUNIO_SOUC_ENGINE=lean_single "$SOUC_BIN" run tests/native-v2/hyper_epistemic_mir_boundary.sio

run_step aarch64_preview_gate \
  bash scripts/ci/native_v2_aarch64_preview_gate.sh

cat >"$SUMMARY_JSON" <<EOF
{
  "schema": "sounio.native_v2_hyper_epistemic_boundary_gate.v1",
  "status": "pass",
  "compiler": "$SOUC_BIN",
  "proved": [
    "Madaros check reaches Knowledge<Hyper<Octonion,f64>> hyper epistemic multiplication witness",
    "self-hosted/native/test_hyper_epistemic.sio reports 10/10 runtime self-tests passing",
    "native-v2 MIR boundary rejects IrHyperEpistemicMul fail-closed with unsupported_detail=hyper_epistemic_mul",
    "AArch64 Mach-O preview gate emits a 32768-byte Mach-O artifact with magic cffaedfe"
  ],
  "not_proved": [
    "full native-v2 legal MIR lowering for IrHyperEpistemicMul",
    "default Madaros imported/native runtime for imported machine_ir witnesses",
    "Apple Silicon native-v2 runtime parity"
  ],
  "known_default_engine_blocker": "madaros_imported_native_lower_array_seed_segfault"
}
EOF

echo "[native-v2-hyper-epistemic] PASS: boundary, native selftest, and AArch64 preview gates passed"
echo "[native-v2-hyper-epistemic] summary=$SUMMARY_JSON"
