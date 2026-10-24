#!/usr/bin/env bash
# Strategy-selection coverage only; not scientific non-vacuity or a preservation proof.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/ci/sco_stack_contract.sh"
MADAROS="${SCO_MADAROS:-$ROOT/artifacts/self-hosted/madaros-sco}"
OUT="${SCO_OUTPUT_DIR:-$ROOT/artifacts/sco}/strategy"
mkdir -p "$OUT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
test -x "$MADAROS" || { echo 'FAIL: missing source-built compiler'; exit 1; }
source_digest() {
  find self-hosted stdlib -type f -name '*.sio' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1
}
receipt="$MADAROS.sco-source.sha256"
test -f "$receipt" || { echo 'FAIL: missing source/ELF-bound SCO receipt'; exit 1; }
grep -qx "source_sha256=$(source_digest)" "$receipt" || { echo 'FAIL: source mismatch'; exit 1; }
grep -qx "elf_sha256=$(sha256sum "$MADAROS" | cut -d' ' -f1)" "$receipt" || { echo 'FAIL: compiler mismatch'; exit 1; }
for mode in enforce ablation; do
  ablation=0; expected=2
  if [[ "$mode" == ablation ]]; then ablation=1; expected=0; fi
  compile_rc=0
  rm -f "$OUT/$mode.elf"
  timeout 180 env SOUNIO_SCO_ABLATION="$ablation" SOUNIO_SCO_STRATEGY_TRACE=1 \
    "$MADAROS" compile examples/sco_corpus/strategy_probe/probe_nonassoc.sio \
    -o "$OUT/$mode.elf" >"$OUT/$mode.compile.log" 2>&1 || compile_rc=$?
  printf 'compile_rc=%s\n' "$compile_rc" >"$OUT/$mode.status"
  [[ "$compile_rc" == 0 ]] || { echo "FAIL: $mode compile ($compile_rc)"; exit 1; }
  [[ -s "$OUT/$mode.elf" ]] || { echo "FAIL: $mode compile produced no fresh ELF"; exit 1; }
  grep '^SCO_STRATEGY function=sco_probe_leaf strategy=' "$OUT/$mode.compile.log" \
    >"$OUT/$mode.probe-trace" || { echo "FAIL: missing named final probe strategy ($mode)"; exit 1; }
  # Every emitted assignment for this exact probe must match; unrelated functions cannot satisfy it.
  if grep -vx "SCO_STRATEGY function=sco_probe_leaf strategy=$expected" "$OUT/$mode.probe-trace"; then
    echo "FAIL: incorrect probe strategy ($mode)"; exit 1
  fi
  chmod +x "$OUT/$mode.elf"
  run_rc=0
  timeout 30 "$OUT/$mode.elf" >"$OUT/$mode.stdout" 2>"$OUT/$mode.stderr" || run_rc=$?
  printf 'run_rc=%s\n' "$run_rc" >>"$OUT/$mode.status"
  [[ "$run_rc" == 0 ]] || { echo "FAIL: $mode runtime ($run_rc)"; exit 1; }
  printf '1\n' >"$OUT/expected.stdout"
  cmp "$OUT/expected.stdout" "$OUT/$mode.stdout"
  echo "PASS: $mode exact named strategy=$expected and runtime sentinel"
done
echo 'SCO_STRATEGY_PROBE_GATE_OK'
echo 'NOTE: identical runtime outputs do not establish scientific non-vacuity.'
