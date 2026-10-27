#!/usr/bin/env bash
# scripts/ci/chbmit_seizure_examples_gate.sh
#
# The CHB-MIT sedenion SSM seizure examples compute the same numbers on
# Madaros and lean_single (#2789).
#
# #2789 recorded a Madaros source build of main (2026-10-06) computing wrong
# values on chb01/03/05 (train MSE 1e2-4e3, a 9.22e18 sentinel). The examples
# were tagged `requires: lean_single`, so nothing ran them on the default
# engine again and the tag would have outlived the defect forever. This gate
# runs all four on both engines:
#   chb01/03/05  ALL PASS on both, identical result lines (MSE, ratios, T1-T5);
#   chb11        T1/T2 PASS and identical numbers on both; T3/T4 FAIL on both,
#                which is what its known-failure records (a scientific result:
#                the subject's data refute the 16ch-most-sensitive hypothesis).
# and refuses a `requires: lean_single` tag on an example that passes here.
#
# GATE_CONTRACT: v0
# GATE_ID: chbmit_seizure_examples
# GATE_CLAIMS: the four CHB-MIT S-SSM examples print identical results on
#              Madaros and lean_single, chb01/03/05 ALL PASS
# GATE_ENGINE: madaros, lean_single
# GATE_RESULT_ON_SKIP: fail
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "chbmit_seizure_examples"

SOUC="${SOUC:-./bin/souc}"
LEAN="${SOUNIO_LEAN_SINGLE_BIN:-./bin/souc-lean-single-x86_64}"
TIMEOUT="${CHBMIT_GATE_TIMEOUT:-900}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
ulimit -s unlimited 2>/dev/null || true
require_executable "$LEAN"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

fails=0
# Result lines only: numbers and verdicts, not banners.
results() { grep -E 'train=|ratio=|^T[0-9]|ALL PASS|FAILURES' "$1"; }

run_engine() {
  local engine="$1" src="$2" tag="$3"
  local elf="$work/$tag.$engine.elf"
  rm -f "$elf"
  if [[ "$engine" == "lean_single" ]]; then
    timeout "$TIMEOUT" "$LEAN" "$src" "$elf" > "$work/$tag.$engine.compile.log" 2>&1
  else
    timeout "$TIMEOUT" "$SOUC" compile "$src" -o "$elf" > "$work/$tag.$engine.compile.log" 2>&1
  fi
  if [[ ! -s "$elf" ]] || ! head -c 4 "$elf" | grep -q ELF; then
    echo "CHBMIT_FAIL $tag $engine: did not compile"; tail -5 "$work/$tag.$engine.compile.log"
    return 2
  fi
  chmod +x "$elf"
  timeout "$TIMEOUT" "$elf" > "$work/$tag.$engine.out" 2>&1
  local rc=$?
  if [[ "$rc" -ge 124 ]]; then
    echo "CHBMIT_FAIL $tag $engine: rc=$rc (timeout or signal)"; return 2
  fi
  return "$rc"
}

for tag in chb01 chb03 chb05 chb11; do
  src="examples/sedenion_ssm_seizure_$tag.sio"
  require_file "$src"
  run_engine madaros "$src" "$tag"; rc_md=$?
  run_engine lean_single "$src" "$tag"; rc_ls=$?
  if [[ "$rc_md" -eq 2 || "$rc_ls" -eq 2 ]]; then fails=$((fails + 1)); continue; fi
  if ! diff <(results "$work/$tag.madaros.out") <(results "$work/$tag.lean_single.out") > "$work/$tag.diff"; then
    echo "CHBMIT_FAIL $tag: Madaros and lean_single print different results"
    cat "$work/$tag.diff"; fails=$((fails + 1)); continue
  fi
  if [[ -z "$(results "$work/$tag.madaros.out")" ]]; then
    echo "CHBMIT_FAIL $tag: no result lines"; fails=$((fails + 1)); continue
  fi
  if [[ "$tag" == "chb11" ]]; then
    if [[ "$rc_md" -eq 1 && "$rc_ls" -eq 1 ]] \
       && grep -q '^T2 PASS' "$work/$tag.madaros.out" \
       && grep -q '^T3 FAIL' "$work/$tag.madaros.out" \
       && grep -q '^T4 FAIL' "$work/$tag.madaros.out" \
       && grep -q '^FAILURES: 2' "$work/$tag.madaros.out" \
       && head -n 5 "$src" | grep -q '^//@ known-failure: the chb11 data refute T3/T4'; then
      echo "CHBMIT_OK $tag: identical on both engines; T3/T4 FAIL as its known-failure records"
    else
      echo "CHBMIT_FAIL $tag: outcome no longer matches its known-failure (rc madaros=$rc_md lean_single=$rc_ls)"
      results "$work/$tag.madaros.out"; fails=$((fails + 1))
    fi
    continue
  fi
  if [[ "$rc_md" -ne 0 || "$rc_ls" -ne 0 ]] || ! grep -qx 'ALL PASS' "$work/$tag.madaros.out"; then
    echo "CHBMIT_FAIL $tag: not ALL PASS (rc madaros=$rc_md lean_single=$rc_ls)"
    results "$work/$tag.madaros.out"; fails=$((fails + 1)); continue
  fi
  if head -n 10 "$src" | grep -q '^//@ requires: lean_single'; then
    echo "CHBMIT_FAIL $tag: passes on Madaros but still declares requires: lean_single"
    fails=$((fails + 1)); continue
  fi
  echo "CHBMIT_OK $tag: ALL PASS, identical results on Madaros and lean_single"
done

if [[ "$fails" -ne 0 ]]; then
  gate_fail "$fails example(s) failed"
fi
gate_pass "4 CHB-MIT examples agree across engines"
