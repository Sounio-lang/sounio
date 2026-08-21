#!/usr/bin/env bash
# The Madaros witness gate.
#
# CI runs lean_single -- the Full Test Suite prints
# SOUNIO_TEST_SOUC_BIN=/tmp/souc-stage2, built from lean_single.sio. So every
# check that exists is blind to the modular Madaros compiler, which is what
# bin/souc resolves to and what users get.
#
# Measured 2026-08-21, all invisible to CI at the time:
#   - `let d: mg = 500.0` could not bind at all under Madaros;
#   - a correct refinement alias and an incorrect one drew the SAME diagnostic,
#     so the refusal carried no information;
#   - a first-order-variance regression sat on main printing 0.000000.
#
# The full corpus cannot close this. 1773 programs serialised exceed the
# 60-minute CI limit; run parallel, the raw ELF SIGSEGVs under memory pressure
# and invents regressions -- 1510 of them, identical across three
# configurations, with one file appearing in both the pass and fail lists. A
# corpus is a periodic Slurm sweep. This is the per-PR gate.
#
# Every witness must assert an ANSWER, not merely survive: `//@ expect-stdout:`
# for a run-pass, `//@ error-pattern:` for a compile-fail. A test that exits 0
# and prints the wrong number is exactly the failure this exists to catch.
set -uo pipefail
cd "$(dirname "$0")/../.."
# shellcheck source=scripts/lib/gate_assert.sh
. scripts/lib/gate_assert.sh

LIST=scripts/ci/madaros_witness_set.list
MADAROS="${SOUNIO_MADAROS_WITNESS_BIN:-}"
ART_DIR=artifacts/gates
mkdir -p "$ART_DIR"

require_nonempty_file "$LIST" "the witness set is missing or empty"
require_nonempty "$MADAROS" "SOUNIO_MADAROS_WITNESS_BIN must name a current-source Madaros ELF"
[[ -x "$MADAROS" ]] || gate_fail "not executable: $MADAROS"

# The raw ELF needs far more stack than the default 8 MB. bin/souc raises it;
# this gate invokes the ELF directly. Without this every compile SIGSEGVs and
# the gate reports a semantic regression that is really a crash.
if ! ulimit -s 524288 2>/dev/null; then
  ulimit -s unlimited 2>/dev/null \
    || gate_fail "cannot raise the stack; the raw Madaros ELF will SIGSEGV on every compile"
fi
echo "[madaros-witness] stack: $(ulimit -s) KiB"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$PWD/stdlib}"
"$MADAROS" --version 2>&1 | head -1 | sed 's/^/[madaros-witness] /'

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
total=0; passed=0; failed=0

while IFS= read -r src; do
  case "$src" in ''|'#'*) continue ;; esac
  [[ -f "$src" ]] || gate_fail "listed but absent: $src"
  total=$((total + 1))
  want_fail=0
  grep -qE '^//@[[:space:]]*compile-fail' "$src" && want_fail=1
  pat=$(sed -n 's|^//@[[:space:]]*error-pattern:[[:space:]]*||p' "$src" | head -1)
  marker=$(sed -n 's|^//@[[:space:]]*expect-stdout:[[:space:]]*||p' "$src" | head -1)

  if [[ $want_fail -eq 1 && -z "$pat" ]]; then
    echo "[madaros-witness] FAIL $src -- compile-fail with no //@ error-pattern:"
    failed=$((failed + 1)); continue
  fi
  # A run-pass witness must assert an ANSWER -- but there is more than one
  # honest way to do that, and the first version of this rule confused "does
  # not assert" with "asserts differently". The R1 Lorenz witnesses carry no
  # expect-stdout and self-check by exit code
  # (`if a * b != 21 { return 10 }`), which the run below already enforces.
  # What must stay refused is a witness that neither prints a marker nor
  # checks anything -- one that passes by merely surviving.
  self_checks=0
  grep -qE 'assert\(|return [1-9][0-9]*' "$src" && self_checks=1
  if [[ $want_fail -eq 0 && -z "$marker" && $self_checks -eq 0 ]]; then
    echo "[madaros-witness] FAIL $src -- run-pass that asserts nothing: no //@ expect-stdout: and no self-check"
    failed=$((failed + 1)); continue
  fi

  elf="$W/w.elf"; log="$W/w.log"
  "$MADAROS" compile "$src" -o "$elf" > "$log" 2>&1
  rc=$?

  if [[ $want_fail -eq 1 ]]; then
    if [[ $rc -eq 0 ]]; then
      echo "[madaros-witness] FAIL $src -- built cleanly, expected refusal"
      failed=$((failed + 1)); continue
    fi
    if grep -qF "$pat" "$log"; then
      echo "[madaros-witness] ok   $src -- refused with: $pat"
      passed=$((passed + 1))
    else
      echo "[madaros-witness] FAIL $src -- refused, but not for: $pat"
      grep -E 'error\[E[0-9]+\]' "$log" | head -2 | sed 's/^/[madaros-witness]      got: /'
      failed=$((failed + 1))
    fi
    continue
  fi

  if [[ $rc -ne 0 ]]; then
    echo "[madaros-witness] FAIL $src -- did not build"
    grep -E 'error\[E[0-9]+\]' "$log" | head -2 | sed 's/^/[madaros-witness]      /'
    failed=$((failed + 1)); continue
  fi
  chmod +x "$elf"
  out=$(timeout 60 "$elf" 2>&1); run_rc=$?
  if [[ $run_rc -ne 0 ]]; then
    echo "[madaros-witness] FAIL $src -- ran with rc=$run_rc"
    failed=$((failed + 1)); continue
  fi
  if [[ -z "$marker" ]]; then
    echo "[madaros-witness] ok   $src -- self-checked by exit code"
    passed=$((passed + 1)); continue
  fi
  case "$out" in
    *"$marker"*) echo "[madaros-witness] ok   $src"; passed=$((passed + 1)) ;;
    *) echo "[madaros-witness] FAIL $src -- stdout missing: $marker"
       echo "$out" | head -4 | sed 's/^/[madaros-witness]      got: /'
       failed=$((failed + 1)) ;;
  esac
done < "$LIST"

# A sweep that found nothing to check must be red, not green.
require_min_count "$total" 1 "witnesses resolved from $LIST"

st=pass
[[ $failed -eq 0 ]] || st=fail
cat > "$ART_DIR/madaros_witness.json" <<JSON
{"status":"$st","metrics":{"total":$total,"passed":$passed,"failed":$failed,"not_run":0}}
JSON
echo "status=$st"
echo "metrics {total=$total, passed=$passed, failed=$failed, not_run=0}"
echo "artifact=$ART_DIR/madaros_witness.json"
[[ $st == pass ]]
