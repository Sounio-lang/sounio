#!/usr/bin/env bash
# KL-5: ε comparisons must follow the operator preserved by the parser.
# Confidence floors (>=) order upward; error bounds (<, <=) order downward;
# equality requires equality. Both engines must refuse the clinical counterexample
# and accept/execute the satisfying control.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
# Witness Gate builds this PR's Madaros into MADAROS_RAW_BIN; Contracts must
# not run this gate against the committed prebuilt (it lags self-hosted/).
if [[ -z "${MADAROS_RAW_BIN:-}${SOUNIO_MADAROS_BIN:-}" ]]; then
  echo "KL5_EPSILON_POLARITY_GATE_FAIL: MADAROS_RAW_BIN or SOUNIO_MADAROS_BIN required" >&2
  exit 1
fi
SOUC="${SOUC:-$ROOT/bin/souc}"
BAD="tests/compile-fail/vancomycin_low_conf.sio"
GOOD="tests/run-pass/kl5_epsilon_confidence_boundary_ok.sio"
WORK="$(mktemp -d /tmp/madaros-kl5-epsilon.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

check_with() {
  local engine="$1" src="$2" log="$3"
  if [[ "$engine" == lean_single ]]; then
    SOUNIO_SOUC_ENGINE=lean_single "$SOUC" check "$src" >"$log" 2>&1
  else
    "$SOUC" check "$src" >"$log" 2>&1
  fi
}

compile_with() {
  local engine="$1" src="$2" elf="$3" log="$4"
  if [[ "$engine" == lean_single ]]; then
    SOUNIO_SOUC_ENGINE=lean_single "$SOUC" compile "$src" -o "$elf" >"$log" 2>&1
  else
    "$SOUC" compile "$src" -o "$elf" >"$log" 2>&1
  fi
}

# A relaxed copy is the positive control for the refusal. If changing only the
# confidence floor from 0.82 to 0.30 does not remove the error, the negative
# fixture is failing for an unrelated reason and this gate must not report green.
RELAXED="$WORK/vancomycin_relaxed.sio"
sed 's/0\.82/0.30/g' "$BAD" >"$RELAXED"
grep -q 'ε >= 0.30' "$RELAXED" || {
  echo "KL5_EPSILON_POLARITY_GATE_FAIL: sabotage did not change the bound" >&2
  exit 1
}

for engine in madaros lean_single; do
  bad_log="$WORK/${engine}_bad.log"
  if check_with "$engine" "$BAD" "$bad_log"; then
    echo "KL5_EPSILON_POLARITY_GATE_FAIL engine=$engine accepted=$BAD" >&2
    cat "$bad_log" >&2
    exit 1
  fi
  grep -qiE 'confidence|epsilon|ε|E036|bound is not tight' "$bad_log" || {
    echo "KL5_EPSILON_POLARITY_GATE_FAIL engine=$engine wrong refusal" >&2
    cat "$bad_log" >&2
    exit 1
  }

  relaxed_log="$WORK/${engine}_relaxed.log"
  if ! check_with "$engine" "$RELAXED" "$relaxed_log"; then
    echo "KL5_EPSILON_POLARITY_GATE_FAIL engine=$engine relaxed control refused" >&2
    cat "$relaxed_log" >&2
    exit 1
  fi

  elf="$WORK/${engine}_good.elf"
  good_log="$WORK/${engine}_good_compile.log"
  if ! compile_with "$engine" "$GOOD" "$elf" "$good_log"; then
    echo "KL5_EPSILON_POLARITY_GATE_FAIL engine=$engine witness compile" >&2
    cat "$good_log" >&2
    exit 1
  fi
  chmod +x "$elf"
  run_log="$WORK/${engine}_good_run.log"
  if ! "$elf" >"$run_log" 2>&1; then
    echo "KL5_EPSILON_POLARITY_GATE_FAIL engine=$engine witness run" >&2
    cat "$run_log" >&2
    exit 1
  fi
  grep -q 'KL5_EPSILON_CONFIDENCE_BOUNDARY_OK' "$run_log" || {
    echo "KL5_EPSILON_POLARITY_GATE_FAIL engine=$engine missing sentinel" >&2
    cat "$run_log" >&2
    exit 1
  }
  echo "KL5_EPSILON_ENGINE_OK engine=$engine refusal=confidence-boundary witness=pass sabotage=pass"
done

echo "MADAROS_KL5_EPSILON_POLARITY_GATE_OK"
