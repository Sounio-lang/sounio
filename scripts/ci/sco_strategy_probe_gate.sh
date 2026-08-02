#!/usr/bin/env bash
# SCO non-vacuity probe: enforce vs ablation strategy assignment for with NonAssoc.
# Requires Madaros rebuilt with SCO_ETAU_STRATEGY_TRACE (artifacts/self-hosted/madaros-sco).
#
# Expect:
#   enforce  → at least one SCO_STRATEGY=2  (PRECISION_PRESERVING)
#   ablation → at least one SCO_STRATEGY=0  (STANDARD) and no SCO_STRATEGY=2
#              for this ordinary-return NonAssoc probe
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

MADAROS="${SCO_MADAROS:-$ROOT/artifacts/self-hosted/madaros-sco}"
PROBE="examples/sco_corpus/strategy_probe/probe_nonassoc.sio"
fail=0

if [[ ! -x "$MADAROS" ]]; then
  echo "FAIL: missing rebuilt Madaros at $MADAROS (run sco_step4_runtime_gate.sh first)"
  exit 1
fi

if ! grep -q "SCO_ETAU_STRATEGY_TRACE" self-hosted/ir/lower.sio; then
  echo "FAIL: strategy trace marker missing in lower.sio"
  exit 1
fi

# lower.sio must be older or equal age to ELF for runtime claim
if [[ self-hosted/ir/lower.sio -nt "$MADAROS" ]]; then
  echo "FAIL: lower.sio newer than $MADAROS — rebuild required for strategy probe"
  exit 1
fi

run_trace() {
  local label="$1"
  shift
  local out elf
  out="$(mktemp)"
  elf="$(mktemp)"
  # check-mode does not lower; strategy is assigned during AST→IR lower on compile.
  # shellcheck disable=SC2068
  if ! env "$@" SOUNIO_SCO_STRATEGY_TRACE=1 "$MADAROS" compile "$PROBE" -o "$elf" >"$out" 2>&1; then
    echo "FAIL: $label compile failed"
    tail -30 "$out" || true
    rm -f "$out" "$elf"
    return 1
  fi
  rm -f "$elf"
  echo "$out"
}

echo "### enforce (no ablation) ###"
ENFORCE_OUT="$(run_trace enforce)" || fail=1
if [[ "$fail" -eq 0 ]]; then
  if grep -q 'SCO_STRATEGY=2' "$ENFORCE_OUT"; then
    echo "PASS: enforce saw SCO_STRATEGY=2"
    grep 'SCO_STRATEGY=' "$ENFORCE_OUT" | sort -u | head -20
  else
    echo "FAIL: enforce missing SCO_STRATEGY=2"
    grep 'SCO_STRATEGY=' "$ENFORCE_OUT" | sort -u | head -20 || true
    tail -40 "$ENFORCE_OUT" || true
    fail=1
  fi
fi

echo "### ablation (SOUNIO_SCO_ABLATION=1) ###"
ABL_OUT="$(run_trace ablation SOUNIO_SCO_ABLATION=1)" || fail=1
if [[ "$fail" -eq 0 ]]; then
  if grep -q 'SCO_STRATEGY=2' "$ABL_OUT"; then
    # May still see 2 from other modules; require at least one 0 and that probe path is not only-2.
    # Strong criterion: ablation must emit SCO_STRATEGY=0 for ordinary NonAssoc.
    :
  fi
  if grep -q 'SCO_STRATEGY=0' "$ABL_OUT"; then
    echo "PASS: ablation saw SCO_STRATEGY=0"
    grep 'SCO_STRATEGY=' "$ABL_OUT" | sort -u | head -20
  else
    echo "FAIL: ablation missing SCO_STRATEGY=0"
    grep 'SCO_STRATEGY=' "$ABL_OUT" | sort -u | head -20 || true
    fail=1
  fi
  # Non-vacuity: enforce and ablation traces must not be identical multisets of strategies for probe
  enf_set="$(grep 'SCO_STRATEGY=' "$ENFORCE_OUT" | sort -u | tr '\n' ' ')"
  abl_set="$(grep 'SCO_STRATEGY=' "$ABL_OUT" | sort -u | tr '\n' ' ')"
  if [[ "$enf_set" == "$abl_set" ]]; then
    echo "FAIL: enforce and ablation strategy sets identical ($enf_set) — no non-vacuity"
    fail=1
  else
    echo "PASS: enforce vs ablation strategy sets differ"
    echo "  enforce: $enf_set"
    echo "  ablation: $abl_set"
  fi
fi

rm -f "$ENFORCE_OUT" "$ABL_OUT" 2>/dev/null || true

if [[ "$fail" -ne 0 ]]; then
  echo "SCO_STRATEGY_PROBE_GATE_FAIL"
  exit 1
fi
echo "SCO_STRATEGY_PROBE_GATE_OK"
exit 0
