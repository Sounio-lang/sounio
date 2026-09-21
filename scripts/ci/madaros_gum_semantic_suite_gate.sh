#!/usr/bin/env bash
# Semantic GUM / H1 / epistemic-PBPK suite for Madaros (parity harvest 1-2-3).
# Requires PASS markers; does not require byte-identical lean_single stdout
# (print newlines / trailing digits differ; GUM math must hold).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

RAW="${MADAROS_RAW_BIN:-}"
if [[ -z "$RAW" ]]; then
  if [[ -x "$ROOT/artifacts/self-hosted/madaros" ]]; then
    RAW="$ROOT/artifacts/self-hosted/madaros"
  elif [[ -x "$ROOT/bin/madaros-linux-x86_64" ]]; then
    RAW="$ROOT/bin/madaros-linux-x86_64"
  else
    echo "[madaros-gum-semantic] FAIL: set MADAROS_RAW_BIN" >&2
    exit 1
  fi
fi

stack_kb="${SOUNIO_MADAROS_GUM_SEMANTIC_STACK_KB:-524288}"
stack_before="$(ulimit -S -s 2>/dev/null || echo unlimited)"
if [[ "$stack_before" != "unlimited" ]] && [[ "$stack_before" =~ ^[0-9]+$ ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || true
fi

WORK=$(mktemp -d /tmp/sounio-gum-semantic.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

run_one() {
  local src="$1"
  local expect_re="$2"
  local tag
  tag="$(basename "$src" .sio)"
  local elf="$WORK/${tag}.elf"
  local log="$WORK/${tag}.log"
  "$RAW" "$ROOT/$src" -o "$elf" >"$WORK/${tag}.compile.log" 2>&1 || {
    cat "$WORK/${tag}.compile.log" >&2
    echo "[madaros-gum-semantic] FAIL compile: $tag" >&2
    return 1
  }
  chmod +x "$elf"
  timeout 90 "$elf" >"$log" 2>&1 || {
    cat "$log" >&2
    echo "[madaros-gum-semantic] FAIL run: $tag" >&2
    return 1
  }
  if ! grep -qE "$expect_re" "$log"; then
    cat "$log" >&2
    echo "[madaros-gum-semantic] FAIL expect /$expect_re/: $tag" >&2
    return 1
  fi
  if grep -qE '\bFAIL\b' "$log" && ! grep -qE 'Failed:[[:space:]]*0' "$log"; then
    # gum_reporting prints "Failed: 0" on success — allow that form only.
    if grep -qE 'Failed:[[:space:]]*[1-9]' "$log" || grep -qE '^FAIL' "$log" || grep -qE 'FAIL ' "$log"; then
      cat "$log" >&2
      echo "[madaros-gum-semantic] FAIL marker: $tag" >&2
      return 1
    fi
  fi
  echo "[madaros-gum-semantic] PASS: $tag"
}

# Batch 1 — core GUM FO
run_one tests/run-pass/gum_compliance.sio 'All 7 tests match GUM'
run_one tests/run-pass/gum_correlated.sio '^PASS$'
run_one tests/run-pass/gum_cross_function.sio '^PASS$'
run_one tests/run-pass/gum_iso_budget.sio '^PASS$'
run_one tests/run-pass/gum_euler_ode.sio '^PASS$'

# Batch 2 — GUM H.1 end-to-end
run_one tests/run-pass/gum_h1_native.sio '\[PASS\] Step-by-step matches analytical'
run_one tests/run-pass/gum_h1_end_gauge.sio '\[PASS\] Step-by-step matches analytical'

# Batch 3 — epistemic PBPK (native multi-drug path)
run_one tests/run-pass/epistemic_pbpk_native.sio 'PASS: T1 brain/plasma'
run_one tests/run-pass/epistemic_pbpk_multidrug.sio 'PASS T1: brain/plasma'
run_one tests/run-pass/darwin_pbpk28_smoke.sio 'PBPK28_CORE_SMOKE_PASS'

echo "[madaros-gum-semantic] ALL PASS (10 witnesses)"
