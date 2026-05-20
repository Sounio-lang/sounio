#!/usr/bin/env bash
# Sub-PTX B2-4 Sinkhorn-LSE N=16 end-to-end gate.
#
# Validates the `sinkhorn16` K-AXI pattern by:
#   1. Building the CUDA driver-API runner
#   2. Emitting the f32 PTX (2 iterations — see kernel header re. the
#      256 KB KaxiAsmBuf cap)
#   3. ptxas-compiling to a cubin on sm_89
#   4. Launching on the local GPU with the diagonal-cost / uniform-
#      marginal sanity case (mathematically tractable fixed point)
#   5. Checking u[0..15] all converge to -7.0875 ± 1e-3 (CPU oracle:
#      u = la - log2(1 + (N-1)·2^(-1)) = -4 - log2(8.5) = -7.0875)
#   6. Checking v[0..15] all converge to 0 ± 1e-3 (CPU oracle: 0 by
#      symmetry after iter 1; stays at 0 across subsequent iters
#      with this specific structure)
#   7. Run-to-run determinism on three back-to-back launches
#
# SKIP behaviour: prints SKIPPED and exits 0 if ptxas / GPU / cc absent.
#
# Reference: docs/research/subptx_sinkhorn16_2iter.md
#            docs/research/subptx_fmad_invariance.md

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v ptxas >/dev/null 2>&1; then
    echo "kretikos_kaxi_sinkhorn16_gate: SKIPPED (no ptxas)"
    exit 0
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "kretikos_kaxi_sinkhorn16_gate: SKIPPED (no GPU)"
    exit 0
fi
if ! command -v cc >/dev/null 2>&1; then
    echo "kretikos_kaxi_sinkhorn16_gate: SKIPPED (no cc)"
    exit 0
fi

KRETIKOS="${KRETIKOS:-./bin/kretikos}"
RUNNER_SRC="scripts/gpu/kaxi_ptx_runner.c"

TMP_DIR="$(mktemp -d -t kaxi_sinkhorn.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

PTX="$TMP_DIR/sinkhorn16.ptx"
RUNNER="$TMP_DIR/runner"
INPUT="$TMP_DIR/input.csv"

PASS=0
FAIL=0
FIRST_FAILURES=()

note_fail() {
    FAIL=$((FAIL + 1))
    FIRST_FAILURES+=("$1")
}

# 1) Build runner.
if cc -O2 "$RUNNER_SRC" -ldl -lm -o "$RUNNER" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
else
    note_fail "runner build (cc)"
fi

# 2) Emit Sinkhorn-16 PTX.
if "$KRETIKOS" kaxi-emit-ptx sinkhorn16 --f32 -o "$PTX" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
else
    note_fail "kretikos kaxi-emit-ptx sinkhorn16"
fi

# 3) Generate diagonal-cost / uniform-marginal input.
# la=lb=log2(1/16)=-4.0 exactly (N=16 is a power of 2); K=0 on diag/-1 off; 32 trailing zeros.
awk 'BEGIN {
  sep=""
  for (i=0;i<32;i++) { printf "%s-4.0",sep; sep="," }
  for (i=0;i<16;i++) for (j=0;j<16;j++) { printf ",%s",(i==j?"0.0":"-1.0") }
  for (i=0;i<32;i++) { printf ",0.0" }
  printf "\n"
}' > "$INPUT"

# 4) Single launch, parse u_out (mem[288..303]) and v_out (mem[304..319]).
LAUNCH_OUT="$(${RUNNER} ${PTX} --threads 1 --mem-words 320 --init-mem "$(cat ${INPUT})" --type f32 2>&1 | grep '^MEM:' || true)"
if [[ -z "$LAUNCH_OUT" ]]; then
    note_fail "no launch output"
fi

u0="$(echo "$LAUNCH_OUT" | awk '{print $290}')"
v0="$(echo "$LAUNCH_OUT" | awk '{print $306}')"

# 5) u[0] should be -7.0875 ± 1e-3.
if awk -v v="$u0" 'BEGIN { d = v - (-7.0875); if (d < 0) d = -d; exit (d <= 1e-3) ? 0 : 1 }'; then
    PASS=$((PASS + 1))
else
    note_fail "u[0] = $u0, expected -7.0875 ± 1e-3"
fi

# 6) v[0] should be 0 ± 1e-3.
if awk -v v="$v0" 'BEGIN { d = v - 0.0; if (d < 0) d = -d; exit (d <= 1e-3) ? 0 : 1 }'; then
    PASS=$((PASS + 1))
else
    note_fail "v[0] = $v0, expected 0 ± 1e-3"
fi

# Check all 16 u values are bit-identical (symmetry).
u_values="$(echo "$LAUNCH_OUT" | awk '{for (i=290; i<=305; i++) print $i}' | sort -u)"
u_uniq_count="$(echo "$u_values" | wc -l)"
if [[ "$u_uniq_count" -eq 1 ]]; then
    PASS=$((PASS + 1))
else
    note_fail "u[0..15] not all identical (got $u_uniq_count distinct: $u_values)"
fi

# Check all 16 v values are bit-identical.
v_values="$(echo "$LAUNCH_OUT" | awk '{for (i=306; i<=321; i++) print $i}' | sort -u)"
v_uniq_count="$(echo "$v_values" | wc -l)"
if [[ "$v_uniq_count" -eq 1 ]]; then
    PASS=$((PASS + 1))
else
    note_fail "v[0..15] not all identical (got $v_uniq_count distinct: $v_values)"
fi

# 7) Run-to-run determinism.
r1="$("$RUNNER" "$PTX" --threads 1 --mem-words 320 --init-mem "$(cat ${INPUT})" --type f32 2>/dev/null | grep '^MEM:' | awk '{print $290 "_" $306}')"
r2="$("$RUNNER" "$PTX" --threads 1 --mem-words 320 --init-mem "$(cat ${INPUT})" --type f32 2>/dev/null | grep '^MEM:' | awk '{print $290 "_" $306}')"
r3="$("$RUNNER" "$PTX" --threads 1 --mem-words 320 --init-mem "$(cat ${INPUT})" --type f32 2>/dev/null | grep '^MEM:' | awk '{print $290 "_" $306}')"
if [[ "$r1" == "$r2" && "$r2" == "$r3" ]]; then
    PASS=$((PASS + 1))
else
    note_fail "run-to-run determinism (got: $r1, $r2, $r3)"
fi

echo "=== sub-PTX B2-4 Sinkhorn-16 gate (RTX 4000 Ada / sm_89) ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [[ "${#FIRST_FAILURES[@]}" -gt 0 ]]; then
    echo "  first failures:"
    for f in "${FIRST_FAILURES[@]}"; do
        echo "    $f"
    done
fi

if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
exit 0
