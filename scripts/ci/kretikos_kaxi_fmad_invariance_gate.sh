#!/usr/bin/env bash
# Sub-PTX regression gate: verify the Cayley-Dickson GPU kernels remain
# FMA-fusion-invariant under ptxas `--fmad={true,false}`.
#
# Finding (docs/research/subptx_fmad_invariance.md, 2026-05-11):
#   The shipped octonion_mul / sedenion_mul / octonion_associator /
#   sedenion_associator kernels emit explicit `mul.rn.f32` and `add.rn.f32`
#   PTX, which prevents ptxas from fusing into `fma.rn.f32` regardless of
#   the `--fmad` flag. SASS is byte-identical across both compile modes.
#
# This gate locks the invariant: any future change to the K-AXI lowering
# that removes the explicit `.rn` modifiers (or introduces an op pattern
# that ptxas can fuse) will be caught here, before reviewers see the
# downstream artifact concern on orbit equivalence / 5.6× regularization /
# G2 bridge nulls.
#
# Acceptance:
#   For each of the four kernels, on sm_89:
#     1. ptxas --fmad=true emits a cubin with zero FFMA instructions
#     2. ptxas --fmad=false emits a cubin with zero FFMA instructions
#     3. The two SASS dumps are byte-identical
#
# SKIP behaviour: if ptxas/cuobjdump are missing or cuobjdump cannot dump
# the generated cubin, prints SKIPPED and exits 0. This is a sub-PTX gate,
# not a core PTX gate; CI without a usable CUDA disassembler must not regress
# for this reason.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v ptxas >/dev/null 2>&1 || ! command -v cuobjdump >/dev/null 2>&1; then
    echo "kretikos_kaxi_fmad_invariance_gate: SKIPPED (no CUDA toolchain)"
    exit 0
fi

SOUC="${SOUC:-./bin/souc}"
KRETIKOS="${KRETIKOS:-./bin/kretikos}"
ARCH="${ARCH:-sm_89}"
KERNELS=(octonion_mul sedenion_mul octonion_associator sedenion_associator lse8 sinkhorn16)

TMP_DIR="$(mktemp -d -t kaxi_fmad.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FIRST_FAILURES=()

note_fail() {
    FAIL=$((FAIL + 1))
    FIRST_FAILURES+=("$1")
}

if [[ ! -x "$KRETIKOS" ]]; then
    echo "FAIL: kretikos not executable at $KRETIKOS" >&2
    exit 2
fi

for kernel in "${KERNELS[@]}"; do
    ptx="$TMP_DIR/${kernel}.ptx"
    cubin_t="$TMP_DIR/${kernel}_true.cubin"
    cubin_f="$TMP_DIR/${kernel}_false.cubin"
    sass_t="$TMP_DIR/${kernel}_true.sass"
    sass_f="$TMP_DIR/${kernel}_false.sass"

    if ! "$KRETIKOS" kaxi-emit-ptx "$kernel" --f32 -o "$ptx" >/dev/null 2>&1; then
        note_fail "$kernel: kretikos kaxi-emit-ptx failed"
        continue
    fi

    if ! ptxas -arch "$ARCH" --fmad=true "$ptx" -o "$cubin_t" >/dev/null 2>&1; then
        note_fail "$kernel: ptxas --fmad=true failed on $ARCH"
        continue
    fi
    if ! ptxas -arch "$ARCH" --fmad=false "$ptx" -o "$cubin_f" >/dev/null 2>&1; then
        note_fail "$kernel: ptxas --fmad=false failed on $ARCH"
        continue
    fi

    if ! cuobjdump --dump-sass "$cubin_t" > "$sass_t" 2>&1; then
        echo "kretikos_kaxi_fmad_invariance_gate: SKIPPED (CUDA disassembler unusable for ${kernel}/${ARCH})"
        exit 0
    fi
    if ! cuobjdump --dump-sass "$cubin_f" > "$sass_f" 2>&1; then
        echo "kretikos_kaxi_fmad_invariance_gate: SKIPPED (CUDA disassembler unusable for ${kernel}/${ARCH})"
        exit 0
    fi

    # 1: zero FFMA in --fmad=true cubin
    if [[ $(grep -c '\bFFMA\b' "$sass_t") -eq 0 ]]; then
        PASS=$((PASS + 1))
    else
        note_fail "$kernel: --fmad=true cubin contains FFMA (ptxas fused something — explicit .rn modifiers may be missing)"
    fi

    # 2: zero FFMA in --fmad=false cubin (always true; sanity-check)
    if [[ $(grep -c '\bFFMA\b' "$sass_f") -eq 0 ]]; then
        PASS=$((PASS + 1))
    else
        note_fail "$kernel: --fmad=false cubin contains FFMA (ptxas bug?)"
    fi

    # 3: SASS byte-identical
    if diff -q "$sass_t" "$sass_f" >/dev/null 2>&1; then
        PASS=$((PASS + 1))
    else
        note_fail "$kernel: SASS differs between --fmad=true and --fmad=false (kernel is no longer fma-fusion-invariant)"
    fi
done

echo "=== sub-PTX FMA-fusion-invariance gate (arch=$ARCH) ==="
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
