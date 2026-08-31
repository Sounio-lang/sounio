#!/usr/bin/env bash
# Golden PTX gate for K-AXI→PTX transpilers -- drift oracle for Phase C
ulimit -s unlimited 2>/dev/null || true
# (5-transpiler unification). Re-emits all (pattern × mode) combinations
# and asserts byte-identical output vs. captured goldens at
# tests/golden/kaxi_ptx/<mode>/<pattern>.{ptx,sha256,unsupported}.
#
# Capture goldens first via scripts/gpu/kaxi_ptx_capture.sh (a capture tool,
# not a gate: it cannot fail -- this script is the assertion).
#
# Exit 0 only if every combo matches its golden (PTX bytes for supported,
# unsupported marker for unsupported).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GOLDEN_DIR="tests/golden/kaxi_ptx"

if [[ ! -d "$GOLDEN_DIR" ]]; then
    echo "FAIL: golden dir $GOLDEN_DIR missing -- run scripts/gpu/kaxi_ptx_capture.sh first" >&2
    exit 2
fi

PATTERNS=(
    exit_only
    vec_add vec_sub vec_mul vec_div vec_min vec_max vec_abs vec_neg vec_sqrt vec_recip vec_lg2 vec_ex2 vec_sin vec_cos vec_rsqrt vec_conf_sqrt vec_sqrt_gate_var vec_sqrt_gate_var_mb fma
    source_vec_add_f64 source_vec_sub_f64 source_vec_mul_f64 source_vec_div_f64 source_fma_f64
    source_vec_add_f32 source_vec_sub_f32 source_vec_mul_f32 source_vec_div_f32 source_fma_f32
    source_epistemic_dual_output_f32 epistemic_elementwise_f32 epistemic_dual_output_f32
    mb_vec_double_f32
    pbpk_euler pbpk_4step pbpk_8step pbpk_2comp pbpk_2comp_4step
    atomic_sum_f32
    pbpk_2comp_mb pbpk_2comp_4step_mb
    x4_tree x4_chain
    pbpk_rapamycin_1comp_epistemic
    pbpk_2comp_epistemic pbpk_2comp_epistemic_4step pbpk_traj_4step
    conf_gate
    octonion_assoc octonion_mul sedenion_mul
    pbpk_2comp_gum_4step
)

MODES=(
    "default:"
    "epistemic:--epistemic"
    "f32:--f32"
    "f32_2c:--f32-2c"
    "f32e:--f32-epistemic"
    "f32_gum:--f32-gum"
)

PASS=0
FAIL=0
MISSING=0
FIRST_FAILURES=()

for mode_entry in "${MODES[@]}"; do
    mode="${mode_entry%%:*}"
    flag="${mode_entry#*:}"

    for pattern in "${PATTERNS[@]}"; do
        gold_ptx="$GOLDEN_DIR/$mode/$pattern.ptx"
        gold_sha="$GOLDEN_DIR/$mode/$pattern.sha256"
        gold_unsup="$GOLDEN_DIR/$mode/$pattern.unsupported"

        if [[ ! -f "$gold_ptx" && ! -f "$gold_unsup" ]]; then
            MISSING=$((MISSING + 1))
            [[ "${#FIRST_FAILURES[@]}" -lt 5 ]] && FIRST_FAILURES+=("MISS  $mode/$pattern (no golden)")
            continue
        fi

        tmp_ptx="$(mktemp -t kaxi-gate.XXXXXX.ptx)"
        tmp_err="$(mktemp -t kaxi-gate.XXXXXX.err)"

        set +e
        if [[ -n "$flag" ]]; then
            ./bin/kretikos kaxi-emit-ptx "$pattern" -o "$tmp_ptx" --no-ptxas "$flag" >/dev/null 2>"$tmp_err"
        else
            ./bin/kretikos kaxi-emit-ptx "$pattern" -o "$tmp_ptx" --no-ptxas >/dev/null 2>"$tmp_err"
        fi
        rc=$?
        set -e

        if [[ -f "$gold_unsup" ]]; then
            # Expected unsupported. Assert rc nonzero (we don't compare stderr text -- it can be noisy).
            if [[ "$rc" -eq 0 ]]; then
                FAIL=$((FAIL + 1))
                [[ "${#FIRST_FAILURES[@]}" -lt 5 ]] && FIRST_FAILURES+=("REGR  $mode/$pattern (was unsupported, now supported)")
            else
                PASS=$((PASS + 1))
            fi
        else
            # Expected supported -- assert byte-identical.
            if [[ "$rc" -ne 0 || ! -s "$tmp_ptx" ]]; then
                FAIL=$((FAIL + 1))
                [[ "${#FIRST_FAILURES[@]}" -lt 5 ]] && FIRST_FAILURES+=("DROP  $mode/$pattern (rc=$rc, was supported)")
            else
                cur_sha="$(sha256sum "$tmp_ptx" | awk '{print $1}')"
                gold_sha_val="$(cat "$gold_sha" 2>/dev/null || echo MISSING)"
                if [[ "$cur_sha" == "$gold_sha_val" ]]; then
                    PASS=$((PASS + 1))
                else
                    FAIL=$((FAIL + 1))
                    [[ "${#FIRST_FAILURES[@]}" -lt 5 ]] && FIRST_FAILURES+=("DIFF  $mode/$pattern (golden=$gold_sha_val current=$cur_sha)")
                fi
            fi
        fi

        rm -f "$tmp_ptx" "$tmp_err"
    done
done

echo "=== kaxi_ptx golden gate ==="
echo "  PASS:    $PASS"
echo "  FAIL:    $FAIL"
echo "  MISSING: $MISSING"
if [[ "${#FIRST_FAILURES[@]}" -gt 0 ]]; then
    echo "  first failures:"
    for f in "${FIRST_FAILURES[@]}"; do
        echo "    $f"
    done
fi

if [[ "$FAIL" -gt 0 || "$MISSING" -gt 0 ]]; then
    exit 1
fi
echo "OK: all golden PTX byte-identical"
