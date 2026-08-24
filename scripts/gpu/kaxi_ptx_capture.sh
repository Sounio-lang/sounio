#!/usr/bin/env bash
# CAPTURE TOOL -- NOT A GATE. It writes golden PTX and cannot fail.
ulimit -s unlimited 2>/dev/null || true
#
# For every (pattern x mode) combination of the K-AXI->PTX transpilers in
# self-hosted/gpu/kaxi_to_ptx.sio this script records the emitted PTX as a
# golden file. Combinations that already have a golden are KEPT untouched, and
# an emitter failure is recorded as a `.unsupported` marker rather than raised.
# Both are deliberate: capturing must be idempotent and must be able to record
# an unsupported combination. The consequence is that this script exits 0 no
# matter what the emitter does, so it is worthless as a check and must never be
# counted as one. It lived in scripts/ci/ until 2026-08-23 and was counted as a
# passing gate by the sweep in docs/audit/ZERO_SECOND_GATE_SABOTAGE_2026-08-23.md;
# replacing bin/kretikos with a script that fails every time AND deleting a
# golden pair still exited 0.
#
# THE ASSERTION LIVES IN scripts/ci/kaxi_ptx_golden_gate.sh. That gate re-emits
# every combination and demands a byte-identical match against these goldens
# (and that an `.unsupported` combination has not silently started working).
# Under the same sabotage it reports FAIL: 318 and exits 1. Do not duplicate
# that comparison here -- an asserting capture tool could not capture.
#
# Usage:
#   scripts/gpu/kaxi_ptx_capture.sh             # only write missing golden files
#   scripts/gpu/kaxi_ptx_capture.sh --force     # overwrite existing goldens
#
# Output layout:
#   tests/golden/kaxi_ptx/<mode>/<pattern>.ptx          (when supported)
#   tests/golden/kaxi_ptx/<mode>/<pattern>.sha256
#   tests/golden/kaxi_ptx/<mode>/<pattern>.unsupported  (when transpiler errors)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

GOLDEN_DIR="tests/golden/kaxi_ptx"

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

# Mode name : flag passed to kaxi-emit-ptx (empty = default f64)
MODES=(
    "default:"
    "epistemic:--epistemic"
    "f32:--f32"
    "f32_2c:--f32-2c"
    "f32e:--f32-epistemic"
    "f32_gum:--f32-gum"
)

NEW=0
KEPT=0
SUPPORTED=0
UNSUPPORTED=0

for mode_entry in "${MODES[@]}"; do
    mode="${mode_entry%%:*}"
    flag="${mode_entry#*:}"

    mode_dir="$GOLDEN_DIR/$mode"
    mkdir -p "$mode_dir"

    for pattern in "${PATTERNS[@]}"; do
        ptx_path="$mode_dir/$pattern.ptx"
        sha_path="$mode_dir/$pattern.sha256"
        unsup_path="$mode_dir/$pattern.unsupported"

        if [[ "$FORCE" -eq 0 ]] && \
           { [[ -f "$ptx_path" && -f "$sha_path" ]] || [[ -f "$unsup_path" ]]; }; then
            KEPT=$((KEPT + 1))
            continue
        fi

        rm -f "$ptx_path" "$sha_path" "$unsup_path"

        tmp_ptx="$(mktemp -t kaxi-golden.XXXXXX.ptx)"
        tmp_err="$(mktemp -t kaxi-golden.XXXXXX.err)"

        set +e
        if [[ -n "$flag" ]]; then
            ./bin/kretikos kaxi-emit-ptx "$pattern" -o "$tmp_ptx" --no-ptxas "$flag" >/dev/null 2>"$tmp_err"
        else
            ./bin/kretikos kaxi-emit-ptx "$pattern" -o "$tmp_ptx" --no-ptxas >/dev/null 2>"$tmp_err"
        fi
        rc=$?
        set -e

        if [[ "$rc" -eq 0 && -s "$tmp_ptx" ]]; then
            mv "$tmp_ptx" "$ptx_path"
            sha256sum "$ptx_path" | awk '{print $1}' > "$sha_path"
            SUPPORTED=$((SUPPORTED + 1))
            printf '  capture %-12s %-40s OK   %s bytes\n' "$mode" "$pattern" "$(wc -c < "$ptx_path")"
        else
            {
                echo "rc=$rc"
                head -n 4 "$tmp_err" 2>/dev/null || true
            } > "$unsup_path"
            rm -f "$tmp_ptx"
            UNSUPPORTED=$((UNSUPPORTED + 1))
            printf '  capture %-12s %-40s N/A  rc=%d\n' "$mode" "$pattern" "$rc"
        fi
        rm -f "$tmp_err"

        NEW=$((NEW + 1))
    done
done

echo ""
echo "=== kaxi_ptx golden capture summary ==="
echo "  new:         $NEW"
echo "  kept:        $KEPT"
echo "  supported:   $SUPPORTED"
echo "  unsupported: $UNSUPPORTED"
echo "  total combos checked: $((NEW + KEPT))"
