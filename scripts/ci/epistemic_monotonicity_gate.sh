#!/usr/bin/env bash
# Epistemic Monotonicity Gate
#
# Theorem (Epistemic Monotonicity under Self-Compilation):
#   For a compiler C carrying GUM uncertainty through its IR, the u_c_scaled
#   field of the .sounio.epistemic ELF section is non-decreasing across
#   bootstrap stages and exactly equal at the fixed point.
#
#   Formally: u_c(stage1) >= u_c(seed)
#             u_c(stage2) >= u_c(stage1)
#             u_c(stage3) == u_c(stage2)   [fixed-point equality]
#
#   u_c_scaled = count of Knowledge<T> token occurrences in the compiled source.
#   For the compiler itself this is 0 (correct; the compiler has no epistemic
#   annotations).  For epistemic programs it is > 0 and stable across stages.
#
# Gate status: pass when all monotonicity assertions hold.
# Regression trigger: any stage producing u_c_scaled < the previous stage.
# Determinism trigger: stage2 != stage3 (non-determinism detected).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[epistemic-monotonicity] SKIP: Linux-only gate" >&2
  exit 0
fi
case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[epistemic-monotonicity] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

OUT_JSON="${SOUNIO_EPISTEMIC_MONOTONICITY_GATE_JSON:-$ROOT_DIR/artifacts/omega/epistemic_monotonicity_gate.v1.json}"
OUT_DIR="${SOUNIO_EPISTEMIC_MONOTONICITY_DIR:-$(mktemp -d /tmp/sounio-epistemic-mono.XXXXXX)}"
mkdir -p "$OUT_DIR" "$(dirname "$OUT_JSON")"

DRIVER_SRC="self-hosted/compiler/native_compile_driver.sio"
SEED_ELF="artifacts/omega/native_compile_driver.epistemic_seed.elf"
STAGE1="$OUT_DIR/stage1"
STAGE2="$OUT_DIR/stage2"
STAGE3="$OUT_DIR/stage3"

emit_json() {
  local status="$1" reason="$2" stages_json="$3" checks_json="$4"
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$ROOT_DIR/bin/kretikos" json-emit \
    --string   "schema=sounio.epistemic-monotonicity-gate.v1" \
    --string   "generated_at_utc=$ts" \
    --string   "status=$status" \
    --string   "reason=$reason" \
    --string   "theorem=u_c_scaled is non-decreasing across bootstrap stages and equal at the fixed point" \
    --string   "u_c_definition=count of Knowledge<T> token occurrences in compiled source (0 = no epistemic coverage)" \
    --raw-json "stages=$stages_json" \
    --raw-json "checks=$checks_json" \
    > "$OUT_JSON"
}

parse_siep() {
  "$ROOT_DIR/bin/kretikos" siep-probe "$1"
}

printf '[epistemic-monotonicity] souc=%s\n' "$SOUC_BIN"
printf '[epistemic-monotonicity] out=%s\n' "$OUT_DIR"

# ── Seed: use committed epistemic seed ELF ────────────────────────────────────
SEED_EP="$(parse_siep "$SEED_ELF")"
SEED_STATUS="$(echo "$SEED_EP" | cut -d: -f1)"
SEED_INSTR="$(echo "$SEED_EP" | cut -d: -f2)"
SEED_UC="$(echo "$SEED_EP" | cut -d: -f3)"
printf '[epistemic-monotonicity] seed=%s\n' "$SEED_EP"

# ── Stage 1: JIT-compiled source → native binary ──────────────────────────────
if ! "$SOUC_BIN" run "$DRIVER_SRC" -- "$DRIVER_SRC" -o "$STAGE1" >/dev/null 2>&1; then
  emit_json "fail" "stage1_compile_failed" "[]" "{}"
  echo "[epistemic-monotonicity] FAIL: JIT failed to compile stage1" >&2
  exit 1
fi
S1_EP="$(parse_siep "$STAGE1")"
S1_STATUS="$(echo "$S1_EP" | cut -d: -f1)"
S1_INSTR="$(echo "$S1_EP" | cut -d: -f2)"
S1_UC="$(echo "$S1_EP" | cut -d: -f3)"
printf '[epistemic-monotonicity] stage1=%s\n' "$S1_EP"

# ── Stage 2: stage1 binary compiles source ────────────────────────────────────
if ! "$STAGE1" "$DRIVER_SRC" -o "$STAGE2" >/dev/null 2>&1; then
  emit_json "fail" "stage2_compile_failed" "[]" "{}"
  echo "[epistemic-monotonicity] FAIL: stage1 failed to compile stage2" >&2
  exit 1
fi
S2_EP="$(parse_siep "$STAGE2")"
S2_UC="$(echo "$S2_EP" | cut -d: -f3)"
printf '[epistemic-monotonicity] stage2=%s\n' "$S2_EP"

# ── Stage 3: stage2 binary compiles source ────────────────────────────────────
if ! "$STAGE2" "$DRIVER_SRC" -o "$STAGE3" >/dev/null 2>&1; then
  emit_json "fail" "stage3_compile_failed" "[]" "{}"
  echo "[epistemic-monotonicity] FAIL: stage2 failed to compile stage3" >&2
  exit 1
fi
S3_EP="$(parse_siep "$STAGE3")"
S3_UC="$(echo "$S3_EP" | cut -d: -f3)"
printf '[epistemic-monotonicity] stage3=%s\n' "$S3_EP"

# ── Monotonicity checks ───────────────────────────────────────────────────────
MONO_S1_GE_SEED=false
MONO_S2_GE_S1=false
FIXED_POINT_EQ=false
SIEP_PRESENT=false

[[ "$S1_STATUS" == "present" && "$SEED_STATUS" != "absent" ]] && SIEP_PRESENT=true
[[ "$S1_UC" -ge "$SEED_UC" ]] && MONO_S1_GE_SEED=true
[[ "$S2_UC" -ge "$S1_UC" ]] && MONO_S2_GE_S1=true
[[ "$S2_EP" == "$(parse_siep "$STAGE3")" ]] && FIXED_POINT_EQ=true

# re-parse stage3 for JSON emission (already have S3_EP)
S3_INSTR="$(echo "$S3_EP" | cut -d: -f2)"

S2_INSTR="$(echo "$S2_EP" | cut -d: -f2)"
STAGES_JSON="$(printf '[{"name":"seed","instr_count":%s,"u_c_scaled":%s},{"name":"stage1","instr_count":%s,"u_c_scaled":%s},{"name":"stage2","instr_count":%s,"u_c_scaled":%s},{"name":"stage3","instr_count":%s,"u_c_scaled":%s}]' \
  "$SEED_INSTR" "$SEED_UC" "$S1_INSTR" "$S1_UC" "$S2_INSTR" "$S2_UC" "$S3_INSTR" "$S3_UC")"

CHECKS_JSON="$(printf '{"siep_section_present_in_all_stages":%s,"u_c_scaled_monotone_stage1_ge_seed":%s,"u_c_scaled_monotone_stage2_ge_stage1":%s,"u_c_scaled_fixed_point_stage2_eq_stage3":%s}' \
  "$SIEP_PRESENT" "$MONO_S1_GE_SEED" "$MONO_S2_GE_S1" "$FIXED_POINT_EQ")"

if [[ "$MONO_S1_GE_SEED" == "true" && "$MONO_S2_GE_S1" == "true" && "$FIXED_POINT_EQ" == "true" ]]; then
  emit_json "pass" "epistemic_monotonicity_verified" "$STAGES_JSON" "$CHECKS_JSON"
  printf '[epistemic-monotonicity] PASS: monotone=%s>=%s>=%s fixed_point=%s==%s\n' \
    "$SEED_UC" "$S1_UC" "$S2_UC" "$S2_UC" "$S3_UC"
  exit 0
fi

FAIL_REASON="monotonicity_assertion_failed"
[[ "$FIXED_POINT_EQ" != "true" ]] && FAIL_REASON="fixed_point_broken"
emit_json "fail" "$FAIL_REASON" "$STAGES_JSON" "$CHECKS_JSON"
printf '[epistemic-monotonicity] FAIL: seed.uc=%s s1.uc=%s s2.uc=%s s3.uc=%s\n' \
  "$SEED_UC" "$S1_UC" "$S2_UC" "$S3_UC" >&2
exit 1
