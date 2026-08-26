#!/usr/bin/env bash
# madaros_f128_f256_v0d_softfloat_gate.sh — V0-D softfloat ladder stage.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-D
# Semantic-Lane-ID: WS-G-V0D-SOFTFLOAT-LIMB-ROUTINES
#
# ENGINE (named — V0-A taught this matters):
#   - Positive-control scaffold probes build with the **lean_single seed ELF**
#     (descriptor / payload infrastructure), same pattern as V0-C scaffolds.
#   - V0-D *green* is **not** a lean_single claim and **not** “Madaros accepts
#     source f128 +”. It requires a compiler-owned softfloat (limb routines)
#     that consumes tests/vectors/f128_f256_v0d/arith_hard_*.jsonl with
#     **bit-identity** to MPFR `result` under RNE — including trap families
#     that a widen-f64 shortcut fails.
#   - Default Madaros source surface may still E249-reject user `+` on f128
#     (V0-B/V0-D ladder: source ops stay compile-fail until later). Softfloat
#     is compiler-internal limb code tested by a dedicated consumer.
#
# CRITICAL SHAPE (V0-B/V0-C):
#   - Positive control MUST fire.
#   - External corpus integrity MUST pass.
#   - Gate MUST FAIL today until a softfloat corpus consumer exists.
#   - Correctness is not “plausible values on easy inputs”: green requires
#     bit-identity on ALL hard rows and explicitly on MUST_TRAP_IDS
#     (halfway / sticky / cancel / rump).
#
# Usage:
#   bash scripts/ci/madaros_f128_f256_v0d_softfloat_gate.sh
#   bash scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0d
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
if [[ -n "${SOUNIO_STDLIB_PATH:-}" && ! -d "${SOUNIO_STDLIB_PATH}" ]]; then
  unset SOUNIO_STDLIB_PATH || true
fi
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SEED_COMPILER="$(realpath "${SOUNIO_F128_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"
if [[ ! -x "$SEED_COMPILER" ]]; then
  echo "FAIL seed compiler not executable: $SEED_COMPILER" >&2
  exit 2
fi
if [[ "$(head -c2 "$SEED_COMPILER" 2>/dev/null)" == '#!' ]]; then
  echo "FAIL seed must be a resolved ELF, not a wrapper: $SEED_COMPILER" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0d.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()

note_pass() {
  PASS=$((PASS + 1))
  echo "PASS $1"
}

note_fail() {
  FAIL=$((FAIL + 1))
  FAILURES+=("$1")
  echo "FAIL $1" >&2
}

echo "=== madaros_f128_f256_ladder_gate stage=v0d ==="
echo "engine_scaffold=lean_single_seed_elf"
echo "engine_contract=compiler_owned_softfloat_limb_routines_vs_MPFR_hard_corpus"
echo "seed_elf=$SEED_COMPILER"
echo "seed_sha256=$(sha256sum "$SEED_COMPILER" | awk '{print $1}')"
echo "corpus_dir=tests/vectors/f128_f256_v0d"
echo "note=V0-D green is bit-identity on hard cases including halfway/sticky/cancel/rump — not easy-input equality"

# ---------------------------------------------------------------------------
# Positive control — limb/format infrastructure alive (seed-built probes).
# ---------------------------------------------------------------------------
run_scaffold_probe() {
  local src="$1"
  local expect_line="$2"
  local label="$3"
  local elf="$TMP_DIR/${label}.elf"
  local blog="$TMP_DIR/${label}.build.log"
  local rlog="$TMP_DIR/${label}.run.log"
  if [[ ! -f "$ROOT_DIR/$src" ]]; then
    note_fail "scaffold_missing:$src"
    return
  fi
  if ! "$SEED_COMPILER" "$ROOT_DIR/$src" "$elf" >"$blog" 2>&1; then
    note_fail "scaffold_build_failed:$label"
    tail -20 "$blog" >&2 || true
    return
  fi
  chmod +x "$elf"
  if ! "$elf" >"$rlog" 2>&1; then
    note_fail "scaffold_run_failed:$label"
    cat "$rlog" >&2 || true
    return
  fi
  if ! grep -Fq "$expect_line" "$rlog"; then
    note_fail "scaffold_missing_receipt:$label"
    cat "$rlog" >&2 || true
    return
  fi
  note_pass "positive_control_scaffold:$label"
}

run_scaffold_probe \
  self-hosted/compiler/f128_f256_format_descriptor_probe.sio \
  "PASS f128_f256_format_descriptor_probe" \
  format_descriptor

run_scaffold_probe \
  self-hosted/compiler/f128_f256_numeric_payload_probe.sio \
  "PASS f128_f256_numeric_payload_probe payloads=3 limbs=8 order=lsw-first duplicate_ids=distinct full=256x4 negative_cases=10" \
  numeric_payload

# ---------------------------------------------------------------------------
# Hard corpus oracle + widen-f64 trap classification + consumer requirement.
# ---------------------------------------------------------------------------
ORACLE="$ROOT_DIR/scripts/dev/ws_g_v0d_softfloat_corpus_oracle.py"
ORACLE_LOG="$TMP_DIR/v0d_oracle.log"
MUST_TRAP_FILE="$TMP_DIR/must_trap_ids.txt"
CATCH_FILE="$TMP_DIR/catch_widen_f64.txt"
MISS_FILE="$TMP_DIR/miss_widen_f64.txt"
: >"$MUST_TRAP_FILE"
: >"$CATCH_FILE"
: >"$MISS_FILE"

if [[ ! -f "$ORACLE" ]]; then
  note_fail "missing_oracle_script:$ORACLE"
else
  set +e
  python3 "$ORACLE" >"$ORACLE_LOG" 2>&1
  o_rc=$?
  set -e
  while IFS= read -r line; do
    case "$line" in
      PASS\ *)
        note_pass "${line#PASS }"
        ;;
      FAIL\ *)
        note_fail "${line#FAIL }"
        ;;
      NOTE\ *)
        echo "$line"
        ;;
      CATCH_WIDEN_F64\ *)
        echo "${line#CATCH_WIDEN_F64 }" >>"$CATCH_FILE"
        echo "$line"
        ;;
      MISS_WIDEN_F64\ *)
        echo "${line#MISS_WIDEN_F64 }" >>"$MISS_FILE"
        echo "$line"
        ;;
      MUST_TRAP_IDS\ *)
        echo "${line#MUST_TRAP_IDS }" | tr ',' '\n' >"$MUST_TRAP_FILE"
        echo "$line"
        ;;
      CATCH_WIDEN_F64_BEGIN|CATCH_WIDEN_F64_END|MISS_WIDEN_F64_BEGIN|MISS_WIDEN_F64_END)
        echo "$line"
        ;;
      *)
        echo "$line"
        ;;
    esac
  done <"$ORACLE_LOG"
  unset o_rc
fi

# Explicit correctness bar (must appear in oracle output)
if grep -Fq 'PASS correctness_contract' "$ORACLE_LOG" 2>/dev/null; then
  note_pass "correctness_bar_declared_bit_identity_plus_must_trap"
else
  note_fail "correctness_bar_missing"
fi

n_catch=$(wc -l <"$CATCH_FILE" | tr -d ' ')
n_miss=$(wc -l <"$MISS_FILE" | tr -d ' ')
n_trap=$(grep -c . "$MUST_TRAP_FILE" 2>/dev/null || echo 0)
echo "widen_f64_catch_count=$n_catch"
echo "widen_f64_miss_count=$n_miss"
echo "must_trap_id_count=$n_trap"

if [[ "${n_catch:-0}" -lt 1 ]]; then
  note_fail "empty_catch_widen_f64_set"
fi

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
echo "engine_scaffold=lean_single_seed_elf"
echo "engine_contract=compiler_owned_softfloat_vs_MPFR_arith_hard_corpus"

if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0d_softfloat ops=add/sub/mul/div/cmp/sqrt/rump limb_routines=green const_fold=exact rounded=ieee754-2019 must_trap=${n_trap} catch_widen_f64=${n_catch} miss_widen_f64=${n_miss} bit_identity=all_hard_rows"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0d"
  exit 0
fi

echo "FAIL madaros_f128_f256_ladder_gate stage=v0d" >&2
echo "first_failures:" >&2
for f in "${FAILURES[@]}"; do
  echo "  - $f" >&2
done

if printf '%s\n' "${FAILURES[@]}" | grep -q 'v0d_softfloat_does_not_consume_hard_corpus'; then
  echo "diagnosis=V0-D_scaffold_alive_but_softfloat_hard_corpus_unconsumed" >&2
  echo "right_reason=arith_hard_f128.jsonl(53)+arith_hard_f256.jsonl(50) not driven through compiler-owned limb softfloat; must_trap_ids untested" >&2
  echo "engine=scaffold_probes_use_lean_single_seed; green_requires_bit_identity_on_MPFR_hard_corpus_including_halfway_sticky_cancel_rump" >&2
  echo "correctness=equality_on_easy_inputs_alone_is_insufficient; consumer must fail if widen-f64 path used on CATCH_WIDEN_F64 set" >&2
fi

# Always print trap summary on failure for implementers
echo "CATCH_WIDEN_F64_count=$n_catch (see gate log CATCH_WIDEN_F64 lines)" >&2
echo "MISS_WIDEN_F64_count=$n_miss (still required for full bit-identity green)" >&2

exit 1
