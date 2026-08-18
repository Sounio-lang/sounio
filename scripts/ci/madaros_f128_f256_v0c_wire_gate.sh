#!/usr/bin/env bash
# madaros_f128_f256_v0c_wire_gate.sh — V0-C wire format / limb-pool ladder stage.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-C
# Semantic-Lane-ID: WS-G-V0C-WIRE-LIMB-POOLS
#
# ENGINE (asserted): dual-path, deliberately named — learned from V0-A split.
#   - Scaffold probes (descriptor/payload/wire) build with the **lean_single
#     seed ELF** (same as madaros_f128_f256_numeric_*_gate.sh). That path is
#     the existing compiler-owned limb/wire *infrastructure* under the seed.
#   - Corpus consumption and V0-C green are **not** claimed on lean_single
#     alone. Full V0-C green requires a codec consumer that maps
#     tests/vectors/f128_f256_v0c/wire_*.jsonl through the limb/wire codec and
#     emits the exact success receipt below. Default Madaros is refused for
#     building self-hosted probes here only when SOUNIO_SOUC_ENGINE=lean_single
#     is forced for the *souc check* surface — scaffolds use SEED explicitly.
#
# CRITICAL SHAPE (same as V0-B):
#   - Positive control MUST fire (descriptor + payload + wire scaffold probes).
#   - External corpus integrity MUST pass (md5 + structural oracle).
#   - Gate MUST FAIL today until a real corpus consumer exists — scaffold
#     probes only exercise hard-coded cases, not wire_f128.jsonl (31) /
#     wire_f256.jsonl (24).
#
# Usage:
#   bash scripts/ci/madaros_f128_f256_v0c_wire_gate.sh
#   bash scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0c
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
if [[ ! -d "$SOUNIO_STDLIB_PATH" ]]; then
  echo "FAIL SOUNIO_STDLIB_PATH missing: $SOUNIO_STDLIB_PATH" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0c.XXXXXX")"
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

echo "=== madaros_f128_f256_ladder_gate stage=v0c ==="
echo "engine_scaffold=lean_single_seed_elf"
echo "engine_contract=Madaros_or_seed_limb_wire_codec_must_consume_external_corpus"
echo "seed_elf=$SEED_COMPILER"
echo "seed_sha256=$(sha256sum "$SEED_COMPILER" | awk '{print $1}')"
echo "stdlib=$SOUNIO_STDLIB_PATH"
echo "corpus_dir=tests/vectors/f128_f256_v0c"
echo "note=V0-A taught engine splits matter; this gate names scaffold vs corpus engines explicitly"

# ---------------------------------------------------------------------------
# Positive control — scaffold probes MUST fire (infrastructure alive).
# Built with lean_single seed (same as numeric_wire/payload gates).
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

run_scaffold_probe \
  self-hosted/compiler/f128_f256_numeric_wire_probe.sio \
  "PASS f128_f256_numeric_wire_probe bytes=136 payloads=2 limbs=6 le=exact high_bit=exact checksum=adler32:57941ddc roundtrip=byte_exact decode_negative=18 encode_negative=1 reseal_negative=2 destination=transactional" \
  numeric_wire

# ---------------------------------------------------------------------------
# External corpus (grok-cli1) — structural oracle + consumer requirement.
# ---------------------------------------------------------------------------
ORACLE="$ROOT_DIR/scripts/dev/ws_g_v0c_wire_corpus_oracle.py"
ORACLE_LOG="$TMP_DIR/v0c_oracle.log"
if [[ ! -f "$ORACLE" ]]; then
  note_fail "missing_oracle_script:$ORACLE"
else
  set +e
  python3 "$ORACLE" >"$ORACLE_LOG" 2>&1
  o_rc=$?
  set -e
  while IFS= read -r line; do
    case "$line" in
      PASS\ *) note_pass "${line#PASS }" ;;
      FAIL\ *) note_fail "${line#FAIL }" ;;
      NOTE\ *) echo "$line" ;;
      *) echo "$line" ;;
    esac
  done <"$ORACLE_LOG"
  if [[ "$o_rc" -ne 0 ]]; then
    # FAILs already counted via FAIL lines; ensure non-zero propagates.
    :
  fi
fi

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
echo "engine_scaffold=lean_single_seed_elf"
echo "engine_note=corpus_green_requires_codec_consumer_not_scaffold_alone"

if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0c_wire limbs=8 order=lsw-first payloads=4 wire_bytes=272 roundtrip=exact decode_negative=24 encode_negative=4 checksum=adler32 ir_emit=green soir_bss=green corpus_f128=31 corpus_f256=24 accept=33 reject=22"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0c"
  exit 0
fi

echo "FAIL madaros_f128_f256_ladder_gate stage=v0c" >&2
echo "first_failures:" >&2
for f in "${FAILURES[@]}"; do
  echo "  - $f" >&2
done

if printf '%s\n' "${FAILURES[@]}" | grep -q 'v0c_codec_does_not_consume_external_corpus'; then
  echo "diagnosis=V0-C_scaffold_alive_but_external_wire_corpus_unconsumed" >&2
  echo "right_reason=wire_f128.jsonl(31)+wire_f256.jsonl(24) not mapped through limb/wire codec; hardcoded probes are not the corpus" >&2
  echo "engine=scaffold_probes_use_lean_single_seed; corpus_contract_is_codec_coverage_not_engine_E218" >&2
fi

exit 1
