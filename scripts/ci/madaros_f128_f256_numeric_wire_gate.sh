#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_SHA='9ccac660ff2cc3722f0811f1a273ccc9d6b9c7a0'
SEED_COMPILER="$(realpath "${SOUNIO_F128_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"
if [[ ! -x "$SEED_COMPILER" ]]; then
  echo "FAIL bootstrap seed is not executable: $SEED_COMPILER" >&2
  exit 2
fi
if [[ "$(head -c2 "$SEED_COMPILER" 2>/dev/null)" == '#!' ]]; then
  echo "FAIL gate requires a resolved seed ELF, not a wrapper: $SEED_COMPILER" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROBE='self-hosted/compiler/f128_f256_numeric_wire_probe.sio'
CODEC='self-hosted/ir/numeric_payload_wire.sio'
PROBE_ELF="$TMP_DIR/f128-f256-numeric-wire-probe.elf"
BUILD_LOG="$TMP_DIR/build.log"
RUN_LOG="$TMP_DIR/run.log"

echo "seed_elf=$SEED_COMPILER"
echo "seed_sha256=$(sha256sum "$SEED_COMPILER" | awk '{print $1}')"
echo "source_head=$(git rev-parse HEAD)"
echo "stack_base=$BASE_SHA"

if ! "$SEED_COMPILER" "$PROBE" "$PROBE_ELF" >"$BUILD_LOG" 2>&1; then
  echo "FAIL standalone numeric wire probe did not compile with the bootstrap seed" >&2
  cat "$BUILD_LOG" >&2
  exit 1
fi
chmod +x "$PROBE_ELF"
if ! "$PROBE_ELF" >"$RUN_LOG" 2>&1; then
  echo "FAIL standalone numeric wire probe returned nonzero" >&2
  cat "$RUN_LOG" >&2
  exit 1
fi
RECEIPT='PASS f128_f256_numeric_wire_probe bytes=136 payloads=2 limbs=6 le=exact high_bit=exact checksum=adler32:57941ddc roundtrip=byte_exact decode_negative=18 encode_negative=1 reseal_negative=2 destination=transactional'
if ! grep -Fxq "$RECEIPT" "$RUN_LOG"; then
  echo "FAIL standalone numeric wire probe omitted the exact receipt" >&2
  cat "$RUN_LOG" >&2
  exit 1
fi

for exact in \
  'module ir::numeric_payload_wire' \
  'IR_NUMERIC_WIRE_MAX_BYTES: i64 = 16384' \
  'IR_NUMERIC_WIRE_MAGIC: i64 = 1347898963' \
  'IR_NUMERIC_WIRE_VERSION: i64 = 1' \
  'IR_NUMERIC_WIRE_HEADER_BYTES: i64 = 40' \
  'IR_NUMERIC_WIRE_ENTRY_BYTES: i64 = 24' \
  'IR_NUMERIC_WIRE_LIMB_BYTES: i64 = 8' \
  'IR_NUMERIC_WIRE_ERR_BAD_MAGIC' \
  'IR_NUMERIC_WIRE_ERR_BAD_VERSION' \
  'IR_NUMERIC_WIRE_ERR_TRUNCATED' \
  'IR_NUMERIC_WIRE_ERR_BAD_COUNTS' \
  'IR_NUMERIC_WIRE_ERR_BAD_LENGTH' \
  'IR_NUMERIC_WIRE_ERR_UNKNOWN_FORMAT' \
  'IR_NUMERIC_WIRE_ERR_NONCANONICAL_SPAN' \
  'IR_NUMERIC_WIRE_ERR_NONCANONICAL_COUNT' \
  'IR_NUMERIC_WIRE_ERR_TRAILING_BYTES' \
  'IR_NUMERIC_WIRE_ERR_CHECKSUM' \
  'fn ir_numeric_payload_wire_checksum_unchecked' \
  '(b << 16) | a' \
  'pub fn ir_numeric_payload_wire_reseal_canonical(' \
  'if (*wire).len < IR_NUMERIC_WIRE_HEADER_BYTES' \
  'if (*wire).len > IR_NUMERIC_WIRE_MAX_BYTES' \
  'Payload limbs retain arena order: limb 0 is least-significant' \
  'The destination is not touched until all checks pass'; do
  if ! grep -Fq "$exact" "$CODEC"; then
    echo "FAIL numeric wire structural invariant missing: $exact" >&2
    exit 1
  fi
done

if grep -Fq 'pub fn ir_numeric_payload_wire_checksum' "$CODEC"; then
  echo "FAIL unchecked checksum primitive is public" >&2
  exit 1
fi

if rg -n 'use ir::serialize|use ir::ir|IrModule|IrInstr|IrOpcode|IrNop' "$CODEC" >"$TMP_DIR/leak.log"; then
  echo "FAIL standalone codec depends on current SOIR or instruction/module IR" >&2
  cat "$TMP_DIR/leak.log" >&2
  exit 1
fi

for protected in \
  self-hosted/ir/serialize.sio \
  self-hosted/ir/ir.sio \
  self-hosted/ir/lower.sio \
  self-hosted/compiler/module_loader.sio \
  self-hosted/compiler/module_frontend.sio; do
  if ! git diff --quiet "$BASE_SHA" -- "$protected"; then
    echo "FAIL V0-C modified protected compiler/SOIR surface: $protected" >&2
    exit 1
  fi
done

{
  git diff --name-only "$BASE_SHA"
  git ls-files --others --exclude-standard
} | sort -u >"$TMP_DIR/write-set.actual"
printf '%s\n' \
  'scripts/ci/madaros_f128_f256_numeric_payload_gate.sh' \
  'scripts/ci/madaros_f128_f256_numeric_wire_gate.sh' \
  'self-hosted/compiler/f128_f256_numeric_wire_probe.sio' \
  'self-hosted/ir/numeric_payload_wire.sio' \
  >"$TMP_DIR/write-set.expected"
if ! diff -u "$TMP_DIR/write-set.expected" "$TMP_DIR/write-set.actual" \
    >"$TMP_DIR/write-set.log" 2>&1; then
  echo "FAIL V0-C write set exceeds the standalone codec/probe/gates" >&2
  cat "$TMP_DIR/write-set.log" >&2
  exit 1
fi

rg -l 'IrNumericPayloadWire|numeric_payload_wire|IR_NUMERIC_WIRE' self-hosted \
  | sort >"$TMP_DIR/wire-references.actual"
printf '%s\n' \
  'self-hosted/compiler/f128_f256_numeric_wire_probe.sio' \
  'self-hosted/ir/numeric_payload_wire.sio' \
  >"$TMP_DIR/wire-references.expected"
if ! diff -u "$TMP_DIR/wire-references.expected" "$TMP_DIR/wire-references.actual" \
    >"$TMP_DIR/integration-leak.log" 2>&1; then
  echo "FAIL standalone wire codec leaked into SOIR, IrModule, IrInstr, lowering, ABI, native, source, or arithmetic" >&2
  cat "$TMP_DIR/integration-leak.log" >&2
  exit 1
fi

echo "probe_elf_sha256=$(sha256sum "$PROBE_ELF" | awk '{print $1}')"
echo "PASS numeric wire: canonical LE header/entries/limbs and byte-exact f128/f256 roundtrip"
echo "PASS fail-closed: 18 decoder mutations + 1 corrupt-source encode + 2 bounded-reseal rejects, structured status, transactional destination"
echo "PASS containment: standalone future-section codec; current SOIR/version/IR/lowering/ABI/native/source/arithmetic untouched"
echo "PASS madaros_f128_f256_numeric_wire_gate"
