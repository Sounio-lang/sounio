#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

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

SOURCE_HEAD="$(git rev-parse HEAD)"
PROBE='self-hosted/compiler/f128_f256_numeric_payload_probe.sio'
PROBE_ELF="$TMP_DIR/f128-f256-numeric-payload-probe.elf"
BUILD_LOG="$TMP_DIR/build.log"
RUN_LOG="$TMP_DIR/run.log"

echo "seed_elf=$SEED_COMPILER"
echo "seed_sha256=$(sha256sum "$SEED_COMPILER" | awk '{print $1}')"
echo "source_head=$SOURCE_HEAD"

if ! "$SEED_COMPILER" "$PROBE" "$PROBE_ELF" >"$BUILD_LOG" 2>&1; then
  echo "FAIL numeric payload probe did not compile with the bootstrap seed" >&2
  cat "$BUILD_LOG" >&2
  exit 1
fi
chmod +x "$PROBE_ELF"
if ! "$PROBE_ELF" >"$RUN_LOG" 2>&1; then
  echo "FAIL numeric payload probe returned nonzero" >&2
  cat "$RUN_LOG" >&2
  exit 1
fi
if ! grep -Fxq 'PASS f128_f256_numeric_payload_probe payloads=3 limbs=8 order=lsw-first duplicate_ids=distinct full=256x4 negative_cases=10' "$RUN_LOG"; then
  echo "FAIL numeric payload probe omitted the exact receipt" >&2
  cat "$RUN_LOG" >&2
  exit 1
fi

PAYLOAD_FILE='self-hosted/ir/numeric_payload.sio'
if ! grep -Fq 'module ir::numeric_payload' "$PAYLOAD_FILE"; then
  echo "FAIL numeric payload pool is not compiler-owned under ir::numeric_payload" >&2
  exit 1
fi
for exact in \
  'IR_WIDE_NUMERIC_MAX_PAYLOADS: i64 = 256' \
  'IR_WIDE_NUMERIC_MAX_LIMBS: i64 = 1024' \
  'IR_WIDE_NUMERIC_MAX_LIMBS_PER_PAYLOAD: i64 = 4' \
  'entries: [IrWideNumericPayloadEntry; 256]' \
  'limbs: [i64; 1024]' \
  'raw_limbs: [i64; 4]' \
  'required > IR_WIDE_NUMERIC_MAX_LIMBS_PER_PAYLOAD' \
  'descriptor.storage_bits / 64' \
  'limb 0 is the least-significant'; do
  if ! grep -Fq "$exact" "$PAYLOAD_FILE"; then
    echo "FAIL numeric payload structural invariant missing: $exact" >&2
    exit 1
  fi
done

rg -l 'IrWideNumericPayload|wide_numeric_payload|numeric_payload|IR_WIDE_NUMERIC' self-hosted \
  | sort >"$TMP_DIR/payload-references.actual"
printf '%s\n' \
  'self-hosted/compiler/f128_f256_numeric_payload_probe.sio' \
  'self-hosted/compiler/f128_f256_numeric_wire_probe.sio' \
  'self-hosted/ir/numeric_payload.sio' \
  'self-hosted/ir/numeric_payload_wire.sio' \
  >"$TMP_DIR/payload-references.expected"
if ! diff -u "$TMP_DIR/payload-references.expected" "$TMP_DIR/payload-references.actual" \
    >"$TMP_DIR/integration-leak.log" 2>&1; then
  echo "FAIL numeric payload surface leaked beyond the V0-B arena and bounded standalone V0-C wire codec" >&2
  cat "$TMP_DIR/integration-leak.log" >&2
  exit 1
fi

echo "PASS f128/f256 payload pool exact roundtrip and deterministic corruption rejection"
echo "PASS containment: no IrInstr/lowering/ABI/SOIR/arithmetic/source integration"
echo "PASS madaros_f128_f256_numeric_payload_gate"
