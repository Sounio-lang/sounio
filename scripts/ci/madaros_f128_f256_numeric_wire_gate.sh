#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

V0C_MERGE_COMMIT='f7b7c85f5c01e40a8372e54887174c4764ec339a'
V0C_FEATURE_COMMIT='12d7163a5f46ede31aa38a83905e6c1c1934bb9e'

v0c_expected_feature_write_set() {
  printf '%s\n' \
    'scripts/ci/madaros_f128_f256_numeric_payload_gate.sh' \
    'scripts/ci/madaros_f128_f256_numeric_wire_gate.sh' \
    'self-hosted/compiler/f128_f256_numeric_wire_probe.sio' \
    'self-hosted/ir/numeric_payload_wire.sio'
}

v0c_feature_write_set_matches() {
  local repo="$1"
  local feature_commit="$2"
  local prefix="$3"

  git -C "$repo" diff-tree --no-commit-id --name-only -r \
    "${feature_commit}^" "$feature_commit" | sort -u >"${prefix}.actual"
  v0c_expected_feature_write_set >"${prefix}.expected"
  diff -u "${prefix}.expected" "${prefix}.actual" >"${prefix}.diff" 2>&1
}

v0c_reviewed_owned_surfaces_unchanged() {
  local repo="$1"
  local feature_commit="$2"
  local drift_file="$3"
  local owned

  for owned in \
    scripts/ci/madaros_f128_f256_numeric_payload_gate.sh \
    self-hosted/compiler/f128_f256_numeric_wire_probe.sio \
    self-hosted/ir/numeric_payload_wire.sio; do
    if ! git -C "$repo" diff --quiet "$feature_commit" -- "$owned"; then
      printf '%s\n' "$owned" >"$drift_file"
      return 1
    fi
  done
  : >"$drift_file"
}

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

v0c_write_set_self_test() {
  local synthetic_repo="$TMP_DIR/synthetic-repo"
  git init -q -b main "$synthetic_repo"
  git -C "$synthetic_repo" config user.name 'Sounio Gate Self-Test'
  git -C "$synthetic_repo" config user.email 'gate-selftest@sounio.local'

  printf '%s\n' 'base' >"$synthetic_repo/base.txt"
  git -C "$synthetic_repo" add base.txt
  git -C "$synthetic_repo" commit -q -m 'base'
  local base_commit
  base_commit="$(git -C "$synthetic_repo" rev-parse HEAD)"

  git -C "$synthetic_repo" checkout -q -b v0c-feature
  while IFS= read -r path; do
    mkdir -p "$(dirname "$synthetic_repo/$path")"
    printf '%s\n' 'v0c' >"$synthetic_repo/$path"
  done < <(v0c_expected_feature_write_set)
  git -C "$synthetic_repo" add .
  git -C "$synthetic_repo" commit -q -m 'synthetic exact V0-C'
  local exact_feature
  exact_feature="$(git -C "$synthetic_repo" rev-parse HEAD)"

  git -C "$synthetic_repo" checkout -q main
  printf '%s\n' 'concurrent main work' >"$synthetic_repo/concurrent-main.txt"
  git -C "$synthetic_repo" add concurrent-main.txt
  git -C "$synthetic_repo" commit -q -m 'advance first parent'
  git -C "$synthetic_repo" merge -q --no-ff -m 'merge synthetic V0-C' "$exact_feature"
  if ! v0c_feature_write_set_matches "$synthetic_repo" "$exact_feature" "$TMP_DIR/synthetic-merge"; then
    echo "FAIL exact V0-C write set was contaminated by an advanced merge first parent" >&2
    cat "$TMP_DIR/synthetic-merge.diff" >&2
    return 1
  fi
  if ! v0c_reviewed_owned_surfaces_unchanged \
    "$synthetic_repo" "$exact_feature" "$TMP_DIR/synthetic-initial-drift"; then
    echo "FAIL synthetic merge changed a reviewed V0-C owned surface" >&2
    cat "$TMP_DIR/synthetic-initial-drift" >&2
    return 1
  fi

  printf '%s\n' 'later codec drift' >>"$synthetic_repo/self-hosted/ir/numeric_payload_wire.sio"
  git -C "$synthetic_repo" add self-hosted/ir/numeric_payload_wire.sio
  git -C "$synthetic_repo" commit -q -m 'later codec-owned drift'
  if v0c_reviewed_owned_surfaces_unchanged \
    "$synthetic_repo" "$exact_feature" "$TMP_DIR/synthetic-later-drift"; then
    echo "FAIL synthetic later codec-owned edit was silently ignored" >&2
    return 1
  fi
  if ! grep -Fxq 'self-hosted/ir/numeric_payload_wire.sio' "$TMP_DIR/synthetic-later-drift"; then
    echo "FAIL synthetic later drift did not identify the codec-owned file" >&2
    cat "$TMP_DIR/synthetic-later-drift" >&2
    return 1
  fi

  git -C "$synthetic_repo" checkout -q -b bad-v0c "$base_commit"
  while IFS= read -r path; do
    mkdir -p "$(dirname "$synthetic_repo/$path")"
    printf '%s\n' 'v0c' >"$synthetic_repo/$path"
  done < <(v0c_expected_feature_write_set)
  mkdir -p "$synthetic_repo/self-hosted/ir"
  printf '%s\n' 'unexpected' >"$synthetic_repo/self-hosted/ir/unexpected_v0c_leak.sio"
  git -C "$synthetic_repo" add .
  git -C "$synthetic_repo" commit -q -m 'synthetic contaminated V0-C'
  local bad_feature
  bad_feature="$(git -C "$synthetic_repo" rev-parse HEAD)"
  if v0c_feature_write_set_matches "$synthetic_repo" "$bad_feature" "$TMP_DIR/synthetic-bad"; then
    echo "FAIL synthetic unexpected V0-C file was accepted" >&2
    return 1
  fi
  if ! grep -Fq '+self-hosted/ir/unexpected_v0c_leak.sio' "$TMP_DIR/synthetic-bad.diff"; then
    echo "FAIL synthetic rejection did not identify the unexpected V0-C file" >&2
    cat "$TMP_DIR/synthetic-bad.diff" >&2
    return 1
  fi

  echo "PASS synthetic merge: advanced first parent does not contaminate the V0-C feature write set"
  echo "PASS synthetic adversary: unexpected V0-C feature file is rejected"
  echo "PASS synthetic drift: later codec-owned edit is rejected explicitly"
  echo "PASS madaros_f128_f256_numeric_wire_gate write-set-self-test"
}

if [[ "${1:-}" == '--write-set-self-test' ]]; then
  v0c_write_set_self_test
  exit 0
fi

SEED_COMPILER="$(realpath "${SOUNIO_F128_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"
if [[ ! -x "$SEED_COMPILER" ]]; then
  echo "FAIL bootstrap seed is not executable: $SEED_COMPILER" >&2
  exit 2
fi
if [[ "$(head -c2 "$SEED_COMPILER" 2>/dev/null)" == '#!' ]]; then
  echo "FAIL gate requires a resolved seed ELF, not a wrapper: $SEED_COMPILER" >&2
  exit 2
fi

PROBE='self-hosted/compiler/f128_f256_numeric_wire_probe.sio'
CODEC='self-hosted/ir/numeric_payload_wire.sio'
PROBE_ELF="$TMP_DIR/f128-f256-numeric-wire-probe.elf"
BUILD_LOG="$TMP_DIR/build.log"
RUN_LOG="$TMP_DIR/run.log"

echo "seed_elf=$SEED_COMPILER"
echo "seed_sha256=$(sha256sum "$SEED_COMPILER" | awk '{print $1}')"
echo "source_head=$(git rev-parse HEAD)"
echo "v0c_merge_commit=$V0C_MERGE_COMMIT"
echo "v0c_feature_commit=$V0C_FEATURE_COMMIT"

if ! git cat-file -e "${V0C_MERGE_COMMIT}^{commit}" 2>/dev/null; then
  echo "FAIL canonical V0-C merge commit is unavailable: $V0C_MERGE_COMMIT" >&2
  exit 1
fi
if [[ "$(git rev-parse "${V0C_MERGE_COMMIT}^2")" != "$V0C_FEATURE_COMMIT" ]]; then
  echo "FAIL canonical V0-C feature commit is not the merge second parent" >&2
  exit 1
fi
if ! git merge-base --is-ancestor "$V0C_MERGE_COMMIT" HEAD; then
  echo "FAIL canonical V0-C merge is not an ancestor of current HEAD" >&2
  exit 1
fi
if ! v0c_feature_write_set_matches "$ROOT_DIR" "$V0C_FEATURE_COMMIT" "$TMP_DIR/canonical-feature"; then
  echo "FAIL canonical V0-C feature commit exceeds the standalone codec/probe/gates write set" >&2
  cat "$TMP_DIR/canonical-feature.diff" >&2
  exit 1
fi

if ! v0c_reviewed_owned_surfaces_unchanged \
  "$ROOT_DIR" "$V0C_FEATURE_COMMIT" "$TMP_DIR/current-owned-drift"; then
  echo "FAIL V0-C owned surface drifted after the reviewed feature commit: $(cat "$TMP_DIR/current-owned-drift")" >&2
  exit 1
fi

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
  if ! git diff --quiet "${V0C_FEATURE_COMMIT}^" "$V0C_FEATURE_COMMIT" -- "$protected"; then
    echo "FAIL V0-C modified protected compiler/SOIR surface: $protected" >&2
    exit 1
  fi
done

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
echo "PASS provenance: canonical V0-C feature write set exact; reviewed owned surfaces unchanged after merge"
echo "PASS containment: standalone future-section codec; current SOIR/version/IR/lowering/ABI/native/source/arithmetic untouched"
echo "PASS madaros_f128_f256_numeric_wire_gate"
