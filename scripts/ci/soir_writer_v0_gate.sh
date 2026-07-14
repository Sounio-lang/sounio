#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WRITER="$ROOT/self-hosted/ir/soir_writer.sio"
WITNESS="$ROOT/self-hosted/test_soir_writer_v0.sio"
SPEC="$ROOT/docs/internal/implementation/SOIR_WRITER_V0.md"

fail() {
  printf 'soir-writer-v0 gate: FAIL: %s\n' "$*" >&2
  exit 1
}

require_fixed() {
  local file="$1"
  local text="$2"
  grep -Fq -- "$text" "$file" || fail "missing '$text' in ${file#$ROOT/}"
}

for file in "$WRITER" "$WITNESS" "$SPEC"; do
  [[ -f "$file" ]] || fail "missing ${file#$ROOT/}"
done

# The prototype owns cursor state, never the 128 KiB bytes.
require_fixed "$WRITER" 'pub struct SoirWritePlan {'
require_fixed "$WRITER" 'struct SoirWriterCursor {'
require_fixed "$WRITER" 'pub let SOIR_WRITER_VERSION_V5: i8 = 5'
require_fixed "$WRITER" 'pub let SOIR_V5_WIRE_PARAM_SLOTS: i64 = 64'
require_fixed "$WRITER" 'pub fn soir_writer_preflight_empty_extensions_v5('
require_fixed "$WRITER" 'pub fn soir_writer_emit_empty_extensions_v5('
require_fixed "$WRITER" 'out_buf: &![i8; 131072]'
require_fixed "$WRITER" 'let verified = soir_writer_preflight_empty_extensions_v5('
require_fixed "$WRITER" '(*function).param_count > IR_MAX_PARAMS'
require_fixed "$WRITER" '(*function).param_count > SOIR_V5_WIRE_PARAM_SLOTS'
require_fixed "$WRITER" 'while i < SOIR_V5_WIRE_PARAM_SLOTS {'

if grep -Eq '(buf|out_buf): \[i8; 131072\]' "$WRITER"; then
  fail 'writer passes the 128 KiB buffer by value'
fi
if grep -Eq '(^|[[:space:]])(pub[[:space:]]+)?var[[:space:]]+[A-Z_]+:' "$WRITER"; then
  fail 'writer introduces mutable global state'
fi
if grep -Fq 'SOIR_WRITE_FAILED' "$WRITER"; then
  fail 'writer depends on legacy global failure state'
fi
if grep -Eq 'emit_.*v4|writer_.*v4' "$WRITER"; then
  fail 'v4 must remain decode/golden only'
fi

# Writer v0 copies the already-established wire tags. Compare the explicit
# tables mechanically so the prototype cannot drift while the legacy core is
# still the oracle.
for enum_name in IrOpcode BinaryOp UnaryOp; do
  if ! cmp -s \
    <(grep -E "${enum_name}::[A-Za-z0-9_]+ => [0-9]+," "$ROOT/self-hosted/ir/soir_core.sio" | sed -E "s/^.*(${enum_name}::[A-Za-z0-9_]+ => [0-9]+,).*$/\\1/" | sort -u) \
    <(grep -E "${enum_name}::[A-Za-z0-9_]+ => [0-9]+," "$WRITER" | sed -E "s/^.*(${enum_name}::[A-Za-z0-9_]+ => [0-9]+,).*$/\\1/" | sort -u); then
    fail "$enum_name wire tags differ from legacy SOIR core"
  fi
done

# Keep the new writer outside the default compiler path during differential
# qualification. These files are read-only for this lane.
for file in \
  "$ROOT/self-hosted/ir/serialize.sio" \
  "$ROOT/self-hosted/ir/soir_core.sio" \
  "$ROOT/self-hosted/ir/heap_storage.sio" \
  "$ROOT/scripts/bootstrap/bootstrap_concat.sh"; do
  if grep -Fq 'soir_writer' "$file"; then
    fail "default path imports prototype in ${file#$ROOT/}"
  fi
done

require_fixed "$WITNESS" 'soir_writer_compare_with_legacy(&module, 320)'
require_fixed "$WITNESS" 'soir_writer_compare_with_legacy(&module, 1632)'
require_fixed "$WITNESS" 'SOIR_WRITER_UNSUPPORTED_OPCODE'
require_fixed "$WITNESS" 'soir_writer_buffer_is_canary'
require_fixed "$WITNESS" 'v4[4] = 4 as i8'
require_fixed "$WITNESS" 'PASS soir_writer_v0_differential'

printf 'soir-writer-v0 gate: static contract PASS\n'

# Import routing for standalone self-hosted modules is not assumed. When the
# packaged compiler can resolve the lane, run the differential witness; when it
# cannot, fail explicitly unless the caller selected static-only qualification.
if [[ "${SOIR_WRITER_STATIC_ONLY:-0}" == "1" ]]; then
  printf 'soir-writer-v0 gate: dynamic witness NOT RUN (SOIR_WRITER_STATIC_ONLY=1)\n'
  exit 0
fi

SOUC="${SOUC:-$ROOT/bin/souc}"
"$SOUC" check "$WRITER"

check_dir="$(mktemp -d -t soir-writer-v0-check.XXXXXX)"
trap 'rm -rf "$check_dir"' EXIT
set +e
"$SOUC" check "$ROOT/self-hosted/ir/serialize.sio" >"$check_dir/legacy.log" 2>&1
legacy_rc=$?
"$SOUC" check "$WITNESS" >"$check_dir/witness.log" 2>&1
witness_rc=$?
set -e

if [[ "$witness_rc" -ne 0 ]]; then
  sed -n 's/.*error\[\(E[0-9][0-9][0-9]\).*/\1/p' "$check_dir/legacy.log" | sort | uniq -c >"$check_dir/legacy.categories"
  sed -n 's/.*error\[\(E[0-9][0-9][0-9]\).*/\1/p' "$check_dir/witness.log" | sort | uniq -c >"$check_dir/witness.categories"
  if [[ "$legacy_rc" -ne 0 ]] && cmp -s "$check_dir/legacy.categories" "$check_dir/witness.categories"; then
    printf 'soir-writer-v0 gate: witness checker BASELINE-EQUIVALENT (legacy import privacy diagnostics)\n'
    cat "$check_dir/witness.categories"
    printf 'soir-writer-v0 gate: dynamic differential NOT RUN (modular import routing blocked)\n'
    if [[ "${SOIR_WRITER_REQUIRE_DYNAMIC:-0}" == "1" ]]; then
      fail 'dynamic differential required but modular import routing is blocked'
    fi
    fail 'dynamic differential not run; use SOIR_WRITER_STATIC_ONLY=1 only for explicit static qualification'
  fi
  cat "$check_dir/witness.log" >&2
  fail 'witness checker introduced diagnostics beyond the legacy baseline'
fi

"$SOUC" run "$WITNESS" | tee "$check_dir/witness.out"
grep -Fxq 'PASS soir_writer_v0_differential' "$check_dir/witness.out" \
  || fail 'differential PASS marker absent'
printf 'soir-writer-v0 gate: dynamic differential PASS\n'
