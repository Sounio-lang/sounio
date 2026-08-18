#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="$ROOT_DIR/examples/research/rna_cd_confirmatory/rna_cd_manifest.sio"
VALIDATOR="$ROOT_DIR/examples/research/rna_cd_confirmatory/validate_manifest.jl"
FIXTURES="$ROOT_DIR/tests/research/rna_cd_confirmatory"
SALT="rna-cd-confirmatory-v1"

# shellcheck source=../lib/resolve_souc.sh
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

JULIA_RESOLVED="${JULIA_BIN:-}"
if [[ -z "$JULIA_RESOLVED" ]]; then
  JULIA_RESOLVED="$(command -v julia || true)"
fi
if [[ -z "$JULIA_RESOLVED" || ! -x "$JULIA_RESOLVED" ]]; then
  echo "RNA_CD_CONFIRMATORY_BLOCKED reason=julia_missing" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

ELF="$TMP_DIR/rna-cd-manifest"
ACTUAL="$TMP_DIR/manifest.tsv"

cd "$ROOT_DIR"
"$SOUC_BIN" check "$SOURCE"
"$SOUC_BIN" compile "$SOURCE" -o "$ELF"
"$ELF" "$FIXTURES/valid.tsv" "$ACTUAL" "$SALT"
cmp "$FIXTURES/expected_manifest.tsv" "$ACTUAL"

negative_cases=(
  "invalid_unbalanced.tsv:STRUCTURE"
  "invalid_order.tsv:RECORD_ID_ORDER"
  "invalid_sha.tsv:SHA256_FORMAT"
  "invalid_no_pairs.tsv:NO_MASKABLE_PAIR"
  "invalid_family_group.tsv:FAMILY_GROUP_CONFLICT"
  "invalid_duplicate_cross_group.tsv:DUPLICATE_SEQUENCE_CROSS_GROUP"
)

for negative_case in "${negative_cases[@]}"; do
  fixture="${negative_case%%:*}"
  expected_reason="${negative_case#*:}"
  set +e
  "$ELF" "$FIXTURES/$fixture" "$TMP_DIR/$fixture.out" "$SALT" \
    >"$TMP_DIR/$fixture.log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 11 ]]; then
    cat "$TMP_DIR/$fixture.log" >&2
    echo "RNA_CD_CONFIRMATORY_FAIL fixture=$fixture expected_rc=11 actual_rc=$rc" >&2
    exit 1
  fi
  if ! grep -Fq "reason=$expected_reason" "$TMP_DIR/$fixture.log"; then
    cat "$TMP_DIR/$fixture.log" >&2
    echo "RNA_CD_CONFIRMATORY_FAIL fixture=$fixture expected_reason=$expected_reason" >&2
    exit 1
  fi
done

"$JULIA_RESOLVED" --startup-file=no "$VALIDATOR" --self-test
"$JULIA_RESOLVED" --startup-file=no "$VALIDATOR" \
  "$FIXTURES/valid.tsv" "$ACTUAL" "$SALT"

for negative_case in "${negative_cases[@]}"; do
  fixture="${negative_case%%:*}"
  set +e
  "$JULIA_RESOLVED" --startup-file=no "$VALIDATOR" \
    "$FIXTURES/$fixture" "$FIXTURES/expected_manifest.tsv" "$SALT" \
    >"$TMP_DIR/julia-$fixture.log" 2>&1
  julia_input_rc=$?
  set -e
  if [[ $julia_input_rc -eq 0 ]]; then
    cat "$TMP_DIR/julia-$fixture.log" >&2
    echo "RNA_CD_CONFIRMATORY_FAIL fixture=$fixture reason=julia_accepted_invalid_input" >&2
    exit 1
  fi
done

set +e
"$JULIA_RESOLVED" --startup-file=no "$VALIDATOR" \
  "$FIXTURES/valid.tsv" "$FIXTURES/invalid_manifest.tsv" "$SALT" \
  >"$TMP_DIR/julia-invalid-artifact.log" 2>&1
julia_rc=$?
set -e
if [[ $julia_rc -eq 0 ]]; then
  cat "$TMP_DIR/julia-invalid-artifact.log" >&2
  echo "RNA_CD_CONFIRMATORY_FAIL reason=julia_accepted_corruption" >&2
  exit 1
fi

echo "RNA_CD_CONFIRMATORY_CONTRACT_PASS records=7 groups=6 nested=3 crossing=4 folds=5 tolerance=0"
