#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../lib/classify_compile.sh
source "$ROOT_DIR/scripts/lib/classify_compile.sh"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

POSITIVE_SRC="tests/frontend/ontology_bundle_directive_snomed.sio"
NEGATIVE_SRC="tests/frontend/ontology_bundle_directive_snomed_reject.sio"
POSITIVE_EXPANDED="$TMP_DIR/ontology_bundle_directive_snomed.expanded.sio"
POSITIVE_EXPANDED_2="$TMP_DIR/ontology_bundle_directive_snomed.expanded.second.sio"
NEGATIVE_EXPANDED="$TMP_DIR/ontology_bundle_directive_snomed_reject.expanded.sio"
CACHE_DIR="$TMP_DIR/ontocache"

bash scripts/ontology/expand_ontology_bundle_directives.sh \
  --cache-dir "$CACHE_DIR" \
  "$POSITIVE_SRC" "$POSITIVE_EXPANDED" \
  >"$TMP_DIR/expand-positive.stdout" 2>"$TMP_DIR/expand-positive.stderr"
bash scripts/ontology/expand_ontology_bundle_directives.sh \
  --cache-dir "$CACHE_DIR" \
  "$POSITIVE_SRC" "$POSITIVE_EXPANDED_2" \
  >"$TMP_DIR/expand-positive-second.stdout" 2>"$TMP_DIR/expand-positive-second.stderr"
if ! grep -q 'cache MISS' "$TMP_DIR/expand-positive.stderr"; then
  printf 'FAIL  first ontology-bundle expansion did not populate .ontocache\n' >&2
  cat "$TMP_DIR/expand-positive.stderr" >&2
  exit 1
fi
if ! grep -q 'cache HIT' "$TMP_DIR/expand-positive-second.stderr"; then
  printf 'FAIL  second ontology-bundle expansion did not reuse .ontocache\n' >&2
  cat "$TMP_DIR/expand-positive-second.stderr" >&2
  exit 1
fi
if ! cmp -s "$POSITIVE_EXPANDED" "$POSITIVE_EXPANDED_2"; then
  printf 'FAIL  cached ontology-bundle expansion is not deterministic\n' >&2
  exit 1
fi
printf 'PASS  ontology-bundle expansion populated and reused deterministic .ontocache\n'

bash scripts/ontology/expand_ontology_bundle_directives.sh \
  --cache-dir "$CACHE_DIR" \
  "$NEGATIVE_SRC" "$NEGATIVE_EXPANDED" \
  >"$TMP_DIR/expand-negative.stdout" 2>"$TMP_DIR/expand-negative.stderr"
if ! grep -q 'cache HIT' "$TMP_DIR/expand-negative.stderr"; then
  printf 'FAIL  negative ontology-bundle expansion did not reuse .ontocache\n' >&2
  cat "$TMP_DIR/expand-negative.stderr" >&2
  exit 1
fi

if bin/souc run "$POSITIVE_EXPANDED" >"$TMP_DIR/positive.log" 2>&1 &&
   grep -q 'ontology bundle directive snomed OK' "$TMP_DIR/positive.log"; then
  printf 'PASS  %s expanded and accepted SNOMED subclass witness\n' "$POSITIVE_SRC"
else
  printf 'FAIL  %s did not pass after ontology-bundle expansion\n' "$POSITIVE_SRC" >&2
  cat "$TMP_DIR/positive.log" >&2
  exit 1
fi

__refusal_rc=0
bin/souc check "$NEGATIVE_EXPANDED" >"$TMP_DIR/negative.log" 2>&1 || __refusal_rc=$?
if [[ "$__refusal_rc" -eq 0 ]]; then
  printf 'FAIL  %s unexpectedly accepted sibling as subclass after expansion\n' "$NEGATIVE_SRC" >&2
  cat "$TMP_DIR/negative.log" >&2
  exit 1
fi
# rc != 0 is not yet a refusal: a crash, timeout or OOM kill is not (#2425).
sounio_expect_failed_exit "$__refusal_rc" "$TMP_DIR/negative.log" || { printf 'FAIL  %s: not a clean refusal: %s\n' "$TMP_DIR/negative.log" "$(sounio_refusal_reason)" >&2; cat "$TMP_DIR/negative.log" >&2; exit 1; }
if grep -q 'cannot prove ontology subsumption' "$TMP_DIR/negative.log"; then
  printf 'PASS  %s expanded and rejected sibling subsumption\n' "$NEGATIVE_SRC"
else
  printf 'FAIL  %s failed without expected ontology diagnostic\n' "$NEGATIVE_SRC" >&2
  cat "$TMP_DIR/negative.log" >&2
  exit 1
fi

cat <<'MSG'
Ontology bundle directive expansion gate passed.
This proves the `//@ ontology-bundle: "..."` source form can be expanded
through the C .dontology importer into ordinary ontology declarations before
checking. It also proves a deterministic pre-native text .ontocache is populated
and reused for repeated expansions. It is a pre-native bridge: the compiler
checker still owns ontology reasoning, and native compile-time side-table
loading remains future work. Bounded native side-table .ontocache persistence
is covered separately by the native scan/load-plan gate.
MSG
