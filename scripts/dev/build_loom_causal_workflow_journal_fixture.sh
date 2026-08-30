#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
MODULE="${SOUNIO_LOOM_CAUSAL_WORKFLOW_JOURNAL_MODULE:-$ROOT_DIR/tools/loom/src/loom_causal_workflow.ml}"
FIXTURE="${SOUNIO_LOOM_CAUSAL_WORKFLOW_JOURNAL_FIXTURE:-$ROOT_DIR/tools/loom/causal_workflow_journal_fixture.ml}"
OUTPUT="${SOUNIO_LOOM_CAUSAL_WORKFLOW_JOURNAL_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-causal-workflow-journal-fixture}"

fail() {
  printf 'build-loom-causal-workflow-journal-fixture: FAIL: %s\n' "$*" >&2
  exit 1
}

[[ -f "$MODULE" && ! -L "$MODULE" ]] || fail 'OCaml journal module is absent or linked'
[[ -f "$FIXTURE" && ! -L "$FIXTURE" ]] || fail 'OCaml fixture is absent or linked'
command -v ocamlfind >/dev/null || fail 'ocamlfind is unavailable'
ocamlfind query cryptokit >/dev/null || fail 'Cryptokit is unavailable'

work="$(mktemp -d "${TMPDIR:-/tmp}/loom-causal-workflow-journal-build.XXXXXX")"
trap 'rm -rf "$work"' EXIT
cp "$MODULE" "$work/loom_causal_workflow.ml"
cp "$FIXTURE" "$work/causal_workflow_journal_fixture.ml"
(
  cd "$work"
  ocamlfind ocamlopt -package unix,cryptokit -linkpkg \
    loom_causal_workflow.ml causal_workflow_journal_fixture.ml \
    -o loom-causal-workflow-journal-fixture
)
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$work/loom-causal-workflow-journal-fixture" "$OUTPUT"
printf 'BUILT_CAUSAL_WORKFLOW_JOURNAL path=%s operational_language=OCaml semantic_authority=Sounio action=9037\n' "$OUTPUT"
