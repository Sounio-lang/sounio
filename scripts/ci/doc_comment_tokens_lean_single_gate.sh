#!/usr/bin/env bash
# ADR-015 row 12, lean_single half: lists the documentation comments of
# tests/doc_comments/basic.sio with `lean_single SRC OUT --emit-docs` and compares the
# `DOC ...` lines with the expected file. The expected file is the one the Madaros gate
# (scripts/ci/doc_comment_tokens_gate.sh) uses: the two engines must list the same comments.
#
# Usage: bash scripts/ci/doc_comment_tokens_lean_single_gate.sh [path/to/lean_single]
#   Default binary: bin/souc-lean-single-x86_64.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
BIN="${1:-bin/souc-lean-single-x86_64}"
FIXTURE="tests/doc_comments/basic.sio"
EXPECTED="tests/doc_comments/basic.expected"
GOT="$(mktemp)"
trap 'rm -f "$GOT"' EXIT
"$BIN" "$FIXTURE" /dev/null --emit-docs 2>&1 | grep -E '^DOC' > "$GOT" || true
if [[ ! -s "$GOT" ]]; then
    echo "[doc-comment-tokens-lean-single] FAIL: $BIN --emit-docs listed no documentation comments" >&2
    exit 1
fi
if ! diff -u "$EXPECTED" "$GOT"; then
    echo "[doc-comment-tokens-lean-single] FAIL: the listed doc comments differ from $EXPECTED" >&2
    exit 1
fi
echo "[doc-comment-tokens-lean-single] PASS: $(wc -l < "$GOT") documentation comments, as expected"
