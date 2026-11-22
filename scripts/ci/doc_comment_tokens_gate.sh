#!/usr/bin/env bash
# ADR-015 row 12: documentation comments (`///`, `//!`, `/** */`, `/*! */`) are recorded
# with the token that follows them, so reference documentation can be generated from the
# source. This gate lists the doc comments of tests/doc_comments/basic.sio with
# `madaros --emit-docs` and compares the `DOC ...` lines with the expected file.
#
# The table is a side table: the token stream the parser reads is unchanged, so a doc
# comment in any position (struct fields, enum variants, before EOF) cannot break a parse.
# `////`, `/**/` and `/***` are ordinary comments and must not be listed.
#
# Usage: bash scripts/ci/doc_comment_tokens_gate.sh [path/to/madaros]
#   Default binary: $MADAROS_RAW_BIN (CI: the fresh current-source build), else bin/madaros
#   (set SOUNIO_MADAROS_BIN to pin a build).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
BIN="${1:-${MADAROS_RAW_BIN:-bin/madaros}}"
# A raw Madaros ELF needs the large stack the wrapper gives it.
ulimit -s 1048576 2>/dev/null || ulimit -s unlimited 2>/dev/null || true
FIXTURE="tests/doc_comments/basic.sio"
EXPECTED="tests/doc_comments/basic.expected"
GOT="$(mktemp)"
trap 'rm -f "$GOT"' EXIT
"$BIN" --emit-docs "$FIXTURE" 2>&1 | grep -E '^DOC' > "$GOT" || true
if [[ ! -s "$GOT" ]]; then
    echo "[doc-comment-tokens] FAIL: $BIN --emit-docs listed no documentation comments" >&2
    exit 1
fi
if ! diff -u "$EXPECTED" "$GOT"; then
    echo "[doc-comment-tokens] FAIL: the listed doc comments differ from $EXPECTED" >&2
    exit 1
fi
echo "[doc-comment-tokens] PASS: $(wc -l < "$GOT") documentation comments, as expected"
