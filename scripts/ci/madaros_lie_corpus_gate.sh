#!/usr/bin/env bash
# The lie corpus must be REFUSED by the Madaros built from this commit.
#
# Why this gate exists
# --------------------
# Every lie-corpus compile-fail fixture (tests/compile-fail/lie*.sio, see
# claude/research/2026-11-22-lie-corpus.md in the project) declares
# `//@ requires: madaros`. The Full Test Suite runs without Madaros and marks
# all of them SKIPPED; the Madaros Corpus Regression job runs tests/run-pass
# only. Measured on PR #3744 (head 04bcb7b9, 2026-11-26): lie01a..lie01h were
# skipped in all four Full Test Suite shards, and no other job checked them.
# "The compiler refuses these lies" was therefore a manual claim, not a
# CI-enforced one: a checker change that stopped refusing one of them would
# have left CI green.
#
# What it checks
# --------------
# For each fixture: `souc check` under the given Madaros, classified with
# scripts/lib/classify_compile.sh. The fixture passes only if the class is
# exactly REFUSED (not CRASHED, TIMEOUT, ACCEPTED, ...) AND every
# `//@ error-pattern:` line it declares appears in the log. A refusal for an
# unrelated reason is not a witness for the lie the fixture names.
# There is no baseline: every lie-corpus fixture must pass.
#
# Usage:
#   MADAROS_RAW_BIN=/path/to/madaros bash scripts/ci/madaros_lie_corpus_gate.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# shellcheck source=../lib/classify_compile.sh
source "$ROOT_DIR/scripts/lib/classify_compile.sh"

SOUC="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
TIMEOUT_SECS="${TIMEOUT_SECS:-120}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-lie-corpus.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "MADAROS_LIE_CORPUS_GATE_START"
echo "souc=$SOUC"
[[ -x "$SOUC" ]] || { echo "error: missing souc at $SOUC" >&2; exit 1; }

ls tests/compile-fail/lie*.sio 2>/dev/null | LC_ALL=C sort > "$WORK_DIR/corpus.txt" || true
n_corpus="$(wc -l < "$WORK_DIR/corpus.txt" | tr -d ' ')"
# The selector narrowing to nothing is itself a failure, never a silent pass.
[[ "$n_corpus" -gt 0 ]] || { echo "error: no tests/compile-fail/lie*.sio fixture found" >&2; exit 1; }
echo "corpus_files=$n_corpus"

fail=0
while IFS= read -r f; do
    base="$(basename "$f" .sio)"
    log="$WORK_DIR/$base.log"
    set +e
    timeout "$TIMEOUT_SECS" "$SOUC" check "$f" >"$log" 2>&1
    rc=$?
    set -e
    sounio_classify_compile "$rc" "$log" ""
    missing=""
    while IFS= read -r want; do
        [[ -z "$want" ]] && continue
        grep -Fq -- "$want" "$log" || missing="${missing}[${want}]"
    done < <(sed -n 's|^//@[[:space:]]*error-pattern:[[:space:]]*\(.*\)$|\1|p' "$f")
    n_patterns="$(grep -c '^//@[[:space:]]*error-pattern:' "$f" || true)"
    if [[ "$SOUNIO_CC_CLASS" == "REFUSED" && -z "$missing" && "$n_patterns" -gt 0 ]]; then
        printf 'PASS\t%s\trc=%s\tpatterns=%s\n' "$base" "$rc" "$n_patterns"
    else
        printf 'FAIL\t%s\trc=%s\tclass=%s\tpatterns=%s\tmissing=%s\n' \
            "$base" "$rc" "$SOUNIO_CC_CLASS" "$n_patterns" "${missing:-none}" >&2
        echo "=== $base (tail) ===" >&2
        tail -n 12 "$log" >&2
        fail=1
    fi
done < "$WORK_DIR/corpus.txt"

if [[ "$fail" -ne 0 ]]; then
    echo "MADAROS_LIE_CORPUS_GATE=FAIL corpus=$n_corpus" >&2
    exit 1
fi
echo "MADAROS_LIE_CORPUS_GATE=PASS corpus=$n_corpus"
