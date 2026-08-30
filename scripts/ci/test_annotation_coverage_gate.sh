#!/usr/bin/env bash
# Refuses a test file that the suite would silently skip.
#
# scripts/dev/run_sio_test_suite_v2.sh executes a file only if it carries one of
# the mode annotations, and skips it — reporting "no-annotation" — otherwise. A
# file sitting in tests/run-pass/ or tests/compile-fail/ with no //@ line
# therefore looks like coverage and runs nothing.
#
# Measured 2026-08-30, before this gate existed: 37 of 649 files in
# tests/compile-fail/ and 60 of 1852 in tests/run-pass/ were skipped this way,
# and NONE of the 97 declared the skip with //@ ignore or //@ known-failure.
# Waking them (#2287, #2290) surfaced eight live defects, among them a program
# with no main that the Madaros engine compiles to an ELF while lean_single
# refuses it with error[E221], and a test that prints "FAIL: let var binding"
# and exits 1 — it had been announcing its own failure unheard.
#
# The sweep is one-shot; this gate is what stops the next file from going
# dormant. //@ ignore and //@ known-failure count as passing: the point is not
# that every test must run, it is that a skip must be DECLARED rather than
# accidental.
#
# Scope deliberately mirrors the harness. run_sio_test_suite_v2.sh globs
# "$ROOT_DIR"/tests/run-pass/*.sio and .../compile-fail/*.sio — a single level,
# no recursion — so files in subdirectories (tests/run-pass/fixtures/,
# tests/run-pass/d3_openslice_len/) are imported leaves and multi-module parts
# that the suite never enumerates. Requiring annotations there would demand them
# of files that are not tests. If the harness ever starts recursing, this gate
# must follow it.
#
# Run with --selftest (or SELFTEST=1) to exercise the controls alone; they run
# automatically before the real check.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Kept in sync with the case arms in run_sio_test_suite_v2.sh: the four mode
# annotations that make a file execute, plus the two that declare a skip.
ANNOT_RE='^//@ *(run-pass|compile-fail|check-only|typecheck-fail|ignore|known-failure)'

# The harness reads a bounded header, not the whole file; matching that keeps a
# stray mention deep in a fixture from counting as an annotation.
HEADER_LINES=20

scan_dir() {
  local dir="$1" fail=0 f
  shopt -s nullglob
  for f in "$dir"/*.sio; do
    if ! head -n "$HEADER_LINES" "$f" | grep -qE "$ANNOT_RE"; then
      echo "$f: no //@ annotation in the first $HEADER_LINES lines." >&2
      echo "  The suite will skip this file and report it as \"no-annotation\"," >&2
      echo "  so it looks like coverage and runs nothing. Add the mode it needs" >&2
      echo "  (//@ run-pass, //@ compile-fail, //@ check-only, //@ typecheck-fail)" >&2
      echo "  or declare the skip (//@ ignore, //@ known-failure) with a reason." >&2
      fail=1
    fi
  done
  return "$fail"
}

selftest() {
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  # 1. a file with no annotation must be refused -- the whole point
  printf 'fn main() -> i64 { 0 }\n' > "$tmp/bare.sio"
  if scan_dir "$tmp" 2>/dev/null; then
    echo "selftest FAILED: an unannotated test file was accepted" >&2
    return 1
  fi
  rm -f "$tmp/bare.sio"

  # 2. an annotation buried past the header must NOT rescue the file, or the
  #    check could be satisfied by a mention the harness never reads
  { printf '// filler\n%.0s' $(seq 1 25); printf '//@ run-pass\nfn main() -> i64 { 0 }\n'; } > "$tmp/late.sio"
  if scan_dir "$tmp" 2>/dev/null; then
    echo "selftest FAILED: an annotation past the header window was accepted" >&2
    return 1
  fi
  rm -f "$tmp/late.sio"

  # 3. each accepted annotation must actually be accepted
  local a
  for a in run-pass compile-fail check-only typecheck-fail ignore known-failure; do
    printf '//@ %s\nfn main() -> i64 { 0 }\n' "$a" > "$tmp/ok.sio"
    if ! scan_dir "$tmp" 2>/dev/null; then
      echo "selftest FAILED: //@ $a was refused" >&2
      return 1
    fi
    rm -f "$tmp/ok.sio"
  done

  # 4. null control -- an empty directory must pass, so the refusals above are
  #    attributable to the files rather than to scan_dir failing generically
  if ! scan_dir "$tmp" 2>/dev/null; then
    echo "selftest FAILED: an empty directory was refused" >&2
    return 1
  fi

  echo "test annotation selftest passed (2 refusals + 6 accepted forms + null control)."
}

selftest

if [[ "${1:-}" == "--selftest" || "${SELFTEST:-}" == "1" ]]; then
  exit 0
fi

RC=0
scan_dir tests/run-pass || RC=1
scan_dir tests/compile-fail || RC=1

if [[ "$RC" -ne 0 ]]; then
  exit 1
fi

echo "test annotation coverage passed: every top-level run-pass and compile-fail file either runs or declares its skip."
