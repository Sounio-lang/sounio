#!/usr/bin/env bash
# Every inline //@ known-failure must say HOW it fails; the set that does not
# may only shrink.
#
# A bare `//@ known-failure: REASON` makes the suite accept ANY nonzero result
# as an expected failure: a later SIGSEGV, a compiler crash, or a different
# wrong answer is counted as the old, audited failure. The only pin that
# existed (tests/known_failures/hardened_diagnostics_full_suite.txt, `path|substring`)
# is loaded solely for unfiltered JUnit runs and is matched against a
# 160-character tail snippet. Found on PR #2699 (2026-09-26): a pinned
# `outside_mask=9` literature failure was still accepted with any other mask
# or a crash under `--filter`.
#
# scripts/dev/run_sio_test_suite_v2.sh now reads two pins from the test itself
# and enforces them in every mode:
#   //@ xfail-expect: TEXT   (repeatable) TEXT must occur in the full output
#   //@ xfail-exit: N        the raw souc exit code must be N (124 = timeout)
# A mismatch is a fresh FAIL; a pass is still XPAS.
#
# This gate:
#   1. Self-tests that harness behaviour against a stub compiler (no souc
#      needed): pinned match -> xfail, wrong text / wrong exit -> fail,
#      pass -> xpas, stranded or malformed pin -> fail, unpinned -> xfail,
#      and the same verdicts under --filter.
#   2. Ratchets the unpinned inline known-failures against
#      tests/known_failures/unpinned_inline_known_failures.txt: a new unpinned
#      one fails; a listed one that is now pinned, deleted or no longer a
#      known failure also fails until the baseline is shrunk. Regenerate the
#      baseline with --write-baseline (only ever to remove lines).
#
# Scope and header rule mirror the harness: the files it enumerates, and
# annotations only in the leading run of `//@ `, `// ` and blank lines (a bare
# `//` line ends the header).
#
# Usage: known_failure_pin_gate.sh [--selftest | --write-baseline]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
HARNESS="$ROOT_DIR/scripts/dev/run_sio_test_suite_v2.sh"
BASELINE="$ROOT_DIR/tests/known_failures/unpinned_inline_known_failures.txt"

# Leading annotation header, exactly as the harness reads it.
header_lines() {
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ ! "$line" =~ ^[[:space:]]*//@\  && ! "$line" =~ ^[[:space:]]*//\  && ! "$line" =~ ^[[:space:]]*$ ]]; then
      break
    fi
    printf '%s\n' "$line"
  done < "$1"
}

# The files run_sio_test_suite_v2.sh enumerates when given no --test-list.
suite_files() {
  shopt -s nullglob
  local f
  for f in tests/run-pass/*.sio tests/compile-fail/*.sio \
           tests/ui/type/*.sio tests/ui/effect/*.sio tests/ui/ownership/*.sio \
           tests/ui/resolve/*.sio tests/ui/pattern/*.sio \
           tests/stdlib/*/test_*.sio tests/gpu/*.sio; do
    [[ -f "$f" ]] && printf '%s\n' "$f"
  done
}

unpinned_known_failures() {
  local f h
  while IFS= read -r f; do
    h="$(header_lines "$f")"
    grep -qE '^[[:space:]]*//@ known-failure' <<<"$h" || continue
    grep -qE '^[[:space:]]*//@ xfail-(expect|exit):' <<<"$h" && continue
    printf '%s\n' "$f"
  done < <(suite_files) | LC_ALL=C sort
}

selftest() {
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  # Stub compiler: `run|check|compile FILE` prints the file's `// stub-out:`
  # lines and exits with its `// stub-rc:` (default 0).
  cat > "$tmp/souc" <<'STUB'
#!/usr/bin/env bash
f="$2"
rc=0
while IFS= read -r l; do
  case "$l" in
    "// stub-rc: "*) rc="${l#// stub-rc: }" ;;
    "// stub-out: "*) printf '%s\n' "${l#// stub-out: }" ;;
  esac
done < "$f"
exit "$rc"
STUB
  chmod +x "$tmp/souc"

  mk() {  # mk NAME RC OUT HEADER...
    local name="$1" rc="$2" out="$3"; shift 3
    { printf '%s\n' "$@"; printf '// stub-rc: %s\n// stub-out: %s\nfn main() {}\n' "$rc" "$out"; } > "$tmp/$name.sio"
    printf '%s\n' "$tmp/$name.sio" >> "$tmp/list"
  }
  local KF='//@ known-failure: selftest'
  local RP='//@ run-pass'
  local EC='//@ expect-stdout-contains: SELFTEST_PASS'
  mk pin_ok        1   'SELFTEST_FAIL_HONEST outside_mask=9' "$RP" "$KF" '//@ xfail-expect: FAIL_HONEST outside_mask=9' '//@ xfail-exit: 1' "$EC"
  mk pin_text      1   'SELFTEST_FAIL_HONEST outside_mask=5' "$RP" "$KF" '//@ xfail-expect: FAIL_HONEST outside_mask=9' "$EC"
  mk pin_exit      139 'SELFTEST_FAIL_HONEST outside_mask=9' "$RP" "$KF" '//@ xfail-expect: FAIL_HONEST outside_mask=9' '//@ xfail-exit: 1' "$EC"
  mk pin_passes    0   'SELFTEST_PASS'                       "$RP" "$KF" '//@ xfail-expect: FAIL_HONEST' "$EC"
  mk pin_stranded  1   'SELFTEST_FAIL_HONEST'                "$RP" '//@ xfail-expect: FAIL_HONEST' "$EC"
  mk pin_bad_exit  1   'SELFTEST_FAIL_HONEST'                "$RP" "$KF" '//@ xfail-exit: one' "$EC"
  mk unpinned      139 'anything at all'                     "$RP" "$KF" "$EC"

  # Verdict per test from the JUnit report, which covers every result
  # (including the early refusals the per-result copy directory never sees).
  status_of() {  # status_of JUNIT_FILE NAME
    awk -v want="$2" '
      /<testcase name="/ {
        n = $0; sub(/.*<testcase name="/, "", n); sub(/".*/, "", n)
        cur = n
        if (cur == want && $0 ~ /\/>[[:space:]]*$/) { print "pass"; found = 1; exit }
        next
      }
      cur == want && /<failure message="stale known-failure/ { print "xpas"; found = 1; exit }
      cur == want && /<failure/                              { print "fail"; found = 1; exit }
      cur == want && /<skipped message="Known failure"/      { print "xfail"; found = 1; exit }
      END { if (!found) print "missing" }
    ' "$1"
  }

  local mode junit name want got bad=0
  for mode in unfiltered filtered; do
    junit="$tmp/junit-$mode.xml"
    local args=(--test-list "$tmp/list" --format junit)
    # Any active filter is enough: the manifest pin is never loaded under one,
    # which is exactly the mode the inline pins must still hold in.
    [[ "$mode" == filtered ]] && args+=(--filter pin)
    SOUNIO_TEST_SOUC_BIN="$tmp/souc" SOUNIO_TEST_JUNIT_FILE="$junit" \
      SOUNIO_TEST_KNOWN_FAILURES_FILE=/dev/null \
      bash "$HARNESS" "${args[@]}" > "$tmp/log-$mode" 2>&1 || true
    for name in pin_ok:xfail pin_text:fail pin_exit:fail pin_passes:xpas \
                pin_stranded:fail pin_bad_exit:fail unpinned:xfail; do
      want="${name#*:}"; name="${name%%:*}"
      got="$(status_of "$junit" "$name")"
      if [[ "$got" != "$want" ]]; then
        echo "selftest FAILED ($mode): $name is '$got', expected '$want'" >&2
        bad=1
      fi
    done
  done
  if [[ "$bad" -ne 0 ]]; then
    sed -n '1,60p' "$tmp/log-unfiltered" >&2
    return 1
  fi

  # The ratchet's header reader must stop where the harness stops.
  printf '//@ run-pass\n//\n//@ known-failure: late\nfn main() {}\n' > "$tmp/late.sio"
  if header_lines "$tmp/late.sio" | grep -q 'known-failure'; then
    echo "selftest FAILED: header reader read past a bare // line" >&2
    return 1
  fi
  echo "known_failure_pin_gate: selftest OK (7 harness cases x 2 modes, header rule)"
}

selftest

if [[ "${1:-}" == "--selftest" || "${SELFTEST:-}" == "1" ]]; then
  exit 0
fi

current="$(unpinned_known_failures)"
if [[ "${1:-}" == "--write-baseline" ]]; then
  {
    echo "# Inline //@ known-failure tests with no //@ xfail-expect / //@ xfail-exit pin."
    echo "# Shrink-only: scripts/ci/known_failure_pin_gate.sh fails on any new entry and"
    echo "# on any entry that is no longer an unpinned known failure. Pin a test, then"
    echo "# regenerate with: bash scripts/ci/known_failure_pin_gate.sh --write-baseline"
    [[ -n "$current" ]] && printf '%s\n' "$current"
  } > "$BASELINE"
  echo "known_failure_pin_gate: wrote $(grep -vc '^#' "$BASELINE" || true) entries to ${BASELINE#$ROOT_DIR/}"
  exit 0
fi

[[ -f "$BASELINE" ]] || { echo "known_failure_pin_gate: FAIL: missing ${BASELINE#$ROOT_DIR/}" >&2; exit 1; }
baseline="$(grep -v '^#' "$BASELINE" | sed '/^[[:space:]]*$/d' | LC_ALL=C sort)"

new="$(LC_ALL=C comm -13 <(printf '%s\n' "$baseline") <(printf '%s\n' "$current") | sed '/^$/d')"
gone="$(LC_ALL=C comm -23 <(printf '%s\n' "$baseline") <(printf '%s\n' "$current") | sed '/^$/d')"
fail=0
if [[ -n "$new" ]]; then
  echo "known_failure_pin_gate: FAIL: new //@ known-failure without an xfail pin:" >&2
  printf '  %s\n' $new >&2
  echo "  Add //@ xfail-expect: <text the failure prints> and/or //@ xfail-exit: <rc>" >&2
  echo "  to the leading annotation block, so a different failure is not absorbed." >&2
  fail=1
fi
if [[ -n "$gone" ]]; then
  echo "known_failure_pin_gate: FAIL: baseline lists tests that are no longer unpinned known failures:" >&2
  printf '  %s\n' $gone >&2
  echo "  Shrink the baseline: bash scripts/ci/known_failure_pin_gate.sh --write-baseline" >&2
  fail=1
fi
[[ "$fail" -eq 0 ]] || exit 1
echo "known_failure_pin_gate: PASS ($(printf '%s\n' "$current" | sed '/^$/d' | wc -l | tr -d ' ') unpinned, none new)"
