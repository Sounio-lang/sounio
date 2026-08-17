#!/usr/bin/env bash
# SIGPIPE hygiene gate (2026-08-17).
#
# Two CI incidents, one mechanism: a verdict-carrying pipeline under
# `set -o pipefail` lost its writer to an early-exiting reader.
#   * scripts/dev/run_sio_test_suite_v2.sh:434 (then-line) logged
#     "echo: write error: Broken pipe" in a run whose single failure was a
#     "missing error" flake on a test that passes elsewhere -- `grep -q`
#     exits on first match, closes the pipe, and the flushing `echo` fails
#     the pipeline; `if ! ...` then reads the match as absent.
#   * scripts/ci/madaros_ir_capacity_probe.sh logged "sort: fflush failed:
#     Broken pipe" and died before its REPORT_ONLY exit 0, from `| head -10`.
#
# The fixes remove the racy shapes (here-strings instead of `echo | grep -q`;
# `sed -n '1,N{...p}'` instead of `| head -N`). This gate keeps them removed.
# Three arms, because an instrument with one arm measures nothing:
#   1. CANARY      -- this environment can still produce the failure class.
#                     If it cannot, the gate refuses rather than certifies.
#   2. FIXED FORM  -- the replacement still decides correctly on huge input.
#   3. SHAPE BAN   -- the guarded files no longer contain the racy shapes.
set -uo pipefail

HARNESS="scripts/dev/run_sio_test_suite_v2.sh"
PROBE="scripts/ci/madaros_ir_capacity_probe.sh"

fail() { echo "SIGPIPE_HYGIENE_GATE_FAIL: $*" >&2; exit 1; }

for f in "$HARNESS" "$PROBE"; do
    [[ -f "$f" ]] || fail "guarded file absent: $f (nothing to scan)"
done

# Big enough to overflow any pipe buffer (Linux default 64 KiB; bash may ask
# for more): a writer of this size is still flushing when a fast reader exits.
BIG="needle_sigpipe_canary$(printf 'y%.0s' $(seq 1 200000))"

# --- 1. CANARY -------------------------------------------------------------
# `true` exits before reading. If this ever returns 0, the environment can no
# longer reproduce the mechanism this gate exists to guard against, and a
# green run would certify nothing. Refuse loudly instead.
canary_rc=0
printf '%s\n' "$BIG" | true || canary_rc=$?
if [[ $canary_rc -eq 0 ]]; then
    fail "canary did not reproduce: pipe-to-instant-exit returned 0 here, so this gate cannot certify anything on this machine"
fi
echo "canary: writer lost to an instant-exit reader under pipefail (rc=$canary_rc) -- the failure class is observable here"

# --- 2. FIXED FORM ----------------------------------------------------------
# The here-string form has no writer process: verdicts must be correct on
# input far larger than any pipe buffer, present or absent needle, exactly
# like the harness uses it.
if ! grep -qF -- "needle_sigpipe_canary" <<<"$BIG"; then
    fail "here-string grep lost a present needle in $BIG bytes"
fi
if grep -qF -- "absent_needle" <<<"$BIG"; then
    fail "here-string grep matched an absent needle"
fi
# The probe's replacement shape: prints exactly the first 10, reads all input,
# never exits early. It must succeed under pipefail with more input than any
# buffer.
probe_lines="$(printf 'x\n%.0s' $(seq 1 500) | sed -n '1,10{s/^/  /p}' | wc -l)"
[[ "$probe_lines" == "10" ]] || fail "sed top-10 shape printed $probe_lines lines, expected 10"
echo "fixed forms: here-string grep and sed top-10 decide correctly on 200 KiB+ input"

# --- 3. SHAPE BAN -----------------------------------------------------------
# Executable lines only: comments may name the forbidden shape while
# documenting why it is forbidden, and that is not a regression.
executable() { grep -v '^[[:space:]]*#' "$1"; }
# No `echo "$captured" | grep -q` in the harness (the false-"missing" shape).
if executable "$HARNESS" | grep -n 'echo "$output" | grep -q\|echo "$test_output" | grep -q' >/dev/null; then
    executable "$HARNESS" | grep -n 'echo "$output" | grep -q\|echo "$test_output" | grep -q' | sed 's/^/  /'
    fail "verdict-carrying echo|grep -q pipeline is back in $HARNESS"
fi
# No `| head` in the probe's executable lines: it runs under set -euo
# pipefail, and any early-exiting reader there can kill the step before its
# own exit path.
if executable "$PROBE" | grep -n '| head' >/dev/null; then
    executable "$PROBE" | grep -n '| head' | sed 's/^/  /'
    fail "early-exiting '| head' reader is back in $PROBE"
fi
echo "shape ban: no echo|grep -q in $HARNESS, no | head in $PROBE"

echo "SIGPIPE_HYGIENE_GATE_OK"
