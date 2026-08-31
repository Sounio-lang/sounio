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

# Every file below carried a verdict-deciding `echo ... | grep -q` (or a
# printf writer) that #1763 exiled from the harness and the wired-presence
# change exiled from the gates: `grep -q` exits at first match, the flushing
# writer fails under pipefail, and a present needle reads as absent.
SHAPE_BAN_GREPQ_FILES=(
    "$HARNESS"
    "$PROBE"
    scripts/ci/ontology_cli_smoke_gate.sh
    scripts/ci/self_falsifying_compiler_gate.sh
    scripts/ci/semantic_orc_depression_orc_gate.sh
    scripts/ci/sedenion_phi_injectivity_gate.sh
    scripts/ci/sedenion_cd_qbig_gate.sh
    scripts/ci/cd_tower_seam_gate.sh
    scripts/ci/sounio_validation.sh
    scripts/ci/mli_s3_bit_identity_gate.sh
    scripts/ci/kretikos_kaxi_phase_y_gate.sh
    scripts/ci/claim_ast_gate.sh
    scripts/ci/native_v2_frontend_convergence_gate.sh
)
# Same mechanism, reader side: `| head` exits after its window and any writer
# still flushing fails under pipefail before its own exit path runs. The
# harness is not listed here: its two `| head` uses are display-only and were
# reviewed as noise-free in #1763.
SHAPE_BAN_HEAD_FILES=(
    "$PROBE"
    scripts/ci/ontology_cli_smoke_gate.sh
    scripts/ci/self_falsifying_compiler_gate.sh
    scripts/ci/semantic_orc_depression_orc_gate.sh
    scripts/ci/sedenion_phi_injectivity_gate.sh
    scripts/ci/sedenion_cd_qbig_gate.sh
    scripts/ci/cd_tower_seam_gate.sh
    scripts/ci/sounio_validation.sh
    scripts/ci/mli_s3_bit_identity_gate.sh
    scripts/ci/kretikos_kaxi_phase_y_gate.sh
    scripts/ci/claim_ast_gate.sh
    scripts/ci/native_v2_frontend_convergence_gate.sh
)
# The ban greps capture whole and assert on the capture: `executable | grep -q`
# here would recreate the banned mechanism inside the gate that bans it.
for f in "${SHAPE_BAN_GREPQ_FILES[@]}"; do
    [[ -f "$f" ]] || fail "guarded file absent: $f (nothing to scan)"
    viol="$(executable "$f" | grep -nE '(echo|printf) [^|]*\| *grep -q' || true)"
    if [[ -n "$viol" ]]; then
        printf '%s\n' "$viol" | sed "s|^|$f:|"
        fail "verdict-carrying echo|grep -q pipeline is back in $f"
    fi
done
for f in "${SHAPE_BAN_HEAD_FILES[@]}"; do
    [[ -f "$f" ]] || fail "guarded file absent: $f (nothing to scan)"
    viol="$(executable "$f" | grep -nE '\| *head\b' || true)"
    if [[ -n "$viol" ]]; then
        printf '%s\n' "$viol" | sed "s|^|$f:|"
        fail "early-exiting '| head' reader is back in $f"
    fi
done
echo "shape ban: no echo|grep -q in ${#SHAPE_BAN_GREPQ_FILES[@]} files, no | head in ${#SHAPE_BAN_HEAD_FILES[@]} files"

echo "SIGPIPE_HYGIENE_GATE_OK"
