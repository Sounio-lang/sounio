# scripts/lib/classify_compile.sh — shared compile-outcome classification.
# Source this file; do not execute directly.
#
# Why this exists
# ---------------
# Across scripts/ci/*.sh, `rc != 0` is read as the proposition "the compiler
# semantically refused this program". It is not. It is the disjunction
#
#     refused ∨ crashed ∨ timed-out ∨ OOM-killed ∨ binary-missing
#             ∨ stdlib-path-wrong ∨ shell-error
#
# and in this repo it is unsound in the OTHER direction too: per
# scripts/ci/souc-native-wrapper.sh, the raw lean_single ELF exits 0 on silent
# lexer/parser failure and writes a ~35 kB stub ELF. (A leaked instance of that
# stub sits at the repo root as the file `-o`.) So `rc == 0` does not mean
# "accepted" either.
#
# The measured scoreboard, over the 18 gates that assert a compile must fail:
# 2 check for a signal, 0 check for a timeout, 6 check that no artifact was
# emitted. a64_compile_fail_parity_gate.sh, which asserts complete x86/arm64
# refusal parity over the whole tests/compile-fail corpus, does none of them --
# its probe() sends both compilers' output to /dev/null and deletes the output
# path without ever testing whether it was written.
#
# This is an EXTRACTION, not an invention. witness_declares_its_sabotage_gate.sh
# already classifies deaths as run / compile-refused / crash / timeout /
# misattributed and fails closed over SOUNIO_WITNESS_UNCLEAN_CEILING -- for
# tests/run-pass only. madaros_global_capacity_gate.sh checks rc>=128 plus
# artifact absence plus the diagnostic. madaros_visibility_context_gate.sh
# checks exact rc plus diagnostic cardinality plus a fatal-text guard.
# lean_single_pub_use_reexport_gate.sh factors its predicate out and self-tests
# it against synthetic adversarial logs. The class vocabulary below matches
# theirs deliberately: the point is one implementation, not a second dialect.
#
# Every run lands in exactly one named class. Nothing falls through to a
# default. A gate that cannot classify an outcome must say so, not average it
# into a verdict.
#
# After sourcing:
#   sounio_classify_compile RC LOG [OUT]  — sets SOUNIO_CC_CLASS / SOUNIO_CC_DETAIL
#   sounio_diag_codes LOG                 — sorted unique diagnostic codes, one per line
#   sounio_primary_diag LOG               — code on the first diagnostic, or empty
#   sounio_is_fatal_log LOG               — true if the log shows a crash
#   sounio_expect_refusal RC LOG OUT RULE — 0 iff REFUSED and rule matches

# Guard against double-sourcing.
if [[ -n "${_SOUNIO_CLASSIFY_COMPILE_LOADED:-}" ]]; then
  return 0
fi
_SOUNIO_CLASSIFY_COMPILE_LOADED=1

# --- classes -----------------------------------------------------------------
# ACCEPTED              rc=0, success marker present, non-empty artifact
# REFUSED               rc in 1..123, diagnostic present, NO artifact
# REFUSED_UNDIAGNOSED   rc in 1..123, no diagnostic — cannot attribute to a rule
# REFUSED_WITH_ARTIFACT rc in 1..123 but an artifact was emitted — contract violation
# SILENT_FAILURE        rc=0, no success marker (the raw-ELF stub pathology)
# CRASHED               rc>=128, or fatal text in the log
# TIMEOUT               rc=124 (GNU timeout) or rc=137 (SIGKILL / --kill-after / OOM)
# INFRA                 rc=126 (not executable), rc=127 (not found), missing log

# `Killed` catches the kernel OOM reaper's message; rc=137 catches the wait
# status. Neither alone is reliable, because which one surfaces depends on
# whether a shell was in the middle.
_SOUNIO_CC_FATAL_RE='segmentation fault|core dumped|terminated by signal|bus error|illegal instruction|double free|stack smashing|out of memory|Killed'

# Real-success markers, per emitter. lean_single prints `compile: fns=N` on a
# completed compile (lean_single.sio); the check path prints `check: OK`; the
# Madaros visibility preflight prints `run_check_mode: verdict=0`.
_SOUNIO_CC_SUCCESS_RE='^compile: fns=|^check: OK|^run_check_mode: verdict=0'

# Verdict markers that mean "the compiler decided against this program", as
# opposed to "the compiler died". lean_single.sio:29509 prints `typecheck:
# failed` and returns 1 before reaching any write_file call.
_SOUNIO_CC_VERDICT_FAIL_RE='^typecheck: failed|^run_check_mode: verdict=1'

sounio_is_fatal_log() {
  local log="$1"
  [[ -f "$log" ]] || return 1
  grep -Eiq "$_SOUNIO_CC_FATAL_RE" "$log"
}

# All diagnostic codes present, sorted unique. Must handle every emitter shape
# in the tree, because there is no single catalog:
#   error[E218] in mod::fn at 3..9: msg     (check.sio:2872)
#   error[E170]: ... at line N              (lean_single.sio hardcoded sites)
#   E200 `name` at line N                   (lean_single.sio:15642, bare)
#
# Both readers are written to be safe under `set -euo pipefail`, which 437 of
# the 452 scripts in scripts/ci/ set. A bare `grep ... | head -1` is not: on no
# match grep exits 1, pipefail propagates it, and set -e kills the *caller*
# mid-gate. a64_compile_fail_parity_gate.sh already hit exactly this and
# patched it with a trailing `|| true` -- "an all-comments baseline made the
# gate exit 1 with no verdict line, which reads as a failure and is in fact
# total success". A shared library must not re-export that trap, so no
# uncaptured pipeline appears below and every path returns 0 explicitly.
sounio_diag_codes() {
  local log="$1"
  [[ -f "$log" ]] || return 0
  local codes
  codes="$(grep -oE '\b[EPW][0-9]{3,4}\b' "$log" 2>/dev/null || true)"
  [[ -n "$codes" ]] || return 0
  printf '%s\n' "$codes" | LC_ALL=C sort -u
  return 0
}

# The code on the FIRST diagnostic the compiler printed -- not the first code
# anywhere in the log. Those differ, and the difference is not academic: on
# tests/compile-fail/alternative_requires_metadata.sio both targets open with an
# identical `error[P0003]: Type mismatch ...`, but x86 goes on to print an E200
# further down and aarch64 does not. Scanning for the first `E###` skipped the
# P0003 that actually stopped the compile, compared two unrelated later lines,
# and reported a divergence where the compilers agreed.
#
# So: locate the first diagnostic line, then read the code off THAT line, or
# report none if it carries no code. Families in the tree are E (161 sites),
# P (1) and W (1); a bare `E200 `x` at line N` form with no `error:` prefix also
# exists (lean_single.sio:15642), so it is matched at line start too.
sounio_primary_diag() {
  local log="$1"
  [[ -f "$log" ]] || return 0
  local first
  first="$(grep -m1 -E '^(error|warning)|^[EPW][0-9]{3,4}\b' "$log" 2>/dev/null || true)"
  [[ -n "$first" ]] || return 0
  local code
  code="$(printf '%s' "$first" | grep -oE '\b[EPW][0-9]{3,4}\b' | head -1 || true)"
  [[ -n "$code" ]] || return 0
  printf '%s\n' "$code"
  return 0
}

# Any diagnostic, coded or not. A large fraction of lean_single.sio diagnostics
# carry no code at all (`error: unresolved function body for call target fn#N`),
# so a coded-only test would misread them as "the compiler said nothing".
_sounio_has_diagnostic() {
  local log="$1"
  [[ -f "$log" ]] || return 1
  grep -qE '\b[EPW][0-9]{3,4}\b|^error:|^error\[|'"$_SOUNIO_CC_VERDICT_FAIL_RE" "$log"
}

# sounio_classify_compile RC LOG [OUT]
# Sets SOUNIO_CC_CLASS and SOUNIO_CC_DETAIL. Always returns 0 — the class is
# the answer, not the exit status. Callers branch on SOUNIO_CC_CLASS.
sounio_classify_compile() {
  local rc="$1"
  local log="$2"
  local out="${3:-}"

  SOUNIO_CC_CLASS=""
  SOUNIO_CC_DETAIL=""

  local have_out=0
  if [[ -n "$out" && -s "$out" ]]; then
    have_out=1
  fi

  if [[ ! -f "$log" ]]; then
    SOUNIO_CC_CLASS="INFRA"
    SOUNIO_CC_DETAIL="log_missing:$log"
    return 0
  fi

  # Order matters. The dangerous confounders are decided before the ordinary
  # interpretations, so that a crash can never be read as a refusal.
  case "$rc" in
    124)
      SOUNIO_CC_CLASS="TIMEOUT"
      SOUNIO_CC_DETAIL="timeout_rc124"
      return 0
      ;;
    137)
      SOUNIO_CC_CLASS="TIMEOUT"
      SOUNIO_CC_DETAIL="sigkill_rc137_timeout_or_oom"
      return 0
      ;;
    126)
      SOUNIO_CC_CLASS="INFRA"
      SOUNIO_CC_DETAIL="not_executable_rc126"
      return 0
      ;;
    127)
      SOUNIO_CC_CLASS="INFRA"
      SOUNIO_CC_DETAIL="not_found_rc127"
      return 0
      ;;
  esac

  if [[ "$rc" -ge 128 ]]; then
    SOUNIO_CC_CLASS="CRASHED"
    SOUNIO_CC_DETAIL="signal_$((rc - 128))_rc${rc}"
    return 0
  fi

  # A wrapper can mask a signal behind a tidy exit code, and souc-native-wrapper.sh
  # is documented to rewrite exit codes. When rc and the log disagree about
  # whether the process died, the log wins.
  if sounio_is_fatal_log "$log"; then
    SOUNIO_CC_CLASS="CRASHED"
    SOUNIO_CC_DETAIL="fatal_text_in_log_rc${rc}"
    return 0
  fi

  if [[ "$rc" -eq 0 ]]; then
    if grep -qE "$_SOUNIO_CC_SUCCESS_RE" "$log" && [[ "$have_out" -eq 1 ]]; then
      SOUNIO_CC_CLASS="ACCEPTED"
      SOUNIO_CC_DETAIL="success_marker+artifact"
      return 0
    fi
    # The raw-ELF pathology: exit 0, no success marker, often a stub artifact.
    # Never ACCEPTED — the compiler did not report doing the work.
    SOUNIO_CC_CLASS="SILENT_FAILURE"
    SOUNIO_CC_DETAIL="rc0_no_success_marker artifact=${have_out}"
    return 0
  fi

  # rc in 1..123 — the compiler decided something. Which?
  if [[ "$have_out" -eq 1 ]]; then
    # A refusal that still wrote an artifact is a contract violation, not a
    # refusal: downstream steps can pick the file up.
    SOUNIO_CC_CLASS="REFUSED_WITH_ARTIFACT"
    SOUNIO_CC_DETAIL="rc${rc}_emitted_artifact:$out"
    return 0
  fi
  if _sounio_has_diagnostic "$log"; then
    SOUNIO_CC_CLASS="REFUSED"
    SOUNIO_CC_DETAIL="rc${rc}_diag:$(sounio_primary_diag "$log" | tr -d '\n')"
    return 0
  fi
  # Rejected, but said nothing — cannot be attributed to any rule, so it cannot
  # witness that a specific check fired.
  SOUNIO_CC_CLASS="REFUSED_UNDIAGNOSED"
  SOUNIO_CC_DETAIL="rc${rc}_no_diagnostic"
  return 0
}

# sounio_expect_refusal RC LOG OUT [RULE]
# Returns 0 only for a clean, attributable refusal. RULE, when given, is either
# an E-code (E218) or a literal message fragment that must appear in the log.
# This is the function that replaces `[[ $rc -ne 0 ]]`.
sounio_expect_refusal() {
  local rc="$1"
  local log="$2"
  local out="${3:-}"
  local rule="${4:-}"

  sounio_classify_compile "$rc" "$log" "$out"
  [[ "$SOUNIO_CC_CLASS" == "REFUSED" ]] || return 1
  [[ -z "$rule" ]] && return 0

  if [[ "$rule" =~ ^[EPW][0-9]{3,4}$ ]]; then
    sounio_diag_codes "$log" | grep -qx "$rule" || return 1
  else
    grep -qF -- "$rule" "$log" || return 1
  fi
  return 0
}
