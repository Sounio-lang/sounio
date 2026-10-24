#!/usr/bin/env bash
# scripts/dev/knowledge_annotation_parser_coverage_source_built.sh
#
# Sister pin to scripts/dev/knowledge_annotation_parser_coverage.sh,
# using a source-built Madaros instead of the shipped ELF. If shipped
# vs source-built agree on every probe, the audit's "behaviour is the
# behaviour of the source" claim is grounded in two independent
# compilers.
#
# Scope (post-2026-09-01 extension):
#
#   1. Wrapper bracket closure (PR #2108): Intervention[f64, ...],
#      Validated[f64, Derived> — the bracket form is not a live
#      surface on the wrapper side; source-built must refuse these
#      identically to the shipped ELF.
#
#   2. Wrapper angle positive control: Intervention<f64, Derived> —
#      the angle form must check OK on the source-built ELF.
#
#   3. Angle-form Knowledge closure (this extension):
#        knowledge_angle_derived.sio   check OK
#        angle_input.sio               check OK (post-PR #2229;
#                                      Input is a lexer keyword; the
#                                      audit addendum from PR #2102
#                                      claimed parse-fail, which is
#                                      now stale.)
#        angle_source.sio              parse fail (Source is an Ident;
#                                      angle path is loud — E241 + the
#                                      angle component loop refuse it.)
#        angle_literature.sio          parse fail (same as Source.)
#
#   This script does NOT retest bracket-form Knowledge probes
#   (source.sio, literature.sio, int_skip.sio, typo_ident.sio,
#   derived_eps.sio). Those are exercised by the shipped pin's
#   dynamic mode and the Madaros Witness Gate; including them here
#   would double the work without adding an ELF/source cross-check
#   the shipped pin already performs.
#
# Usage:
#   bash scripts/dev/knowledge_annotation_parser_coverage_source_built.sh /path/to/source-built-madaros
#
# The pin script intentionally has no fall-back to bin/souc: if the
# source-built Madaros is missing, the script fails loudly. Mixing the
# two would defeat the purpose.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${1:-}"
if [[ -z "$SOUC" ]]; then
  echo "usage: $0 /path/to/source-built-madaros" >&2
  exit 2
fi
if [[ ! -x "$SOUC" ]]; then
  echo "error: $SOUC missing or not executable" >&2
  exit 2
fi

PROBE_DIR="docs/audit/probes/knowledge-annotation-parser-coverage-2026-08-19"

fail=0
note() { printf '%s\n' "$*"; }
pin_fail() { printf 'PIN FAIL: %s\n' "$*" >&2; fail=1; }

souc_ver=$(ulimit -s 524288 && env -u SOUC_BIN -u SOUNIO_SOUC_BIN "$SOUC" --version 2>/dev/null | head -1 || true)
note "SOURCE-BUILT compiler: ${souc_ver:-unknown} path=$SOUC"

check_one() {
  local file="$1"
  ( ulimit -s 524288
    env -u SOUC_BIN -u SOUNIO_SOUC_BIN \
      SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
      "$SOUC" check "$file"
  ) >/tmp/kcov_sb_check.out 2>&1
}

# Wrapper bracket closure — three probes that parse-fail on shipped ELF
# and must parse-fail identically on a source-built Madaros. If they
# disagree, ELF/source drift is the only remaining explanation.
sb_fail=(intervention_bracket_only intervention_bracket_derived validated_bracket_derived)
for name in "${sb_fail[@]}"; do
  f="$PROBE_DIR/${name}.sio"
  if [[ ! -f "$f" ]]; then
    pin_fail "missing source-built fail probe $f"
    continue
  fi
  if check_one "$f"; then
    pin_fail "source-built fail-probe $name: expected parse/check failure, got check OK"
  fi
done

# Wrapper angle form — Intervention<f64, Derived> must check OK on a
# source-built Madaros. This is the empirical counterpart to the
# bracket-form failures above: same family, different syntax.
sb_pass=(intervention_angle_derived)
for name in "${sb_pass[@]}"; do
  f="$PROBE_DIR/${name}.sio"
  if [[ ! -f "$f" ]]; then
    pin_fail "missing source-built pass probe $f"
    continue
  fi
  if check_one "$f"; then
    :
  else
    pin_fail "source-built pass-probe $name: expected check OK"
    tail -5 /tmp/kcov_sb_check.out >&2
  fi
done

# Angle-form Knowledge — the source-current clock for `Knowledge<T>` in
# angle syntax. The shipped pin's dynamic mode tests bracket-form
# Knowledge probes only; this section is the matching angle-form half.
#
# Expected matrix (current main, post-PR #2229):
#
#   knowledge_angle_derived    Knowledge<f64, Derived>     check OK
#   angle_input                Knowledge<f64, Input>      check OK
#     — Input became a lexer keyword under PR #2229 (commit 1adec5e731).
#       The audit addendum from PR #2102 still calls angle_input a
#       parse-fail probe; that is now stale. The probe files were
#       added in this branch; the empirical post-#2229 outcome is
#       check OK, identical on shipped bin/souc and on a freshly
#       rebuilt source-built Madaros.
#   angle_source               Knowledge<f64, Source>     parse fail
#   angle_literature           Knowledge<f64, Literature> parse fail
#     — Source and Literature are still Ident epsilon-sinks in the
#       bracket path, but the angle path (parse_knowledge_type's
#       comma-component loop) refuses an unknown component before
#       the sink can swallow it. This is the empirical discriminator
#       the audit addendum names "angle form is loud, bracket form
#       is silent".
#
# If any of these flip on the source-built ELF, either the parser
# changed (the audit must be re-read) or the source-built ELF drifted
# from main (the build pipeline has a problem).
sb_angle_fail=(angle_source angle_literature)
for name in "${sb_angle_fail[@]}"; do
  f="$PROBE_DIR/${name}.sio"
  if [[ ! -f "$f" ]]; then
    pin_fail "missing source-built angle-fail probe $f"
    continue
  fi
  if check_one "$f"; then
    pin_fail "source-built angle-fail-probe $name: expected parse/check failure, got check OK"
  fi
done

sb_angle_pass=(knowledge_angle_derived angle_input)
for name in "${sb_angle_pass[@]}"; do
  f="$PROBE_DIR/${name}.sio"
  if [[ ! -f "$f" ]]; then
    pin_fail "missing source-built angle-pass probe $f"
    continue
  fi
  if check_one "$f"; then
    :
  else
    pin_fail "source-built angle-pass-probe $name: expected check OK"
    tail -5 /tmp/kcov_sb_check.out >&2
  fi
done

if [[ "$fail" -ne 0 ]]; then
  note "KNOWLEDGE_ANNOTATION_PARSER_COVERAGE_SOURCE_BUILT: FAIL"
  exit 1
fi
note "KNOWLEDGE_ANNOTATION_PARSER_COVERAGE_SOURCE_BUILT: PASS"
exit 0
