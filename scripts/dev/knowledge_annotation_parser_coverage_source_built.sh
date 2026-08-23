#!/usr/bin/env bash
# scripts/dev/knowledge_annotation_parser_coverage_source_built.sh
#
# Same pin as scripts/dev/knowledge_annotation_parser_coverage.sh, but
# using a source-built Madaros instead of the shipped ELF. This lets us
# empirically confirm or refute ELF/source drift: if shipped vs
# source-built agree on every probe, the audit's "behaviour is the
# behaviour of the source" claim is grounded in two independent
# compilers.
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
PIN="scripts/dev/knowledge_annotation_parser_coverage.sh"

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

pass_probes=(
  derived computed measured valid validuntil validwhile
  source literature input source_eps int_skip typo_ident
  knowledge_angle_derived
)
for name in "${pass_probes[@]}"; do
  f="$PROBE_DIR/${name}.sio"
  if [[ ! -f "$f" ]]; then
    pin_fail "missing pass probe $f"
    continue
  fi
  if check_one "$f"; then
    :
  else
    pin_fail "pass-probe $name: expected check OK, got rc=$?"
    tail -5 /tmp/kcov_sb_check.out >&2
  fi
done

fail_probes=(derived_eps)
for name in "${fail_probes[@]}"; do
  f="$PROBE_DIR/${name}.sio"
  if [[ ! -f "$f" ]]; then
    pin_fail "missing fail probe $f"
    continue
  fi
  if check_one "$f"; then
    pin_fail "fail-probe $name: expected parse/check failure, got check OK"
  fi
done

# Source-built wrappers — empirically close the wrapper bracket question.
# If shipped vs source-built disagree, the audit's behaviour claim breaks.
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

sb_fail=(intervention_bracket_only intervention_bracket_derived validated_bracket_derived angle_source angle_literature angle_input)
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

if [[ "$fail" -ne 0 ]]; then
  note "KNOWLEDGE_ANNOTATION_PARSER_COVERAGE_SOURCE_BUILT: FAIL"
  exit 1
fi
note "KNOWLEDGE_ANNOTATION_PARSER_COVERAGE_SOURCE_BUILT: PASS"
exit 0
