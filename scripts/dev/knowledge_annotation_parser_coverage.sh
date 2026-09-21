#!/usr/bin/env bash
# scripts/dev/knowledge_annotation_parser_coverage.sh
#
# Executable pin for docs/audit/KNOWLEDGE_ANNOTATION_PARSER_COVERAGE_2026-08-19.md
#
# Two clocks, labeled:
#   STATIC  — current self-hosted/ text (declared enum cases vs parser
#             construction sites vs lexer keywords). This is the authority
#             for "what the source can produce".
#   DYNAMIC — ./bin/souc check of the discriminator probes. This is the
#             default user-facing ELF (Madaros), NOT a from-source rebuild.
#             Skip with SOUNIO_KCOV_SKIP_DYNAMIC=1.
#
# Exit 1 if a pin moves. That is the tripwire: either the gap closed, or
# the gap widened, and the audit must be re-read.
#
# Stack: Madaros raw ELF SEGVs under the pod default 8MB stack. Always
# raise it. Unset SOUC_BIN so a poisoned env cannot silently measure a
# different checkout.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PROBE_DIR="docs/audit/probes/knowledge-annotation-parser-coverage-2026-08-19"
AST="self-hosted/parser/ast.sio"
TYPES="self-hosted/parser/types.sio"
LEX_TABLES="self-hosted/lexer/tables.sio"
LEX_PARSER="self-hosted/parser/parser.sio"

fail=0
note() { printf '%s\n' "$*"; }
pin_fail() { printf 'PIN FAIL: %s\n' "$*" >&2; fail=1; }

# ---------------------------------------------------------------------------
# STATIC
# ---------------------------------------------------------------------------

declared=$(
  awk '/^pub enum AstProvenanceKind/,/^}/ {
    if ($1 ~ /^AstProv/ && $1 !~ /Kind/) {
      gsub(/,/, "", $1)
      print $1
    }
  }' "$AST" | sort
)
expected_declared=$'AstProvComputed\nAstProvDerived\nAstProvInput\nAstProvLiterature\nAstProvMeasured\nAstProvSource'
if [[ "$declared" != "$expected_declared" ]]; then
  pin_fail "AstProvenanceKind cases moved"
  printf '  got:\n%s\n  expected:\n%s\n' "$declared" "$expected_declared" >&2
fi

# Construction sites in the Knowledge / wrapper parsers only.
constructed=$(
  grep -E 'AstProvenanceKind::AstProv[A-Za-z]+' "$TYPES" \
    | sed -E 's/.*AstProvenanceKind:://' \
    | sed -E 's/[^A-Za-z].*$//' \
    | sort -u
)
expected_constructed=$'AstProvComputed\nAstProvDerived\nAstProvMeasured'
if [[ "$constructed" != "$expected_constructed" ]]; then
  pin_fail "parser construction sites for AstProvenanceKind moved"
  printf '  got:\n%s\n  expected:\n%s\n' "$constructed" "$expected_constructed" >&2
fi

# The three unreachable cases must still have zero construction sites
# under self-hosted/parser/.
for dead in AstProvSource AstProvLiterature AstProvInput; do
  if grep -q "AstProvenanceKind::${dead}" self-hosted/parser/*.sio; then
    pin_fail "$dead gained a parser construction site"
  fi
done

# Lexer keywords: live Madaros table + the parser.sio duplicate.
# Presence.
for word in Derived Computed Measured Valid ValidUntil ValidWhile; do
  if ! grep -q "return TokenKind::${word}" "$LEX_TABLES"; then
    pin_fail "lexer/tables.sio lost TokenKind::${word}"
  fi
  if ! grep -q "return TokenKind::${word}" "$LEX_PARSER"; then
    pin_fail "parser.sio ident-table lost TokenKind::${word}"
  fi
done

# Absence of the three unreachable provenance words as keywords.
# E241 refuses a *bare* Source identifier; it does not make Source a provenance word.
for word in Source Literature Input; do
  if grep -q "return TokenKind::${word}" "$LEX_TABLES" "$LEX_PARSER"; then
    pin_fail "$word became a lexer keyword — that would mint a provenance surface we did not ask for"
  fi
done

# Honesty pin: the silent skip / default-CmpLt sink must stay closed.
if grep -q 'Unknown component — skip' "$TYPES"; then
  pin_fail "unknown-component skip comment returned in types.sio"
fi
if ! grep -q 'error\[E241\]' "$TYPES"; then
  pin_fail "E241 diagnostic missing from types.sio"
fi
if ! grep -q 'report_unknown_knowledge_component' "$TYPES"; then
  pin_fail "report_unknown_knowledge_component missing from types.sio"
fi
if ! grep -q 'saw_cmp' "$TYPES"; then
  pin_fail "Ident-as-epsilon no longer requires a comparison operator"
fi

# Wrapper path still only constructs the same three (no ValidUntil/ValidWhile).
# Bounded to parse_epistemic_wrapper_type by reading that function body.
wrapper_body=$(
  awk '/fn parse_epistemic_wrapper_type/,/fn parse_validated_type/' "$TYPES"
)
for word in ValidUntil ValidWhile; do
  if printf '%s' "$wrapper_body" | grep -q "TokenKind::${word}"; then
    pin_fail "parse_epistemic_wrapper_type gained a ${word} branch"
  fi
done

note "STATIC: declared=6 constructed=3 unreachable=Source,Literature,Input E241=present Ident-epsilon-requires-cmp"

# ---------------------------------------------------------------------------
# DYNAMIC — only against a source-built Madaros (SOUNIO_KCOV_DYNAMIC=1).
# The committed ELF still swallows unknown components; putting this in
# Contracts would fail every PR until the ELF is rebuilt. The live-refuse
# gate under Madaros Witness is the source-current clock.
# ---------------------------------------------------------------------------

if [[ "${SOUNIO_KCOV_DYNAMIC:-0}" != "1" ]]; then
  note "DYNAMIC: skipped (set SOUNIO_KCOV_DYNAMIC=1 against a source-built Madaros)"
else
  SOUC="$ROOT_DIR/bin/souc"
  if [[ ! -x "$SOUC" ]]; then
    pin_fail "bin/souc missing or not executable"
  else
    souc_ver=$(env -u SOUC_BIN -u SOUNIO_SOUC_BIN "$SOUC" --version 2>/dev/null | head -1 || true)
    note "DYNAMIC compiler: ${souc_ver:-unknown} path=$SOUC (shipped ELF, not source-built)"

    check_one() {
      local file="$1"
      ( ulimit -s 524288
        env -u SOUC_BIN -u SOUNIO_SOUC_BIN \
          SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
          "$SOUC" check "$file"
      ) >/tmp/kcov_check.out 2>&1
    }

    pass_probes=(
      derived computed measured valid validuntil validwhile
      source_eps knowledge_angle_derived
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
        pin_fail "pass-probe $name: expected check OK, rc=$?"
      fi
    done

    fail_probes=(derived_eps source literature input int_skip typo_ident)
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
  fi
fi

if [[ "$fail" -ne 0 ]]; then
  note "KNOWLEDGE_ANNOTATION_PARSER_COVERAGE: FAIL"
  exit 1
fi
note "KNOWLEDGE_ANNOTATION_PARSER_COVERAGE: PASS"
exit 0
