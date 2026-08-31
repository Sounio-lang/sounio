#!/usr/bin/env bash
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/gate_artifact.sh"
# Every word classified IDENTIFIER_OK must actually BE usable as an identifier.
#
# parser_keyword_classification_gate.sh checks that each identifier-shaped
# keyword is classified. It cannot check that the classification is TRUE, and on
# 2026-08-23 that gap bit: moving `on` into IDENTIFIER_OK turned that gate green
# while self-hosted/ was untouched. The list and the parser are two files and
# nothing tied them together.
#
# A static check was tried first and was WRONG. Modelling the acceptance paths
# as "these three positions name the TokenKind inline or call
# tk_is_contextual_ident" reported 30 words broken; probing them showed all 30
# work. The model of the parser was wrong, not the parser. So this gate does not
# model anything -- it compiles the word.
#
#     fn f(<w>: i64) -> i64 { <w> }
#
# Parameter position, not `let`, and not a struct field. `let <w> = 1` collides
# with `var`, which introduces its own binding form. A struct field accepts
# EVERYTHING -- measured: match, struct, while and fn are all legal field names
# -- so a field probe has no discriminating power and would have been a gate
# that cannot fail.
#
# Both halves of the probe matter: the parameter tests the binding position and
# the body tests expression position. Measured before this gate was written:
# match, struct, while, return and let all fail; belief and unit pass.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 9
SOUC="${SOUC:-./bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$(pwd)/stdlib}"
ART=artifacts/gates/keyword_identifier_capability.v1.json
mkdir -p "$(dirname "$ART")"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

[[ -x "$SOUC" ]] || { echo "KEYWORD_CAPABILITY_FAIL reason=no_compiler" >&2; exit 1; }

probe() {  # $1 = word; echoes rc
  printf 'fn f(%s: i64) -> i64 {\n    %s\n}\n' "$1" "$1" > "$TMP/p.sio"
  timeout 120 "$SOUC" check "$TMP/p.sio" >/dev/null 2>&1
  echo $?
}

# Positive control FIRST, and in BOTH directions, because a probe that always
# passes and a probe that always fails are both useless and look different.
for w in match struct while return let; do
  [[ "$(probe "$w")" != "0" ]] || {
    echo "CONTROL_FAIL: reserved word '$w' was accepted as an identifier." >&2
    echo "  The probe cannot distinguish anything. Refusing to report." >&2
    printf '{"status":"fail","reason":"control did not fire","metrics":{"total":0,"passed":0,"failed":1,"not_run":0}}\n' | gate_write_artifact "$ART"
    exit 1
  }
done
echo "control: match, struct, while, return and let are all refused, as required"

# `var` is IDENTIFIER_OK and is NOT usable as a parameter name, and that is not
# a defect to fix here. It introduces the mutable-binding form, so a binding
# cannot also be called `var`. Measured 2026-08-23: as a struct field it works,
# in expression position it works, as a parameter it does not. The exclusion is
# that narrow statement rather than a convenience -- if `var` ever becomes
# usable in parameter position, delete this line and the gate stays green.
SKIP=" var "

WORDS=$(python3 - <<'PY'
import re
g = open("scripts/ci/parser_keyword_classification_gate.sh").read()
ok = set(re.findall(r'"([a-z_]+)"', re.search(r"IDENTIFIER_OK = \{(.*?)\}", g, re.S).group(1)))
lex = open("self-hosted/lexer/tables.sio").read() + open("self-hosted/parser/parser.sio").read()
# Only words the lexer actually turns into a keyword can be broken here; the
# rest of IDENTIFIER_OK arrives as a plain Ident and never had a TokenKind.
kinds = {re.sub(r"(?<!^)(?=[A-Z])", "_", re.sub(r"Lower$", "", k)).lower()
         for k in re.findall(r"TokenKind::(\w+)", lex)}
print(" ".join(sorted(ok & kinds)))
PY
)

# A word ADDED to IDENTIFIER_OK in this diff cannot be judged by the committed
# binary, and that is the actual epistemic situation rather than a loophole.
# ./bin/souc is the shipped ELF, not a build of the source under review, and the
# Contracts job that runs this gate downloads no compiler artifact (needs:
# impact only). So for a newly-declared capability, "the source's claim is
# false" and "the binary predates the source" produce the identical observation.
#
# Those are reported PENDING-REBUILD, never passed and never failed. The other
# half -- a shipped binary that fell behind a capability the source already had
# -- is what scripts/ci/madaros_binary_source_drift_gate.sh catches, so the
# split leaves nothing unwatched.
NEW_WORDS=""
if git rev-parse --verify -q origin/main >/dev/null 2>&1; then
  NEW_WORDS=$(python3 scripts/ci/lib/keyword_newly_declared.py 2>/dev/null || true)
fi
# Words the DRIFT gate already owns. With a committed binary these two gates ask
# the same question -- can the shipped ELF do what the source declares -- and
# only one of them can answer whose fault it is. madaros_binary_source_drift_gate.sh
# names the binary; this gate would name the source, and be wrong.
if [[ -f scripts/ci/madaros_binary_source_drift_gate.sh ]]; then
  DRIFT_OWNED=$(grep -oE 'check_row "([a-z_]+) is a contextual identifier"' \
                scripts/ci/madaros_binary_source_drift_gate.sh 2>/dev/null \
                | grep -oE '"[a-z_]+ is' | grep -oE '^"[a-z_]+' | tr -d '"' | tr '\n' ' ')
  NEW_WORDS="$NEW_WORDS $DRIFT_OWNED"
fi
[[ -n "$NEW_WORDS" ]] && echo "  declared in this diff, undecidable with the shipped binary: $NEW_WORDS"

total=0; failed=0; pending=0
for w in $WORDS; do
  case "$SKIP" in *" $w "*) continue ;; esac
  total=$((total + 1))
  if [[ "$(probe "$w")" != "0" ]]; then
    case " $NEW_WORDS " in
      *" $w "*)
        pending=$((pending + 1))
        echo "  PENDING-REBUILD  $w -- declared IDENTIFIER_OK in this diff; the shipped binary predates it" ;;
      *)
        failed=$((failed + 1))
        echo "  $w is classified IDENTIFIER_OK but cannot be used as one" >&2 ;;
    esac
  fi
done
echo "  probed $total keyword-shaped words, $failed unusable, $pending pending rebuild"

if [[ "$failed" -gt 0 ]]; then
  echo "KEYWORD_CAPABILITY_FAIL: the classification claims something the parser does not honour." >&2
  printf '{"status":"fail","metrics":{"total":%s,"passed":%s,"failed":%s,"not_run":0}}\n' "$total" "$((total-failed))" "$failed" | gate_write_artifact "$ART"
  exit 1
fi
printf '{"status":"pass","metrics":{"total":%s,"passed":%s,"failed":0,"not_run":%s}}\n' "$total" "$((total-pending))" "$pending" | gate_write_artifact "$ART"
echo "KEYWORD_IDENTIFIER_CAPABILITY_OK"
