#!/usr/bin/env bash
# The CLAUDE.md section 7 syntax table, executed.
#
# GATE_CONTRACT: v0
# GATE_ID: claude_md_syntax_executable
# GATE_CLAIMS: every claim in the entry document's syntax table is compiled and checked
# GATE_ENGINE: Madaros default (bin/souc)
# GATE_RESULT_ON_SKIP: fail
#
# WHY.
#
# CLAUDE.md is the door. Every agent in this repository reads section 7 before
# writing a line of Sounio, and on 2026-08-20 five of its seven rows were measured
# to be style rather than enforcement: a trailing semicolon compiles and runs,
# unary minus computes correctly, `x >> 4` needs no u8 suffix. Several agents —
# including the one that measured it — had been writing `0 - x` for no reason.
#
# The table did not go stale because anyone was careless. It went stale because
# NOTHING EXECUTED IT. A claim with no runner cannot go red, so it ages in silence
# and propagates into every session that reads it.
#
# This gate executes it. Each row of the table has a fragment and a verdict in
# scripts/ci/fixtures/claude_md_syntax/rows.tsv, and the fragment is compiled. If
# the language changes under a row, this fails the day it changes rather than the
# day someone notices.
#
# It deliberately encodes the CURRENT truth, including the rows that are NOT
# enforced. `accepted` is a real verdict here: it pins "this is style" so that if
# the compiler ever starts refusing it, the door gets updated on purpose.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SOUC="${SOUC:-./bin/souc}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
FIX="scripts/ci/fixtures/claude_md_syntax/rows.tsv"
OUT="${GATE_ARTIFACT:-artifacts/gates/claude_md_syntax_executable.json}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/claude-md-syntax.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

[[ -f "$FIX" ]] || { echo "claude_md_syntax: FAIL: missing fixture $FIX" >&2; exit 1; }

mapfile -t rows < <(grep -vE '^\s*#|^\s*$' "$FIX")
n="${#rows[@]}"

# NEGATIVE CONTROL 1 -- an empty fixture must not pass for free.
if (( n == 0 )); then
  echo "claude_md_syntax: FAIL: fixture parsed to zero rows" >&2
  exit 1
fi

# NEGATIVE CONTROL 2 -- the fixture must still correspond to the table. Count the
# table's data rows in CLAUDE.md; if the table grows or shrinks without the
# fixture following, the two have drifted and neither can be trusted.
table_rows=$(awk '/^## 7\. Sounio syntax/,/^Helpers must be defined/' CLAUDE.md 2>/dev/null \
  | grep -cE '^\| ' )
table_rows=$(( table_rows - 2 ))   # header + separator
# One table row may need more than one executable case -- the assert row needs two,
# because the point is that assert! and assert differ. So the rule is coverage plus
# a frozen row count: every row must be covered, and the table cannot change size
# without someone updating the number on purpose.
FROZEN_ROWS="scripts/ci/fixtures/claude_md_syntax/table_rows.frozen"
frozen_rows=$(tr -dc '0-9' < "$FROZEN_ROWS" 2>/dev/null | head -c 8)
[[ -n "$frozen_rows" ]] || { echo "claude_md_syntax: FAIL: missing $FROZEN_ROWS" >&2; exit 1; }
if (( table_rows != frozen_rows )); then
  echo "claude_md_syntax: FAIL: CLAUDE.md table has ${table_rows} rows, frozen at ${frozen_rows}" >&2
  echo "  the door changed size. Add or remove the matching case in the fixture and" >&2
  echo "  update ${FROZEN_ROWS} in the same commit." >&2
  exit 1
fi
if (( n < table_rows )); then
  echo "claude_md_syntax: FAIL: ${table_rows} table rows but only ${n} executable cases" >&2
  echo "  a row of the door is claimed and never run" >&2
  exit 1
fi

pass=0; fail=0
for row in "${rows[@]}"; do
  id=$(cut -f1 <<<"$row"); verdict=$(cut -f2 <<<"$row"); frag=$(cut -f3- <<<"$row")
  f="$WORK/$id.sio"
  printf '%b\n' "$frag" > "$f"
  out=$(timeout 200 "$SOUC" check "$f" 2>&1); 
  refused=0
  grep -qiE 'error|failed to parse' <<<"$out" && refused=1
  ok=0
  case "$verdict" in
    refused-check)  (( refused == 1 )) && ok=1 ;;
    accepted)       (( refused == 0 )) && ok=1 ;;
    runs-through|runs-halts)
      if (( refused == 1 )); then ok=0
      else
        rout=$( ulimit -s 524288 2>/dev/null; timeout 300 "$SOUC" run "$f" 2>&1 )
        seen=0; grep -q 'SENTINEL' <<<"$rout" && seen=1
        if [[ "$verdict" == "runs-through" ]]; then
          (( seen == 1 )) && ok=1
        else
          (( seen == 0 )) && ok=1
        fi
      fi ;;
    *) echo "claude_md_syntax: FAIL: unknown verdict '$verdict' for $id" >&2; exit 1 ;;
  esac
  if (( ok == 1 )); then
    pass=$(( pass + 1 )); echo "CLAUDE_MD_SYNTAX_OK   $id ($verdict)"
  else
    fail=$(( fail + 1 ))
    echo "CLAUDE_MD_SYNTAX_FAIL $id: claimed '$verdict', compiler disagrees" >&2
    echo "$out" | tail -3 | sed 's/^/    /' >&2
  fi
done

mkdir -p "$(dirname "$OUT")"
status=pass; rc=0
if (( fail > 0 )); then
  status=fail; rc=1
  echo "REFUSE: ${fail} of ${n} rows of the CLAUDE.md syntax table no longer hold." >&2
  echo "  The door is what every agent reads first. Update the row AND this fixture" >&2
  echo "  together, so the claim and its runner never drift again." >&2
else
  echo "OK: all ${n} rows of the CLAUDE.md syntax table hold."
fi
cat > "$OUT" <<JSON
{
  "gate": "claude_md_syntax_executable",
  "status": "${status}",
  "claims": "every row of the entry document's syntax table is compiled and checked",
  "metrics": { "total": ${n}, "passed": ${pass}, "failed": ${fail}, "not_run": 0 }
}
JSON
exit "$rc"
