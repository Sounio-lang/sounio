#!/usr/bin/env bash
# Corpus census: how much of stdlib/ and examples/ does the CURRENT compiler accept?
#
# Written because the published figures do not reconcile -- 638/2003, 767, ~633, ~689
# predicted -- and all of them come from docs/EPISTEMIC_RELEASE_STATUS.md, which lives
# on the ORPHAN branch integration/native-v2-honest and was measured in June against a
# DIFFERENT tree. The corpus has since grown by ~500 files. Quoting that percentage for
# todays main is a claim about one tree supported by a measurement of another.
#
# The verdict here is `souc check`, per CLAUDE.md principle 3: most files under stdlib/
# and examples/ are libraries, not executables, so `check` plus a caller is the existence
# test. This does NOT measure whether they compile to a working ELF -- that is a
# strictly stronger question and this script deliberately does not answer it.
#
# Usage: bash scripts/dev/corpus_census.sh [path-to-souc] [jobs]
set -u
SOUC="${1:-./bin/souc}"
JOBS="${2:-8}"
SOUC="$(cd "$(dirname "$SOUC")" && pwd)/$(basename "$SOUC")"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
OUT="${CENSUS_OUT:-/tmp/corpus_census.tsv}"
PER_FILE_TIMEOUT="${CENSUS_TIMEOUT:-90}"

export SOUC PER_FILE_TIMEOUT
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"

check_one() {
  f="$1"
  timeout "$PER_FILE_TIMEOUT" "$SOUC" check "$f" >/dev/null 2>&1
  rc=$?
  if [ "$rc" = "0" ]; then
    st=OK
  elif [ "$rc" = "124" ]; then
    st=TIMEOUT
  elif [ "$rc" -ge 128 ]; then
    st=CRASH
  else
    st=REJECT
  fi
  printf "%s\\t%s\\t%s\\n" "$f" "$rc" "$st"
}
export -f check_one

git ls-files -- stdlib examples | grep "\\.sio$" | \
  xargs -P "$JOBS" -I{} bash -c "check_one {}" > "$OUT"

echo "census written: $OUT"

summarise() {
  area="$1"
  tot=$(awk -F"\\t" -v a="$area" "index(\$1,a)==1" "$OUT" | wc -l)
  ok=$(awk -F"\\t" -v a="$area" "index(\$1,a)==1 && \$3==\"OK\"" "$OUT" | wc -l)
  rej=$(awk -F"\\t" -v a="$area" "index(\$1,a)==1 && \$3==\"REJECT\"" "$OUT" | wc -l)
  to=$(awk -F"\\t" -v a="$area" "index(\$1,a)==1 && \$3==\"TIMEOUT\"" "$OUT" | wc -l)
  cr=$(awk -F"\\t" -v a="$area" "index(\$1,a)==1 && \$3==\"CRASH\"" "$OUT" | wc -l)
  pct=0
  if [ "$tot" -gt 0 ]; then pct=$((ok * 100 / tot)); fi
  printf "%-10s total=%-6s OK=%-6s (%s%%)  REJECT=%-6s TIMEOUT=%-4s CRASH=%s\\n" \
    "$area" "$tot" "$ok" "$pct" "$rej" "$to" "$cr"
}

echo
echo "=== corpus census: $($SOUC --version 2>&1 | head -1) ==="
summarise stdlib
summarise examples
echo
tot=$(wc -l < "$OUT")
ok=$(awk -F"\\t" "\$3==\"OK\"" "$OUT" | wc -l)
echo "TOTAL      $ok / $tot accepted by souc check"
