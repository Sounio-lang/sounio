#!/usr/bin/env bash
# gen_examples_index.sh — run every example that has a `main` on both engines
# and write examples/INDEX.md.
#
# For each tracked examples/**/*.sio outside examples/_attic/:
#   - no `fn main`  -> listed as "library", not run (CLAUDE.md §6: running a
#                      library file and calling it broken is a category error);
#   - has `fn main` -> `./bin/souc run` (Madaros, the default engine) and
#                      `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run`, each
#                      under a timeout, stdin from /dev/null, from the repo root.
#
# Status cells: pass (rc 0) | fail rc=N [first error codes] | timeout | crash (signal).
# Wall time is per engine, in seconds, including compilation.
#
# Usage: bash scripts/dev/gen_examples_index.sh [--out FILE]
# Env:   INDEX_TIMEOUT (s, default 120)  INDEX_JOBS (default nproc)
#        INDEX_LIMIT (run only the first N programs; for smoke-testing the script)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="examples/INDEX.md"
[[ "${1:-}" == "--out" ]] && OUT="$2"
TO="${INDEX_TIMEOUT:-120}"
JOBS="${INDEX_JOBS:-$(nproc 2>/dev/null || echo 4)}"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export TO

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Materialize the committed Madaros ELF once, before the parallel runs race on it.
./bin/souc --version >/dev/null 2>&1 || true

# Tracked files only (git ls-files), or every file when not in a git checkout
# (a `git archive` extraction, as staged on the cluster).
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git ls-files 'examples/*.sio'
else
  find examples -name '*.sio' -type f
fi | grep -v '^examples/_attic/' | LC_ALL=C sort > "$WORK/all"

has_main() {
  # A `fn main` at line start that is not inside a // comment.
  grep -qE '^[[:space:]]*(pub[[:space:]]+)?fn[[:space:]]+main[[:space:]]*\(' "$1"
}

describe() {
  # First comment line with words in it, harness lines (//@) and rulers skipped.
  local d
  d="$(grep -m1 -E '^[[:space:]]*(//[/!]?|/\*+|\*)[[:space:]]*[A-Za-z0-9]' "$1" \
        | grep -vE '^[[:space:]]*//@' \
        | sed -E 's#^[[:space:]]*(//[/!]?|/\*+|\*)[[:space:]]*##; s#[[:space:]]*\*/[[:space:]]*$##')"
  if [[ -z "$d" ]]; then
    d="$(grep -E '^[[:space:]]*(//[/!]?|/\*+|\*)[[:space:]]*[A-Za-z0-9]' "$1" | grep -vE '^[[:space:]]*//@' | head -1 \
          | sed -E 's#^[[:space:]]*(//[/!]?|/\*+|\*)[[:space:]]*##; s#[[:space:]]*\*/[[:space:]]*$##')"
  fi
  d="${d//|/\\|}"
  d="${d//</&lt;}"
  [[ ${#d} -gt 110 ]] && d="${d:0:107}..."
  printf '%s' "${d:-—}"
}

: > "$WORK/progs"
: > "$WORK/libs"
while IFS= read -r f; do
  if has_main "$f"; then echo "$f" >> "$WORK/progs"; else echo "$f" >> "$WORK/libs"; fi
done < "$WORK/all"
if [[ -n "${INDEX_LIMIT:-}" ]]; then head -n "$INDEX_LIMIT" "$WORK/progs" > "$WORK/p2"; mv "$WORK/p2" "$WORK/progs"; fi

run_one() {
  local f="$1" eng="$2" tmp s rc t codes st
  tmp="$(mktemp -d)"
  s=$(date +%s%N)
  if [[ "$eng" == madaros ]]; then
    timeout "$TO" ./bin/souc run "$f" </dev/null >"$tmp/o" 2>"$tmp/e"; rc=$?
  else
    SOUNIO_SOUC_ENGINE=lean_single timeout "$TO" ./bin/souc run "$f" </dev/null >"$tmp/o" 2>"$tmp/e"; rc=$?
  fi
  t=$(( ($(date +%s%N) - s) / 1000000 ))
  codes=$(cat "$tmp/o" "$tmp/e" | grep -oE '\bE[0-9]{3}\b' | awk '!seen[$0]++' | head -3 | tr '\n' ' ')
  codes="${codes% }"
  if [[ $rc -eq 0 ]]; then st="pass"
  elif [[ $rc -eq 124 ]]; then st="timeout"
  elif [[ $rc -gt 128 ]]; then st="crash (signal $((rc - 128)))"
  else st="fail rc=$rc${codes:+ ($codes)}"
  fi
  printf '%s\t%s\t%s\t%s\n' "$f" "$eng" "$st" "$t"
  rm -rf "$tmp"
}
export -f run_one

awk '{print $0" madaros"; print $0" lean_single"}' "$WORK/progs" \
  | xargs -P "$JOBS" -L1 bash -c 'run_one "$0" "$1"' > "$WORK/results.tsv"

status_of() { awk -F'\t' -v f="$1" -v e="$2" '$1==f && $2==e {print $3; exit}' "$WORK/results.tsv"; }
time_of()   { awk -F'\t' -v f="$1" -v e="$2" '$1==f && $2==e {printf "%.1f", $4/1000; exit}' "$WORK/results.tsv"; }

n_all=$(wc -l < "$WORK/all" | tr -d ' ')
n_prog=$(wc -l < "$WORK/progs" | tr -d ' ')
n_lib=$(wc -l < "$WORK/libs" | tr -d ' ')
mad_pass=$(awk -F'\t' '$2=="madaros" && $3=="pass"' "$WORK/results.tsv" | wc -l | tr -d ' ')
ls_pass=$(awk -F'\t' '$2=="lean_single" && $3=="pass"' "$WORK/results.tsv" | wc -l | tr -d ' ')
both_pass=$(awk -F'\t' '$3=="pass" {c[$1]++} END {n=0; for (k in c) if (c[k]==2) n++; print n}' "$WORK/results.tsv")
neither=$(awk -F'\t' '{if ($3=="pass") p[$1]=1; else seen[$1]=1} END {n=0; for (k in seen) if (!(k in p)) n++; print n}' "$WORK/results.tsv")
mad_to=$(awk -F'\t' '$2=="madaros" && $3=="timeout"' "$WORK/results.tsv" | wc -l | tr -d ' ')
ls_to=$(awk -F'\t' '$2=="lean_single" && $3=="timeout"' "$WORK/results.tsv" | wc -l | tr -d ' ')

commit="$(git rev-parse --short=12 HEAD 2>/dev/null || echo "${INDEX_COMMIT:-unknown}")"
version="$(./bin/souc --version 2>/dev/null | grep -m1 -E 'Madaros v[0-9]' | sed -E 's/.*(Madaros v[0-9.]+).*/\1/')"

{
  echo "# Examples index"
  echo
  echo "Generated $(date -u +%Y-%m-%d) at commit \`${commit}\` by \`bash scripts/dev/gen_examples_index.sh\`."
  echo "Do not edit by hand; re-run the script."
  echo
  echo "Each example with a \`fn main\` was run from the repository root on both engines, with a ${TO}s timeout and stdin from \`/dev/null\`:"
  echo
  echo "- **Madaros**: \`./bin/souc run <file>\` (the default engine${version:+, $version})"
  echo "- **lean_single**: \`SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run <file>\`"
  echo
  echo "Files without \`fn main\` are libraries (modules other files import); they are listed, not run."
  echo "Quarantined examples (stubs, pre-module legacy syntax) are in [\`_attic/\`](_attic/README.md)."
  echo
  echo "## Summary"
  echo
  echo "| | count |"
  echo "|---|---:|"
  echo "| \`.sio\` files indexed | ${n_all} |"
  echo "| programs (have \`fn main\`) | ${n_prog} |"
  echo "| libraries (no \`fn main\`) | ${n_lib} |"
  echo "| programs passing on Madaros | ${mad_pass} |"
  echo "| programs passing on lean_single | ${ls_pass} |"
  echo "| programs passing on both | ${both_pass} |"
  echo "| programs passing on neither | ${neither} |"
  echo "| timeouts (Madaros / lean_single) | ${mad_to} / ${ls_to} |"
  echo
  echo "Status: \`pass\` = exit code 0; \`fail rc=N (Exxx)\` = non-zero exit with the first diagnostic codes printed; \`timeout\` = killed after ${TO}s; \`crash (signal N)\` = killed by a signal. Wall time includes compilation."
  echo
  echo "## Programs and libraries"
  echo
  echo "| file | what it is | Madaros | lean_single | wall time (s) Madaros / lean_single |"
  echo "|---|---|---|---|---|"
  while IFS= read -r f; do
    rel="${f#examples/}"
    desc="$(describe "$f")"
    if grep -qxF "$f" "$WORK/progs"; then
      echo "| [\`${rel}\`](${rel}) | ${desc} | $(status_of "$f" madaros) | $(status_of "$f" lean_single) | $(time_of "$f" madaros) / $(time_of "$f" lean_single) |"
    elif grep -qxF "$f" "$WORK/libs"; then
      echo "| [\`${rel}\`](${rel}) | ${desc} | library | library | — |"
    fi
  done < "$WORK/all"
} > "$OUT"

echo "wrote $OUT: files=$n_all programs=$n_prog libraries=$n_lib madaros_pass=$mad_pass lean_single_pass=$ls_pass both=$both_pass neither=$neither"
