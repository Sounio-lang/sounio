#!/usr/bin/env bash
# Assemble changelog.d/ fragments into a CHANGELOG.md release section
# (ADR-018 §4). Run by the release manager on release/X.Y at an RC.
#
# Usage:
#   changelog_assemble.sh VERSION DATE [--dry-run]
#   changelog_assemble.sh --selftest
#
# Fragments are grouped by kind in release-note order; `internal` fragments
# are counted but not printed. Each bullet ends with the PR that added the
# fragment, read from the commit that introduced the file ("(#N)" in a squash
# subject or "Merge pull request #N"). Without --dry-run the section is
# inserted above the newest release in CHANGELOG.md and the consumed
# fragments are removed with `git rm`.
set -euo pipefail

ORDER=(breaking refusal semantics numerical-correction numerical added changed deprecated removed fixed performance security)
declare -A TITLE=(
  [breaking]="Breaking" [refusal]="Refusals (soundness)" [semantics]="Semantics"
  [numerical-correction]="Numerical corrections" [numerical]="Numerical changes (consistent)"
  [added]="Added" [changed]="Changed" [deprecated]="Deprecated" [removed]="Removed"
  [fixed]="Fixed" [performance]="Performance" [security]="Security"
)

pr_of() { # print " (#N)" for the PR that added file $1, or nothing
  local subj n
  subj=$(git log --diff-filter=A --format=%s -1 -- "$1" 2>/dev/null || true)
  if [[ "$subj" =~ \(#([0-9]+)\)$ ]]; then n=${BASH_REMATCH[1]}
  elif [[ "$subj" =~ ^Merge\ pull\ request\ #([0-9]+) ]]; then n=${BASH_REMATCH[1]}
  else return 0; fi
  printf ' ([#%s](https://github.com/Sounio-lang/sounio/pull/%s))' "$n" "$n"
}

render() { # <dir> <version> <date>  -> section on stdout
  local dir=$1 version=$2 date=$3 kind f body internal=0 printed=0
  printf '## [%s] - %s\n' "$version" "$date"
  for kind in "${ORDER[@]}"; do
    local files=()
    for f in "$dir"/*."$kind".md; do [[ -e "$f" ]] && files+=("$f"); done
    (( ${#files[@]} )) || continue
    printf '\n### %s\n\n' "${TITLE[$kind]}"
    for f in "${files[@]}"; do
      body=$(sed -e '/^[[:space:]]*$/d' "$f" | paste -sd' ' -)
      printf -- '- %s%s\n' "$body" "$(pr_of "$f")"
      printed=$((printed + 1))
    done
  done
  for f in "$dir"/*.internal.md; do [[ -e "$f" ]] && internal=$((internal + 1)); done
  printf '\n<!-- assembled: %d entries, %d internal fragments not shown -->\n' "$printed" "$internal"
}

insert() { # <section file> <changelog>: place section above the first "## [" header
  awk -v sec="$1" 'BEGIN{done=0} /^## \[/ && !done {while((getline l < sec)>0) print l; print ""; done=1} {print} END{if(!done){while((getline l < sec)>0) print l}}' "$2"
}

selftest() {
  local w; w=$(mktemp -d); trap 'rm -rf "$w"' RETURN
  mkdir -p "$w/changelog.d"
  printf 'Module-local calls bind to the local function.\n' > "$w/changelog.d/a.breaking.md"
  printf '`atan(0.5)` is correct.\n\nSecond line.\n' > "$w/changelog.d/b.numerical-correction.md"
  printf 'CI tidy.\n' > "$w/changelog.d/c.internal.md"
  printf '# Changelog\n\nIntro.\n\n## [2.1.0] - 2026-06-04\n\nOld.\n' > "$w/CHANGELOG.md"
  render "$w/changelog.d" 3.0.1 2026-11-01 > "$w/sec.md"
  insert "$w/sec.md" "$w/CHANGELOG.md" > "$w/out.md"
  local fails=0
  grep -q '^## \[3.0.1\] - 2026-11-01$' "$w/out.md" || { echo "FAIL header"; fails=1; }
  grep -q '^### Breaking$' "$w/out.md" || { echo "FAIL breaking title"; fails=1; }
  grep -q '^- `atan(0.5)` is correct. Second line.$' "$w/out.md" || { echo "FAIL paragraph join"; fails=1; }
  grep -q 'CI tidy' "$w/out.md" && { echo "FAIL internal printed"; fails=1; }
  [[ $(grep -n '^## \[3.0.1\]' "$w/out.md" | cut -d: -f1) -lt $(grep -n '^## \[2.1.0\]' "$w/out.md" | cut -d: -f1) ]] || { echo "FAIL order"; fails=1; }
  [[ $(grep -n '^### Breaking' "$w/out.md" | cut -d: -f1) -lt $(grep -n '^### Numerical corrections' "$w/out.md" | cut -d: -f1) ]] || { echo "FAIL kind order"; fails=1; }
  grep -q 'assembled: 2 entries, 1 internal' "$w/out.md" || { echo "FAIL counts"; fails=1; }
  (( fails )) && { echo CHANGELOG_ASSEMBLE_SELFTEST_FAIL; return 1; }
  echo CHANGELOG_ASSEMBLE_SELFTEST_PASS
}

[[ "${1:-}" == "--selftest" ]] && { selftest; exit; }
version=${1:?usage: changelog_assemble.sh VERSION DATE [--dry-run]}
date=${2:?usage: changelog_assemble.sh VERSION DATE [--dry-run]}
root=$(git rev-parse --show-toplevel)
cd "$root"
grep -q "^## \[$version\]" CHANGELOG.md && { echo "CHANGELOG.md already has $version" >&2; exit 1; }
sec=$(mktemp); render changelog.d "$version" "$date" > "$sec"
if [[ "${3:-}" == "--dry-run" ]]; then cat "$sec"; rm -f "$sec"; exit 0; fi
insert "$sec" CHANGELOG.md > CHANGELOG.md.new && mv CHANGELOG.md.new CHANGELOG.md
rm -f "$sec"
shopt -s nullglob
consumed=(changelog.d/*.*.md)
(( ${#consumed[@]} )) && git rm -q -- "${consumed[@]}"
echo "CHANGELOG_ASSEMBLED version=$version consumed=${#consumed[@]}"
