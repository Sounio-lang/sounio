#!/usr/bin/env bash
# Every line the ORIGINAL commits added must still exist in the REBASED result.
#
# A rebase resolves conflicts by picking hunks, so it can silently drop an added
# line that lives inside a function whose name and brace count are unchanged.
# Neither a brace-delta check nor a function-name set-diff can see that; this can.
# Ordering and position are deliberately ignored -- only survival is asserted.
#
# usage: rebase_line_survival.sh <orig-base> <orig-head> <rebased-head>
set -uo pipefail
ORIG_BASE="${1:?orig-base}"; ORIG_HEAD="${2:?orig-head}"; REBASED="${3:?rebased-head}"

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT

# Lines added by the original range, minus those too short or too common to
# identify a hunk uniquely (a bare brace matches everywhere and proves nothing).
git diff "$ORIG_BASE".."$ORIG_HEAD" -- '*.sio' \
  | grep '^+' | grep -v '^+++' | sed 's/^+//' \
  | awk '{ s=$0; gsub(/[ \t]/,"",s); if (length(s) >= 12 && s !~ /^\/\//) print }' \
  | sort -u > "$work/added"

# The whole rebased tree as one line pool: survival is what we assert, not place.
git grep -h '' "$REBASED" -- '*.sio' | sort -u > "$work/pool"

grep -Fxv -f "$work/pool" -- "$work/added" > "$work/lost" 2>/dev/null

checked=$(wc -l < "$work/added"); lost=$(wc -l < "$work/lost")
sed 's/^/LOST: /' "$work/lost"
printf 'checked=%s lost=%s\n' "$checked" "$lost"
[ "$lost" -eq 0 ]
