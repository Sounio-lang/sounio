#!/usr/bin/env bash
# ci_reuse_green_ancestor.sh — find an earlier head of this pull request whose
# CI Decision already passed and that differs from HEAD only in paths no heavy
# job reads. Prints that commit's sha on stdout, or nothing.
#
# WHY THIS EXISTS
#
# The main ruleset requires the branch to be up to date and CI Decision green.
# A compiler PR's CI takes ~80 min. On 2026-11-26 a website PR reached main
# inside that window again and again, so a compiler PR fell behind, merged main
# in, and restarted every heavy job. #3746 went green twice and lost both
# times. The merged-in commits touched only website/, docs and changelog
# fragments. No compiler, stdlib, test or workflow input differed from the head
# that had already passed.
#
# RULE (decision 2026-11-26, "CI rápido quando só o site mudou")
#
# Let R be a commit on HEAD's first-parent chain, other than HEAD and at most
# 12 back, such that:
#   1. every path in `git diff --name-only R HEAD` is reusable: website/**,
#      docs/**, a top-level *.md, or changelog.d/*.md; and
#   2. the check run "CI Decision" on R concluded success.
# Then every input of every heavy job is byte-identical between R and HEAD.
# The impact of HEAD can be classified by the R..HEAD diff, and the heavy jobs
# keep R's verdict. A workflow file, a script, stdlib/, tests/ or self-hosted/
# in the delta disqualifies R. Fail-closed: if the API is unavailable, a call
# fails or no candidate qualifies, nothing is printed and the caller
# classifies by the full BASE..HEAD diff as before.
#
# ENV
#   CI_HEAD_SHA               head to classify (default HEAD)
#   CI_REUSE_GREEN_SHAS       space-separated shas treated as green, instead of
#                             asking the GitHub API (selftests only)
#   GITHUB_REPOSITORY, GH_TOKEN  for the API lookup
#   CI_REUSE_DISABLE=1        never reuse
set -uo pipefail

HEAD_SHA="${CI_HEAD_SHA:-HEAD}"
[[ "${CI_REUSE_DISABLE:-0}" == 1 ]] && exit 0

reusable_path() {
  case "$1" in
    website/*|docs/*|changelog.d/*.md) return 0 ;;
    */*) return 1 ;;
    *.md) return 0 ;;
  esac
  return 1
}

is_green() {
  local c="$1"
  if [[ -n "${CI_REUSE_GREEN_SHAS+set}" ]]; then
    [[ " $CI_REUSE_GREEN_SHAS " == *" $c "* ]]
    return
  fi
  [[ -n "${GITHUB_REPOSITORY:-}" ]] || return 1
  command -v gh >/dev/null 2>&1 || return 1
  local concl
  concl="$(gh api "repos/$GITHUB_REPOSITORY/commits/$c/check-runs?check_name=CI%20Decision&filter=latest" \
            --jq '[.check_runs[] | select(.status == "completed") | .conclusion] | first // ""' 2>/dev/null)" || return 1
  [[ "$concl" == "success" ]]
}

head_full="$(git rev-parse --verify -q "$HEAD_SHA^{commit}")" || exit 0
mapfile -t chain < <(git rev-list --first-parent --max-count=13 "$head_full" 2>/dev/null | tail -n +2)
for c in "${chain[@]}"; do
  [[ -n "$c" ]] || continue
  delta="$(git diff --no-renames --name-status "$c" "$head_full" 2>/dev/null)" || continue
  [[ -n "$delta" ]] || continue
  ok=1
  while IFS=$'\t' read -r st p; do
    [[ -z "$p" ]] && continue
    reusable_path "$p" || { ok=0; break; }
    # A changelog fragment may be added or edited in the delta, never deleted:
    # the fragment gate reads this classification, and a PR's own fragment
    # vanishing between R and HEAD must not pass as a docs-only change.
    if [[ "$p" == changelog.d/* && "$st" != A && "$st" != M ]]; then ok=0; break; fi
  done <<<"$delta"
  # A qualifying candidate that is not green (its run was cancelled when a
  # newer push superseded it, say) does not end the search: an older head can
  # still qualify on its own delta to HEAD.
  if [[ $ok == 1 ]] && is_green "$c"; then
    echo "$c"
    exit 0
  fi
done
exit 0
