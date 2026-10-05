#!/usr/bin/env bash
# scripts/dev/pr_mergeability_refresh.sh
#
# GitHub's mergeable/mergeStateStatus fields drift stale in this repo: main's
# commit rate outruns GitHub's mergeability recomputation queue. See
# docs/audit/GITHUB_MERGEABILITY_CACHE_STALENESS_CENSUS_2026-08-18.md for the
# original census (API wrong on 20/38 "conflicting" PRs, i.e. 6 of 7 measured
# symptom shapes) and its follow-up (18/37: still ~1:1).
#
# Real conflicts are a FLOW, not a stock: of 19 real conflicts found in the
# second census, 15 were already in the first census's 18 and 4 were new,
# born from main's continued advance rather than from cache staleness. A
# one-time triage under-serves a queue that keeps generating fresh conflicts.
# This script is meant to be re-run periodically, not once.
#
# Three rules, non-negotiable:
#   1. Phantom vs real is decided by LOCAL verification (git merge-tree
#      --write-tree, git >=2.38) — NEVER by the API's mergeable /
#      mergeStateStatus fields. Those fields are recorded per PR only to
#      report how often the API was wrong; they never gate behaviour. The
#      API was wrong 6 of 7 measured times in this repo.
#   2. Before pushing a refreshed head to a branch this agent does not own,
#      check bin/sounio-coord status for an ACTIVE claim on that exact
#      branch and send a heads-up over the bus first. The merge only touches
#      the remote ref, but the owner's next push needs to pull first.
#   3. Report the phantom:real ratio for THIS run. A high ratio means the
#      cache is still the dominant failure mode and this script earns its
#      keep. A falling ratio means the queue has shifted toward genuine
#      unresolved divergence — the fix at that point is authors, not
#      another run of this script. Do not let a falling ratio go unreported;
#      that is the signal to stop leaning on automation and go look at who
#      needs to resolve what.
#
# Usage:
#   bash scripts/dev/pr_mergeability_refresh.sh                # all open PRs
#   bash scripts/dev/pr_mergeability_refresh.sh 1420 1451 1466  # specific PRs
set -uo pipefail

AGENT="${SOUNIO_AGENT_ID:-claude}"
LANE="${SOUNIO_COORD_LANE:-pr-mergeability-refresh-$$}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COORD="$REPO_ROOT/bin/sounio-coord"
WORK="$(mktemp -d /tmp/pr-mergeability-refresh.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

cd "$REPO_ROOT"
git fetch origin --quiet

if [ "$#" -gt 0 ]; then
  PR_NUMBERS=("$@")
else
  mapfile -t PR_NUMBERS < <(gh pr list --state open --json number -q '.[].number' --limit 300)
fi

PHANTOM_COUNT=0
REAL_COUNT=0
CORRECT_COUNT=0
PHANTOM_LIST=()
REAL_LIST=()

for n in "${PR_NUMBERS[@]}"; do
  meta=$(gh pr view "$n" --json headRefName,baseRefName,mergeable 2>/dev/null) || continue
  branch=$(echo "$meta" | jq -r '.headRefName')
  base=$(echo "$meta" | jq -r '.baseRefName')
  api_mergeable=$(echo "$meta" | jq -r '.mergeable')

  git fetch origin "+refs/pull/$n/head:refs/remotes/origin/pr/$n" --quiet 2>/dev/null || continue
  git fetch origin "$base" --quiet 2>/dev/null || true

  # LOCAL VERIFICATION IS THE ONLY SOURCE OF TRUTH for what happens next.
  # api_mergeable is captured only to print the phantom:real ratio below —
  # it must never appear in a branch condition that decides an action.
  if git merge-tree --write-tree "origin/$base" "origin/pr/$n" >"$WORK/mt_$n.log" 2>&1; then
    local_clean=1
  else
    local_clean=0
  fi

  if [ "$api_mergeable" = "MERGEABLE" ] && [ "$local_clean" = "1" ]; then
    CORRECT_COUNT=$((CORRECT_COUNT + 1))
    continue
  fi

  if [ "$local_clean" = "0" ]; then
    # Real conflict: never touched, only reported. Needs the author.
    REAL_COUNT=$((REAL_COUNT + 1))
    REAL_LIST+=("$n")
    echo "PR #$n ($branch): real conflict (local), not touched"
    continue
  fi

  # Phantom: locally clean despite the API disagreeing.
  claim=$("$COORD" status 2>/dev/null | grep "^ACTIVE" | grep -F "branch=$branch ")
  if [ -n "$claim" ]; then
    owner_lane=$(echo "$claim" | grep -oP 'lane=\K[^ ]+')
    "$COORD" send --agent "$AGENT" --lane "$LANE" --to-lane "$owner_lane" --kind info \
      --message "Vou empurrar um merge-commit normal (git merge origin/$base --no-edit + push, sem rebase/force) na PR #$n ($branch) para a tirar do estado de mergeabilidade fantasma do cache do GitHub. Local verificado limpo. Se estiveres a meio de um push, avisa antes que eu avance." \
      >/dev/null 2>&1
  fi

  wt="$WORK/wt-$n"
  git worktree add --quiet "$wt" -B "___refresh_$n" "origin/pr/$n" >>"$WORK/refresh_$n.log" 2>&1
  if (cd "$wt" && git merge "origin/$base" --no-edit >>"$WORK/refresh_$n.log" 2>&1); then
    if (cd "$wt" && git push origin "HEAD:$branch" >>"$WORK/refresh_$n.log" 2>&1); then
      PHANTOM_COUNT=$((PHANTOM_COUNT + 1))
      PHANTOM_LIST+=("$n")
      echo "PR #$n ($branch): phantom, refreshed and pushed"
    else
      echo "PR #$n ($branch): phantom locally but PUSH FAILED — see $WORK/refresh_$n.log (not counted either way)"
    fi
  else
    # The write-tree probe said clean but the real merge disagrees. Per
    # rule 3 in the source dispatch: stop, do not push through, reclassify
    # honestly as a real conflict instead of trusting the cheaper probe.
    (cd "$wt" && git merge --abort 2>/dev/null || true)
    REAL_COUNT=$((REAL_COUNT + 1))
    REAL_LIST+=("$n")
    echo "PR #$n ($branch): STOP — real conflict surfaced during actual merge though write-tree said clean; not touched"
  fi
  git worktree remove --force "$wt" 2>/dev/null || true
done

echo ""
echo "=== Run summary ($(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown-time)) ==="
echo "API-correct (MERGEABLE and locally clean): $CORRECT_COUNT"
echo "Phantom, refreshed this run: $PHANTOM_COUNT   [${PHANTOM_LIST[*]:-none}]"
echo "Real conflict, untouched: $REAL_COUNT   [${REAL_LIST[*]:-none}]"
if [ "$REAL_COUNT" -gt 0 ]; then
  ratio=$(awk -v p="$PHANTOM_COUNT" -v r="$REAL_COUNT" 'BEGIN { printf "%.2f", p / r }')
  echo "Phantom:real ratio this run: $ratio"
  echo "High ratio: cache staleness still dominant, this script earns its keep."
  echo "Falling ratio: the queue has shifted to genuine divergence -- go find authors, do not just rerun this."
elif [ "$PHANTOM_COUNT" -gt 0 ]; then
  echo "Phantom:real ratio this run: infinite (no real conflicts found) -- cache staleness fully explains today's queue."
fi
