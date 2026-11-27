#!/usr/bin/env bash
# ci_reuse_green_ancestor.sh — find an earlier head of this pull request whose
# CI run already covered every job HEAD needs, and that differs from HEAD only
# in paths no heavy job reads. Prints that commit's sha on stdout, or nothing.
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
# 12 back. R qualifies when all of these hold:
#   1. Delta. Every path in R..HEAD is website/**, docs/**, a top-level *.md
#      other than CHANGELOG.md, or an added or edited changelog.d/*.md.
#   2. Same PR, same workflow, green. The GitHub Actions run of
#      .github/workflows/ci.yml for head_sha R has event pull_request, lists
#      THIS pull request (CI_PR_NUMBER), and concluded success.
#   3. Coverage. That run's own Impact job selected, by itself (its reused_from
#      is empty), every lane that HEAD's full BASE..HEAD classification selects
#      (CI_REUSE_REQUIRED). A green docs-only run, a run on a stacked base that
#      already held the compiler changes, or a run that itself reused a verdict
#      does not qualify. Since CI Decision passed on R, every job its
#      classification selected passed. Those jobs cover HEAD's jobs, and by
#      condition 4 their inputs are byte-identical.
#   4. Effective inputs. The tree R's run tested, which is its synthetic merge
#      "Merge R into <base>" (read from its Impact log and recomputed with
#      git merge-tree), differs from the tree tested now (the checkout,
#      CI_TESTED_SHA) only in the same reusable paths. A base change that R's
#      run tested and main later reverted is invisible in R..HEAD.
# The caller then classifies the impact of HEAD by the R..HEAD delta, for job
# scheduling only. Policy gates (the changelog fragment gate) keep the full
# BASE..HEAD classification. Fail-closed: if the API is unavailable, a call
# fails, evidence is missing or ambiguous, or no candidate qualifies, nothing
# is printed, and the caller classifies by the full diff.
#
# ENV
#   CI_HEAD_SHA            head to classify (default HEAD)
#   CI_REUSE_REQUIRED      space-separated impact keys HEAD's full
#                          classification selects (e.g. "compiler stdlib tests")
#   CI_PR_NUMBER           this pull request's number (required for the API)
#   CI_TESTED_SHA          the commit this run tests (default CI_HEAD_SHA; on a
#                          pull_request run, the checked-out synthetic merge)
#   GITHUB_REPOSITORY, GH_TOKEN   for the API lookup
#   CI_REUSE_GREEN_SHAS    selftests only, instead of the API: entries
#                          "<sha>:<key>,<key>,...[@<base>]" (the keys R's own
#                          run selected; <base>: the base its merge used)
#   CI_REUSE_DISABLE=1     never reuse
set -uo pipefail

HEAD_SHA="${CI_HEAD_SHA:-HEAD}"
REQUIRED="${CI_REUSE_REQUIRED:-}"
[[ "${CI_REUSE_DISABLE:-0}" == 1 ]] && exit 0

# skip <commit> <reason>: say on stderr why a candidate was not reused, and fail.
# The Impact log keeps stderr, so a reuse that did not happen can be diagnosed
# from the run itself (first live miss: #3750 @ 3aefd8f93a, 2026-11-27).
skip() { echo "ci-reuse: skip ${1:0:10}: $2" >&2; return 1; }

reusable_path() {
  case "$1" in
    website/*|docs/*|changelog.d/*.md) return 0 ;;
    CHANGELOG.md) return 1 ;;
    */*) return 1 ;;
    *.md) return 0 ;;
  esac
  return 1
}

# covers <selected keys (space-separated)>: rc 0 iff every REQUIRED key is in it.
covers() {
  local have=" $1 " k
  for k in $REQUIRED; do
    [[ "$have" == *" $k "* ]] || return 1
  done
  return 0
}

# own_run_selection <sha>: prints "<base> <keys...>": the base sha that R's own
# PR run of ci.yml merged R into (its synthetic merge, from the checkout line
# "Merge <R> into <base>" of its Impact log), then the impact keys that run
# selected. Fails when the run is missing, not green, not this PR's, its base
# cannot be read, or it itself reused an earlier verdict.
own_run_selection() {
  local c="$1"
  if [[ -n "${CI_REUSE_GREEN_SHAS+set}" ]]; then
    local e
    for e in $CI_REUSE_GREEN_SHAS; do
      if [[ "${e%%:*}" == "$c" ]]; then
        # "<sha>:<keys>" tests the raw head; "<sha>:<keys>@<base>" tests the
        # synthetic merge of <sha> into <base>.
        local rest="${e#*:}" mb="$c"
        [[ "$rest" == *@* ]] && { mb="${rest#*@}"; rest="${rest%@*}"; }
        printf '%s %s\n' "$mb" "$(tr ',' ' ' <<<"$rest")"
        return 0
      fi
    done
    return 1
  fi
  [[ -n "${GITHUB_REPOSITORY:-}" && -n "${CI_PR_NUMBER:-}" ]] || { skip "$c" "GITHUB_REPOSITORY or CI_PR_NUMBER unset"; return 1; }
  command -v gh >/dev/null 2>&1 || { skip "$c" "gh not on PATH"; return 1; }
  local run_id job_id log keys k v reused="" err
  err="$(mktemp)"; trap 'rm -f "$err"' RETURN
  # One line per green pull_request run of ci.yml for this head: "<id> <PR numbers>".
  # Kept apart so that "no green run" and "a green run that does not list
  # this PR" give different reasons (review of #3753); both still refuse.
  local runs
  runs="$(gh api "repos/$GITHUB_REPOSITORY/actions/runs?head_sha=$c&event=pull_request&status=completed&per_page=30" \
      --jq "[.workflow_runs[] | select(.path == \".github/workflows/ci.yml\" and .head_sha == \"$c\" and .conclusion == \"success\")] | .[] | \"\\(.id) \\([.pull_requests[].number | tostring] | join(\",\"))\"" 2>"$err")" \
    || { skip "$c" "runs query failed: $(head -c 200 "$err")"; return 1; }
  [[ -n "$runs" ]] || { skip "$c" "no successful pull_request run of ci.yml for this head"; return 1; }
  run_id="$(awk -v pr="$CI_PR_NUMBER" '{ n = split($2, a, ","); for (i = 1; i <= n; i++) if (a[i] == pr) { print $1; exit } }' <<<"$runs")"
  [[ "$run_id" =~ ^[0-9]+$ ]] || { skip "$c" "green run(s) $(awk '{printf "%s[prs=%s] ", $1, ($2 == "" ? "none" : $2)}' <<<"$runs")do not list PR $CI_PR_NUMBER"; return 1; }
  job_id="$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$run_id/jobs?per_page=100" \
      --jq '[.jobs[] | select(.name == "Impact" and .conclusion == "success") | .id] | first // empty'  2>"$err")" || { skip "$c" "jobs query failed for run $run_id: $(head -c 200 "$err")"; return 1; }
  [[ "$job_id" =~ ^[0-9]+$ ]] || { skip "$c" "run $run_id has no successful Impact job"; return 1; }
  log="$(gh api "repos/$GITHUB_REPOSITORY/actions/jobs/$job_id/logs"  2>"$err")" || { skip "$c" "log fetch failed for job $job_id: $(head -c 200 "$err")"; return 1; }
  keys=""
  while IFS= read -r line; do
    line="${line#*Z }"
    k="${line%%=*}"; v="${line#*=}"
    case "$k" in
      docs|website|compiler|runtime|stdlib|tests|lean|bootstrap|math|ontology|clinical|sio|full)
        [[ "$v" == true ]] && keys="$keys $k" ;;
      reused_from) reused="$v" ;;
    esac
  done < <(grep -E 'Z (docs|website|compiler|runtime|stdlib|tests|lean|bootstrap|math|ontology|clinical|sio|full|reused_from)=' <<<"$log")
  [[ -z "$reused" ]] || { skip "$c" "run $run_id itself reused $reused"; return 1; }
  [[ -n "$keys" ]] || { skip "$c" "no impact rows in the Impact log of job $job_id"; return 1; }
  local base
  base="$(grep -oE "HEAD is now at [0-9a-f]+ Merge $c into [0-9a-f]{40}" <<<"$log" | head -1 | grep -oE "[0-9a-f]{40}\$")"
  [[ "$base" =~ ^[0-9a-f]{40}$ ]] || { skip "$c" "no \"Merge <R> into <base>\" checkout line in job $job_id"; return 1; }
  printf '%s %s\n' "$base" "$keys"
}

# delta_reusable <a> <b>: rc 0 iff a..b is non-empty and every path in it is
# reusable, with changelog fragments only added or edited.
delta_reusable() {
  local delta st p
  delta="$(git diff --no-renames --name-status "$1" "$2" 2>/dev/null)" || return 1
  [[ -n "$delta" ]] || return 1
  while IFS=$'\t' read -r st p; do
    [[ -z "$p" ]] && continue
    reusable_path "$p" || return 1
    # A changelog fragment may be added or edited in the delta, never deleted.
    if [[ "$p" == changelog.d/* && "$st" != A && "$st" != M ]]; then return 1; fi
  done <<<"$delta"
  return 0
}

# tested_tree <R> <base>: the tree R's run actually tested. That is R itself
# when base is R; otherwise the synthetic merge GitHub built ("Merge R into
# base"), recomputed with git merge-tree. A conflicted or failed merge fails.
tested_tree() {
  if [[ "$1" == "$2" ]]; then git rev-parse --verify -q "$1^{tree}"; return; fi
  git cat-file -e "$2^{commit}" 2>/dev/null || return 1
  local out
  out="$(git merge-tree --write-tree "$1" "$2" 2>/dev/null)" || return 1
  head -1 <<<"$out"
}

head_full="$(git rev-parse --verify -q "$HEAD_SHA^{commit}")" || exit 0
# What this run tests: the checkout (the synthetic merge of HEAD into the
# current base, on a pull_request run). Defaults to HEAD itself.
tested_now="$(git rev-parse --verify -q "${CI_TESTED_SHA:-$head_full}^{tree}")" || exit 0
mapfile -t chain < <(git rev-list --first-parent --max-count=13 "$head_full" 2>/dev/null | tail -n +2)
for c in "${chain[@]}"; do
  [[ -n "$c" ]] || continue
  # 1. The raw delta (which the caller schedules jobs by) is reusable.
  delta_reusable "$c" "$head_full" || { skip "$c" "raw delta to HEAD is empty or touches a non-reusable path"; continue; }
  # 2-3. R's own covering run of this PR. A cancelled or non-covering run does
  # not end the search: an older head can still qualify on its own.
  sel="$(own_run_selection "$c")" || continue
  base="${sel%% *}"; keys="${sel#* }"
  covers "$keys" || { skip "$c" "its run selected ($keys ) but HEAD needs ($REQUIRED)"; continue; }
  # 4. Effective inputs. The tree R's run tested (its synthetic merge with the
  # base of that moment) differs from the tree tested now only in reusable
  # paths. The raw delta alone is not enough: a base change that R's run
  # tested and main later reverted is absent from R..HEAD (review of #3751).
  t_r="$(tested_tree "$c" "$base")" || { skip "$c" "cannot rebuild its tested merge with base ${base:0:10}"; continue; }
  [[ -n "$t_r" ]] || { skip "$c" "empty tested tree"; continue; }
  if [[ "$t_r" == "$tested_now" ]] || delta_reusable "$t_r" "$tested_now"; then
    echo "$c"
    exit 0
  fi
  skip "$c" "its tested tree differs from the tree tested now outside the reusable paths"
done
exit 0
