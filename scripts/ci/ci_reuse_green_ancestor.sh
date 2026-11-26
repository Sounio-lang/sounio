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
#      classification selected passed. Those jobs cover HEAD's jobs, and their
#      inputs are byte-identical between R and HEAD.
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
#   GITHUB_REPOSITORY, GH_TOKEN   for the API lookup
#   CI_REUSE_GREEN_SHAS    selftests only, instead of the API: entries
#                          "<sha>:<key>,<key>,..." (the keys R's own run selected)
#   CI_REUSE_DISABLE=1     never reuse
set -uo pipefail

HEAD_SHA="${CI_HEAD_SHA:-HEAD}"
REQUIRED="${CI_REUSE_REQUIRED:-}"
[[ "${CI_REUSE_DISABLE:-0}" == 1 ]] && exit 0

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

# own_run_selection <sha>: prints the impact keys that R's own PR run of ci.yml
# selected, or fails. Fails when the run is missing, not green, not this PR's,
# or itself reused an earlier verdict.
own_run_selection() {
  local c="$1"
  if [[ -n "${CI_REUSE_GREEN_SHAS+set}" ]]; then
    local e
    for e in $CI_REUSE_GREEN_SHAS; do
      if [[ "${e%%:*}" == "$c" ]]; then
        printf '%s\n' "${e#*:}" | tr ',' ' '
        return 0
      fi
    done
    return 1
  fi
  [[ -n "${GITHUB_REPOSITORY:-}" && -n "${CI_PR_NUMBER:-}" ]] || return 1
  command -v gh >/dev/null 2>&1 || return 1
  local run_id job_id log keys k v reused=""
  run_id="$(gh api "repos/$GITHUB_REPOSITORY/actions/runs?head_sha=$c&event=pull_request&status=completed&per_page=30" \
      --jq "[.workflow_runs[] | select(.path == \".github/workflows/ci.yml\" and .head_sha == \"$c\" and .conclusion == \"success\" and any(.pull_requests[]; .number == $CI_PR_NUMBER)) | .id] | first // empty" 2>/dev/null)" || return 1
  [[ "$run_id" =~ ^[0-9]+$ ]] || return 1
  job_id="$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$run_id/jobs?per_page=100" \
      --jq '[.jobs[] | select(.name == "Impact" and .conclusion == "success") | .id] | first // empty' 2>/dev/null)" || return 1
  [[ "$job_id" =~ ^[0-9]+$ ]] || return 1
  log="$(gh api "repos/$GITHUB_REPOSITORY/actions/jobs/$job_id/logs" 2>/dev/null)" || return 1
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
  [[ -z "$reused" ]] || return 1
  [[ -n "$keys" ]] || return 1
  printf '%s\n' "$keys"
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
    # A changelog fragment may be added or edited in the delta, never deleted.
    if [[ "$p" == changelog.d/* && "$st" != A && "$st" != M ]]; then ok=0; break; fi
  done <<<"$delta"
  [[ $ok == 1 ]] || continue
  # A qualifying candidate whose run was cancelled, or that does not cover HEAD,
  # does not end the search: an older head can still qualify on its own.
  sel="$(own_run_selection "$c")" || continue
  if covers "$sel"; then
    echo "$c"
    exit 0
  fi
done
exit 0
