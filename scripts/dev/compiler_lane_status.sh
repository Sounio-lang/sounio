#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN_REF="${SOUNIO_COMPILER_MAIN_REF:-origin/main}"
FRONTIER_REF="${SOUNIO_COMPILER_FRONTIER_REF:-origin/canon/madaros-v2-sota}"
ACTIVITY_SECONDS="${SOUNIO_COMPILER_ACTIVITY_SECONDS:-14400}"
CURRENT_ONLY=0
VERBOSE=0

usage() {
  cat <<'USAGE'
Usage: scripts/dev/compiler_lane_status.sh [options]

Read-only classification of compiler worktrees. The scanner never assigns
ownership, changes branches, removes worktrees, or promotes a compiler claim.

Options:
  --current-only       inspect only the current worktree
  --verbose            print compiler paths changed by each dirty lane
  --activity-seconds N recent-file threshold (default: 14400)
  --main-ref REF       integrated-product reference (default: origin/main)
  --frontier-ref REF   compiler-frontier reference
  -h, --help           show this help
USAGE
}

while (($#)); do
  case "$1" in
    --current-only) CURRENT_ONLY=1 ;;
    --verbose) VERBOSE=1 ;;
    --activity-seconds)
      shift
      [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]] || {
        echo "error: --activity-seconds requires a nonnegative integer" >&2
        exit 2
      }
      ACTIVITY_SECONDS="$1"
      ;;
    --main-ref) shift; [[ $# -gt 0 ]] || exit 2; MAIN_REF="$1" ;;
    --frontier-ref) shift; [[ $# -gt 0 ]] || exit 2; FRONTIER_REF="$1" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

git -C "$ROOT_DIR" rev-parse --verify "$MAIN_REF^{commit}" >/dev/null 2>&1 || {
  echo "error: main reference is unavailable: $MAIN_REF" >&2
  exit 1
}

is_compiler_path() {
  case "$1" in
    self-hosted/compiler/*|self-hosted/parser/*|self-hosted/check/*|self-hosted/ir/*|self-hosted/codegen/*|\
    scripts/lib/resolve_souc.sh|scripts/lib/resolve_madaros.sh|scripts/ci/build_native_souc.sh|\
    scripts/ci/build_modular_madaros.sh|scripts/run_sio_test_suite.sh|bin/souc|bin/madaros)
      return 0 ;;
    *) return 1 ;;
  esac
}

branch_for() {
  local wt="$1" branch
  branch="$(git -C "$wt" branch --show-current 2>/dev/null || true)"
  [[ -n "$branch" ]] || branch="detached@$(git -C "$wt" rev-parse --short=10 HEAD 2>/dev/null || echo unknown)"
  printf '%s' "$branch"
}

mapfile -t WORKTREES < <(
  if [[ "$CURRENT_ONLY" == "1" ]]; then
    printf '%s\n' "$ROOT_DIR"
  else
    git -C "$ROOT_DIR" worktree list --porcelain | sed -n 's/^worktree //p'
  fi
)

now="$(date +%s)"
integrated=0
content_integrated=0
frontier=0
frontier_integrated=0
active=0
review_ready=0
stale=0
scratch=0
unclassified=0

printf 'Sounio compiler lane status\n'
printf 'snapshot_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'main_ref=%s main_sha=%s\n' "$MAIN_REF" "$(git -C "$ROOT_DIR" rev-parse --short=10 "$MAIN_REF")"
if git -C "$ROOT_DIR" rev-parse --verify "$FRONTIER_REF^{commit}" >/dev/null 2>&1; then
  printf 'frontier_ref=%s frontier_sha=%s\n' "$FRONTIER_REF" "$(git -C "$ROOT_DIR" rev-parse --short=10 "$FRONTIER_REF")"
else
  printf 'frontier_ref=%s frontier_sha=unavailable\n' "$FRONTIER_REF"
fi
printf 'worktrees_seen=%s\n\n' "${#WORKTREES[@]}"

for wt in "${WORKTREES[@]}"; do
  [[ -d "$wt" ]] || continue
  branch="$(branch_for "$wt")"
  head="$(git -C "$wt" rev-parse HEAD 2>/dev/null || true)"
  short_head="${head:0:10}"
  head_epoch="$(git -C "$wt" show -s --format=%ct HEAD 2>/dev/null || echo 0)"
  dirty_count=0
  committed_count=0
  committed_same_main=0
  latest_epoch=0
  compiler_paths=()
  committed_paths=()

  while IFS= read -r status_line; do
    [[ -n "$status_line" ]] || continue
    path="${status_line:3}"
    [[ "$path" == *' -> '* ]] && path="${path##* -> }"
    is_compiler_path "$path" || continue
    compiler_paths+=("$path")
    dirty_count=$((dirty_count + 1))
    file_epoch="$(stat -c %Y "$wt/$path" 2>/dev/null || echo 0)"
    ((file_epoch == 0)) && file_epoch="$head_epoch"
    ((file_epoch > latest_epoch)) && latest_epoch="$file_epoch"
  done < <(git -C "$wt" status --porcelain=v1 --untracked-files=all 2>/dev/null || true)

  if [[ -n "$head" ]]; then
    if git -C "$ROOT_DIR" merge-base --is-ancestor "$head" "$MAIN_REF" 2>/dev/null; then
      committed_source=(git -C "$wt" diff-tree --no-commit-id --name-only -r "$head")
    else
      committed_source=(git -C "$wt" diff --name-only "$MAIN_REF...$head")
    fi
    while IFS= read -r path; do
      [[ -n "$path" ]] || continue
      is_compiler_path "$path" || continue
      committed_paths+=("$path")
      committed_count=$((committed_count + 1))
      head_blob="$(git -C "$wt" rev-parse "$head:$path" 2>/dev/null || echo missing)"
      main_blob="$(git -C "$wt" rev-parse "$MAIN_REF:$path" 2>/dev/null || echo missing)"
      [[ "$head_blob" == "$main_blob" ]] && committed_same_main=$((committed_same_main + 1))
    done < <("${committed_source[@]}" 2>/dev/null || true)
  fi

  # A clean docs/research worktree is not a compiler lane merely because its
  # branch is not an ancestor of main.
  ((dirty_count > 0 || committed_count > 0)) || continue

  age_seconds=-1
  if ((latest_epoch == 0 && committed_count > 0)); then
    latest_epoch="$head_epoch"
  fi
  if ((latest_epoch > 0)); then
    age_seconds=$((now - latest_epoch))
    ((age_seconds < 0)) && age_seconds=0
  fi

  if [[ "$wt" == */scratchpad/* && "$dirty_count" -gt 0 ]]; then
    state="SCRATCH_COPY"; scratch=$((scratch + 1))
  elif [[ "$dirty_count" -gt 0 && "$age_seconds" -le "$ACTIVITY_SECONDS" ]]; then
    state="ACTIVE"; active=$((active + 1))
  elif [[ "$dirty_count" -gt 0 ]]; then
    state="STALE_WITH_RESIDUE"; stale=$((stale + 1))
  elif [[ -n "$head" ]] && git -C "$ROOT_DIR" merge-base --is-ancestor "$head" "$MAIN_REF" 2>/dev/null; then
    state="INTEGRATED"; integrated=$((integrated + 1))
  elif ((committed_count > 0 && committed_same_main == committed_count)); then
    state="CONTENT_INTEGRATED"; content_integrated=$((content_integrated + 1))
  elif [[ -n "$head" ]] && git -C "$ROOT_DIR" rev-parse --verify "$FRONTIER_REF^{commit}" >/dev/null 2>&1 && \
       git -C "$ROOT_DIR" merge-base --is-ancestor "$head" "$FRONTIER_REF" 2>/dev/null; then
    if [[ "$head" == "$(git -C "$ROOT_DIR" rev-parse "$FRONTIER_REF")" ]]; then
      state="FRONTIER"; frontier=$((frontier + 1))
    else
      state="FRONTIER_INTEGRATED"; frontier_integrated=$((frontier_integrated + 1))
    fi
  elif [[ -n "$head" ]]; then
    state="REVIEW_READY"; review_ready=$((review_ready + 1))
  else
    state="UNCLASSIFIED"; unclassified=$((unclassified + 1))
  fi

  printf 'state=%s branch=%s head=%s dirty_compiler_paths=%s committed_compiler_paths=%s same_as_main=%s age_seconds=%s worktree=%s\n' \
    "$state" "$branch" "${short_head:-unknown}" "$dirty_count" "$committed_count" "$committed_same_main" "$age_seconds" "$wt"
  if [[ "$VERBOSE" == "1" ]]; then
    for path in "${compiler_paths[@]}"; do printf '  path=%s\n' "$path"; done
    for path in "${committed_paths[@]}"; do printf '  committed_path=%s\n' "$path"; done
  fi
done

printf '\n== Summary ==\n'
printf 'integrated=%s\ncontent_integrated=%s\nfrontier=%s\nfrontier_integrated=%s\nactive=%s\nreview_ready=%s\nstale_with_residue=%s\nscratch_copy=%s\nunclassified=%s\n' \
  "$integrated" "$content_integrated" "$frontier" "$frontier_integrated" "$active" "$review_ready" "$stale" "$scratch" "$unclassified"
printf 'scanner_mode=%s\n' "$( [[ "$CURRENT_ONLY" == "1" ]] && echo current-only || echo all-worktrees )"
