#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGISTRY="$ROOT_DIR/docs/internal/concepts/registry.tsv"
BINDINGS="$ROOT_DIR/docs/internal/concepts/bindings.tsv"
STALE_SECONDS="${SOUNIO_SEMANTIC_STALE_SECONDS:-14400}"
ACTIVITY_SECONDS="${SOUNIO_SEMANTIC_ACTIVITY_SECONDS:-14400}"
STRICT_OVERLAP=0
CURRENT_ONLY=0
SCAN_PROCESSES=1
VERBOSE_WRITERS=0

usage() {
  cat <<'USAGE'
Usage: scripts/dev/sounio_semantic_status.sh [options]

Read-only view of concepts, dirty semantic writers, overlapping worktree
edits, pending interfaces, and long-running compiler/runtime processes.

Options:
  --current-only       inspect only the current worktree (fast gate mode)
  --no-processes       skip process-age inspection
  --verbose-writers    print every concept/path/worktree binding
  --stale-seconds N    process alert threshold (default: 14400)
  --activity-seconds N recent dirty-writer threshold (default: 14400)
  --strict-overlap     exit nonzero when an exact dirty path has >1 writer
  -h, --help           show this help
USAGE
}

while (($#)); do
  case "$1" in
    --current-only) CURRENT_ONLY=1 ;;
    --no-processes) SCAN_PROCESSES=0 ;;
    --verbose-writers) VERBOSE_WRITERS=1 ;;
    --strict-overlap) STRICT_OVERLAP=1 ;;
    --stale-seconds)
      shift
      [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]] || {
        echo "error: --stale-seconds requires a nonnegative integer" >&2
        exit 2
      }
      STALE_SECONDS="$1"
      ;;
    --activity-seconds)
      shift
      [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]] || {
        echo "error: --activity-seconds requires a nonnegative integer" >&2
        exit 2
      }
      ACTIVITY_SECONDS="$1"
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ -f "$REGISTRY" ]] || { echo "error: missing $REGISTRY" >&2; exit 1; }
[[ -f "$BINDINGS" ]] || { echo "error: missing $BINDINGS" >&2; exit 1; }

declare -a BIND_CONCEPT=()
declare -a BIND_PATTERN=()
declare -a BIND_ROLE=()
while IFS=$'\t' read -r concept pattern role extra; do
  [[ -z "$concept" || "$concept" == \#* ]] && continue
  [[ -n "$pattern" && -n "$role" && -z "${extra:-}" ]] || {
    echo "error: malformed binding row for $concept" >&2
    exit 1
  }
  BIND_CONCEPT+=("$concept")
  BIND_PATTERN+=("$pattern")
  BIND_ROLE+=("$role")
done < "$BINDINGS"

declare -A PATH_WRITER_COUNT=()
declare -A PATH_WRITERS=()
declare -A PATH_WORKTREE_SEEN=()
declare -A PATH_ACTIVE_WORKTREE_SEEN=()
declare -A PATH_ACTIVE_WRITER_COUNT=()
declare -A CONCEPT_WRITERS=()
declare -A CONCEPT_PATHS=()
declare -A WORKTREE_SEMANTIC_PATHS=()
declare -A WORKTREE_LATEST_EPOCH=()
declare -A WORKTREE_BRANCH=()

concept_matches_for_path() {
  local path="$1"
  local i
  for ((i=0; i<${#BIND_PATTERN[@]}; i++)); do
    if [[ "$path" == ${BIND_PATTERN[$i]} ]]; then
      printf '%s\t%s\n' "${BIND_CONCEPT[$i]}" "${BIND_ROLE[$i]}"
    fi
  done
}

short_branch() {
  local wt="$1"
  local branch
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

printf 'Sounio semantic status\n'
printf 'snapshot_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'repo=%s\n' "$ROOT_DIR"
printf 'worktrees_seen=%s\n' "${#WORKTREES[@]}"

printf '\n== Concept Registry ==\n'
while IFS=$'\t' read -r concept status authority contract canonical pending extra; do
  [[ -z "$concept" || "$concept" == \#* ]] && continue
  printf '%-36s status=%-11s canonical=%s pending=%s\n' \
    "$concept" "$status" "$canonical" "$pending"
done < "$REGISTRY"

printf '\n== Dirty Semantic Writers ==\n'
writer_rows=0
snapshot_epoch="$(date +%s)"
for wt in "${WORKTREES[@]}"; do
  [[ -d "$wt" ]] || continue
  branch="$(short_branch "$wt")"
  WORKTREE_BRANCH["$wt"]="$branch"
  head_epoch="$(git -C "$wt" show -s --format=%ct HEAD 2>/dev/null || echo 0)"
  while IFS= read -r status_line; do
    [[ -n "$status_line" ]] || continue
    path="${status_line:3}"
    if [[ "$path" == *' -> '* ]]; then path="${path##* -> }"; fi
    matches="$(concept_matches_for_path "$path")"
    [[ -n "$matches" ]] || continue
    file_epoch="$(stat -c %Y "$wt/$path" 2>/dev/null || echo 0)"
    latest_epoch="$file_epoch"
    # Deleted paths have no mtime; only then use the branch tip as an age proxy.
    ((latest_epoch == 0)) && latest_epoch="$head_epoch"
    ((latest_epoch > ${WORKTREE_LATEST_EPOCH[$wt]:-0})) && WORKTREE_LATEST_EPOCH["$wt"]="$latest_epoch"
    WORKTREE_SEMANTIC_PATHS["$wt"]=$(( ${WORKTREE_SEMANTIC_PATHS["$wt"]:-0} + 1 ))
    age_seconds=$((snapshot_epoch - latest_epoch))
    ((age_seconds < 0)) && age_seconds=0
    while IFS=$'\t' read -r concept role; do
      [[ -n "$concept" ]] || continue
      if [[ "$VERBOSE_WRITERS" == "1" ]]; then
        printf 'writer concept=%s role=%s path=%s branch=%s worktree=%s\n' \
          "$concept" "$role" "$path" "$branch" "$wt"
      fi
      writer_rows=$((writer_rows + 1))
      path_wt_key="$path|$wt"
      if [[ -z "${PATH_WORKTREE_SEEN[$path_wt_key]:-}" ]]; then
        PATH_WORKTREE_SEEN[$path_wt_key]=1
        PATH_WRITER_COUNT["$path"]=$(( ${PATH_WRITER_COUNT["$path"]:-0} + 1 ))
        if [[ -n "${PATH_WRITERS["$path"]:-}" ]]; then
          PATH_WRITERS["$path"]+=" | $branch@$wt"
        else
          PATH_WRITERS["$path"]="$branch@$wt"
        fi
      fi
      if ((age_seconds <= ACTIVITY_SECONDS)) && [[ -z "${PATH_ACTIVE_WORKTREE_SEEN[$path_wt_key]:-}" ]]; then
        PATH_ACTIVE_WORKTREE_SEEN[$path_wt_key]=1
        PATH_ACTIVE_WRITER_COUNT["$path"]=$(( ${PATH_ACTIVE_WRITER_COUNT["$path"]:-0} + 1 ))
      fi
      CONCEPT_WRITERS["$concept"]=$(( ${CONCEPT_WRITERS["$concept"]:-0} + 1 ))
      CONCEPT_PATHS["$concept"]="${CONCEPT_PATHS["$concept"]:-} $path"
    done <<< "$matches"
  done < <(git -C "$wt" status --porcelain=v1 --untracked-files=all 2>/dev/null || true)
done
if [[ "$writer_rows" -eq 0 ]]; then
  echo 'none'
elif [[ "$VERBOSE_WRITERS" == "0" ]]; then
  echo "details=suppressed semantic_bindings=$writer_rows use=--verbose-writers"
fi

printf '\n== Dirty Worktree Activity ==\n'
if ((${#WORKTREE_SEMANTIC_PATHS[@]})); then
  while IFS= read -r wt; do
    latest_epoch="${WORKTREE_LATEST_EPOCH[$wt]:-0}"
    age_seconds=$((snapshot_epoch - latest_epoch))
    ((age_seconds < 0)) && age_seconds=0
    if [[ "$wt" == */scratchpad/* ]]; then
      activity="scratch-copy"
    elif ((age_seconds <= ACTIVITY_SECONDS)); then
      activity="recent-dirty"
    else
      activity="stale-residue"
    fi
    printf 'activity=%s age_seconds=%s semantic_bindings=%s branch=%s worktree=%s\n' \
      "$activity" "$age_seconds" "${WORKTREE_SEMANTIC_PATHS[$wt]}" "${WORKTREE_BRANCH[$wt]}" "$wt"
  done < <(printf '%s\n' "${!WORKTREE_SEMANTIC_PATHS[@]}" | sort)
else
  echo 'none'
fi

printf '\n== Exact Path Collisions ==\n'
collision_count=0
if ((${#PATH_WRITER_COUNT[@]})); then
  while IFS= read -r path; do
    count="${PATH_WRITER_COUNT[$path]}"
    if ((count > 1)); then
      active_count="${PATH_ACTIVE_WRITER_COUNT[$path]:-0}"
      kind="historical-overlap"
      ((active_count > 1)) && kind="live-collision"
      printf 'collision kind=%s path=%s writers=%s recent_writers=%s lanes=%s\n' \
        "$kind" "$path" "$count" "$active_count" "${PATH_WRITERS[$path]}"
      collision_count=$((collision_count + 1))
    fi
  done < <(printf '%s\n' "${!PATH_WRITER_COUNT[@]}" | sort)
fi
[[ "$collision_count" -gt 0 ]] || echo 'none'

printf '\n== Concept Activity ==\n'
while IFS=$'\t' read -r concept status authority contract canonical pending extra; do
  [[ -z "$concept" || "$concept" == \#* ]] && continue
  printf '%-36s dirty_bindings=%s pending=%s\n' \
    "$concept" "${CONCEPT_WRITERS[$concept]:-0}" "$pending"
done < "$REGISTRY"

printf '\n== Runtime Alerts ==\n'
runtime_alerts=0
if [[ "$SCAN_PROCESSES" == "1" ]]; then
  while read -r pid etimes pcpu command; do
    [[ -n "${pid:-}" && "${etimes:-}" =~ ^[0-9]+$ ]] || continue
    if ((etimes >= STALE_SECONDS)); then
      case "$command" in
        *madaros-run*|*build_modular_madaros*|*gen_seed.elf*|*souc-stage*)
          printf 'long-run pid=%s age_seconds=%s cpu=%s command=%s\n' \
            "$pid" "$etimes" "$pcpu" "${command:0:240}"
          runtime_alerts=$((runtime_alerts + 1))
          ;;
      esac
    fi
  done < <(ps -eo pid=,etimes=,pcpu=,args= 2>/dev/null || true)
else
  echo 'skipped'
fi
if [[ "$SCAN_PROCESSES" == "1" && "$runtime_alerts" -eq 0 ]]; then echo 'none'; fi

printf '\n== Summary ==\n'
printf 'semantic_writer_rows=%s\n' "$writer_rows"
printf 'exact_path_collisions=%s\n' "$collision_count"
printf 'runtime_alerts=%s\n' "$runtime_alerts"
printf 'scanner_mode=%s\n' "$( [[ "$CURRENT_ONLY" == "1" ]] && echo current-only || echo all-worktrees )"

if [[ "$STRICT_OVERLAP" == "1" && "$collision_count" -gt 0 ]]; then
  exit 1
fi
