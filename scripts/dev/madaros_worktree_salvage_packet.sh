#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR=""
PLAN_TSV=""
AUDIT_TSV=""
MAKE_TARBALL=0

usage() {
  cat <<'USAGE'
Usage: scripts/dev/madaros_worktree_salvage_packet.sh [options]

Build a non-destructive salvage packet from the Madaros cleanup plan. The packet
contains per-worktree status, tracked/staged binary diffs, short logs, untracked
file manifests, and size indexes. It never pushes branches, removes worktrees,
deletes files, resets, or cleans.

Options:
  --out-dir DIR       write packet under DIR (default: mktemp /tmp dir)
  --plan-tsv PATH     use an existing madaros-cleanup-plan.tsv
  --audit-tsv PATH    pass an existing worktree audit TSV through the cleanup
                      planner before building the packet
  --tarball           also write DIR.tar.gz plus DIR.tar.gz.sha256
  --no-tar            do not write a tarball (default)
  -h, --help          show this help

Environment:
  SOUNIO_MADAROS_CLEANUP_ALLOW_RE   passed through to the cleanup planner
USAGE
}

while (($#)); do
  case "$1" in
    --out-dir)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "error: --out-dir requires a directory" >&2
        exit 2
      fi
      OUT_DIR="$1"
      ;;
    --plan-tsv)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "error: --plan-tsv requires a path" >&2
        exit 2
      fi
      PLAN_TSV="$1"
      ;;
    --audit-tsv)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "error: --audit-tsv requires a path" >&2
        exit 2
      fi
      AUDIT_TSV="$1"
      ;;
    --tarball)
      MAKE_TARBALL=1
      ;;
    --no-tar)
      MAKE_TARBALL=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(mktemp -d /tmp/madaros-cleanup-salvage-packet.XXXXXX)"
fi
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/per-worktree"

if [[ -n "$PLAN_TSV" && -n "$AUDIT_TSV" ]]; then
  echo "error: use either --plan-tsv or --audit-tsv, not both" >&2
  exit 2
fi

if [[ -n "$PLAN_TSV" ]]; then
  [[ -f "$PLAN_TSV" ]] || {
    echo "error: plan TSV not found: $PLAN_TSV" >&2
    exit 2
  }
  plan_abs="$(cd "$(dirname "$PLAN_TSV")" && pwd)/$(basename "$PLAN_TSV")"
  plan_target="$(cd "$OUT_DIR" && pwd)/madaros-cleanup-plan.tsv"
  if [[ "$plan_abs" != "$plan_target" ]]; then
    cp "$PLAN_TSV" "$plan_target"
  fi
  PLAN_TSV="$OUT_DIR/madaros-cleanup-plan.tsv"
else
  cleanup_args=(--out-dir "$OUT_DIR")
  if [[ -n "$AUDIT_TSV" ]]; then
    [[ -f "$AUDIT_TSV" ]] || {
      echo "error: audit TSV not found: $AUDIT_TSV" >&2
      exit 2
    }
    cleanup_args+=(--audit-tsv "$AUDIT_TSV")
  fi
  scripts/dev/madaros_worktree_cleanup_plan.sh "${cleanup_args[@]}" \
    > "$OUT_DIR/logs/cleanup-planner.stdout"
  PLAN_TSV="$OUT_DIR/madaros-cleanup-plan.tsv"
fi

[[ -f "$PLAN_TSV" ]] || {
  echo "error: cleanup plan did not produce $PLAN_TSV" >&2
  exit 2
}

slug_for_path() {
  basename "$1" | sed 's/[^A-Za-z0-9._-]\+/-/g; s/^-//; s/-$//'
}

write_if_git() {
  local wt_path="$1"
  shift
  git -C "$wt_path" "$@" 2>&1 || true
}

{
  printf 'salvage_packet=%s\n' "$OUT_DIR"
  printf 'generated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'greenline_tip=%s\n' "$(git rev-parse HEAD 2>/dev/null || true)"
  printf 'plan_tsv=%s\n' "$PLAN_TSV"
  printf 'note=%s\n' 'Non-destructive packet: tracked/staged diffs, status, logs, and untracked manifests only. No branches pushed or worktrees removed.'
} > "$OUT_DIR/README.txt"

awk -F '\t' -v OFS=$'\034' '
  NR > 1 {
    print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
      $14, $15, $16, $17, $18, $19, $20, $21, $22
  }
' "$PLAN_TSV" | while IFS=$'\034' read -r category wt_path branch head upstream state dirty_count ahead behind remote_ref prs unique_origin_main unique_upstream tracked_dirty untracked_dirty tracked_diff_files tracked_diff_added tracked_diff_deleted salvage_ref critical_dirty critical_vs_base disposition; do
  slug="$(slug_for_path "$wt_path")"
  wt_dir="$OUT_DIR/per-worktree/$slug"
  mkdir -p "$wt_dir"

  {
    printf 'path=%s\n' "$wt_path"
    printf 'branch=%s\n' "$branch"
    printf 'category=%s\n' "$category"
    printf 'head=%s\n' "$head"
    printf 'upstream=%s\n' "$upstream"
    printf 'state=%s\n' "$state"
    printf 'dirty_count=%s\n' "$dirty_count"
    printf 'ahead=%s\n' "$ahead"
    printf 'behind=%s\n' "$behind"
    printf 'remote_ref=%s\n' "$remote_ref"
    printf 'prs=%s\n' "$prs"
    printf 'unique_commits_origin_main=%s\n' "$unique_origin_main"
    printf 'unique_commits_upstream=%s\n' "${unique_upstream:-}"
    printf 'tracked_dirty_files=%s\n' "$tracked_dirty"
    printf 'untracked_dirty_files=%s\n' "$untracked_dirty"
    printf 'tracked_diff_files=%s\n' "$tracked_diff_files"
    printf 'tracked_diff_added=%s\n' "$tracked_diff_added"
    printf 'tracked_diff_deleted=%s\n' "$tracked_diff_deleted"
    printf 'suggested_salvage_ref=%s\n' "$salvage_ref"
    printf 'critical_dirty=%s\n' "$critical_dirty"
    printf 'critical_vs_base=%s\n' "$critical_vs_base"
    printf 'disposition=%s\n' "$disposition"
    printf 'actual_head=%s\n' "$(git -C "$wt_path" rev-parse HEAD 2>/dev/null || true)"
  } > "$wt_dir/metadata.txt"

  write_if_git "$wt_path" status --short --branch > "$wt_dir/status.short.txt"
  write_if_git "$wt_path" diff --stat > "$wt_dir/diff.stat.txt"
  git -C "$wt_path" diff --binary > "$wt_dir/tracked.diff" 2>/dev/null || true
  git -C "$wt_path" diff --cached --binary > "$wt_dir/staged.diff" 2>/dev/null || true
  write_if_git "$wt_path" log --oneline --decorate --max-count=40 > "$wt_dir/log.head.txt"
  write_if_git "$wt_path" log --oneline origin/main..HEAD > "$wt_dir/log.origin-main-ahead.txt"
  write_if_git "$wt_path" ls-files --others --exclude-standard > "$wt_dir/untracked.files.txt"

  : > "$wt_dir/untracked.sizes.tsv"
  if [[ -s "$wt_dir/untracked.files.txt" ]]; then
    while IFS= read -r rel; do
      [[ -n "$rel" ]] || continue
      if [[ -e "$wt_path/$rel" ]]; then
        bytes="$(du -sb "$wt_path/$rel" 2>/dev/null | awk '{print $1}')"
        printf '%s\t%s\n' "${bytes:-?}" "$rel"
      fi
    done < "$wt_dir/untracked.files.txt" > "$wt_dir/untracked.sizes.tsv"
  fi
done

find "$OUT_DIR/per-worktree" -mindepth 2 -maxdepth 2 -type f \
  | sed "s#^$OUT_DIR/##" \
  | sort > "$OUT_DIR/file-index.txt"

find "$OUT_DIR/per-worktree" -name tracked.diff -type f -printf '%s\t%p\n' \
  | sort -nr \
  | sed "s#\t$OUT_DIR/#\t#" > "$OUT_DIR/tracked-diff-sizes.tsv"

find "$OUT_DIR/per-worktree" -name untracked.files.txt -type f \
  -exec sh -c 'for f; do c=$(wc -l < "$f"); printf "%s\t%s\n" "$c" "$f"; done' sh {} + \
  | sort -nr \
  | sed "s#\t$OUT_DIR/#\t#" > "$OUT_DIR/untracked-counts.tsv"

echo "packet_dir=$OUT_DIR"
echo "plan_tsv=$PLAN_TSV"
echo "file_index=$OUT_DIR/file-index.txt"

if [[ "$MAKE_TARBALL" == "1" ]]; then
  tarball="$OUT_DIR.tar.gz"
  tar -C "$(dirname "$OUT_DIR")" -czf "$tarball" "$(basename "$OUT_DIR")"
  sha256sum "$tarball" > "$tarball.sha256"
  echo "tarball=$tarball"
  echo "tarball_sha256=$tarball.sha256"
fi

awk -F '\t' '
  NR > 1 {
    total++
    by_category[$1]++
  }
  END {
    printf "planned_worktrees=%d\n", total
    for (category in by_category) {
      printf "category[%s]=%d\n", category, by_category[category]
    }
  }
' "$PLAN_TSV" | sort
