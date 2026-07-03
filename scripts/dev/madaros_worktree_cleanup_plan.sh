#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_ALLOW_RE='^(/workspace/sounio|/workspace/sounio-effects|/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b)$'
ALLOW_RE="${SOUNIO_MADAROS_CLEANUP_ALLOW_RE:-$DEFAULT_ALLOW_RE}"
OUT_DIR=""
AUDIT_TSV=""

usage() {
  cat <<'USAGE'
Usage: scripts/dev/madaros_worktree_cleanup_plan.sh [options]

Build a non-destructive cleanup plan for Madaros critical dirty worktrees.
The script writes an audit TSV, a classified cleanup TSV, and a commented shell
plan. It never runs git push, git reset, git clean, git branch -D, or
git worktree remove.

Options:
  --out-dir DIR       write outputs under DIR (default: mktemp /tmp dir)
  --audit-tsv PATH    use an existing worktree_branch_audit TSV instead of
                      generating a fresh one
  -h, --help          show this help

Environment:
  SOUNIO_MADAROS_CLEANUP_ALLOW_RE   awk regex for allowed critical dirty paths
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
    --audit-tsv)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "error: --audit-tsv requires a path" >&2
        exit 2
      fi
      AUDIT_TSV="$1"
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
  OUT_DIR="$(mktemp -d /tmp/sounio-madaros-cleanup-plan.XXXXXX)"
fi
mkdir -p "$OUT_DIR"

if [[ -z "$AUDIT_TSV" ]]; then
  AUDIT_TSV="$OUT_DIR/worktree-audit.tsv"
  SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE="$ALLOW_RE" \
    scripts/dev/worktree_branch_audit.sh "$AUDIT_TSV" > "$OUT_DIR/worktree-audit.log"
elif [[ ! -f "$AUDIT_TSV" ]]; then
  echo "error: audit TSV not found: $AUDIT_TSV" >&2
  exit 2
fi

PLAN_TSV="$OUT_DIR/madaros-cleanup-plan.tsv"
PLAN_SH="$OUT_DIR/madaros-cleanup-plan.commands.sh"

have() {
  command -v "$1" >/dev/null 2>&1
}

remote_ref_for_branch() {
  local branch="$1"
  if [[ "$branch" == "detached" || -z "$branch" ]]; then
    printf 'detached'
    return 0
  fi
  if git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
    printf 'origin/%s' "$branch"
  else
    printf 'none'
  fi
}

prs_for_branch() {
  local branch="$1"
  if [[ "$branch" == "detached" || -z "$branch" ]] || ! have gh; then
    printf ''
    return 0
  fi
  gh pr list --repo Sounio-lang/sounio --state all --head "$branch" --limit 5 \
    --json number,state,isDraft,baseRefName,mergeable,url \
    --jq 'map("#"+(.number|tostring)+":"+.state+":draft="+(.isDraft|tostring)+":base="+.baseRefName+":mergeable="+(.mergeable//"UNKNOWN")+":"+.url) | join(",")' \
    2>/dev/null || true
}

category_for_row() {
  local path="$1" branch="$2"

  case "$path" in
    /tmp/sounio-madaros-greenline-codex|/workspace/sounio-greenfirst)
      printf 'greenline_leftover'
      return 0
      ;;
    /tmp/sounio-madaros-main-port-20260702|/tmp/sounio-madaros-lower-known-test-20260702|/tmp/sounio-phase03-step5-clean|/workspace/sounio-gc-fix-20260701)
      printf 'detached_risky'
      return 0
      ;;
  esac

  if [[ "$branch" == "detached" ]]; then
    printf 'detached_risky'
    return 0
  fi

  case "$path" in
    /tmp/sounio-active-compact-ir-20260702|/tmp/sounio-active-lowerfix|/tmp/sounio-bdf64-bridge|/tmp/sounio-madaros-plan-mainline-20260702|/tmp/sounio-phase03-4e68|/tmp/sounio-phase03-step5-fix|/tmp/sounio-phase03-step5-lower-revert|/tmp/sounio-phase03-step5-lowerfix-min)
      printf 'active_other_lane_wip'
      ;;
    /tmp/sounio-abide-madaros-rebuild-20260630|/tmp/sounio-abide-madaros-singlemodule-20260630|/tmp/sounio-project-spine-slice-20260630|/tmp/sounio-madaros-fncount-20260701|/tmp/sounio-madaros-retire-lean-single-20260627)
      printf 'stale_local_temp'
      ;;
    *)
      printf 'unclassified'
      ;;
  esac
}

disposition_for_category() {
  local category="$1" remote_ref="$2" prs="$3"
  case "$category" in
    greenline_leftover)
      printf 'inspect dirty patch; if no unique value remains, retire after explicit approval'
      ;;
    detached_risky)
      printf 'create archive branch and patch before any removal'
      ;;
    active_other_lane_wip)
      printf 'do not remove; confirm owner or move to explicit active lane'
      ;;
    stale_local_temp)
      if [[ "$remote_ref" == "none" && -z "$prs" ]]; then
        printf 'check unique commits; push archive/salvage before removal'
      else
        printf 'review remote/PR state before removal'
      fi
      ;;
    *)
      printf 'manual review required'
      ;;
  esac
}

quote_sh() {
  printf '%q' "$1"
}

slug_for_path() {
  basename "$1" | sed 's/[^A-Za-z0-9._-]\+/-/g; s/^-//; s/-$//'
}

emit_unallowed_rows() {
  awk -F '\t' -v allow_re="$ALLOW_RE" -v OFS=$'\034' '
    NR > 1 && $9 != "" {
      if (allow_re != "" && $1 ~ allow_re) {
        next
      }
      print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
    }
  ' "$AUDIT_TSV"
}

emit_plan_rows() {
  awk -F '\t' -v OFS=$'\034' '
    NR > 1 {
      print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
    }
  ' "$PLAN_TSV"
}

{
  printf 'category\tpath\tbranch\thead\tupstream\tstate\tdirty_count\tahead\tbehind\tremote_ref\tprs\tcritical_dirty\tdisposition\n'

  emit_unallowed_rows | while IFS=$'\034' read -r path branch head upstream state dirty_count ahead behind critical_dirty critical_vs pr; do
    remote_ref="$(remote_ref_for_branch "$branch")"
    prs="$(prs_for_branch "$branch")"
    category="$(category_for_row "$path" "$branch")"
    disposition="$(disposition_for_category "$category" "$remote_ref" "$prs")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$category" "$path" "$branch" "$head" "$upstream" "$state" "$dirty_count" \
      "$ahead" "$behind" "$remote_ref" "$prs" "$critical_dirty" "$disposition"
  done
} > "$PLAN_TSV"

{
  cat <<'HEADER'
#!/usr/bin/env bash
set -euo pipefail

# Generated by scripts/dev/madaros_worktree_cleanup_plan.sh.
# This file is intentionally non-executing: every mutating command is commented.
# Review each block, push/archive before deletion, and obtain explicit operator
# approval before uncommenting any git push/worktree remove/branch delete command.

HEADER

  emit_plan_rows | while IFS=$'\034' read -r category path branch head upstream state dirty_count ahead behind remote_ref prs critical_dirty disposition; do
    path_q="$(quote_sh "$path")"
    archive_branch="archive/madaros-$(slug_for_path "$path")"
    archive_q="$(quote_sh "$archive_branch")"
    echo "# category=$category branch=$branch head=$head remote_ref=$remote_ref"
    echo "# disposition=$disposition"
    echo "git -C $path_q status --short --branch"
    echo "git -C $path_q diff --stat"
    echo "git -C $path_q diff > /tmp/$(basename "$path").dirty.patch"
    if [[ "$category" == "active_other_lane_wip" || "$category" == "unclassified" ]]; then
      echo "# owner confirmation required before any archive, push, or removal"
      echo
      continue
    fi
    if [[ "$branch" == "detached" ]]; then
      echo "# git -C $path_q switch -c $archive_q"
      echo "# git push origin $archive_q"
    elif [[ "$remote_ref" == "none" ]]; then
      echo "git -C $path_q log --oneline origin/main..HEAD"
      echo "# git push origin HEAD:refs/heads/$archive_q"
    else
      echo "git -C $path_q log --oneline @{u}..HEAD || true"
    fi
    echo "# git worktree remove $path_q"
    if [[ "$branch" != "detached" ]]; then
      branch_q="$(quote_sh "$branch")"
      echo "# git branch -d $branch_q"
    fi
    echo
  done
} > "$PLAN_SH"
chmod +x "$PLAN_SH"

echo "audit_tsv=$AUDIT_TSV"
echo "plan_tsv=$PLAN_TSV"
echo "plan_commands=$PLAN_SH"
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
