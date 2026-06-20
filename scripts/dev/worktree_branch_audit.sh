#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${SOUNIO_AUDIT_BASE_REF:-origin/main}"
INCLUDE_PRS="${SOUNIO_AUDIT_INCLUDE_PRS:-0}"
OUT="${1:-}"

if [[ -z "$OUT" ]]; then
  OUT="$(mktemp /tmp/sounio-worktree-audit.XXXXXX.tsv)"
fi

CRITICAL_PATHS=(
  self-hosted/compiler
  self-hosted/ir
  self-hosted/native
  bin/madaros
  bin/madaros-linux-x86_64
  bin/souc
  bin/souc-linux-x86_64
  scripts/ci
  scripts/lib
  .github/workflows
  artifacts/omega/agent_handoff.log.md
)

print_header() {
  printf 'path\tbranch\thead\tupstream\tstate\tdirty_count\tahead\tbehind\tcritical_dirty\tcritical_vs_base\tpr\n'
}

pr_for_branch() {
  local branch="$1"
  if [[ "$INCLUDE_PRS" != "1" || "$branch" == "detached" ]]; then
    printf ''
    return 0
  fi
  gh pr list \
    --repo Sounio-lang/sounio \
    --head "$branch" \
    --state open \
    --json number \
    --jq '.[0].number // ""' 2>/dev/null || true
}

write_rows() {
  git worktree list --porcelain | awk '
    /^worktree / {
      if (wt) print wt "\t" head "\t" branch "\t" prunable
      wt=$2; head=""; branch=""; prunable=""
    }
    /^HEAD / { head=$2 }
    /^branch / { branch=$2 }
    /^detached$/ { branch="detached" }
    /^prunable / { prunable=$0 }
    END {
      if (wt) print wt "\t" head "\t" branch "\t" prunable
    }
  ' | while IFS=$'\t' read -r wt_path head branchref prunable; do
    local branch upstream dirty_count wt_state ahead behind counts critical_dirty critical_vs pr

    if [[ ! -d "$wt_path/.git" && ! -f "$wt_path/.git" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$wt_path" "${branchref#refs/heads/}" "${head:0:12}" "" \
        "PRUNABLE:${prunable#prunable }" "" "" "" "" "" ""
      continue
    fi

    branch="$(git -C "$wt_path" branch --show-current 2>/dev/null || true)"
    [[ -n "$branch" ]] || branch="detached"
    upstream="$(git -C "$wt_path" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
    dirty_count="$(git -C "$wt_path" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    wt_state="clean"
    [[ "$dirty_count" == "0" ]] || wt_state="dirty"

    if [[ -n "$upstream" ]]; then
      counts="$(git -C "$wt_path" rev-list --left-right --count "$upstream"...HEAD 2>/dev/null || printf '? ?')"
      behind="$(printf '%s' "$counts" | awk '{print $1}')"
      ahead="$(printf '%s' "$counts" | awk '{print $2}')"
    else
      counts="$(git -C "$wt_path" rev-list --left-right --count "$BASE_REF"...HEAD 2>/dev/null || printf '? ?')"
      behind="${BASE_REF}:$(printf '%s' "$counts" | awk '{print $1}')"
      ahead="${BASE_REF}:$(printf '%s' "$counts" | awk '{print $2}')"
    fi

    critical_dirty="$(
      git -C "$wt_path" status --porcelain -- "${CRITICAL_PATHS[@]}" 2>/dev/null \
        | sed 's/[[:space:]]\+/ /g' \
        | paste -sd ';' -
    )"
    critical_vs="$(
      git -C "$wt_path" diff --name-only "$BASE_REF"...HEAD -- "${CRITICAL_PATHS[@]}" 2>/dev/null \
        | paste -sd ',' -
    )"
    pr="$(pr_for_branch "$branch")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$wt_path" "$branch" "${head:0:12}" "$upstream" "$wt_state" \
      "$dirty_count" "$ahead" "$behind" "$critical_dirty" "$critical_vs" "$pr"
  done
}

{
  print_header
  write_rows
} > "$OUT"

echo "audit_tsv=$OUT"
awk -F '\t' '
  NR > 1 {
    total++
    if ($5 != "clean") dirty++
    if ($5 ~ /^PRUNABLE/) prunable++
    if ($9 != "") critical_dirty++
    if ($10 != "") critical_vs_base++
    if ($11 != "") open_pr++
  }
  END {
    printf "total=%d dirty=%d prunable=%d critical_dirty=%d critical_vs_base=%d open_pr=%d\n",
      total, dirty, prunable, critical_dirty, critical_vs_base, open_pr
  }
' "$OUT"

echo "critical_dirty_worktrees:"
awk -F '\t' 'NR > 1 && $9 != "" { print "- " $1 " | " $2 " | " $9 }' "$OUT"
