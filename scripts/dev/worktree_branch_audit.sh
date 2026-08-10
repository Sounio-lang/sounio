#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${SOUNIO_AUDIT_BASE_REF:-origin/main}"
INCLUDE_PRS="${SOUNIO_AUDIT_INCLUDE_PRS:-0}"
CHECK_MODE=0
OUT=""

usage() {
  cat <<'USAGE'
Usage: scripts/dev/worktree_branch_audit.sh [--check] [OUT.tsv]

Writes a TSV inventory of git worktrees and prints summary counts.

Options:
  --check        fail if governance thresholds are exceeded
  -h, --help     show this help

Check-mode environment:
  SOUNIO_AUDIT_MAX_PRUNABLE              default: 0
  SOUNIO_AUDIT_MAX_CRITICAL_DIRTY        default: 0
  SOUNIO_AUDIT_MAX_DIRTY                 optional
  SOUNIO_AUDIT_MAX_OPEN_PR               optional
  SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE   optional awk regex for allowed paths
USAGE
}

while (($#)); do
  case "$1" in
    --check)
      CHECK_MODE=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$OUT" ]]; then
        echo "error: multiple output paths provided" >&2
        usage >&2
        exit 2
      fi
      OUT="$1"
      ;;
  esac
  shift
done

if [[ -z "$OUT" ]]; then
  OUT="$(mktemp /tmp/sounio-worktree-audit.XXXXXX.tsv)"
fi

CRITICAL_PATHS=(
  AGENTS.md
  CLAUDE.md
  CLAUDE_HANDOFF.md
  ONBOARDING.md
  .claude/AGENT_OFFLOAD_POLICY.md
  .claude/PARALLEL_BLOCKER_CONTRACT.md
  self-hosted/compiler
  self-hosted/ir
  self-hosted/native
  bin/madaros
  bin/madaros-linux-x86_64
  bin/souc
  bin/souc-linux-x86_64
  scripts/ci
  scripts/dev
  scripts/lib
  scripts/ops
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
    # `git status` can fail on a worktree this process cannot read — a
    # root-owned .git, a stale gitdir, a deleted checkout. Under
    # `set -euo pipefail` that non-zero status crossed the pipe and killed the
    # audit, and the `2>/dev/null` swallowed the only clue: the gate exited 128
    # having printed nothing, indistinguishable from the script crashing.
    # Measured 2026-07-30 on /workspace/.source-fresh/worktree-*, whose .git is
    # owned by root ("fatal: detected dubious ownership").
    #
    # An unreadable worktree is NOT a clean one. Defaulting it to dirty_count=0
    # would report a clean audit over a worktree the audit never read, which is
    # a worse failure than the crash it replaces.
    local st_out st_rc=0
    st_out="$(git -C "$wt_path" status --porcelain 2>&1)" || st_rc=$?
    if [[ "$st_rc" -ne 0 ]]; then
      wt_state="UNREADABLE:$(printf '%s' "$st_out" | head -1 | cut -c1-72)"
      dirty_count="?"
    else
      dirty_count="$(printf '%s' "$st_out" | grep -c . || true)"
      wt_state="clean"
      [[ "$dirty_count" == "0" ]] || wt_state="dirty"
    fi

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
      { git -C "$wt_path" status --porcelain -- "${CRITICAL_PATHS[@]}" 2>/dev/null || true; } \
        | sed 's/[[:space:]]\+/ /g' \
        | paste -sd ';' -
    )"
    critical_vs="$(
      { git -C "$wt_path" diff --name-only "$BASE_REF"...HEAD -- "${CRITICAL_PATHS[@]}" 2>/dev/null || true; } \
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
summary="$(
  awk -F '\t' '
    BEGIN {
      allow_critical_dirty_re = ENVIRON["SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE"]
    }
    NR > 1 {
      total++
      # UNREADABLE is counted on its own, not folded into dirty: "dirty" means
      # measured and found modified, and these were never measured.
      if ($5 ~ /^UNREADABLE/) unreadable++
      else if ($5 != "clean") dirty++
      if ($5 ~ /^PRUNABLE/) prunable++
      if ($9 != "") {
        critical_dirty++
        if (allow_critical_dirty_re == "" || $1 !~ allow_critical_dirty_re) {
          unallowed_critical_dirty++
        }
      }
      if ($10 != "") critical_vs_base++
      if ($11 != "") open_pr++
    }
    END {
      printf "total=%d dirty=%d unreadable=%d prunable=%d critical_dirty=%d unallowed_critical_dirty=%d critical_vs_base=%d open_pr=%d\n",
        total, dirty, unreadable, prunable, critical_dirty, unallowed_critical_dirty, critical_vs_base, open_pr
    }
  ' "$OUT"
)"
echo "$summary"

summary_value() {
  local key="$1"
  printf '%s\n' "$summary" | tr ' ' '\n' | awk -F= -v key="$key" '$1 == key { print $2 }'
}

check_threshold() {
  local label="$1" actual="$2" max="$3"
  if [[ -n "$max" && "$actual" =~ ^[0-9]+$ && "$max" =~ ^[0-9]+$ && "$actual" -gt "$max" ]]; then
    echo "gate_violation: $label=$actual exceeds max=$max" >&2
    return 1
  fi
  return 0
}

echo "critical_dirty_worktrees:"
awk -F '\t' '
  BEGIN {
    allow_critical_dirty_re = ENVIRON["SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE"]
  }
  NR > 1 {
    if ($9 != "") {
      prefix = "unallowed"
      if (allow_critical_dirty_re != "" && $1 ~ allow_critical_dirty_re) {
        prefix = "allowed"
      }
      print "- [" prefix "] " $1 " | " $2 " | " $9
    }
  }
' "$OUT"

echo "unreadable_worktrees:"
awk -F '\t' 'NR > 1 && $5 ~ /^UNREADABLE/ { print "- " $1 " | " $5 }' "$OUT"

if [[ "$CHECK_MODE" == "1" ]]; then
  max_unreadable="${SOUNIO_AUDIT_MAX_UNREADABLE:-0}"
  max_prunable="${SOUNIO_AUDIT_MAX_PRUNABLE:-0}"
  max_critical_dirty="${SOUNIO_AUDIT_MAX_CRITICAL_DIRTY:-0}"
  max_dirty="${SOUNIO_AUDIT_MAX_DIRTY:-}"
  max_open_pr="${SOUNIO_AUDIT_MAX_OPEN_PR:-}"

  failed=0
  # A worktree the audit could not read is a hole in the audit's own coverage,
  # so it is a gate violation by default rather than a silent omission. Raise
  # SOUNIO_AUDIT_MAX_UNREADABLE if a host legitimately has unreadable checkouts.
  check_threshold "unreadable" "$(summary_value unreadable)" "$max_unreadable" || failed=1
  check_threshold "prunable" "$(summary_value prunable)" "$max_prunable" || failed=1
  check_threshold "unallowed_critical_dirty" "$(summary_value unallowed_critical_dirty)" "$max_critical_dirty" || failed=1
  check_threshold "dirty" "$(summary_value dirty)" "$max_dirty" || failed=1
  check_threshold "open_pr" "$(summary_value open_pr)" "$max_open_pr" || failed=1

  if [[ "$failed" != "0" ]]; then
    echo "worktree audit gate failed; inspect $OUT" >&2
    exit 1
  fi
  echo "worktree audit gate passed"
fi
