#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${SOUNIO_AUDIT_BASE_REF:-origin/main}"
INCLUDE_PRS="${SOUNIO_AUDIT_INCLUDE_PRS:-0}"
ALLOW_MANIFEST="${SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_MANIFEST:-}"
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
  SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_MANIFEST
                                             optional cleanup approval TSV;
                                             approve_keep_active_allowlist rows
                                             allow exact worktree paths
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

if [[ -n "$ALLOW_MANIFEST" ]]; then
  [[ -f "$ALLOW_MANIFEST" ]] || {
    echo "error: allow manifest not found: $ALLOW_MANIFEST" >&2
    exit 2
  }
  scripts/dev/madaros_worktree_cleanup_approval.sh validate \
    --manifest-tsv "$ALLOW_MANIFEST" >/dev/null
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
if [[ -n "$ALLOW_MANIFEST" ]]; then
  echo "allow_manifest=$ALLOW_MANIFEST"
  echo "allow_manifest_decision=approve_keep_active_allowlist"
fi
summary="$(
  awk -F '\t' '
    function strip_cr(value) {
      sub(/\r$/, "", value)
      return value
    }
    function load_allow_manifest(manifest, line, fields, n, header_seen) {
      if (manifest == "") {
        return
      }
      while ((getline line < manifest) > 0) {
        line = strip_cr(line)
        if (line == "" || line ~ /^#/) {
          continue
        }
        if (!header_seen) {
          header_seen = 1
          continue
        }
        n = split(line, fields, "\t")
        if (n >= 6 && fields[1] == "approve_keep_active_allowlist" && fields[6] != "") {
          allow_manifest_path[fields[6]] = 1
        }
      }
      close(manifest)
    }
    function path_allowed(path) {
      if (allow_critical_dirty_re != "" && path ~ allow_critical_dirty_re) {
        return 1
      }
      if (path in allow_manifest_path) {
        return 1
      }
      return 0
    }
    BEGIN {
      allow_critical_dirty_re = ENVIRON["SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE"]
      allow_critical_dirty_manifest = ENVIRON["SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_MANIFEST"]
      load_allow_manifest(allow_critical_dirty_manifest)
    }
    NR > 1 {
      total++
      if ($5 != "clean") dirty++
      if ($5 ~ /^PRUNABLE/) prunable++
      if ($9 != "") {
        critical_dirty++
        if (!path_allowed($1)) {
          unallowed_critical_dirty++
        }
      }
      if ($10 != "") critical_vs_base++
      if ($11 != "") open_pr++
    }
    END {
      printf "total=%d dirty=%d prunable=%d critical_dirty=%d unallowed_critical_dirty=%d critical_vs_base=%d open_pr=%d\n",
        total, dirty, prunable, critical_dirty, unallowed_critical_dirty, critical_vs_base, open_pr
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
  function strip_cr(value) {
    sub(/\r$/, "", value)
    return value
  }
  function load_allow_manifest(manifest, line, fields, n, header_seen) {
    if (manifest == "") {
      return
    }
    while ((getline line < manifest) > 0) {
      line = strip_cr(line)
      if (line == "" || line ~ /^#/) {
        continue
      }
      if (!header_seen) {
        header_seen = 1
        continue
      }
      n = split(line, fields, "\t")
      if (n >= 6 && fields[1] == "approve_keep_active_allowlist" && fields[6] != "") {
        allow_manifest_path[fields[6]] = 1
      }
    }
    close(manifest)
  }
  function path_allowed(path) {
    if (allow_critical_dirty_re != "" && path ~ allow_critical_dirty_re) {
      return 1
    }
    if (path in allow_manifest_path) {
      return 1
    }
    return 0
  }
  BEGIN {
    allow_critical_dirty_re = ENVIRON["SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE"]
    allow_critical_dirty_manifest = ENVIRON["SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_MANIFEST"]
    load_allow_manifest(allow_critical_dirty_manifest)
  }
  NR > 1 {
    if ($9 != "") {
      prefix = "unallowed"
      if (path_allowed($1)) {
        prefix = "allowed"
      }
      print "- [" prefix "] " $1 " | " $2 " | " $9
    }
  }
' "$OUT"

if [[ "$CHECK_MODE" == "1" ]]; then
  max_prunable="${SOUNIO_AUDIT_MAX_PRUNABLE:-0}"
  max_critical_dirty="${SOUNIO_AUDIT_MAX_CRITICAL_DIRTY:-0}"
  max_dirty="${SOUNIO_AUDIT_MAX_DIRTY:-}"
  max_open_pr="${SOUNIO_AUDIT_MAX_OPEN_PR:-}"

  failed=0
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
