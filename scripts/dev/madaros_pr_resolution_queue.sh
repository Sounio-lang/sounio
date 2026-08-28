#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

REPO="${SOUNIO_GITHUB_REPO:-Sounio-lang/sounio}"
LIMIT="${SOUNIO_MADAROS_PR_QUEUE_LIMIT:-50}"
OUT=""
CHECK_MODE=0

usage() {
  cat <<'USAGE'
Usage: scripts/dev/madaros_pr_resolution_queue.sh [--check] [OUT.tsv]

Writes a TSV queue for open PR disposition during Madaros production-readiness
work. The queue is advisory by default; --check fails while any PR overlaps
the active Claude compiler lane.

Options:
  --check        fail if a PR overlaps Claude-owned compiler files
  -h, --help     show this help

Environment:
  SOUNIO_GITHUB_REPO                  default: Sounio-lang/sounio
  SOUNIO_MADAROS_PR_QUEUE_LIMIT       default: 50
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
  OUT="$(mktemp /tmp/sounio-madaros-pr-resolution.XXXXXX.tsv)"
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh is required for PR resolution queue" >&2
  exit 2
fi

owned_file_re='^(self-hosted/compiler/main\.sio|self-hosted/native/codegen\.sio|self-hosted/native/codegen_x86_linux\.sio|self-hosted/native/lower_ir\.sio|self-hosted/native/suite\.sio)$'
compiler_surface_re='^(self-hosted/compiler/|self-hosted/native/|bin/(souc|madaros)(-|$)|scripts/(ci|dev|lib|ops)/)'
docs_governance_re='^(docs/governance/|docs/audit/|AGENTS\.md|CLAUDE\.md|CLAUDE_HANDOFF\.md|ONBOARDING\.md|\.claude/)'

tmp_dir="$(mktemp -d /tmp/sounio-madaros-pr-resolution.XXXXXX)"
trap 'rm -rf "$tmp_dir"' EXIT

printf 'pr\ttitle\tbase\thead\tdraft\tmergeable\tcategory\toverlap_files\taction\turl\n' >"$OUT"

pr_list="$(
  gh pr list --repo "$REPO" --state open --limit "$LIMIT" \
    --json number --jq '.[].number' 2>/dev/null || true
)"

if [[ -n "$pr_list" ]]; then
  while IFS= read -r pr; do
    [[ -n "$pr" ]] || continue

    meta_path="$tmp_dir/pr-$pr.json"
    if ! gh pr view "$pr" --repo "$REPO" \
      --json number,title,baseRefName,headRefName,isDraft,mergeable,url,files \
      >"$meta_path" 2>/dev/null; then
      printf '#%s\tunavailable\tunknown\tunknown\tunknown\tunknown\tmetadata_unavailable\t\tinspect gh pr view failure\t\n' "$pr" >>"$OUT"
      continue
    fi

    python3 - "$meta_path" "$owned_file_re" "$compiler_surface_re" "$docs_governance_re" >>"$OUT" <<'PY'
import json
import re
import sys

path, owned_re, compiler_re, docs_re = sys.argv[1:5]
owned = re.compile(owned_re)
compiler = re.compile(compiler_re)
docs = re.compile(docs_re)

with open(path, "r", encoding="utf-8") as fh:
    pr = json.load(fh)

files = [item.get("path", "") for item in pr.get("files", [])]
owned_hits = [name for name in files if owned.search(name)]
compiler_hits = [name for name in files if compiler.search(name)]
docs_hits = [name for name in files if docs.search(name)]

base = pr.get("baseRefName") or "unknown"
mergeable = pr.get("mergeable") or "UNKNOWN"
is_draft = pr.get("isDraft")

if owned_hits:
    category = "compiler_owner_overlap"
    overlap = ",".join(owned_hits)
    action = "resolve ownership or close/rebuild before Madaros production-ready"
elif base != "main":
    category = "non_main_base"
    overlap = ",".join(compiler_hits[:8])
    action = "exclude from main production-readiness queue unless explicitly promoted"
elif compiler_hits:
    category = "compiler_adjacent_review"
    overlap = ",".join(compiler_hits[:8])
    action = "require compiler-owner review and focused gates before merge"
elif mergeable == "CONFLICTING":
    category = "stale_conflicting"
    overlap = ""
    action = "rebase/reconstruct from current origin/main or close stale branch"
elif is_draft:
    category = "draft_not_merge_candidate"
    overlap = ""
    action = "keep out of production-readiness evidence until marked ready"
elif docs_hits:
    category = "docs_governance_conflict"
    overlap = ",".join(docs_hits[:8])
    action = "refresh docs registry/governance metadata before merge"
else:
    category = "ordinary_open_pr"
    overlap = ""
    action = "normal PR review; no current Madaros ownership blocker detected"

fields = [
    f"#{pr.get('number')}",
    pr.get("title") or "",
    base,
    pr.get("headRefName") or "",
    str(bool(is_draft)).lower(),
    mergeable,
    category,
    overlap,
    action,
    pr.get("url") or "",
]

def cell(value: str) -> str:
    return str(value).replace("\t", " ").replace("\n", " ")

print("\t".join(cell(field) for field in fields))
PY
  done <<<"$pr_list"
fi

echo "pr_resolution_tsv=$OUT"
awk -F '\t' '
  NR > 1 {
    total++
    category[$7]++
    if ($3 == "main" && $5 != "true" && $6 == "CONFLICTING") {
      ready_conflicting_main_pr++
    }
    if ($7 == "compiler_owner_overlap") {
      if ($5 == "true") {
        compiler_owner_overlap_draft++
      } else {
        compiler_owner_overlap_ready++
      }
    }
  }
  END {
    printf "summary total=%d", total
    for (name in category) {
      printf " %s=%d", name, category[name]
    }
    printf " compiler_owner_overlap_draft=%d compiler_owner_overlap_ready=%d",
      compiler_owner_overlap_draft, compiler_owner_overlap_ready
    printf " ready_conflicting_main_pr=%d", ready_conflicting_main_pr
    printf "\n"
  }
' "$OUT"

echo "resolution_queue:"
awk -F '\t' '
  NR > 1 {
    printf "- %s category=%s draft=%s mergeable=%s action=%s url=%s\n", $1, $7, $5, $6, $9, $10
    if ($8 != "") {
      printf "  overlap=%s\n", $8
    }
  }
' "$OUT"

if [[ "$CHECK_MODE" == "1" ]]; then
  if awk -F '\t' 'NR > 1 && $3 == "main" && $5 != "true" && $6 == "CONFLICTING" { found = 1 } END { exit found ? 0 : 1 }' "$OUT"; then
    echo "madaros PR resolution queue failed: ready conflicting main PR remains" >&2
    exit 1
  fi
  if awk -F '\t' 'NR > 1 && $7 == "compiler_owner_overlap" { found = 1 } END { exit found ? 0 : 1 }' "$OUT"; then
    echo "madaros PR resolution queue failed: compiler ownership overlap remains" >&2
    exit 1
  fi
  echo "madaros PR resolution queue passed"
fi
