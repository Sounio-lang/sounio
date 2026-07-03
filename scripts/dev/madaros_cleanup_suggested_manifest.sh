#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST_TSV=""
DECISIONS_TSV=""
OUT_TSV=""

usage() {
  cat <<'USAGE'
Usage:
  scripts/dev/madaros_cleanup_suggested_manifest.sh \
    --manifest-tsv FILE \
    --decisions-tsv FILE \
    --out-tsv FILE

Create an operator-review manifest from a cleanup approval template plus a
reviewed recommendations TSV. This helper is non-destructive and does not
approve anything: actionable rows are emitted with TODO_REQUIRED approval
fields, so validation intentionally fails until the operator fills them.

Inputs:
  --manifest-tsv FILE   madaros-cleanup-approval.tsv template or manifest
  --decisions-tsv FILE  TSV with columns:
                        path, suggested_action,
                        manifest_decision_if_operator_approves,
                        requires_approver_fields, note
  --out-tsv FILE        output suggested approval manifest
  -h, --help            show this help

Supported suggested_action values:
  approve_salvage_remove
  approve_discard_remove
  approve_remove_clean
  keep_active_allowlist  -> approve_keep_active_allowlist
  needs_owner_review     -> hold
USAGE
}

while (($#)); do
  case "$1" in
    --manifest-tsv)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "error: --manifest-tsv requires a path" >&2
        exit 2
      fi
      MANIFEST_TSV="$1"
      ;;
    --decisions-tsv)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "error: --decisions-tsv requires a path" >&2
        exit 2
      fi
      DECISIONS_TSV="$1"
      ;;
    --out-tsv)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "error: --out-tsv requires a path" >&2
        exit 2
      fi
      OUT_TSV="$1"
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

[[ -n "$MANIFEST_TSV" ]] || { echo "error: --manifest-tsv is required" >&2; exit 2; }
[[ -n "$DECISIONS_TSV" ]] || { echo "error: --decisions-tsv is required" >&2; exit 2; }
[[ -n "$OUT_TSV" ]] || { echo "error: --out-tsv is required" >&2; exit 2; }
[[ -f "$MANIFEST_TSV" ]] || { echo "error: manifest TSV not found: $MANIFEST_TSV" >&2; exit 2; }
[[ -f "$DECISIONS_TSV" ]] || { echo "error: decisions TSV not found: $DECISIONS_TSV" >&2; exit 2; }

expected_manifest_header='decision	approver	approved_utc	approval_id	category	path	branch	head	state	dirty_count	remote_ref	salvage_ref	disposition	critical_dirty	critical_vs_base	operator_note'
expected_decisions_header='path	suggested_action	manifest_decision_if_operator_approves	requires_approver_fields	note'

IFS= read -r actual_manifest_header < "$MANIFEST_TSV"
[[ "$actual_manifest_header" == "$expected_manifest_header" ]] || {
  echo "error: manifest header drifted: $actual_manifest_header" >&2
  exit 1
}

IFS= read -r actual_decisions_header < "$DECISIONS_TSV"
[[ "$actual_decisions_header" == "$expected_decisions_header" ]] || {
  echo "error: decisions header drifted: $actual_decisions_header" >&2
  exit 1
}

mkdir -p "$(dirname "$OUT_TSV")"

awk -F '\t' -v OFS='\t' -v decisions="$DECISIONS_TSV" '
  function strip_cr(value) {
    sub(/\r$/, "", value)
    return value
  }
  function decision_for(action, manifest_decision) {
    if (action == "approve_salvage_remove" ||
        action == "approve_discard_remove" ||
        action == "approve_remove_clean") {
      return action
    }
    if (action == "keep_active_allowlist" ||
        manifest_decision == "approve_keep_active_allowlist") {
      return "approve_keep_active_allowlist"
    }
    if (action == "needs_owner_review" || action == "hold" ||
        manifest_decision == "hold") {
      return "hold"
    }
    printf "error: unsupported suggested_action for %s: %s\n", current_path, action > "/dev/stderr"
    failed = 1
    return "hold"
  }
  BEGIN {
    while ((getline line < decisions) > 0) {
      line = strip_cr(line)
      if (line == "" || line ~ /^#/) {
        continue
      }
      if (!header_seen) {
        header_seen = 1
        continue
      }
      n = split(line, fields, "\t")
      if (n != 5) {
        printf "error: decisions row has %d fields, expected 5: %s\n", n, line > "/dev/stderr"
        failed = 1
        continue
      }
      current_path = fields[1]
      suggested_action[current_path] = fields[2]
      manifest_decision[current_path] = fields[3]
      note[current_path] = fields[5]
    }
    close(decisions)
    if (failed) {
      exit 1
    }
  }
  NR == 1 {
    print
    next
  }
  NF != 16 {
    printf "error: manifest row %d has %d fields, expected 16\n", NR, NF > "/dev/stderr"
    failed = 1
    next
  }
  {
    path = $6
    if (path in suggested_action) {
      current_path = path
      next_decision = decision_for(suggested_action[path], manifest_decision[path])
      if (next_decision == "hold") {
        $1 = "hold"
        $2 = "TODO"
        $3 = "TODO"
        $4 = "TODO"
      } else {
        $1 = next_decision
        $2 = "TODO_REQUIRED"
        $3 = "TODO_REQUIRED"
        $4 = "APPROVAL-TODO_REQUIRED"
      }
      if (note[path] != "") {
        $16 = note[path]
      }
    }
    print
  }
  END {
    if (failed) {
      exit 1
    }
  }
' "$MANIFEST_TSV" > "$OUT_TSV"

approval_rows="$(awk 'NR > 1 { rows++ } END { print rows + 0 }' "$OUT_TSV")"
actionable_rows="$(awk -F '\t' 'NR > 1 && $1 != "hold" { rows++ } END { print rows + 0 }' "$OUT_TSV")"
keep_active_rows="$(awk -F '\t' 'NR > 1 && $1 == "approve_keep_active_allowlist" { rows++ } END { print rows + 0 }' "$OUT_TSV")"

echo "suggested_manifest=$OUT_TSV"
echo "approval_rows=$approval_rows"
echo "actionable_rows=$actionable_rows"
echo "approve_keep_active_allowlist_rows=$keep_active_rows"
