#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_REF="${SOUNIO_MADAROS_CLEANUP_BASE_REF:-origin/canon/madaros-greenline}"
DEFAULT_ALLOW_RE='^(/workspace/sounio|/workspace/sounio-effects|/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b)$'
ALLOW_RE="${SOUNIO_MADAROS_CLEANUP_ALLOW_RE:-$DEFAULT_ALLOW_RE}"
OUT_DIR=""
AUDIT_TSV=""
DECISIONS_TSV=""
LATEST_POINTER="${SOUNIO_MADAROS_CLEANUP_LATEST_POINTER:-/tmp/madaros-latest-decision-packet.path}"
MAKE_TARBALL=0
UPDATE_LATEST=1

usage() {
  cat <<'USAGE'
Usage: scripts/dev/madaros_cleanup_decision_packet.sh [options]

Create a fresh non-destructive Madaros cleanup decision packet. The packet
bundles the live worktree audit, cleanup plan, approval template, suggested
operator-review manifest, salvage evidence packet, and README/draft handoff.
It never fills operator approval fields, pushes branches, removes worktrees,
deletes files, resets, or cleans.

Options:
  --out-dir DIR       write packet under DIR (default: /tmp timestamped dir)
  --audit-tsv PATH    use an existing worktree_branch_audit TSV instead of
                      generating a fresh one
  --base-ref REF      base ref for fresh worktree audit
                      (default: origin/canon/madaros-greenline)
  --allow-re REGEX    allowed critical-dirty path regex for audit/planner
  --decisions-tsv PATH
                      reviewed recommendation TSV for suggested manifest;
                      when omitted, a conservative seed file is generated
  --latest-pointer PATH
                      write PATH with the packet dir (default:
                      /tmp/madaros-latest-decision-packet.path)
  --no-latest-pointer do not update the latest-pointer file
  --tarball           include salvage-packet.tar.gz and sha256
  --no-tar            do not include a salvage tarball (default)
  -h, --help          show this help

Environment:
  SOUNIO_MADAROS_CLEANUP_BASE_REF        default --base-ref
  SOUNIO_MADAROS_CLEANUP_ALLOW_RE        default --allow-re
  SOUNIO_MADAROS_CLEANUP_LATEST_POINTER  default --latest-pointer
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
    --base-ref)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "error: --base-ref requires a ref" >&2
        exit 2
      fi
      BASE_REF="$1"
      ;;
    --allow-re)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "error: --allow-re requires a regex" >&2
        exit 2
      fi
      ALLOW_RE="$1"
      ;;
    --decisions-tsv)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "error: --decisions-tsv requires a path" >&2
        exit 2
      fi
      DECISIONS_TSV="$1"
      ;;
    --latest-pointer)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "error: --latest-pointer requires a path" >&2
        exit 2
      fi
      LATEST_POINTER="$1"
      ;;
    --no-latest-pointer)
      UPDATE_LATEST=0
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

timestamp_utc="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="/tmp/madaros-cleanup-decision-packet-$timestamp_utc"
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

packet_path() {
  printf '%s/%s\n' "$OUT_DIR" "$1"
}

copy_input() {
  local src="$1" dest="$2"
  [[ -f "$src" ]] || {
    echo "error: file not found: $src" >&2
    exit 2
  }
  local src_abs dest_abs
  src_abs="$(cd "$(dirname "$src")" && pwd)/$(basename "$src")"
  dest_abs="$(cd "$(dirname "$dest")" && pwd)/$(basename "$dest")"
  if [[ "$src_abs" != "$dest_abs" ]]; then
    cp "$src" "$dest"
  fi
}

write_seed_decisions() {
  local plan_tsv="$1" decisions_tsv="$2"
  {
    printf 'path\tsuggested_action\tmanifest_decision_if_operator_approves\trequires_approver_fields\tnote\n'
    awk -F '\t' 'BEGIN { OFS = "\t" }
      NR == 1 { next }
      {
        action = "needs_owner_review"
        manifest_decision = "hold"
        required = "owner_review"
        note = "operator must classify this row before cleanup"
        if ($1 == "active_other_lane_wip") {
          action = "keep_active_allowlist"
          manifest_decision = "hold_or_allowlist"
          required = "operator_owner_ack"
          note = "active lane candidate; keep only with explicit operator allowlist approval"
        }
        print $2, action, manifest_decision, required, note
      }
    ' "$plan_tsv"
  } > "$decisions_tsv"
}

validate_expected_maybe_fails() {
  local manifest="$1" out="$2"
  set +e
  scripts/dev/madaros_worktree_cleanup_approval.sh validate \
    --manifest-tsv "$manifest" >"$out" 2>&1
  local rc=$?
  set -e
  printf '%s' "$rc"
}

audit_out="$(packet_path worktree-audit.tsv)"
audit_log="$(packet_path worktree-audit.log)"

if [[ -n "$AUDIT_TSV" ]]; then
  copy_input "$AUDIT_TSV" "$audit_out"
  audit_rc=0
  {
    echo "audit_source=$AUDIT_TSV"
    echo "audit_mode=copied"
  } > "$audit_log"
else
  set +e
  env \
    "SOUNIO_AUDIT_BASE_REF=$BASE_REF" \
    "SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE=$ALLOW_RE" \
    scripts/dev/worktree_branch_audit.sh --check "$audit_out" >"$audit_log" 2>&1
  audit_rc=$?
  set -e
fi

cleanup_plan_dir="$(packet_path cleanup-plan)"
approval_template_dir="$(packet_path approval-template)"
approval_render_dir="$(packet_path approval-rendered)"
salvage_dir="$(packet_path salvage-packet)"
mkdir -p "$cleanup_plan_dir" "$approval_template_dir" "$approval_render_dir" "$salvage_dir"

env "SOUNIO_MADAROS_CLEANUP_ALLOW_RE=$ALLOW_RE" \
  scripts/dev/madaros_worktree_cleanup_plan.sh \
  --audit-tsv "$audit_out" \
  --out-dir "$cleanup_plan_dir" \
  > "$(packet_path cleanup-plan.stdout)"

plan_tsv="$cleanup_plan_dir/madaros-cleanup-plan.tsv"

scripts/dev/madaros_worktree_cleanup_approval.sh template \
  --plan-tsv "$plan_tsv" \
  --out-dir "$approval_template_dir" \
  > "$(packet_path approval-template.stdout)"

approval_manifest="$approval_template_dir/madaros-cleanup-approval.tsv"
scripts/dev/madaros_worktree_cleanup_approval.sh validate \
  --manifest-tsv "$approval_manifest" \
  > "$(packet_path approval-validate.stdout)"

scripts/dev/madaros_worktree_cleanup_approval.sh render \
  --manifest-tsv "$approval_manifest" \
  --out-dir "$approval_render_dir" \
  > "$(packet_path approval-rendered.stdout)"

if [[ -n "$DECISIONS_TSV" ]]; then
  copy_input "$DECISIONS_TSV" "$(packet_path suggested-cleanup-decisions.tsv)"
else
  write_seed_decisions "$plan_tsv" "$(packet_path suggested-cleanup-decisions.tsv)"
fi

suggested_manifest="$(packet_path madaros-cleanup-approval.suggested-unapproved.tsv)"
scripts/dev/madaros_cleanup_suggested_manifest.sh \
  --manifest-tsv "$approval_manifest" \
  --decisions-tsv "$(packet_path suggested-cleanup-decisions.tsv)" \
  --out-tsv "$suggested_manifest" \
  > "$(packet_path suggested-manifest.stdout)"

suggested_validate_rc="$(
  validate_expected_maybe_fails \
    "$suggested_manifest" \
    "$(packet_path suggested-unapproved-validate.stdout)"
)"

salvage_args=(--plan-tsv "$plan_tsv" --out-dir "$salvage_dir")
if [[ "$MAKE_TARBALL" == "1" ]]; then
  salvage_args+=(--tarball)
else
  salvage_args+=(--no-tar)
fi
scripts/dev/madaros_worktree_salvage_packet.sh "${salvage_args[@]}" \
  > "$(packet_path salvage-packet.stdout)"

{
  cat <<EOF
# Madaros Cleanup Decision Packet

Generated: $timestamp_utc

This packet is non-destructive and not an approval. It bundles the current
audit, cleanup plan, approval template, suggested operator-review manifest,
and salvage evidence so an operator can decide what stays active and what can
be salvaged/removed later.

## Key Files

- README: \`README.txt\`
- audit TSV: \`worktree-audit.tsv\`
- cleanup plan: \`cleanup-plan/madaros-cleanup-plan.tsv\`
- cleanup commands draft: \`cleanup-plan/madaros-cleanup-plan.commands.sh\`
- approval template: \`approval-template/madaros-cleanup-approval.tsv\`
- suggested decisions: \`suggested-cleanup-decisions.tsv\`
- suggested unapproved manifest: \`madaros-cleanup-approval.suggested-unapproved.tsv\`
- rendered hold-only commands: \`approval-rendered/madaros-cleanup-approved.commands.sh\`
- salvage evidence: \`salvage-packet/\`

## Operator Boundary

Any non-\`hold\` decision must be edited by the operator to fill:

- \`approver\`
- \`approved_utc\`
- \`approval_id\`
- \`operator_note\`

The suggested manifest intentionally uses \`TODO_REQUIRED\` for those fields.
That means validation may fail until the operator fills the fields. Agents must
not fill them.

## Recheck After Approval

\`\`\`bash
scripts/dev/madaros_worktree_cleanup_approval.sh validate \\
  --manifest-tsv $approval_manifest

SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_MANIFEST=$approval_manifest \\
  scripts/dev/madaros_readiness_status.sh --strict
\`\`\`

## Render Commands After Approval

\`\`\`bash
scripts/dev/madaros_worktree_cleanup_approval.sh render \\
  --manifest-tsv $approval_manifest \\
  --out-dir /tmp/madaros-cleanup-approved
\`\`\`

Rendered mutating commands remain commented unless the explicit
\`SOUNIO_MADAROS_CLEANUP_APPROVAL=I_ACCEPT_PUSH_BEFORE_DELETE\` token is set
with \`--allow-mutating-output\`.
EOF
} > "$(packet_path operator-approval-draft.md)"

summary_line="$(grep -E '^total=' "$audit_log" | tail -n 1 || true)"
planner_counts="$(grep -E '^(planned_worktrees=|category\[)' "$(packet_path cleanup-plan.stdout)" || true)"
approval_validate="$(cat "$(packet_path approval-validate.stdout)")"
suggested_counts="$(cat "$(packet_path suggested-manifest.stdout)")"
greenline_tip="$(git rev-parse --verify "$BASE_REF" 2>/dev/null || git rev-parse HEAD 2>/dev/null || true)"

{
  printf 'packet_root=%s\n' "$OUT_DIR"
  printf 'generated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'base_ref=%s\n' "$BASE_REF"
  printf 'greenline_tip=%s\n' "$greenline_tip"
  printf 'allow_re=%s\n' "$ALLOW_RE"
  printf 'audit_rc=%s\n' "$audit_rc"
  [[ -z "$summary_line" ]] || printf '%s\n' "$summary_line"
  grep -E '^gate_violation:' "$audit_log" || true
  printf '\nplanner_counts:\n'
  [[ -z "$planner_counts" ]] || printf '%s\n' "$planner_counts"
  printf '\napproval_validate:\n%s\n' "$approval_validate"
  printf '\nsuggested_manifest_validate_rc=%s\n' "$suggested_validate_rc"
  printf 'suggested_unapproved_manifest_validate=%s\n' \
    "$(if [[ "$suggested_validate_rc" == "0" ]]; then echo pass_no_actionable_or_already_approved; else echo expected_fail_until_operator_fills_approver_fields; fi)"
  printf '\nsuggested_counts:\n%s\n' "$suggested_counts"
  if [[ -f "$salvage_dir.tar.gz.sha256" ]]; then
    printf '\nsalvage_tarball_sha256:\n'
    cat "$salvage_dir.tar.gz.sha256"
  fi
  printf '\noperator_approval_draft=%s\n' "$(packet_path operator-approval-draft.md)"
  printf 'suggested_cleanup_decisions=%s\n' "$(packet_path suggested-cleanup-decisions.tsv)"
  printf 'suggested_unapproved_manifest=%s\n' "$suggested_manifest"
} > "$(packet_path README.txt)"

if [[ "$UPDATE_LATEST" == "1" ]]; then
  mkdir -p "$(dirname "$LATEST_POINTER")"
  printf '%s\n' "$OUT_DIR" > "$LATEST_POINTER"
fi

echo "packet_root=$OUT_DIR"
echo "readme=$OUT_DIR/README.txt"
echo "audit_rc=$audit_rc"
[[ -z "$summary_line" ]] || echo "$summary_line"
echo "approval_manifest=$approval_manifest"
echo "suggested_unapproved_manifest=$suggested_manifest"
echo "suggested_manifest_validate_rc=$suggested_validate_rc"
if [[ "$UPDATE_LATEST" == "1" ]]; then
  echo "latest_pointer=$LATEST_POINTER"
fi
