#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-}"
if [[ -n "$MODE" ]]; then
  shift
fi

OUT_DIR=""
PLAN_TSV=""
AUDIT_TSV=""
MANIFEST_TSV=""
ALLOW_MUTATING_OUTPUT=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/dev/madaros_worktree_cleanup_approval.sh template [options]
  scripts/dev/madaros_worktree_cleanup_approval.sh validate --manifest-tsv FILE
  scripts/dev/madaros_worktree_cleanup_approval.sh render --manifest-tsv FILE [options]

Create and validate a machine-readable approval manifest for Madaros worktree
cleanup. This script is non-destructive: it never runs git push, git reset,
git clean, git branch -D, or git worktree remove. The render mode writes a
commands file for operator review; mutating commands stay commented unless both
--allow-mutating-output and SOUNIO_MADAROS_CLEANUP_APPROVAL=I_ACCEPT_PUSH_BEFORE_DELETE
are set.

Modes:
  template           write madaros-cleanup-approval.tsv from the cleanup plan
  validate           validate an edited approval manifest
  render             validate and render post-approval commands

Options:
  --out-dir DIR       write outputs under DIR (default: mktemp /tmp dir)
  --plan-tsv PATH     use an existing madaros-cleanup-plan.tsv
  --audit-tsv PATH    pass an existing worktree audit TSV through the cleanup
                      planner before building the approval template
  --manifest-tsv PATH approval TSV for validate/render
  --allow-mutating-output
                      render uncommented mutating commands only with the
                      SOUNIO_MADAROS_CLEANUP_APPROVAL confirmation token
  -h, --help          show this help

Manifest decisions:
  hold                    default; no cleanup action
  approve_salvage_remove  archive/salvage first, then remove
  approve_discard_remove  save patch evidence, then discard/remove
  approve_remove_clean    remove only if the row is clean in the source plan
USAGE
}

if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

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
    --manifest-tsv)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "error: --manifest-tsv requires a path" >&2
        exit 2
      fi
      MANIFEST_TSV="$1"
      ;;
    --allow-mutating-output)
      ALLOW_MUTATING_OUTPUT=1
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

case "$MODE" in
  template|validate|render) ;;
  *)
    echo "error: unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(mktemp -d /tmp/madaros-cleanup-approval.XXXXXX)"
fi
mkdir -p "$OUT_DIR"

approval_header() {
  printf 'decision\tapprover\tapproved_utc\tapproval_id\tcategory\tpath\tbranch\thead\tstate\tdirty_count\tremote_ref\tsalvage_ref\tdisposition\tcritical_dirty\tcritical_vs_base\toperator_note\n'
}

expected_approval_header="$(approval_header)"

ensure_plan() {
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
    PLAN_TSV="$plan_target"
    return 0
  fi

  cleanup_args=(--out-dir "$OUT_DIR")
  if [[ -n "$AUDIT_TSV" ]]; then
    [[ -f "$AUDIT_TSV" ]] || {
      echo "error: audit TSV not found: $AUDIT_TSV" >&2
      exit 2
    }
    cleanup_args+=(--audit-tsv "$AUDIT_TSV")
  fi
  scripts/dev/madaros_worktree_cleanup_plan.sh "${cleanup_args[@]}" \
    > "$OUT_DIR/cleanup-planner.stdout"
  PLAN_TSV="$OUT_DIR/madaros-cleanup-plan.tsv"
}

slug_for_path() {
  basename "$1" | sed 's/[^A-Za-z0-9._-]\+/-/g; s/^-//; s/-$//'
}

write_template() {
  ensure_plan
  manifest="$OUT_DIR/madaros-cleanup-approval.tsv"
  {
    approval_header
    awk -F '\t' 'BEGIN { OFS = "\t" }
      NR > 1 {
        decision = "hold"
        approver = "TODO"
        approved_utc = "TODO"
        approval_id = "TODO"
        category = $1
        path = $2
        branch = $3
        head = $4
        state = $6
        dirty_count = $7
        remote_ref = $10
        salvage_ref = $19
        critical_dirty = $20
        critical_vs_base = $21
        disposition = $22
        operator_note = "TODO"
        print decision, approver, approved_utc, approval_id, category, path,
          branch, head, state, dirty_count, remote_ref, salvage_ref, disposition,
          critical_dirty, critical_vs_base, operator_note
      }
    ' "$PLAN_TSV"
  } > "$manifest"
  echo "approval_manifest=$manifest"
  echo "plan_tsv=$PLAN_TSV"
  echo "edit_decisions=hold|approve_salvage_remove|approve_discard_remove|approve_remove_clean"
}

validate_manifest() {
  [[ -n "$MANIFEST_TSV" ]] || {
    echo "error: --manifest-tsv is required" >&2
    exit 2
  }
  [[ -f "$MANIFEST_TSV" ]] || {
    echo "error: approval manifest not found: $MANIFEST_TSV" >&2
    exit 2
  }

  local actual_header
  IFS= read -r actual_header < "$MANIFEST_TSV"
  [[ "$actual_header" == "$expected_approval_header" ]] || {
    echo "error: approval manifest header drifted: $actual_header" >&2
    exit 1
  }

  awk -F '\t' '
    NR == 1 { next }
    NF != 16 {
      printf "error: row %d has %d fields, expected 16\n", NR, NF > "/dev/stderr"
      failed = 1
      next
    }
    $1 != "hold" &&
    $1 != "approve_salvage_remove" &&
    $1 != "approve_discard_remove" &&
    $1 != "approve_remove_clean" {
      printf "error: row %d has invalid decision: %s\n", NR, $1 > "/dev/stderr"
      failed = 1
    }
    $6 == "" {
      printf "error: row %d missing path\n", NR > "/dev/stderr"
      failed = 1
    }
    $1 != "hold" {
      actionable++
      if ($2 == "" || $2 == "TODO") {
        printf "error: row %d approved decision lacks approver\n", NR > "/dev/stderr"
        failed = 1
      }
      if ($3 !~ /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$/) {
        printf "error: row %d approved decision needs approved_utc=YYYY-MM-DDTHH:MM:SSZ\n", NR > "/dev/stderr"
        failed = 1
      }
      if ($4 !~ /^APPROVAL-[A-Za-z0-9._:-]+$/) {
        printf "error: row %d approved decision needs approval_id=APPROVAL-...\n", NR > "/dev/stderr"
        failed = 1
      }
      if ($1 == "approve_remove_clean" && ($9 != "clean" || $10 != "0")) {
        printf "error: row %d approve_remove_clean requires state=clean dirty_count=0\n", NR > "/dev/stderr"
        failed = 1
      }
      if ($1 == "approve_salvage_remove" && $12 !~ /^archive\/madaros-/) {
        printf "error: row %d approve_salvage_remove needs archive/madaros-* salvage_ref\n", NR > "/dev/stderr"
        failed = 1
      }
    }
    END {
      if (failed) {
        exit 1
      }
      rows = NR > 0 ? NR - 1 : 0
      printf "approval_rows=%d\n", rows
      printf "actionable_rows=%d\n", actionable + 0
    }
  ' "$MANIFEST_TSV"
}

render_line() {
  local line="$1"
  local mutating="${2:-0}"
  if [[ "$mutating" == "1" && "$ALLOW_MUTATING_OUTPUT" != "1" ]]; then
    printf '# %s\n' "$line"
  else
    printf '%s\n' "$line"
  fi
}

render_commands() {
  validate_manifest >/dev/null
  if [[ "$ALLOW_MUTATING_OUTPUT" == "1" && "${SOUNIO_MADAROS_CLEANUP_APPROVAL:-}" != "I_ACCEPT_PUSH_BEFORE_DELETE" ]]; then
    echo "error: --allow-mutating-output requires SOUNIO_MADAROS_CLEANUP_APPROVAL=I_ACCEPT_PUSH_BEFORE_DELETE" >&2
    exit 2
  fi

  commands="$OUT_DIR/madaros-cleanup-approved.commands.sh"
  {
    cat <<'HEADER'
#!/usr/bin/env bash
set -euo pipefail

# Generated by scripts/dev/madaros_worktree_cleanup_approval.sh render.
# Review before running. Mutating commands are commented unless the renderer was
# explicitly invoked with --allow-mutating-output and the approval env token.

HEADER
    awk -F '\t' 'NR > 1 { print }' "$MANIFEST_TSV" | while IFS=$'\t' read -r decision approver approved_utc approval_id category path branch head state dirty_count remote_ref salvage_ref disposition critical_dirty critical_vs_base operator_note; do
      path_q="$(printf '%q' "$path")"
      branch_q="$(printf '%q' "$branch")"
      salvage_q="$(printf '%q' "$salvage_ref")"
      slug="$(slug_for_path "$path")"
      patch_q="$(printf '%q' "/tmp/${slug}.approved-cleanup.patch")"

      echo "# path=$path"
      echo "# decision=$decision approver=$approver approved_utc=$approved_utc approval_id=$approval_id"
      echo "# category=$category state=$state dirty_count=$dirty_count remote_ref=$remote_ref"
      echo "# disposition=$disposition"
      if [[ -n "$operator_note" ]]; then
        echo "# operator_note=$operator_note"
      fi
      render_line "git -C $path_q status --short --branch"
      render_line "git -C $path_q diff --stat"

      case "$decision" in
        hold)
          echo "# no cleanup action approved for this row"
          ;;
        approve_salvage_remove)
          render_line "git -C $path_q diff --binary > $patch_q"
          if [[ "$branch" == "detached" ]]; then
            render_line "git -C $path_q switch -c $salvage_q" 1
            render_line "git push origin $salvage_q" 1
          else
            render_line "git push origin HEAD:refs/heads/$salvage_q" 1
          fi
          render_line "git worktree remove $path_q" 1
          if [[ "$branch" != "detached" ]]; then
            render_line "git branch -D $branch_q" 1
          fi
          ;;
        approve_discard_remove)
          render_line "git -C $path_q diff --binary > $patch_q"
          render_line "git worktree remove --force $path_q" 1
          if [[ "$branch" != "detached" ]]; then
            render_line "git branch -D $branch_q" 1
          fi
          ;;
        approve_remove_clean)
          render_line "git worktree remove $path_q" 1
          if [[ "$branch" != "detached" ]]; then
            render_line "git branch -d $branch_q" 1
          fi
          ;;
      esac
      echo
    done
  } > "$commands"
  chmod +x "$commands"
  echo "approval_commands=$commands"
  if [[ "$ALLOW_MUTATING_OUTPUT" == "1" ]]; then
    echo "mutating_output=enabled"
  else
    echo "mutating_output=commented"
  fi
}

case "$MODE" in
  template)
    write_template
    ;;
  validate)
    validate_manifest
    ;;
  render)
    render_commands
    ;;
esac
