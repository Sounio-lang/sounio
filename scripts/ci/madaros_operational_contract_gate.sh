#!/usr/bin/env bash
# Guard the committed operational contract that tells agents how to reason about
# Madaros status. This is intentionally cheap: the heavy compiler proof remains
# `make madaros-full-gate`.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HISTORICAL_MAIN_GREEN_BASE="17d1157be540d32bb583dd03ca7072a6026e2027"
CANONICAL_GREENLINE_PROOF_BASE="bf46bda919596ce71b8fc35dc29cb3a31ff01d7b"

fail() {
  echo "[madaros-contract] FAIL: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || fail "missing file: $1"
}

require_executable() {
  [[ -x "$1" ]] || fail "missing executable: $1"
}

require_grep() {
  local pattern="$1"
  local file="$2"
  grep -Fq -- "$pattern" "$file" || fail "missing marker in $file: $pattern"
}

require_cleanup_planner_evidence() {
  local tmp_dir audit_tsv out_dir plan_tsv plan_sh real_out real_plan real_sh head actual_header expected_header row_count evidence_count
  tmp_dir="$(mktemp -d /tmp/madaros-contract-cleanup.XXXXXX)"
  audit_tsv="$tmp_dir/worktree-audit.tsv"
  out_dir="$tmp_dir/out"
  head="$(git rev-parse --short=12 HEAD 2>/dev/null || printf 'unknown')"

  printf 'path\tbranch\thead\tupstream\tstate\tdirty_count\tahead\tbehind\tcritical_dirty\tcritical_vs_base\tpr\n' > "$audit_tsv"
  printf '%s\tdetached\t%s\t\tdirty\t1\t0\t0\tM scripts/dev/madaros_worktree_cleanup_plan.sh\tscripts/dev/madaros_worktree_cleanup_plan.sh\t\n' \
    "$ROOT_DIR" "$head" >> "$audit_tsv"

  SOUNIO_MADAROS_CLEANUP_ALLOW_RE='^$' \
    scripts/dev/madaros_worktree_cleanup_plan.sh --audit-tsv "$audit_tsv" --out-dir "$out_dir" >/dev/null

  plan_tsv="$out_dir/madaros-cleanup-plan.tsv"
  plan_sh="$out_dir/madaros-cleanup-plan.commands.sh"
  require_file "$plan_tsv"
  require_file "$plan_sh"

  expected_header='category	path	branch	head	upstream	state	dirty_count	ahead	behind	remote_ref	prs	unique_commits_origin_main	unique_commits_upstream	tracked_dirty_files	untracked_dirty_files	tracked_diff_files	tracked_diff_added	tracked_diff_deleted	salvage_ref	critical_dirty	critical_vs_base	disposition'
  IFS= read -r actual_header < "$plan_tsv"
  [[ "$actual_header" == "$expected_header" ]] ||
    fail "cleanup planner header drifted: $actual_header"

  awk -F '\t' '
    NR == 2 {
      if (NF != 22) {
        exit 1
      }
      if ($12 == "" || $14 == "" || $15 == "" || $16 == "" || $17 == "" || $18 == "") {
        exit 1
      }
      if ($19 !~ /^archive\/madaros-/ || $21 == "") {
        exit 1
      }
      found = 1
    }
    END {
      exit found ? 0 : 1
    }
  ' "$plan_tsv" || fail "cleanup planner evidence row missing or malformed"

  grep -Eq '^# evidence=origin_main_unique:[^ ]+ upstream_unique:[^ ]+ tracked_dirty:[0-9]+ untracked_dirty:[0-9]+ tracked_diff:[0-9]+ files \+[0-9]+ -[0-9]+ critical_vs_base:' "$plan_sh" ||
    fail "cleanup planner command evidence comment missing"

  require_no_uncommented_mutation "$plan_sh"

  # Live audit validates row shape and non-destructive output; cleanup counts may drift.
  real_out="$tmp_dir/real"
  scripts/dev/madaros_worktree_cleanup_plan.sh --out-dir "$real_out" >/dev/null
  real_plan="$real_out/madaros-cleanup-plan.tsv"
  real_sh="$real_out/madaros-cleanup-plan.commands.sh"
  require_file "$real_plan"
  require_file "$real_sh"
  IFS= read -r actual_header < "$real_plan"
  [[ "$actual_header" == "$expected_header" ]] ||
    fail "real cleanup planner header drifted: $actual_header"
  awk -F '\t' 'NR > 1 && NF != 22 { exit 1 }' "$real_plan" ||
    fail "real cleanup planner emitted a malformed row"
  row_count="$(awk 'NR > 1 { rows++ } END { print rows + 0 }' "$real_plan")"
  evidence_count="$(grep -c '^# evidence=origin_main_unique:' "$real_sh" || true)"
  [[ "$row_count" == "$evidence_count" ]] ||
    fail "real cleanup planner evidence comments do not match rows: rows=$row_count evidence=$evidence_count"
  require_no_uncommented_mutation "$real_sh"
}

require_salvage_packet_evidence() {
  local tmp_dir audit_tsv out_dir packet_root metadata file_index actual_header expected_header
  tmp_dir="$(mktemp -d /tmp/madaros-contract-salvage.XXXXXX)"
  audit_tsv="$tmp_dir/worktree-audit.tsv"
  out_dir="$tmp_dir/packet"

  printf 'path\tbranch\thead\tupstream\tstate\tdirty_count\tahead\tbehind\tcritical_dirty\tcritical_vs_base\tpr\n' > "$audit_tsv"
  printf '%s\tdetached\t%s\t\tdirty\t1\t0\t0\tM scripts/dev/madaros_worktree_salvage_packet.sh\tscripts/dev/madaros_worktree_salvage_packet.sh\t\n' \
    "$ROOT_DIR" "$(git rev-parse --short=12 HEAD 2>/dev/null || printf unknown)" >> "$audit_tsv"

  SOUNIO_MADAROS_CLEANUP_ALLOW_RE='^$' \
    scripts/dev/madaros_worktree_salvage_packet.sh --audit-tsv "$audit_tsv" --out-dir "$out_dir" --no-tar >/dev/null

  require_file "$out_dir/README.txt"
  require_file "$out_dir/madaros-cleanup-plan.tsv"
  require_file "$out_dir/file-index.txt"
  require_file "$out_dir/tracked-diff-sizes.tsv"
  require_file "$out_dir/untracked-counts.tsv"
  require_file "$out_dir/logs/cleanup-planner.stdout"

  expected_header='category	path	branch	head	upstream	state	dirty_count	ahead	behind	remote_ref	prs	unique_commits_origin_main	unique_commits_upstream	tracked_dirty_files	untracked_dirty_files	tracked_diff_files	tracked_diff_added	tracked_diff_deleted	salvage_ref	critical_dirty	critical_vs_base	disposition'
  IFS= read -r actual_header < "$out_dir/madaros-cleanup-plan.tsv"
  [[ "$actual_header" == "$expected_header" ]] ||
    fail "salvage packet cleanup-plan header drifted: $actual_header"

  packet_root="$out_dir/per-worktree/$(basename "$ROOT_DIR")"
  metadata="$packet_root/metadata.txt"
  file_index="$out_dir/file-index.txt"
  require_file "$metadata"
  require_file "$packet_root/status.short.txt"
  require_file "$packet_root/tracked.diff"
  require_file "$packet_root/staged.diff"
  require_file "$packet_root/untracked.files.txt"
  require_file "$packet_root/untracked.sizes.tsv"
  require_grep 'suggested_salvage_ref=archive/madaros-' "$metadata"
  require_grep 'per-worktree/' "$file_index"
  [[ ! -e "$out_dir.tar.gz" ]] || fail "salvage packet wrote tarball despite --no-tar"
}

require_cleanup_approval_evidence() {
  local tmp_dir audit_tsv out_dir manifest approved render_out commands actual_header expected_header validate_out
  tmp_dir="$(mktemp -d /tmp/madaros-contract-approval.XXXXXX)"
  audit_tsv="$tmp_dir/worktree-audit.tsv"
  out_dir="$tmp_dir/template"

  printf 'path\tbranch\thead\tupstream\tstate\tdirty_count\tahead\tbehind\tcritical_dirty\tcritical_vs_base\tpr\n' > "$audit_tsv"
  printf '%s\tdetached\t%s\t\tdirty\t1\t0\t0\tM scripts/dev/madaros_worktree_cleanup_approval.sh\tscripts/dev/madaros_worktree_cleanup_approval.sh\t\n' \
    "$ROOT_DIR" "$(git rev-parse --short=12 HEAD 2>/dev/null || printf unknown)" >> "$audit_tsv"

  SOUNIO_MADAROS_CLEANUP_ALLOW_RE='^$' \
    scripts/dev/madaros_worktree_cleanup_approval.sh template --audit-tsv "$audit_tsv" --out-dir "$out_dir" >/dev/null

  manifest="$out_dir/madaros-cleanup-approval.tsv"
  require_file "$manifest"

  expected_header='decision	approver	approved_utc	approval_id	category	path	branch	head	state	dirty_count	remote_ref	salvage_ref	disposition	critical_dirty	critical_vs_base	operator_note'
  IFS= read -r actual_header < "$manifest"
  [[ "$actual_header" == "$expected_header" ]] ||
    fail "cleanup approval template header drifted: $actual_header"

  validate_out="$(scripts/dev/madaros_worktree_cleanup_approval.sh validate --manifest-tsv "$manifest")"
  grep -Fq 'actionable_rows=0' <<<"$validate_out" ||
    fail "cleanup approval default template should have zero actionable rows"

  approved="$tmp_dir/approved.tsv"
  awk -F '\t' 'BEGIN { OFS = "\t" }
    NR == 1 { print; next }
    NR == 2 {
      $1 = "approve_salvage_remove"
      $2 = "operator"
      $3 = "2026-07-03T00:00:00Z"
      $4 = "APPROVAL-fixture"
      $16 = "fixture approval row"
      print
      $1 = "approve_keep_active_allowlist"
      $4 = "APPROVAL-fixture-keep-active"
      $16 = "fixture keep-active allowlist row"
      print
      next
    }
    { print }
  ' "$manifest" > "$approved"

  validate_out="$(scripts/dev/madaros_worktree_cleanup_approval.sh validate --manifest-tsv "$approved")"
  grep -Fq 'actionable_rows=2' <<<"$validate_out" ||
    fail "cleanup approval approved fixture should have two actionable rows"

  render_out="$tmp_dir/render"
  scripts/dev/madaros_worktree_cleanup_approval.sh render --manifest-tsv "$approved" --out-dir "$render_out" >/dev/null
  commands="$render_out/madaros-cleanup-approved.commands.sh"
  require_file "$commands"
  require_grep '# git push origin' "$commands"
  require_grep '# git worktree remove' "$commands"
  require_grep 'decision=approve_keep_active_allowlist' "$commands"
  require_grep 'keep-active allowlist approved; no cleanup action rendered' "$commands"
  require_no_uncommented_mutation "$commands"

  if scripts/dev/madaros_worktree_cleanup_approval.sh render --manifest-tsv "$approved" --out-dir "$tmp_dir/blocked" --allow-mutating-output >/dev/null 2>&1; then
    fail "cleanup approval render allowed mutating output without confirmation token"
  fi
}

require_cleanup_suggested_manifest_evidence() {
  local tmp_dir manifest decisions suggested approved validate_out render_out commands
  tmp_dir="$(mktemp -d /tmp/madaros-contract-suggested-manifest.XXXXXX)"
  manifest="$tmp_dir/approval-template.tsv"
  decisions="$tmp_dir/suggested-cleanup-decisions.tsv"
  suggested="$tmp_dir/suggested-unapproved.tsv"
  approved="$tmp_dir/approved.tsv"

  printf 'decision\tapprover\tapproved_utc\tapproval_id\tcategory\tpath\tbranch\thead\tstate\tdirty_count\tremote_ref\tsalvage_ref\tdisposition\tcritical_dirty\tcritical_vs_base\toperator_note\n' > "$manifest"
  printf 'hold\tTODO\tTODO\tTODO\tactive_other_lane_wip\t/tmp/active-lane\twork/active\t%s\tdirty\t1\tnone\tarchive/madaros-active-lane\tdo not remove; confirm owner\tM self-hosted/ir/lower.sio\tself-hosted/ir/lower.sio\tTODO\n' \
    "$(git rev-parse --short=12 HEAD 2>/dev/null || printf unknown)" >> "$manifest"
  printf 'hold\tTODO\tTODO\tTODO\tstale_local_temp\t/tmp/salvage-lane\twork/salvage\t%s\tdirty\t1\tnone\tarchive/madaros-salvage-lane\tcheck unique commits; push archive/salvage before removal\tM self-hosted/compiler/module_native_driver.sio\tself-hosted/compiler/module_native_driver.sio\tTODO\n' \
    "$(git rev-parse --short=12 HEAD 2>/dev/null || printf unknown)" >> "$manifest"

  printf 'path\tsuggested_action\tmanifest_decision_if_operator_approves\trequires_approver_fields\tnote\n' > "$decisions"
  printf '/tmp/active-lane\tkeep_active_allowlist\thold_or_allowlist\toperator_owner_ack\tfixture active lane stays alive\n' >> "$decisions"
  printf '/tmp/salvage-lane\tapprove_salvage_remove\tapprove_salvage_remove\tyes\tfixture salvage before remove\n' >> "$decisions"

  scripts/dev/madaros_cleanup_suggested_manifest.sh \
    --manifest-tsv "$manifest" \
    --decisions-tsv "$decisions" \
    --out-tsv "$suggested" >/dev/null
  require_file "$suggested"
  awk -F '\t' '
    NR > 1 && $6 == "/tmp/active-lane" {
      found_active = ($1 == "approve_keep_active_allowlist" &&
        $2 == "TODO_REQUIRED" && $3 == "TODO_REQUIRED" &&
        $4 == "APPROVAL-TODO_REQUIRED")
    }
    NR > 1 && $6 == "/tmp/salvage-lane" {
      found_salvage = ($1 == "approve_salvage_remove" &&
        $2 == "TODO_REQUIRED" && $3 == "TODO_REQUIRED" &&
        $4 == "APPROVAL-TODO_REQUIRED")
    }
    END {
      exit (found_active && found_salvage) ? 0 : 1
    }
  ' "$suggested" || fail "suggested manifest did not mark active/salvage rows correctly"

  if scripts/dev/madaros_worktree_cleanup_approval.sh validate --manifest-tsv "$suggested" >/dev/null 2>&1; then
    fail "suggested unapproved manifest validated before operator approval fields were filled"
  fi

  awk -F '\t' 'BEGIN { OFS = "\t" }
    NR == 1 { print; next }
    $1 != "hold" {
      $2 = "operator"
      $3 = "2026-07-03T00:00:00Z"
      $4 = "APPROVAL-fixture-" NR
    }
    { print }
  ' "$suggested" > "$approved"

  validate_out="$(scripts/dev/madaros_worktree_cleanup_approval.sh validate --manifest-tsv "$approved")"
  grep -Fq 'actionable_rows=2' <<<"$validate_out" ||
    fail "approved suggested manifest should have two actionable rows"

  render_out="$tmp_dir/render"
  scripts/dev/madaros_worktree_cleanup_approval.sh render --manifest-tsv "$approved" --out-dir "$render_out" >/dev/null
  commands="$render_out/madaros-cleanup-approved.commands.sh"
  require_file "$commands"
  require_grep 'decision=approve_keep_active_allowlist' "$commands"
  require_grep 'keep-active allowlist approved; no cleanup action rendered' "$commands"
  require_no_uncommented_mutation "$commands"
}

require_cleanup_decision_packet_evidence() {
  local tmp_dir audit_tsv decisions out_dir latest_pointer stdout_path suggested commands pointer_value
  tmp_dir="$(mktemp -d /tmp/madaros-contract-decision-packet.XXXXXX)"
  audit_tsv="$tmp_dir/worktree-audit.tsv"
  decisions="$tmp_dir/suggested-cleanup-decisions.tsv"
  out_dir="$tmp_dir/packet"
  latest_pointer="$tmp_dir/latest.path"
  stdout_path="$tmp_dir/stdout.txt"

  printf 'path\tbranch\thead\tupstream\tstate\tdirty_count\tahead\tbehind\tcritical_dirty\tcritical_vs_base\tpr\n' > "$audit_tsv"
  printf '%s\tdetached\t%s\t\tdirty\t1\t0\t0\tM scripts/dev/madaros_cleanup_decision_packet.sh\tscripts/dev/madaros_cleanup_decision_packet.sh\t\n' \
    "$ROOT_DIR" "$(git rev-parse --short=12 HEAD 2>/dev/null || printf unknown)" >> "$audit_tsv"

  printf 'path\tsuggested_action\tmanifest_decision_if_operator_approves\trequires_approver_fields\tnote\n' > "$decisions"
  printf '%s\tapprove_salvage_remove\tapprove_salvage_remove\tyes\tfixture suggested salvage remains unapproved\n' "$ROOT_DIR" >> "$decisions"

  SOUNIO_MADAROS_CLEANUP_ALLOW_RE='^$' \
    scripts/dev/madaros_cleanup_decision_packet.sh \
      --audit-tsv "$audit_tsv" \
      --decisions-tsv "$decisions" \
      --out-dir "$out_dir" \
      --latest-pointer "$latest_pointer" \
      --no-tar > "$stdout_path"

  require_file "$out_dir/README.txt"
  require_file "$out_dir/operator-approval-draft.md"
  require_file "$out_dir/worktree-audit.tsv"
  require_file "$out_dir/cleanup-plan/madaros-cleanup-plan.tsv"
  require_file "$out_dir/approval-template/madaros-cleanup-approval.tsv"
  require_file "$out_dir/suggested-cleanup-decisions.tsv"
  require_file "$out_dir/madaros-cleanup-approval.suggested-unapproved.tsv"
  require_file "$out_dir/salvage-packet/README.txt"
  require_file "$latest_pointer"

  pointer_value="$(cat "$latest_pointer")"
  [[ "$pointer_value" == "$out_dir" ]] ||
    fail "decision packet latest pointer drifted: $pointer_value"

  require_grep 'suggested_manifest_validate_rc=1' "$out_dir/README.txt"
  require_grep 'expected_fail_until_operator_fills_approver_fields' "$out_dir/README.txt"
  require_grep 'Agents must' "$out_dir/operator-approval-draft.md"
  require_grep 'packet_root=' "$stdout_path"
  require_grep 'latest_pointer=' "$stdout_path"

  suggested="$out_dir/madaros-cleanup-approval.suggested-unapproved.tsv"
  if scripts/dev/madaros_worktree_cleanup_approval.sh validate --manifest-tsv "$suggested" >/dev/null 2>&1; then
    fail "decision packet suggested manifest validated before operator approval fields were filled"
  fi

  commands="$out_dir/approval-rendered/madaros-cleanup-approved.commands.sh"
  require_file "$commands"
  require_no_uncommented_mutation "$commands"
  [[ ! -e "$out_dir/salvage-packet.tar.gz" ]] ||
    fail "decision packet wrote salvage tarball despite --no-tar"
}

require_no_uncommented_mutation() {
  local plan_sh="$1"
  awk '
    /^[[:space:]]*git[[:space:]]+push([[:space:]]|$)/ { exit 1 }
    /^[[:space:]]*git[[:space:]]+worktree[[:space:]]+remove([[:space:]]|$)/ { exit 1 }
    /^[[:space:]]*git[[:space:]]+branch[[:space:]]+-[dD]([[:space:]]|$)/ { exit 1 }
    /^[[:space:]]*git[[:space:]]+-C[[:space:]][^[:space:]]+[[:space:]]+(reset|clean)([[:space:]]|$)/ { exit 1 }
  ' "$plan_sh" || fail "cleanup planner emitted an uncommented mutating command: $plan_sh"
}

require_madaros_proof_anchor() {
  local -a accepted_anchors=(
    "$CANONICAL_GREENLINE_PROOF_BASE"
    "$HISTORICAL_MAIN_GREEN_BASE"
  )
  local anchor

  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    return 0
  fi

  for anchor in "${accepted_anchors[@]}"; do
    if git merge-base --is-ancestor "$anchor" HEAD; then
      return 0
    fi
  done

  fail "HEAD does not contain an accepted Madaros proof anchor: canonical greenline $CANONICAL_GREENLINE_PROOF_BASE or historical main $HISTORICAL_MAIN_GREEN_BASE"
}

require_file docs/MADAROS_STATUS.md
require_file docs/audit/MADAROS_WORKTREE_CLEANUP_LEDGER_2026-07-03.md
require_file docs/audit/MADAROS_GPU_KERNEL_IR_LOWER_TO_PTX_PTX_MODULE_COMBINATION_2026-07-02.md
require_file docs/status/madaros_main_proof_17d115.md
require_file AGENTS.md
require_file CLAUDE.md
require_file Makefile
require_file .gitignore
require_executable bin/souc
require_executable bin/madaros
require_file scripts/lib/resolve_madaros.sh
require_file scripts/ci/build_modular_madaros.sh
require_file scripts/ci/madaros_full_gate.sh
require_file scripts/ci/madaros_source_to_elf_gate.sh
require_executable scripts/ci/madaros_gpu_ptx_pairwise_import_witness.sh
require_executable scripts/dev/madaros_two_gate.sh
require_executable scripts/dev/madaros_worktree_cleanup_plan.sh
require_executable scripts/dev/madaros_worktree_cleanup_approval.sh
require_executable scripts/dev/madaros_cleanup_suggested_manifest.sh
require_executable scripts/dev/madaros_worktree_salvage_packet.sh
require_file .github/workflows/ci.yml

require_madaros_proof_anchor

require_grep 'bin/souc` routes to Madaros' docs/MADAROS_STATUS.md
require_grep 'bf46bda919596ce71b8fc35dc29cb3a31ff01d7b' docs/MADAROS_STATUS.md
require_grep '17d1157be' docs/MADAROS_STATUS.md
require_grep 'receipt-gated' docs/MADAROS_STATUS.md
require_grep 'bin/madaros-relocgate' docs/MADAROS_STATUS.md
require_grep 'artifacts/self-hosted/madaros.gate-receipt' docs/MADAROS_STATUS.md
require_grep 'imported-SMT solver gate (6/6)' docs/MADAROS_STATUS.md
require_grep 'bin/madaros-linux-x86_64' docs/MADAROS_STATUS.md
require_grep 'make madaros-full-gate' docs/MADAROS_STATUS.md
require_grep 'ungated' docs/MADAROS_STATUS.md
require_grep 'artifacts/self-hosted/madaros' docs/MADAROS_STATUS.md
require_grep 'binary is **not evidence**' docs/MADAROS_STATUS.md
require_grep 'scripts/ci/madaros_operational_contract_gate.sh' docs/MADAROS_STATUS.md
require_grep 'scripts/ci/madaros_gpu_ptx_pairwise_import_witness.sh' docs/MADAROS_STATUS.md
require_grep 'scripts/dev/madaros_worktree_cleanup_plan.sh' docs/audit/MADAROS_WORKTREE_CLEANUP_LEDGER_2026-07-03.md
require_grep 'PRE-EXISTING MODULE-COMBINATION BREAKAGE' docs/audit/MADAROS_GPU_KERNEL_IR_LOWER_TO_PTX_PTX_MODULE_COMBINATION_2026-07-02.md
require_grep 'gpu::kernel_ir' docs/audit/MADAROS_GPU_KERNEL_IR_LOWER_TO_PTX_PTX_MODULE_COMBINATION_2026-07-02.md

require_grep 'scripts/lib/resolve_madaros.sh' AGENTS.md
require_grep 'docs/MADAROS_STATUS.md' AGENTS.md
require_grep 'default compiler entrypoint and now routes to Madaros' CLAUDE.md

require_grep 'madaros-full-gate: build-madaros' Makefile
require_grep 'bash scripts/ci/madaros_full_gate.sh' Makefile
require_grep 'Madaros Stage1 full-functioning gate' scripts/dev/e2e_gate.sh
require_grep 'make -C "$ROOT_DIR" madaros-full-gate' scripts/dev/e2e_gate.sh
require_grep 'SOUNIO_SKIP_MADAROS' scripts/dev/e2e_gate.sh

require_grep 'RAW_MADAROS=' scripts/ci/madaros_full_gate.sh
require_grep '"$RAW_MADAROS" --check /tmp' scripts/ci/madaros_full_gate.sh
require_grep 'wrapper_tmp_dir.log' scripts/ci/madaros_full_gate.sh
require_grep 'error[E175' scripts/ci/madaros_full_gate.sh
require_grep 'error[E176' scripts/ci/madaros_full_gate.sh
require_grep 'error[E177' scripts/ci/madaros_full_gate.sh
require_grep '--native-v2-emit-sret' scripts/ci/madaros_full_gate.sh
require_grep 'pkg self-test' scripts/ci/madaros_full_gate.sh
require_grep 'resolve_raw_madaros' scripts/ci/madaros_full_gate.sh
require_grep 'receipt_ok' scripts/ci/madaros_full_gate.sh
require_grep 'SOUNIO_MADAROS_FULL_GATE_SKIP_SMT' scripts/ci/madaros_full_gate.sh
require_grep 'imported-SMT solver gate' scripts/ci/madaros_full_gate.sh
require_grep 'write_gate_receipt' scripts/ci/madaros_full_gate.sh
require_grep 'kernel_ir__lower_to_ptx__ptx' scripts/ci/madaros_gpu_ptx_pairwise_import_witness.sh
require_grep 'error\[E175' scripts/ci/madaros_gpu_ptx_pairwise_import_witness.sh
require_grep 'error\[E177' scripts/ci/madaros_gpu_ptx_pairwise_import_witness.sh
require_grep 'error\[E046' scripts/ci/madaros_gpu_ptx_pairwise_import_witness.sh
require_grep 'Madaros Greenline Gate' .github/workflows/ci.yml
require_grep 'make madaros-full-gate' .github/workflows/ci.yml
require_grep 'scripts/dev/madaros_two_gate.sh artifacts/self-hosted/madaros' .github/workflows/ci.yml
require_grep 'scripts/ci/madaros_gpu_ptx_pairwise_import_witness.sh' .github/workflows/ci.yml
require_grep 'scripts/ci/madaros_gpu_ptx_pairwise_import_witness.sh' .github/workflows/madaros-prebuilt-refresh.yml
require_grep 'madaros_source_to_elf_gate.sh' .github/workflows/madaros-prebuilt-refresh.yml
require_grep 'source-to-ELF' .github/workflows/madaros-prebuilt-refresh.yml

require_grep 'DEFAULT ENGINE: Madaros' bin/souc
require_grep '.gate-receipt' bin/souc
require_grep 'bin/madaros-relocgate' bin/souc
require_grep '.gate-receipt' bin/madaros
require_grep 'bin/madaros-relocgate' bin/madaros
require_grep '.gate-receipt' scripts/lib/resolve_madaros.sh
require_grep 'bin/madaros-relocgate' scripts/lib/resolve_madaros.sh
require_grep 'bin/madaros-linux-x86_64' bin/souc
require_grep 'exec env MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros"' bin/souc
require_grep 'artifacts/self-hosted/madaros.gate-receipt' .gitignore
require_grep 'cleanup_plan_command=scripts/dev/madaros_worktree_cleanup_plan.sh' scripts/dev/madaros_readiness_status.sh
require_grep 'decision_packet_command=scripts/dev/madaros_cleanup_decision_packet.sh' scripts/dev/madaros_readiness_status.sh
require_grep 'audit_allow_manifest_decision=approve_keep_active_allowlist' scripts/dev/madaros_readiness_status.sh
require_grep 'scripts/dev/madaros_cleanup_decision_packet.sh --tarball' scripts/dev/madaros_readiness_status.sh
require_grep 'git push, git reset, git clean, git branch -D' scripts/dev/madaros_worktree_cleanup_plan.sh
require_grep 'owner confirmation required before any archive, push, or removal' scripts/dev/madaros_worktree_cleanup_plan.sh
require_grep 'unique_commits_origin_main' scripts/dev/madaros_worktree_cleanup_plan.sh
require_grep 'tracked_diff_added' scripts/dev/madaros_worktree_cleanup_plan.sh
require_grep 'critical_vs_base' scripts/dev/madaros_worktree_cleanup_plan.sh
require_grep '# evidence=origin_main_unique:' scripts/dev/madaros_worktree_cleanup_plan.sh
require_grep 'cleanup_approval_command=scripts/dev/madaros_worktree_cleanup_approval.sh' scripts/dev/madaros_readiness_status.sh
require_grep 'I_ACCEPT_PUSH_BEFORE_DELETE' scripts/dev/madaros_worktree_cleanup_approval.sh
require_grep 'approve_keep_active_allowlist' scripts/dev/madaros_worktree_cleanup_approval.sh
require_grep 'madaros_cleanup_suggested_manifest.sh' docs/audit/MADAROS_WORKTREE_CLEANUP_LEDGER_2026-07-03.md
require_grep 'madaros_cleanup_decision_packet.sh --tarball' docs/audit/MADAROS_WORKTREE_CLEANUP_LEDGER_2026-07-03.md
require_grep 'keep_active_allowlist  -> approve_keep_active_allowlist' scripts/dev/madaros_cleanup_suggested_manifest.sh
require_grep 'SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_MANIFEST' scripts/dev/worktree_branch_audit.sh
require_grep 'tracked/staged binary diffs' scripts/dev/madaros_worktree_salvage_packet.sh
require_grep 'No branches pushed or worktrees removed' scripts/dev/madaros_worktree_salvage_packet.sh
require_cleanup_planner_evidence
require_cleanup_approval_evidence
require_cleanup_suggested_manifest_evidence
require_cleanup_decision_packet_evidence
require_salvage_packet_evidence

echo "[madaros-contract] PASS: status doc, agent contract, default wrapper, and gate wiring are aligned"
