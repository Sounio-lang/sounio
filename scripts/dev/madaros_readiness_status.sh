#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

REPO="${SOUNIO_GITHUB_REPO:-Sounio-lang/sounio}"
RUN_AUDIT=1
RUN_OPEN_BLOCKERS=0
RUN_SOURCE_TO_ELF=0
CHECK_COMPILER_LANE=0
CHECK_COMPILER_PR_OVERLAP=0
CHECK_PR_RESOLUTION_QUEUE=0
IN_PRODUCTION_READY_CHECK=0
COMPILER_LANE_PATH="${SOUNIO_MADAROS_COMPILER_LANE:-/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b}"
REFRESH_GH=0
STRICT=0
PRODUCTION_READY=0
PRODUCTION_READY_FAILED=0
COMPILER_PR_OVERLAP_FAILED=0
PR_RESOLUTION_QUEUE_FAILED=0
OPEN_BLOCKERS_FAILED=0
OPEN_BLOCKERS_OUT_DIR=""
OPEN_BLOCKERS_REPORT=""
ISSUE_356_STATE="unknown"
MAIN_CI_STATUS="unknown"
MAIN_CI_CONCLUSION="unknown"
MAIN_CI_HEAD="unknown"
MAIN_CI_URL="unknown"

usage() {
  cat <<'USAGE'
Usage: scripts/dev/madaros_readiness_status.sh [options]

Print the current Madaros production-readiness control surface: baseline,
local audit status, GitHub issue/PR/CI state when gh is available, and the
next gates that close the active blocker.

Options:
  --refresh-gh          git fetch origin before reading origin/main
  --no-audit           skip worktree governance audit
  --check-compiler-lane
                       inspect the active compiler lane and run read-only
                       check-clean probes for the Claude-owned compiler files
  --check-compiler-pr-overlap
                       inspect open PRs for overlap with Claude-owned compiler
                       files and fail under --strict if any overlap exists
  --check-pr-resolution-queue
                       print the open PR resolution queue and fail under
                       --strict if compiler ownership overlap remains
  --compiler-lane DIR  override active compiler lane path
  --run-open-blockers  run scripts/ci/madaros_open_blockers_probe.sh
  --run-source-to-elf  run scripts/ci/madaros_source_to_elf_gate.sh
  --strict             exit nonzero if audit or optional gates fail
  --production-ready   exit nonzero if current blocker records still prevent
                       saying Madaros is production-ready; also runs the
                       known-open blocker probe
  -h, --help           show this help
USAGE
}

while (($#)); do
  case "$1" in
    --refresh-gh)
      REFRESH_GH=1
      ;;
    --no-audit)
      RUN_AUDIT=0
      ;;
    --check-compiler-lane)
      CHECK_COMPILER_LANE=1
      ;;
    --check-compiler-pr-overlap)
      CHECK_COMPILER_PR_OVERLAP=1
      ;;
    --check-pr-resolution-queue)
      CHECK_PR_RESOLUTION_QUEUE=1
      ;;
    --compiler-lane)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "error: --compiler-lane requires a directory" >&2
        usage >&2
        exit 2
      fi
      COMPILER_LANE_PATH="$1"
      ;;
    --run-open-blockers)
      RUN_OPEN_BLOCKERS=1
      ;;
    --run-source-to-elf)
      RUN_SOURCE_TO_ELF=1
      ;;
    --strict)
      STRICT=1
      ;;
    --production-ready)
      STRICT=1
      PRODUCTION_READY=1
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

have() {
  command -v "$1" >/dev/null 2>&1
}

section() {
  printf '\n== %s ==\n' "$1"
}

run_status_command() {
  local label="$1"
  shift

  echo "+ $*"
  if "$@"; then
    echo "status[$label]=pass"
    return 0
  fi

  local rc=$?
  echo "status[$label]=fail rc=$rc"
  if [[ "$STRICT" == "1" ]]; then
    return "$rc"
  fi
  return 0
}

summarize_check_log() {
  local log_path="$1"

  if [[ ! -s "$log_path" ]]; then
    echo "  diagnostics=none"
    return 0
  fi

  if have rg; then
    local total_errors
    total_errors="$(rg -c '^error\[' "$log_path" 2>/dev/null || true)"
    [[ -n "$total_errors" ]] || total_errors=0
    echo "  error_lines=$total_errors"
    rg -o 'error\[E[0-9]+' "$log_path" 2>/dev/null \
      | sed 's/error\[//' \
      | sort | uniq -c | awk '{ print "  error[" $2 "]=" $1 }' || true
    echo "  tail:"
    tail -n 20 "$log_path" | sed 's/^/    /'
    return 0
  fi

  echo "  tail:"
  tail -n 20 "$log_path" | sed 's/^/    /'
}

run_compiler_lane_check() {
  local lane_path="$1"
  local lane_failed=0

  echo "compiler_lane=$lane_path"
  if [[ ! -d "$lane_path" ]]; then
    echo "status[compiler_lane]=missing"
    if [[ "$STRICT" == "1" ]]; then
      return 1
    fi
    return 0
  fi

  local lane_head lane_branch dirty_files out_dir
  lane_head="$(git -C "$lane_path" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
  lane_branch="$(git -C "$lane_path" branch --show-current 2>/dev/null || true)"
  [[ -n "$lane_branch" ]] || lane_branch="detached"
  out_dir="${SOUNIO_MADAROS_COMPILER_LANE_LOG_DIR:-$(mktemp -d /tmp/sounio-madaros-compiler-lane.XXXXXX)}"

  echo "compiler_lane_branch=$lane_branch"
  echo "compiler_lane_head=$lane_head"
  echo "compiler_lane_logs=$out_dir"
  echo "compiler_lane_dirty_files:"
  dirty_files="$(git -C "$lane_path" status --short -- \
    self-hosted/compiler/main.sio \
    self-hosted/native/codegen.sio \
    self-hosted/native/codegen_x86_linux.sio \
    self-hosted/native/lower_ir.sio \
    self-hosted/native/suite.sio \
    2>/dev/null || true)"
  if [[ -n "$dirty_files" ]]; then
    printf '%s\n' "$dirty_files" | sed 's/^/  /'
  else
    echo "  none"
  fi

  local check_files=(
    self-hosted/native/lower_ir.sio
    self-hosted/native/codegen.sio
    self-hosted/native/codegen_x86_linux.sio
  )

  local file safe_file log_path rc
  for file in "${check_files[@]}"; do
    safe_file="${file//\//__}"
    log_path="$out_dir/${safe_file}.check.log"
    echo "+ compiler_lane_check $file"
    set +e
    env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
      "$lane_path/bin/souc" check "$file" >"$log_path" 2>&1
    rc=$?
    set -e

    if [[ "$rc" -eq 0 ]]; then
      echo "status[compiler_lane:$file]=pass"
    else
      echo "status[compiler_lane:$file]=fail rc=$rc"
      summarize_check_log "$log_path"
      lane_failed=1
    fi
  done

  if [[ "$lane_failed" == "0" ]]; then
    echo "status[compiler_lane]=pass"
    return 0
  fi

  echo "status[compiler_lane]=fail"
  if [[ "$STRICT" == "1" ]]; then
    return 1
  fi
  return 0
}

run_compiler_pr_overlap_check() {
  local overlap_count=0
  local owned_files=(
    self-hosted/compiler/main.sio
    self-hosted/native/codegen.sio
    self-hosted/native/codegen_x86_linux.sio
    self-hosted/native/lower_ir.sio
    self-hosted/native/suite.sio
  )

  if ! have gh; then
    echo "status[compiler_pr_overlap]=unavailable gh_missing"
    COMPILER_PR_OVERLAP_FAILED=1
    if [[ "$STRICT" == "1" && "$IN_PRODUCTION_READY_CHECK" != "1" ]]; then
      return 1
    fi
    return 0
  fi

  echo "compiler_pr_overlap_owned_files:"
  printf '  %s\n' "${owned_files[@]}"
  echo "compiler_pr_overlap_prs:"

  local pr_list
  pr_list="$(gh pr list --repo "$REPO" --state open --limit 40 --json number --jq '.[].number' 2>/dev/null || true)"
  if [[ -z "$pr_list" ]]; then
    echo "  none"
    echo "status[compiler_pr_overlap]=pass"
    return 0
  fi

  local pr files overlap title url
  while IFS= read -r pr; do
    [[ -n "$pr" ]] || continue
    files="$(
      gh pr view "$pr" --repo "$REPO" --json files \
        --jq '.files[].path' 2>/dev/null || true
    )"
    [[ -n "$files" ]] || continue

    overlap="$(
      printf '%s\n' "$files" | awk '
        BEGIN {
          owned["self-hosted/compiler/main.sio"] = 1
          owned["self-hosted/native/codegen.sio"] = 1
          owned["self-hosted/native/codegen_x86_linux.sio"] = 1
          owned["self-hosted/native/lower_ir.sio"] = 1
          owned["self-hosted/native/suite.sio"] = 1
        }
        owned[$0] { print }
      ' | paste -sd ',' -
    )"
    [[ -n "$overlap" ]] || continue

    title="$(gh pr view "$pr" --repo "$REPO" --json title --jq '.title' 2>/dev/null || echo unavailable)"
    url="$(gh pr view "$pr" --repo "$REPO" --json url --jq '.url' 2>/dev/null || echo unavailable)"
    echo "  pr=#$pr files=$overlap url=$url"
    echo "    title=$title"
    overlap_count=$((overlap_count + 1))
  done <<<"$pr_list"

  if [[ "$overlap_count" == "0" ]]; then
    echo "  none"
    echo "status[compiler_pr_overlap]=pass"
    return 0
  fi

  echo "status[compiler_pr_overlap]=fail count=$overlap_count"
  COMPILER_PR_OVERLAP_FAILED=1
  if [[ "$STRICT" == "1" && "$IN_PRODUCTION_READY_CHECK" != "1" ]]; then
    return 1
  fi
  return 0
}

run_open_blockers_probe() {
  local out_dir="${SOUNIO_MADAROS_READINESS_OPEN_BLOCKERS_DIR:-$(mktemp -d /tmp/sounio-madaros-readiness-open-blockers.XXXXXX)}"
  local log_path="$out_dir/probe.log"
  local rc

  OPEN_BLOCKERS_FAILED=0
  OPEN_BLOCKERS_OUT_DIR="$out_dir"
  OPEN_BLOCKERS_REPORT="$out_dir/report.tsv"

  mkdir -p "$out_dir"
  echo "open_blockers_out=$OPEN_BLOCKERS_OUT_DIR"
  echo "+ env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN SOUNIO_MADAROS_OPEN_BLOCKERS_DIR=$out_dir scripts/ci/madaros_open_blockers_probe.sh"

  set +e
  env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
    SOUNIO_MADAROS_OPEN_BLOCKERS_DIR="$out_dir" \
    scripts/ci/madaros_open_blockers_probe.sh >"$log_path" 2>&1
  rc=$?
  set -e

  cat "$log_path"
  echo "open_blockers_report=$OPEN_BLOCKERS_REPORT"

  if [[ "$rc" != "0" ]]; then
    echo "status[open_blockers]=fail rc=$rc"
    echo "reason=open_blockers_probe_changed_or_failed"
    OPEN_BLOCKERS_FAILED=1
  elif [[ ! -s "$OPEN_BLOCKERS_REPORT" ]]; then
    echo "status[open_blockers]=fail"
    echo "reason=open_blockers_report_missing"
    OPEN_BLOCKERS_FAILED=1
  elif awk -F '\t' 'NR > 1 && $6 == "still_open" { found = 1 } END { exit found ? 0 : 1 }' "$OPEN_BLOCKERS_REPORT"; then
    echo "status[open_blockers]=fail"
    echo "reason=open_blockers_still_open"
    OPEN_BLOCKERS_FAILED=1
  else
    echo "status[open_blockers]=pass"
    echo "reason=no_known_open_witnesses_reproduced"
  fi

  if [[ "$OPEN_BLOCKERS_FAILED" != "0" && "$STRICT" == "1" && "$IN_PRODUCTION_READY_CHECK" != "1" ]]; then
    return 1
  fi
  return 0
}

run_pr_resolution_queue_check() {
  local queue_failed=0

  set +e
  scripts/dev/madaros_pr_resolution_queue.sh --check
  queue_failed=$?
  set -e

  if [[ "$queue_failed" != "0" ]]; then
    PR_RESOLUTION_QUEUE_FAILED=1
    if [[ "$STRICT" == "1" && "$IN_PRODUCTION_READY_CHECK" != "1" ]]; then
      return "$queue_failed"
    fi
  fi
  return 0
}

if [[ "$REFRESH_GH" == "1" ]]; then
  git fetch --prune origin
fi

current_branch="$(git branch --show-current 2>/dev/null || true)"
[[ -n "$current_branch" ]] || current_branch="detached"
current_sha="$(git rev-parse --short=12 HEAD)"
origin_main_sha="$(git rev-parse --short=12 origin/main 2>/dev/null || echo unknown)"
full_origin_main_sha="$(git rev-parse origin/main 2>/dev/null || echo unknown)"

section "Baseline"
echo "repo=$ROOT_DIR"
echo "branch=$current_branch"
echo "head=$current_sha"
echo "origin_main=$origin_main_sha"
echo "origin_main_full=$full_origin_main_sha"

if [[ -e /workspace/sounio/.git ]]; then
  primary_head="$(git -C /workspace/sounio rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
  primary_status="$(git -C /workspace/sounio status --short 2>/dev/null | wc -l | tr -d ' ')"
  primary_relation="$(git -C /workspace/sounio status --short --branch 2>/dev/null | head -n1 || true)"
  echo "primary_checkout=/workspace/sounio"
  echo "primary_head=$primary_head"
  echo "primary_dirty_entries=$primary_status"
  echo "primary_relation=$primary_relation"
fi

section "Production Influence Set"
cat <<'EOF'
1. clean origin/main
2. issue #356 blocker records
3. scripts/ci/madaros_open_blockers_probe.sh
4. scripts/ci/madaros_source_to_elf_gate.sh
5. one active compiler owner for the BSS/global blocker
6. scripts/dev/madaros_pr_resolution_queue.sh
EOF

section "Active Blockers"
cat <<'EOF'
BLK-20260621-codex-source-elf-normal-bss
  class=compiler-semantics
  owner=Claude compiler/codegen lane unless ownership transfers explicitly
  acceptance=global_read_exit4 exits 4; global_store_exit7 exits 7; open-blocker probe converted from known-open to closed expectation; source-to-ELF gate green

BLK-20260621-codex-madaros-build-segfault
  class=platform-resource
  owner=integration shepherd / workspace-runtime lane unless compiler owner proves semantic root
  acceptance=local promoted workspace build agrees with GitHub prebuilt refresh, or remains explicitly non-authoritative for production readiness
EOF

section "GitHub"
if have gh; then
  issue_356_fields="$(
    gh issue view 356 --repo "$REPO" \
      --json state,title,url,updatedAt \
      --jq '[.state, .updatedAt, .url, .title] | @tsv' \
      2>/dev/null || true
  )"
  if [[ -n "$issue_356_fields" ]]; then
    IFS=$'\t' read -r ISSUE_356_STATE issue_356_updated issue_356_url issue_356_title <<<"$issue_356_fields"
    echo "issue_356_state=$ISSUE_356_STATE updated_at=$issue_356_updated url=$issue_356_url title=$issue_356_title"
  else
    echo "issue_356_state=unavailable"
  fi

  gh pr list --repo "$REPO" --state open --limit 40 \
    --json number,title,headRefName,baseRefName,isDraft,mergeable,url \
    --jq '
      "open_prs=" + (length|tostring),
      "known_conflicting_prs=" + ([.[] | select(.mergeable == "CONFLICTING")] | length | tostring),
      "unknown_mergeability_prs=" + ([.[] | select(.mergeable == "UNKNOWN")] | length | tostring),
      (.[] | "pr=#\(.number) mergeable=\(.mergeable) draft=\(.isDraft) base=\(.baseRefName) head=\(.headRefName) url=\(.url)")
    ' 2>/dev/null || echo "open_prs=unavailable"

  main_ci_fields="$(
    gh run list --repo "$REPO" --branch main --limit 10 \
      --json databaseId,name,status,conclusion,headSha,url \
      --jq '
        [.[] | select(.name == "CI")][0]
        | if . == null then empty else [.status, (if ((.conclusion // "") == "") then "none" else .conclusion end), .headSha, .url] | @tsv end
      ' 2>/dev/null || true
  )"
  if [[ -n "$main_ci_fields" ]]; then
    IFS=$'\t' read -r MAIN_CI_STATUS MAIN_CI_CONCLUSION MAIN_CI_HEAD MAIN_CI_URL <<<"$main_ci_fields"
  fi

  gh run list --repo "$REPO" --branch main --limit 10 \
    --json databaseId,name,status,conclusion,headSha,url \
    --jq '
      .[]
      | select(.name == "CI" or .name == "Madaros Prebuilt Refresh")
      | "run=\(.databaseId) name=\(.name) status=\(.status) conclusion=\(.conclusion) head=\(.headSha[0:12]) url=\(.url)"
    ' 2>/dev/null || echo "runs=unavailable"
else
  echo "gh=missing"
fi

if [[ "$RUN_AUDIT" == "1" ]]; then
  section "Worktree Audit"
  audit_allow_default="^(/workspace/sounio|/workspace/sounio-effects|/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b|$ROOT_DIR)$"
  audit_out="${SOUNIO_MADAROS_READINESS_AUDIT_OUT:-$(mktemp /tmp/sounio-madaros-readiness-audit.XXXXXX.tsv)}"
  run_status_command audit \
    env SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE="${SOUNIO_AUDIT_ALLOW_CRITICAL_DIRTY_RE:-$audit_allow_default}" \
      scripts/dev/worktree_branch_audit.sh --check "$audit_out"
fi

if [[ "$CHECK_COMPILER_LANE" == "1" ]]; then
  section "Compiler Lane"
  run_compiler_lane_check "$COMPILER_LANE_PATH"
fi

if [[ "$CHECK_COMPILER_PR_OVERLAP" == "1" ]]; then
  section "Compiler PR Overlap"
  run_compiler_pr_overlap_check
fi

if [[ "$CHECK_PR_RESOLUTION_QUEUE" == "1" ]]; then
  section "PR Resolution Queue"
  run_pr_resolution_queue_check
fi

if [[ "$PRODUCTION_READY" == "1" ]]; then
  if [[ "$CHECK_COMPILER_PR_OVERLAP" != "1" ]]; then
    section "Compiler PR Overlap"
    IN_PRODUCTION_READY_CHECK=1
    run_compiler_pr_overlap_check
    IN_PRODUCTION_READY_CHECK=0
  fi

  if [[ "$CHECK_PR_RESOLUTION_QUEUE" != "1" ]]; then
    section "PR Resolution Queue"
    IN_PRODUCTION_READY_CHECK=1
    run_pr_resolution_queue_check
    IN_PRODUCTION_READY_CHECK=0
  fi

  section "Open Blocker Probe"
  IN_PRODUCTION_READY_CHECK=1
  run_open_blockers_probe
  IN_PRODUCTION_READY_CHECK=0

  section "Production Ready Verdict"
  echo "main_ci_status=$MAIN_CI_STATUS"
  echo "main_ci_conclusion=$MAIN_CI_CONCLUSION"
  echo "main_ci_head=$MAIN_CI_HEAD"
  echo "main_ci_url=$MAIN_CI_URL"
  if [[ "$ISSUE_356_STATE" == "CLOSED" && "$COMPILER_PR_OVERLAP_FAILED" == "0" && "$PR_RESOLUTION_QUEUE_FAILED" == "0" && "$OPEN_BLOCKERS_FAILED" == "0" && "$MAIN_CI_STATUS" == "completed" && "$MAIN_CI_CONCLUSION" == "success" && "$MAIN_CI_HEAD" == "$full_origin_main_sha" ]]; then
    echo "status[production_ready]=pass"
    echo "reason=issue_356_closed_no_compiler_pr_overlap_pr_queue_clear_open_blockers_closed_and_current_main_ci_green"
  else
    echo "status[production_ready]=fail"
    if [[ "$MAIN_CI_STATUS" != "completed" || "$MAIN_CI_CONCLUSION" != "success" || "$MAIN_CI_HEAD" != "$full_origin_main_sha" ]]; then
      echo "reason=main_ci_not_green_for_origin_main"
      echo "origin_main_full=$full_origin_main_sha"
      echo "required=latest main CI for current origin/main must be completed/success"
    fi
    if [[ "$ISSUE_356_STATE" != "CLOSED" ]]; then
      echo "reason=issue_356_state_${ISSUE_356_STATE}"
      echo "blocking_issue=https://github.com/Sounio-lang/sounio/issues/356"
      echo "required=close or explicitly narrow BLK-20260621-codex-source-elf-normal-bss and BLK-20260621-codex-madaros-build-segfault"
    fi
    if [[ "$COMPILER_PR_OVERLAP_FAILED" != "0" ]]; then
      echo "reason=compiler_pr_overlap"
      echo "required=resolve open PR overlap with the active Claude compiler lane or transfer ownership explicitly"
    fi
    if [[ "$PR_RESOLUTION_QUEUE_FAILED" != "0" ]]; then
      echo "reason=pr_resolution_queue_blocked"
      echo "required=resolve the open PR disposition queue entries that overlap compiler ownership"
    fi
    if [[ "$OPEN_BLOCKERS_FAILED" != "0" ]]; then
      echo "reason=open_blockers_not_closed"
      echo "open_blockers_report=$OPEN_BLOCKERS_REPORT"
      echo "open_blockers_out=$OPEN_BLOCKERS_OUT_DIR"
      echo "required=known-open blocker witnesses must stop reproducing, or expectations must be updated after the compiler fix"
    fi
    PRODUCTION_READY_FAILED=1
  fi
fi

section "Next Gates"
cat <<'EOF'
Compiler owner:
  env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
    scripts/ci/madaros_open_blockers_probe.sh

To localize the current BSS/global phase before editing compiler files:
  env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
    scripts/ci/madaros_open_blockers_probe.sh --diagnose-lowering

After BSS/global behavior changes:
  update scripts/ci/madaros_open_blockers_probe.sh from known-open expectations to closed expectations
  bash scripts/ci/madaros_source_to_elf_gate.sh
  PR CI green
  post-merge main CI green

Integration shepherd:
  scripts/ci/madaros_blocker_contract_gate.sh
  scripts/dev/madaros_readiness_status.sh --strict
  scripts/dev/madaros_readiness_status.sh --check-compiler-pr-overlap
  scripts/dev/madaros_readiness_status.sh --check-pr-resolution-queue
  scripts/dev/madaros_readiness_status.sh --production-ready
EOF

if [[ "$RUN_OPEN_BLOCKERS" == "1" && "$PRODUCTION_READY" != "1" ]]; then
  section "Open Blocker Probe"
  run_open_blockers_probe
fi

if [[ "$RUN_SOURCE_TO_ELF" == "1" ]]; then
  section "Source-To-ELF Gate"
  run_status_command source_to_elf \
    env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
      bash scripts/ci/madaros_source_to_elf_gate.sh
fi

if [[ "$PRODUCTION_READY" == "1" && "$PRODUCTION_READY_FAILED" != "0" ]]; then
  exit 1
fi
