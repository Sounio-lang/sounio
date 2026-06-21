#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

REPO="${SOUNIO_GITHUB_REPO:-Sounio-lang/sounio}"
RUN_AUDIT=1
RUN_OPEN_BLOCKERS=0
RUN_SOURCE_TO_ELF=0
CHECK_COMPILER_LANE=0
COMPILER_LANE_PATH="${SOUNIO_MADAROS_COMPILER_LANE:-/workspace/sounio/.claude/worktrees/agent-adc1cd8b9d52ba53b}"
REFRESH_GH=0
STRICT=0

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
  --compiler-lane DIR  override active compiler lane path
  --run-open-blockers  run scripts/ci/madaros_open_blockers_probe.sh
  --run-source-to-elf  run scripts/ci/madaros_source_to_elf_gate.sh
  --strict             exit nonzero if audit or optional gates fail
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
  gh issue view 356 --repo "$REPO" \
    --json state,title,url,updatedAt \
    --jq '"issue_356_state=\(.state) updated_at=\(.updatedAt) url=\(.url) title=\(.title)"' \
    2>/dev/null || echo "issue_356_state=unavailable"

  gh pr list --repo "$REPO" --state open --limit 40 \
    --json number,title,headRefName,baseRefName,isDraft,mergeable,url \
    --jq '
      "open_prs=" + (length|tostring),
      "known_conflicting_prs=" + ([.[] | select(.mergeable == "CONFLICTING")] | length | tostring),
      "unknown_mergeability_prs=" + ([.[] | select(.mergeable == "UNKNOWN")] | length | tostring),
      (.[] | "pr=#\(.number) mergeable=\(.mergeable) draft=\(.isDraft) base=\(.baseRefName) head=\(.headRefName) url=\(.url)")
    ' 2>/dev/null || echo "open_prs=unavailable"

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

section "Next Gates"
cat <<'EOF'
Compiler owner:
  env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
    scripts/ci/madaros_open_blockers_probe.sh

After BSS/global behavior changes:
  update scripts/ci/madaros_open_blockers_probe.sh from known-open expectations to closed expectations
  bash scripts/ci/madaros_source_to_elf_gate.sh
  PR CI green
  post-merge main CI green

Integration shepherd:
  scripts/dev/madaros_readiness_status.sh --strict
EOF

if [[ "$RUN_OPEN_BLOCKERS" == "1" ]]; then
  section "Open Blocker Probe"
  run_status_command open_blockers \
    env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
      scripts/ci/madaros_open_blockers_probe.sh
fi

if [[ "$RUN_SOURCE_TO_ELF" == "1" ]]; then
  section "Source-To-ELF Gate"
  run_status_command source_to_elf \
    env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN \
      bash scripts/ci/madaros_source_to_elf_gate.sh
fi
