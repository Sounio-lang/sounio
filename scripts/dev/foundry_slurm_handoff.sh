#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

requested_by="${USER:-agent}"
gate_requested="full compiler stress"
command_or_foundry_target="sounio-forge submit full-compiler --source <commit> --gpu auto"
reason="heavy validation requested from workspace"
gpu_requirement="auto"
slurm_requirement="yes"
expected_artifact_root="artifacts/foundry/<run-id>/"
acceptance_criterion="target completes or returns first classified blocker"
known_blockers="none"
return_payload="summary, artifact root, command used, first failing log if any, failure class"
out_file=""

usage() {
  cat <<'EOF'
Usage: bash scripts/dev/foundry_slurm_handoff.sh [options]

Generate a Foundry/Slurm heavy-validation handoff request from the current
checked-out branch and commit. This script does not submit jobs.

Options:
  --requested-by VALUE       Requesting agent/operator name
  --gate VALUE               Gate requested
  --target VALUE             Foundry target or command shape
  --reason VALUE             Why the heavy run is needed
  --gpu VALUE                GPU requirement
  --slurm VALUE              Slurm requirement
  --artifact-root VALUE      Expected artifact root
  --acceptance VALUE         Acceptance criterion
  --known-blockers VALUE     Known blockers
  --return-payload VALUE     Expected return payload
  --out FILE                 Write handoff to FILE instead of stdout
  -h, --help                 Show this help
EOF
}

require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "error: $flag requires a value" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --requested-by)
      require_value "$1" "${2:-}"
      requested_by="$2"
      shift 2
      ;;
    --gate)
      require_value "$1" "${2:-}"
      gate_requested="$2"
      shift 2
      ;;
    --target)
      require_value "$1" "${2:-}"
      command_or_foundry_target="$2"
      shift 2
      ;;
    --reason)
      require_value "$1" "${2:-}"
      reason="$2"
      shift 2
      ;;
    --gpu)
      require_value "$1" "${2:-}"
      gpu_requirement="$2"
      shift 2
      ;;
    --slurm)
      require_value "$1" "${2:-}"
      slurm_requirement="$2"
      shift 2
      ;;
    --artifact-root)
      require_value "$1" "${2:-}"
      expected_artifact_root="$2"
      shift 2
      ;;
    --acceptance)
      require_value "$1" "${2:-}"
      acceptance_criterion="$2"
      shift 2
      ;;
    --known-blockers)
      require_value "$1" "${2:-}"
      known_blockers="$2"
      shift 2
      ;;
    --return-payload)
      require_value "$1" "${2:-}"
      return_payload="$2"
      shift 2
      ;;
    --out)
      require_value "$1" "${2:-}"
      out_file="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT_DIR"

branch="$(git branch --show-current)"
if [[ -z "$branch" ]]; then
  branch="detached"
fi

commit="$(git rev-parse --short HEAD)"
repo_path_seen="$(pwd)"
host_name="$(hostname 2>/dev/null || true)"

if [[ "$repo_path_seen" == /workspace/* ]]; then
  source_surface="workspace pod"
elif [[ "$repo_path_seen" == /home/devsounio/* ]]; then
  source_surface="host/control-plane"
elif [[ "$host_name" == sounio-workspace-* ]]; then
  source_surface="workspace pod"
else
  source_surface="unknown"
fi

dirty_count="$(git status --porcelain=v1 | wc -l | tr -d ' ')"
if [[ "$dirty_count" == "0" ]]; then
  dirty_state="clean"
else
  dirty_state="dirty (${dirty_count} status entries)"
fi

if [[ "$command_or_foundry_target" == *"<commit>"* ]]; then
  command_or_foundry_target="${command_or_foundry_target//<commit>/$commit}"
fi

handoff="$(cat <<EOF
Heavy Validation Handoff
requested_by: ${requested_by}
source_surface: ${source_surface}
repo_path_seen: ${repo_path_seen}
branch: ${branch}
commit: ${commit}
dirty_state: ${dirty_state}
gate_requested: ${gate_requested}
command_or_foundry_target: ${command_or_foundry_target}
reason: ${reason}
gpu_requirement: ${gpu_requirement}
slurm_requirement: ${slurm_requirement}
expected_artifact_root: ${expected_artifact_root}
acceptance_criterion: ${acceptance_criterion}
known_blockers: ${known_blockers}
return_payload: ${return_payload}
EOF
)"

if [[ -n "$out_file" ]]; then
  mkdir -p "$(dirname "$out_file")"
  printf '%s\n' "$handoff" >"$out_file"
  echo "wrote $out_file"
else
  printf '%s\n' "$handoff"
fi
