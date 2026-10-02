#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

compute_class=""
gate=""
source_ref="HEAD"
output_format="text"
out_file=""

usage() {
  cat <<'EOF'
Usage: bash scripts/dev/sounio_compute_plan.sh --class CLASS --gate GATE [options]

Create a non-submitting, source-pinned compute routing plan.

Classes:
  compiler.cpu-heavy
  compiler.gpu
  compiler.fpga-u250
  research.batch
  foundry.release

Options:
  --source REF       Git commit/ref to pin (default: HEAD)
  --format FORMAT    text or tsv (default: text)
  --out FILE         Write the plan to FILE instead of stdout
  -h, --help         Show this help

This command never submits a job.
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
    --class)
      require_value "$1" "${2:-}"
      compute_class="$2"
      shift 2
      ;;
    --gate)
      require_value "$1" "${2:-}"
      gate="$2"
      shift 2
      ;;
    --source)
      require_value "$1" "${2:-}"
      source_ref="$2"
      shift 2
      ;;
    --format)
      require_value "$1" "${2:-}"
      output_format="$2"
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

if [[ -z "$compute_class" ]]; then
  echo "error: --class is required" >&2
  exit 2
fi
if [[ -z "$gate" ]]; then
  echo "error: --gate is required" >&2
  exit 2
fi
if [[ "$output_format" != "text" && "$output_format" != "tsv" ]]; then
  echo "error: --format must be text or tsv" >&2
  exit 2
fi

cd "$ROOT_DIR"
if ! source_sha="$(git rev-parse --verify --end-of-options "${source_ref}^{commit}" 2>/dev/null)"; then
  echo "error: source ref is not a commit: $source_ref" >&2
  exit 2
fi
source_tree="$(git rev-parse "${source_sha}^{tree}")"
source_branch="$(git branch --show-current)"
[[ -n "$source_branch" ]] || source_branch="detached"

backend=""
route=""
resource_request=""
artifact_transport=""
live_probe="required"
route_boundary=""

case "$compute_class" in
  compiler.cpu-heavy)
    backend="slurm"
    route="partition=all,constraint=gpuorangefs-multi,gres=none"
    resource_request="cpu=8,mem=48G,gpu=0,fpga=0"
    artifact_transport="orangefs_receipt_and_terminal_summary"
    live_probe="required"
    route_boundary="avoid_gpu_orangefs_exclusive_for_cpu_only_work"
    ;;
  compiler.gpu)
    backend="slurm"
    route="partition=gpu-orangefs,gres=gpu:1"
    resource_request="cpu=8,mem=48G,gpu=1,fpga=0"
    artifact_transport="orangefs_receipt_and_terminal_summary"
    live_probe="required"
    route_boundary="gpu_claim_requires_runtime_device_receipt"
    ;;
  compiler.fpga-u250)
    backend="kubernetes"
    route="node_selector=sounio.dev/fpga=u250,resource=sounio.dev/u250=1"
    resource_request="cpu=8,mem=32G,gpu=0,fpga=u250:1"
    artifact_transport="job_receipt_and_xrt_device_attestation"
    live_probe="required"
    route_boundary="u250_is_kubernetes_resource_not_current_slurm_node"
    ;;
  research.batch)
    backend="slurm"
    route="partition=all,array=caller_defined"
    resource_request="cpu=caller_defined,mem=caller_defined,gpu=explicit_only,fpga=0"
    artifact_transport="manifest_receipt_and_shard_ledger"
    live_probe="required"
    route_boundary="batch_claim_requires_complete_shard_manifest"
    ;;
  foundry.release)
    backend="beagle-foundry"
    route="profile=full-compiler,source=${source_sha}"
    resource_request="gpu=auto,slurm=foundry_managed"
    artifact_transport="foundry_artifact_manifest"
    live_probe="required"
    route_boundary="beagle_auth_and_profile_probe_must_pass_before_submit"
    ;;
  *)
    echo "error: unknown compute class: $compute_class" >&2
    exit 2
    ;;
esac

keys=(
  schema
  plan_state
  submission_performed
  compute_class
  backend
  route
  resource_request
  source_ref
  source_sha
  source_tree
  source_branch
  gate
  artifact_transport
  live_probe
  route_boundary
  receipt_contract
)
values=(
  sounio.compute.plan.v1
  PLANNED
  no
  "$compute_class"
  "$backend"
  "$route"
  "$resource_request"
  "$source_ref"
  "$source_sha"
  "$source_tree"
  "$source_branch"
  "$gate"
  "$artifact_transport"
  "$live_probe"
  "$route_boundary"
  job_id,backend,node,source_sha,source_tree,state,exit_code,artifact_root,first_failure,classification
)

render_plan() {
  local i
  for ((i = 0; i < ${#keys[@]}; i++)); do
    if [[ "$output_format" == "tsv" ]]; then
      printf '%s\t%s\n' "${keys[$i]}" "${values[$i]}"
    else
      printf '%s: %s\n' "${keys[$i]}" "${values[$i]}"
    fi
  done
}

if [[ -n "$out_file" ]]; then
  mkdir -p "$(dirname "$out_file")"
  render_plan >"$out_file"
  echo "wrote $out_file"
else
  render_plan
fi
