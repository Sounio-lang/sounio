#!/usr/bin/env bash
set -euo pipefail

compute_class=""
output_format="text"

usage() {
  cat <<'EOF'
Usage: bash scripts/dev/sounio_compute_probe.sh --class CLASS [--format text|tsv]

Run a read-only availability probe for a Sounio compute route. The command
never submits a job or mutates cluster state.

Classes:
  compiler.cpu-heavy
  compiler.gpu
  compiler.fpga-u250
  research.batch
  foundry.release

Fixture environment variables used by the self-test:
  SOUNIO_COMPUTE_PROBE_SLURM_FIXTURE
  SOUNIO_COMPUTE_PROBE_K8S_FIXTURE
  SOUNIO_COMPUTE_PROBE_BEAGLE_FIXTURE
  SOUNIO_COMPUTE_PROBE_BEAGLE_RC
EOF
}

require_value() {
  if [[ -z "${2:-}" ]]; then
    echo "error: $1 requires a value" >&2
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
    --format)
      require_value "$1" "${2:-}"
      output_format="$2"
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
if [[ "$output_format" != "text" && "$output_format" != "tsv" ]]; then
  echo "error: --format must be text or tsv" >&2
  exit 2
fi

backend=""
classification="BLOCKED"
reason="probe_not_run"
observed="none"

normalize() {
  tr '\n\t' '  ' <<<"$1" | tr -s ' ' | cut -c1-240
}

read_slurm() {
  if [[ -n "${SOUNIO_COMPUTE_PROBE_SLURM_FIXTURE:-}" ]]; then
    cat "$SOUNIO_COMPUTE_PROBE_SLURM_FIXTURE"
    return
  fi

  local kubectl_bin="${SOUNIO_KUBECTL_BIN:-kubectl}"
  local namespace="${SOUNIO_SLURM_NAMESPACE:-slurm-pilot}"
  local login="${SOUNIO_SLURM_LOGIN:-deploy/slurm-pilot-login-slinky}"
  "$kubectl_bin" -n "$namespace" exec "$login" -- \
    sinfo -h -o '%P|%a|%D|%G|%f|%T'
}

probe_slurm() {
  local route_kind="$1"
  local output=""
  local rc=0
  output="$(read_slurm 2>&1)" || rc=$?
  observed="$(normalize "$output")"
  if ((rc != 0)); then
    reason="slurm_query_failed_rc_${rc}"
    return
  fi

  case "$route_kind" in
    cpu)
      if awk -F'|' '$1 ~ /^all\*?$/ && $2 == "up" && $5 ~ /gpuorangefs-multi/ && $6 !~ /(down|drain|fail)/ {found=1} END {exit !found}' <<<"$output"; then
        classification="READY"
        reason="slurm_cpu_route_available"
      else
        reason="no_usable_all_gpuorangefs_multi_route"
      fi
      ;;
    gpu)
      if awk -F'|' '$1 ~ /^gpu-orangefs\*?$/ && $2 == "up" && $4 ~ /gpu:/ && $6 !~ /(down|drain|fail)/ {found=1} END {exit !found}' <<<"$output"; then
        classification="READY"
        reason="slurm_gpu_route_available"
      else
        reason="no_usable_gpu_orangefs_route"
      fi
      ;;
    batch)
      if awk -F'|' '$1 ~ /^all\*?$/ && $2 == "up" && $6 !~ /(down|drain|fail)/ {found=1} END {exit !found}' <<<"$output"; then
        classification="READY"
        reason="slurm_batch_route_available"
      else
        reason="no_usable_all_partition_route"
      fi
      ;;
  esac
}

read_k8s_u250() {
  if [[ -n "${SOUNIO_COMPUTE_PROBE_K8S_FIXTURE:-}" ]]; then
    cat "$SOUNIO_COMPUTE_PROBE_K8S_FIXTURE"
    return
  fi

  local kubectl_bin="${SOUNIO_KUBECTL_BIN:-kubectl}"
  "$kubectl_bin" get nodes -l 'sounio.dev/fpga=u250' \
    -o 'jsonpath={range .items[*]}{.metadata.name}{"|"}{range .status.conditions[?(@.type=="Ready")]}{.status}{end}{"|"}{.status.allocatable.sounio\.dev/u250}{"\n"}{end}'
}

probe_u250() {
  local output=""
  local rc=0
  output="$(read_k8s_u250 2>&1)" || rc=$?
  observed="$(normalize "$output")"
  if ((rc != 0)); then
    reason="kubernetes_query_failed_rc_${rc}"
  elif awk -F'|' '$2 == "True" && ($3 + 0) >= 1 {found=1} END {exit !found}' <<<"$output"; then
    classification="READY"
    reason="kubernetes_u250_resource_available"
  else
    reason="no_ready_u250_allocatable_node"
  fi
}

read_beagle() {
  if [[ -n "${SOUNIO_COMPUTE_PROBE_BEAGLE_FIXTURE:-}" ]]; then
    cat "$SOUNIO_COMPUTE_PROBE_BEAGLE_FIXTURE"
    return "${SOUNIO_COMPUTE_PROBE_BEAGLE_RC:-0}"
  fi
  local beagle_bin="${SOUNIO_BEAGLE_BIN:-beagle}"
  "$beagle_bin" hpc profiles
}

probe_beagle() {
  local output=""
  local rc=0
  output="$(read_beagle 2>&1)" || rc=$?
  observed="$(normalize "$output")"
  if ((rc != 0)); then
    if grep -Eqi '(401|unauthori|auth|token|credential)' <<<"$output"; then
      reason="beagle_auth_or_profile_unavailable"
    else
      reason="beagle_profile_query_failed_rc_${rc}"
    fi
  elif grep -Eq '(^|[[:space:]])full-compiler($|[[:space:]])' <<<"$output"; then
    classification="READY"
    reason="beagle_full_compiler_profile_available"
  else
    reason="beagle_full_compiler_profile_missing"
  fi
}

case "$compute_class" in
  compiler.cpu-heavy)
    backend="slurm"
    probe_slurm cpu
    ;;
  compiler.gpu)
    backend="slurm"
    probe_slurm gpu
    ;;
  compiler.fpga-u250)
    backend="kubernetes"
    probe_u250
    ;;
  research.batch)
    backend="slurm"
    probe_slurm batch
    ;;
  foundry.release)
    backend="beagle-foundry"
    probe_beagle
    ;;
  *)
    echo "error: unknown compute class: $compute_class" >&2
    exit 2
    ;;
esac

keys=(schema probe_state submission_performed compute_class backend classification reason observed)
values=(sounio.compute.probe.v1 OBSERVED no "$compute_class" "$backend" "$classification" "$reason" "$observed")

for ((i = 0; i < ${#keys[@]}; i++)); do
  if [[ "$output_format" == "tsv" ]]; then
    printf '%s\t%s\n' "${keys[$i]}" "${values[$i]}"
  else
    printf '%s: %s\n' "${keys[$i]}" "${values[$i]}"
  fi
done

