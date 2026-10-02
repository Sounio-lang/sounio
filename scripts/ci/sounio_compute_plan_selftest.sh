#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLANNER="$ROOT_DIR/scripts/dev/sounio_compute_plan.sh"
TEST_DIR="$(mktemp -d /tmp/sounio-compute-plan-selftest.XXXXXX)"
trap 'rm -rf "$TEST_DIR"' EXIT

fail() {
  echo "SOUNIO_COMPUTE_PLAN_SELFTEST_FAIL: $*" >&2
  exit 1
}

assert_field() {
  local output="$1"
  local key="$2"
  local expected="$3"
  local actual
  actual="$(awk -F '\t' -v key="$key" '$1 == key { print $2; exit }' <<<"$output")"
  [[ "$actual" == "$expected" ]] || fail "$key expected '$expected', got '$actual'"
}

run_plan() {
  bash "$PLANNER" --format tsv --class "$1" --gate "$2" --source HEAD
}

cpu_plan="$(run_plan compiler.cpu-heavy scripts/ci/madaros_imported_runtime_source_fresh_gate.sh)"
assert_field "$cpu_plan" backend slurm
assert_field "$cpu_plan" route "partition=all,constraint=gpuorangefs-multi,gres=none"
assert_field "$cpu_plan" submission_performed no
assert_field "$cpu_plan" route_boundary avoid_gpu_orangefs_exclusive_for_cpu_only_work

gpu_plan="$(run_plan compiler.gpu scripts/ci/kretikos_cross_backend_cuda_runtime_gate.sh)"
assert_field "$gpu_plan" backend slurm
assert_field "$gpu_plan" route "partition=gpu-orangefs,gres=gpu:1"

fpga_plan="$(run_plan compiler.fpga-u250 scripts/ci/san_imagenet_fpga_dl380_gate.sh)"
assert_field "$fpga_plan" backend kubernetes
assert_field "$fpga_plan" route "node_selector=sounio.dev/fpga=u250,resource=sounio.dev/u250=1"
assert_field "$fpga_plan" route_boundary u250_is_kubernetes_resource_not_current_slurm_node

batch_plan="$(run_plan research.batch scripts/ci/example_batch_gate.sh)"
assert_field "$batch_plan" backend slurm
assert_field "$batch_plan" artifact_transport manifest_receipt_and_shard_ledger

foundry_plan="$(run_plan foundry.release full-compiler)"
assert_field "$foundry_plan" backend beagle-foundry
assert_field "$foundry_plan" route_boundary beagle_auth_and_profile_probe_must_pass_before_submit

sha="$(git -C "$ROOT_DIR" rev-parse HEAD)"
tree="$(git -C "$ROOT_DIR" rev-parse HEAD^{tree})"
assert_field "$cpu_plan" source_sha "$sha"
assert_field "$cpu_plan" source_tree "$tree"

out_file="$TEST_DIR/plan.tsv"
bash "$PLANNER" --format tsv --class compiler.cpu-heavy --gate smoke --out "$out_file" >/dev/null
[[ -s "$out_file" ]] || fail "--out did not create a plan"

if bash "$PLANNER" --class unknown --gate smoke >"$TEST_DIR/unknown.out" 2>&1; then
  fail "unknown class was accepted"
fi
if bash "$PLANNER" --class compiler.cpu-heavy --gate smoke --source definitely-not-a-ref >"$TEST_DIR/ref.out" 2>&1; then
  fail "invalid source ref was accepted"
fi
if bash "$PLANNER" --class compiler.cpu-heavy >"$TEST_DIR/gate.out" 2>&1; then
  fail "missing gate was accepted"
fi

echo "SOUNIO_COMPUTE_PLAN_SELFTEST_OK"
