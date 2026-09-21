#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

EVIDENCE=tools/loom/evidence/loom-process-witness-effect-hypercube-v11-host-20260829.txt
POLICY_MANIFEST=tools/loom/process_witness_effect_policy_plan_v11.freeze.v1
POLICY_EXECUTABLE=tools/loom/_build/default/src/sounio-loom-process-witness-effect-policy-plan-v11

fail() {
  printf 'sounio-loom-process-witness-effect-hypercube-v11-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

file_hash() {
  sha256sum "$1" | cut -d ' ' -f 1
}

record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "$key occurs $count times in $path"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

field() {
  record_field "$EVIDENCE" "$1"
}

expect_field() {
  local key="$1" expected="$2" actual
  actual="$(field "$key")"
  [[ "$actual" == "$expected" ]] ||
    fail "$key drifted: expected=$expected actual=$actual"
}

expect_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || fail "frozen path is absent or linked: $path"
  [[ "$(file_hash "$path")" == "$expected" ]] || fail "frozen path hash drifted: $path"
}

expect_hash "$EVIDENCE" \
  57bc9730b0b5662a548af8271bdca6ed1651c5684c7999182e6c3d6e6ad53738
expect_hash "$POLICY_MANIFEST" \
  adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c
expect_hash tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V11.md \
  c065f6f30c721711ffa9cff74b3961043a9ca5ce1ab4938b440d984afab524cb
expect_hash tools/loom/process_witness_effect_policy_plan_v11_main.sio \
  42fe8c08510f00159f0ad63cb0aac620c35776d661a41691d290ffd44ae402e2
expect_hash tools/loom/src/loom_process_witness_effect_hypercube_v11.cpp \
  424d8cd2d5b8b32880cfce7b9ab2825c66932404f1fb6e34f9f78692c6526d5a
expect_hash scripts/dev/build_loom_process_witness_effect_hypercube_v11.sh \
  bd2bbd6fe6d241c080fa4685302ed7362442d7f15c02812bab0ed5c2c9f3c428
expect_hash scripts/ci/sounio_loom_process_witness_effect_hypercube_v11_selftest.sh \
  a40769e69282315b51d79f1eae43cc12f3aa8aaf4e0bba36bdc6d03ed62145f3
expect_hash scripts/dev/build_loom_process_witness_effect_hypercube_root_v11.sh \
  2b68d87e58d8a74a8876b4113252fd6005160ffda9575ef7643e9d4585ea7d7f
expect_hash scripts/ci/sounio_loom_process_witness_effect_hypercube_root_v11_selftest.sh \
  be59a05200d9655d4dc11c58fa193510f58f84ea99d1b875122183e5dc2783c6
expect_hash scripts/ci/sounio_loom_process_witness_effect_hypercube_v11_host_gate.sh \
  2ba290e28ca6bd3d48e14707bc35cfc7345278966619e0750bd16b61f72ef7ab
expect_hash scripts/dev/run_loom_process_witness_effect_hypercube_v11_host_probe.sh \
  ee87578404c99d42dd648ee8295cf1e7f26a905ec98c3a7d051c010f3ab5cc0f

expect_field schema loom-process-witness-effect-hypercube-v11-host-evidence-v1
expect_field stage MATERIAL_HYPERCUBE
expect_field producing_language C++20
expect_field language_role MATERIAL_PARITY
expect_field semantic_authority Sounio
expect_field semantic_decision false
expect_field action 9025
expect_field garden_commit bdacaebe62d7d1a22c3462264a83cb4739bf3489
expect_field sounio_executable_commit 23b3756e1b40771e46f2e0a0ceebd1d5f7a8412b
expect_field sounio_freeze_commit 98d01bd759af38967ec8f9229e69a5d2f4c3687b
expect_field policy_manifest_sha256 adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c
expect_field expected_bundle_sha256 876dce5e9445a5c29236689699719e53ebf79930afae75f8ad5ff21544664394
expect_field hardware_host t560-proxmox
expect_field hardware_arch x86_64
expect_field hardware_kernel Linux_7.0.2-5-pve
expect_field systemd_version 257
expect_field transport kubectl+hostPID+nsenter
expect_field cell_sha256 dc0d5cd0bd4ac372ac1b0de6708824b521f2c465ad1d122720ab90c8d3a519f5
expect_field tree_sha256 c0e2634e6bbaa793df79aa54e29ea30225590e9c3470be2769c277f9efee2cb4
expect_field family_count 12
expect_field probe_count 13
expect_field mechanism_dimension_count 18
expect_field vertex_count 40
expect_field refusal_count 25
expect_field completion_count 15
expect_field extinction_count 15
expect_field mincut_count_expected 13
expect_field crossed_named_rule_count 0
expect_field experiment_unavailable_count 0
expect_field invariant_stable true
expect_field delta_distinct true
expect_field triple_hash_binding true
expect_field vfs_read_only_toggled true
expect_field private_network_toggled true
expect_field unix_endpoint_absence_toggled true
expect_field lock_personality_toggled true
expect_field proc_treatment_toggled CAPSULE_EMPTY_BIND+LIVE_PROCFS
expect_field endpoint_extinction true
expect_field process_extinction true
expect_field scratch_extinction true
expect_field apparatus_falsifier RestrictNamespaces_yes_masked_clone3_open_vertex
expect_field apparatus_correction RestrictNamespaces_no_for_both_family_2_vertices
expect_field command 'bash scripts/dev/run_loom_process_witness_effect_hypercube_v11_host_probe.sh'
expect_field command_sha256 4c8d0da893b768cec321a2c2b3c1602b3a5db72315d3a979d7b0d5fbe1daaccc
expect_field command_result_sha256 f0a31473d9e530de862a1b25911ecb6df0c3b69de343961899208dcb42c5155d
expect_field host_output_sha256 fcbd43e5acbe1a0e2d5037d4f9eb67c78efa5635d9d7dc44ddc73c1f8330976e
expect_field host_result_line_count 42
expect_field normalized_vertex_certificate_count 40
expect_field normalized_vertex_certificate_bundle_sha256 1c92fcd7c97a5df4e8316b722f769f6777ea5979edcd09c207e88f9930f8d3dd
expect_field material_hypercube true
for boundary in material_coverage complete_effects material_execution action_9025_judged production_activation launch_open recycle_open exec_attached commit_attached ci_attached claim_ready; do
  expect_field "$boundary" false
done

[[ "$(printf '%s' "$(field command)" | sha256sum | cut -d ' ' -f 1)" == "$(field command_sha256)" ]] ||
  fail 'host command hash drifted'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-hypercube-v11-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
certificates="$work/certificates"
expected="$work/expected"
observed="$work/observed"

sed -n '/^normalized_vertex_certificates_begin$/,/^normalized_vertex_certificates_end$/p' "$EVIDENCE" |
  sed '1d;$d' >"$certificates"
[[ "$(file_hash "$certificates")" == "$(field normalized_vertex_certificate_bundle_sha256)" ]] ||
  fail 'normalized certificate bundle hash drifted'
[[ "$(grep -c '^vertex_certificate ' "$certificates")" == 40 ]] ||
  fail 'normalized certificate count drifted'

awk '
  function value(key, i, p, a) {
    for (i = 1; i <= NF; i++) {
      split($i, a, "=")
      if (a[1] == key) return a[2]
    }
    return ""
  }
  /^vertex_certificate / {
    family = value("family")
    probe = value("probe")
    bits = value("bits")
    observation = value("observation")
    result = value("syscall_result")
    witness = value("witness_kind")
    extinct = value("witness_extinct")
    invariant = value("invariant_sha256")
    delta = value("delta_sha256")
    witness_hash = value("witness_sha256")
    identity = family "/" probe "/" bits
    if (seen[identity]++) exit 10
    if (family !~ /^[0-9]+$/ || bits !~ /^[01]+$/) exit 11
    if (invariant !~ /^[0-9a-f]{64}$/ || delta !~ /^[0-9a-f]{64}$/ || witness_hash !~ /^[0-9a-f]{64}$/) exit 12
    cube = family "/" probe
    if (cube_invariant[cube] != "" && cube_invariant[cube] != invariant) exit 13
    cube_invariant[cube] = invariant
    if (seen_delta[cube "/" delta]++) exit 14
    if (observation == "REFUSED_BEFORE_EFFECT") {
      refused++
      if (extinct != "false") exit 15
    } else if (observation == "EFFECT_COMPLETED") {
      completed++
      if (extinct != "true" || witness == "NONE") exit 16
      extinctions++
    } else {
      exit 17
    }
    if (bits !~ /0/ && observation != "REFUSED_BEFORE_EFFECT") exit 18
    if (bits !~ /1/ && observation != "EFFECT_COMPLETED") exit 19
    printf "%s %s %s %s %s %s\n", family, probe, bits, observation, result, witness
  }
  END {
    if (NR != 40 || refused != 25 || completed != 15 || extinctions != 15) exit 20
    for (cube in cube_invariant) cubes++
    if (cubes != 13) exit 21
  }
' "$certificates" | sort >"$observed" || fail 'typed certificate validation failed'

policy_result="$(bash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v11_freeze_selftest.sh)"
[[ "$policy_result" == sounio-loom-process-witness-effect-policy-plan-v11-freeze-selftest:\ PASS* ]] ||
  fail 'source-fresh frozen Sounio policy gate failed'
[[ -x "$POLICY_EXECUTABLE" ]] || fail 'source-fresh Sounio policy executable is absent'

"$POLICY_EXECUTABLE" | awk '
  function value(key, i, a) {
    for (i = 1; i <= NF; i++) {
      split($i, a, "=")
      if (a[1] == key) return a[2]
    }
    return ""
  }
  /^VERTEX / {
    printf "%s %s %s %s %s %s\n", value("family"), value("probe"), value("bits"), value("expected"), value("syscall_result"), value("witness_kind")
  }
' | sort >"$expected"
[[ "$(wc -l <"$expected")" == 40 ]] || fail 'Sounio policy omitted expected vertices'
cmp -s "$expected" "$observed" || fail 'host observations diverged from Sounio semantic authority'

material_result="$(bash scripts/ci/sounio_loom_process_witness_effect_hypercube_v11_selftest.sh)"
[[ "$material_result" == sounio-loom-process-witness-effect-hypercube-v11-selftest:\ PASS* ]] ||
  fail 'source-fresh local material-parity gate failed'
[[ "$material_result" == *'semantic_decision=false'* &&
   "$material_result" == *'material_hypercube=false material_coverage=false'* ]] ||
  fail 'local C++ parity promoted itself to semantic or host authority'

printf 'sounio-loom-process-witness-effect-hypercube-v11-freeze-selftest: PASS semantic_authority=Sounio material_producer=C++20 role=MATERIAL_PARITY action=9025 evidence_sha256=%s certificate_bundle_sha256=%s host=t560-proxmox kernel=7.0.2-5-pve families=12 probes=13 mechanism_dimensions=18 vertices=40 refusals=25 completions=15 extinctions=15 mincuts=13 invariant_stable=true delta_distinct=true triple_hash_binding=true material_hypercube=true material_coverage=false complete_effects=false material_execution=false action_9025_judged=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false claim_ready=false\n' \
  "$(file_hash "$EVIDENCE")" "$(field normalized_vertex_certificate_bundle_sha256)"
