#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-hypercube-root-v11.XXXXXX")"
ROOT_ONE="$TEST_ROOT/root-one"
ROOT_TWO="$TEST_ROOT/root-two"

cleanup() {
  chmod -R u+w "$TEST_ROOT" >/dev/null 2>&1 || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-effect-hypercube-root-v11-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for output in "$ROOT_ONE" "$ROOT_TWO"; do
  SOUNIO_LOOM_EFFECT_HYPERCUBE_ROOT_V11_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_loom_process_witness_effect_hypercube_root_v11.sh" \
      >/dev/null
done
diff -r "$ROOT_ONE" "$ROOT_TWO" >/dev/null ||
  fail 'two source-fresh V11 root capsules differ'

expected_paths='./dev
./loom
./loom/effect-cell
./loom/effect-policy-v11.freeze.v1
./proc
./run
./run/systemd
./run/systemd/incoming
./sys
./tmp
./var
./var/tmp'
actual_paths="$(cd "$ROOT_ONE" && find . -mindepth 1 -printf '%p\n' | sort)"
[[ "$actual_paths" == "$expected_paths" ]] || fail 'V11 capsule path set drifted'
for directory in "$ROOT_ONE" "$ROOT_ONE/loom" "$ROOT_ONE/dev" \
                 "$ROOT_ONE/proc" "$ROOT_ONE/tmp" "$ROOT_ONE/run" \
                 "$ROOT_ONE/run/systemd" "$ROOT_ONE/run/systemd/incoming" \
                 "$ROOT_ONE/sys" "$ROOT_ONE/var" "$ROOT_ONE/var/tmp"; do
  [[ "$(stat -c '%F:%a' "$directory")" == 'directory:555' ]] ||
    fail "V11 root directory mode drifted: $directory"
done
[[ "$(stat -c '%F:%a:%h' "$ROOT_ONE/loom/effect-cell")" == \
   'regular file:555:1' ]] || fail 'V11 material cell metadata drifted'
[[ "$(stat -c '%F:%a:%h' "$ROOT_ONE/loom/effect-policy-v11.freeze.v1")" == \
   'regular file:444:1' ]] || fail 'V11 policy manifest metadata drifted'
if readelf -l "$ROOT_ONE/loom/effect-cell" | grep -q 'INTERP'; then
  fail 'V11 root material cell retained a dynamic interpreter'
fi
[[ "$(sha256sum "$ROOT_ONE/loom/effect-policy-v11.freeze.v1" | cut -d ' ' -f 1)" == \
  adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c ]] ||
  fail 'V11 capsule policy manifest drifted'

native="$($ROOT_ONE/loom/effect-cell --selftest \
  --policy-manifest "$ROOT_ONE/loom/effect-policy-v11.freeze.v1")"
[[ "$native" == LOOM_PROCESS_WITNESS_EFFECT_HYPERCUBE_V11_SELFTEST\ PASS* &&
   "$native" == *'compiled_filters=12'* &&
   "$native" == *'semantic_decision=false'* &&
   "$native" == *'material_hypercube=false material_coverage=false'* ]] ||
  fail 'V11 capsule material apparatus gate diverged'

dependencies="$(ldd "$ROOT_ONE/loom/effect-cell" 2>&1 || true)"
if ! printf '%s\n' "$dependencies" | grep -Eq 'not a dynamic executable|statically linked'; then
  fail 'V11 capsule did not prove static linkage'
fi
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'V11 capsule has a prohibited runtime dependency'
fi

printf 'sounio-loom-process-witness-effect-hypercube-root-v11-selftest: PASS semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY transitory=true semantic_decision=false action=9025 capsule_paths=12 cell_sha256=%s static_cell=true self_exec_target=true deterministic=true families=12 probes=13 vertices=40 dev_null=host_materialization_required host_root_ownership=false root_read_only=false host_gate_required=true material_hypercube=false material_coverage=false complete_effects=false material_execution=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_ONE/loom/effect-cell" | cut -d ' ' -f 1)"
