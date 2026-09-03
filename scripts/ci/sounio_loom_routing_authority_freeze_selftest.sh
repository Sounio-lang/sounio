#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/routing_authority.freeze.v1"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-routing-freeze.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT
fail() { printf 'sounio-loom-routing-authority-freeze-selftest: FAIL: %s\n' "$*" >&2; exit 1; }
field() { sed -n "s/^$1=//p" "$MANIFEST"; }
hash() { sha256sum "$1" | awk '{print $1}'; }

[[ -f "$MANIFEST" ]] || fail 'freeze manifest is missing'
[[ "$(field stage)" == SEMANTICS_FROZEN ]] || fail 'stage is not SEMANTICS_FROZEN'
[[ "$(field action)" == 9032 ]] || fail 'wrong action'
[[ "$(field producing_language)" == Sounio ]] || fail 'producer is not Sounio'
[[ "$(field language_role)" == SEMANTIC_AUTHORITY ]] || fail 'role is not SEMANTIC_AUTHORITY'
[[ "$(field parity_open)" == false && "$(field claim_ready)" == false ]] || fail 'freeze improperly promoted parity or claims'
[[ "$(field python_executed)" == false && "$(field rust_executed)" == false ]] || fail 'prohibited language execution recorded'

source_path="$ROOT_DIR/$(field source_path)"
entrypoint_path="$ROOT_DIR/$(field entrypoint_path)"
[[ "$(hash "$source_path")" == "$(field source_sha256)" ]] || fail 'source hash drifted'
[[ "$(hash "$entrypoint_path")" == "$(field entrypoint_sha256)" ]] || fail 'entrypoint hash drifted'
sed -n '1,$p' "$source_path" "$entrypoint_path" > "$TEST_ROOT/semantics.sio"
[[ "$(hash "$TEST_ROOT/semantics.sio")" == "$(field semantics_sha256)" ]] || fail 'semantics bundle drifted'
[[ "$(hash "$ROOT_DIR/$(field build_script_path)")" == "$(field build_script_sha256)" ]] || fail 'build script drifted'
[[ "$(hash "$ROOT_DIR/$(field semantic_gate_path)")" == "$(field semantic_gate_sha256)" ]] || fail 'semantic gate drifted'
[[ "$(hash "$ROOT_DIR/$(field toolchain_wrapper_path)")" == "$(field toolchain_wrapper_sha256)" ]] || fail 'compiler wrapper drifted'
[[ "$(hash "$ROOT_DIR/$(field toolchain_compiler_path)")" == "$(field toolchain_compiler_sha256)" ]] || fail 'compiler binary drifted'
[[ "$(hash "$ROOT_DIR/$(field language_authority_manifest_path)")" == "$(field language_authority_manifest_sha256)" ]] || fail 'language authority parent drifted'

SOUNIO_LOOM_ROUTING_OUTPUT="$TEST_ROOT/a" bash "$ROOT_DIR/$(field build_script_path)" >/dev/null
SOUNIO_LOOM_ROUTING_OUTPUT="$TEST_ROOT/b" bash "$ROOT_DIR/$(field build_script_path)" >/dev/null
actual_a="$(hash "$TEST_ROOT/a")"
actual_b="$(hash "$TEST_ROOT/b")"
[[ "$actual_a" == "$actual_b" ]] || fail 'rebuilds are nondeterministic'
[[ "$actual_a" == "$(field executable_sha256)" ]] || fail 'executable hash drifted'

result="$(bash "$ROOT_DIR/$(field semantic_gate_path)")"
[[ "$result" == "$(field result)" ]] || fail "semantic result drifted: $result"
[[ "$(printf '%s' "$result" | sha256sum | awk '{print $1}')" == "$(field result_sha256)" ]] || fail 'result hash drifted'

printf '%s\n' "sounio-loom-routing-authority-freeze-selftest: PASS action=9032 semantics=$(field semantics_sha256) executable=$actual_a cases=$(field expected_result_cases) parity_open=false claim_ready=false"
