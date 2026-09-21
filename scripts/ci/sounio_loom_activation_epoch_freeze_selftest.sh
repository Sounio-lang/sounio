#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
M="$ROOT/tools/loom/activation_epoch.freeze.v2"
fail(){ printf 'sounio-loom-activation-epoch-freeze-selftest: FAIL: %s\n' "$*" >&2; exit 1; }
value(){ local n; n="$(grep -c "^$1=" "$M" || true)"; [[ $n == 1 ]] || fail "$1 count $n"; sed -n "s/^$1=//p" "$M"; }
expect(){ [[ "$(value "$1")" == "$2" ]] || fail "$1 drifted"; }
verify(){ local p e; p="$(value "$1_path")"; e="$(value "$1_sha256")"; [[ -f "$ROOT/$p" && "$(sha256sum "$ROOT/$p"|cut -d' ' -f1)" == "$e" ]] || fail "$p drifted"; }
expect schema loom-activation-epoch-freeze-v2
expect stage SEMANTICS_FROZEN
expect semantic_authority Sounio
expect action 9049
expect parent_freeze_sha256 0f29211004af425cd9946f35be8c94a5b2f44a1758a22066a88a410bb13baef4
expect append_only_predecessor_required true
expect immutable_pin_set_required true
expect atomic_compatibility_head_required true
expect fail_closed_required true
expect python_oracle_attempt DENY717
expect python_executed false
expect rust_executed false
expect disposable_oracle_executed false
expect change_class ENTRYPOINT_INPUT_ROBUSTNESS
expect semantics_module_changed false
verify predecessor_manifest
[[ "$(value source_sha256)" == "$(sed -n 's/^source_sha256=//p' "$ROOT/$(value predecessor_manifest_path)")" ]] ||
  fail 'semantic module differs from predecessor'
for k in parent_freeze garden source entrypoint build_script selftest first_manifest first_evidence; do verify "$k"; done
[[ "$(cat "$ROOT/$(value source_path)" "$ROOT/$(value entrypoint_path)"|sha256sum|cut -d' ' -f1)" == "$(value semantics_sha256)" ]] || fail 'semantics drifted'
work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-activation-freeze.XXXXXX")"; trap 'rm -rf "$work"' EXIT
for n in one two; do SOUNIO_LOOM_ACTIVATION_EPOCH_OUTPUT="$work/$n" bash "$ROOT/scripts/dev/build_sounio_loom_activation_epoch.sh" >/dev/null; done
cmp "$work/one" "$work/two" || fail 'nondeterministic executable'
[[ "$(sha256sum "$work/one"|cut -d' ' -f1)" == "$(value executable_sha256)" ]] || fail 'executable drifted'
printf 'sounio-loom-activation-epoch-freeze-selftest: PASS semantic_authority=Sounio action=9049 stage=SEMANTICS_FROZEN deterministic=true python_executed=false rust_executed=false parity_open=false claim_ready=false\n'
