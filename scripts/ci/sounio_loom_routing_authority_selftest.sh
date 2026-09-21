#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-routing-selftest.XXXXXX")"
RUNTIME="$TEST_ROOT/sounio-routing-authority"
trap 'rm -rf "$TEST_ROOT"' EXIT
fail() { printf 'sounio-loom-routing-authority-selftest: FAIL: %s\n' "$*" >&2; exit 1; }

SOUNIO_LOOM_ROUTING_OUTPUT="$RUNTIME" bash "$ROOT_DIR/scripts/dev/build_sounio_loom_routing_authority.sh" >/dev/null
[[ "$(printf '0\n' | "$RUNTIME")" == 'SOUNIO_ROUTING_AUTHORITY_SELFTEST PASS cases=29' ]] || fail 'Sounio selftest failed'

# schema stage op policy frozen config task candidate observer fresh index predecessor-chain
# predecessor-denials ownership quota allow-estimated pool adapter model operational-language
# operational-role provider-language provider-role promoted plan-bound plan-fresh plan-match
# provider-plan exec-grant receipt-bound
plan='9032 3 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 9 8 6 6 0 0 0 0 0 0 0'
dispatch='9032 3 2 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 9 8 6 6 0 1 1 1 1 1 1'
python_oracle='9032 3 2 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 7 8 6 6 0 1 1 1 1 1 1'

[[ "$(printf '%s\n' "$plan" | "$RUNTIME")" == *'ALLOW code=0'* ]] || fail 'positive PLAN denied'
[[ "$(printf '%s\n' "$dispatch" | "$RUNTIME")" == *'ALLOW code=0'* ]] || fail 'positive DISPATCH denied'
[[ "$(printf '%s\n' "$python_oracle" | "$RUNTIME")" == *'DENY code=617'* ]] || fail 'Python oracle was not refused before dispatch'
[[ "$(printf '9032 3 2\n' | "$RUNTIME")" == *'DENY code=605'* ]] || fail 'malformed frame did not fail closed'

# Causal control: removing only the ownership rule admits the unchanged block.
needle='if ownership_state != 1 { return 607 }'
module="$ROOT_DIR/stdlib/coordination/loom_routing_authority.sio"
[[ "$(grep -Fc "$needle" "$module")" -eq 1 ]] || fail 'ownership sabotage point is not unique'
sed "s/$needle//" "$module" > "$TEST_ROOT/module-sabotaged.sio"
sed -n '1,$p' "$TEST_ROOT/module-sabotaged.sio" "$ROOT_DIR/tools/loom/routing_authority_main.sio" > "$TEST_ROOT/runtime-sabotaged.sio"
SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" compile "$TEST_ROOT/runtime-sabotaged.sio" -o "$TEST_ROOT/sabotaged" >/dev/null
chmod 0755 "$TEST_ROOT/sabotaged"
ownership_block='9032 3 1 1 1 1 1 1 1 1 0 1 1 2 1 0 1 1 1 9 8 6 6 0 0 0 0 0 0 0'
[[ "$(printf '%s\n' "$ownership_block" | "$RUNTIME")" == *'DENY code=607'* ]] || fail 'ownership block not denied'
[[ "$(printf '%s\n' "$ownership_block" | "$TEST_ROOT/sabotaged")" == *'ALLOW code=0'* ]] || fail 'causal sabotage did not admit blocked ownership'

printf '%s\n' 'sounio-loom-routing-authority-selftest: PASS language=Sounio action=9032 cases=29 plan=ALLOW dispatch=ALLOW python_oracle=DENY617 malformed=DENY605 sabotage_ownership=admits'
