#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLI="$ROOT/bin/llm-offload"
DRIVER="$ROOT/scripts/mcp/llm-offload.sh"
ROUTING="$ROOT/.claude/offload-routing.md"
POLICY="$ROOT/.claude/AGENT_OFFLOAD_POLICY.md"
EXPECTED_MODEL="grok-4.6"
EXPECTED_ENDPOINT="https://api.x.ai/v1"

fail() {
    echo "LLM_OFFLOAD_GROK_IDENTITY_SELFTEST_FAIL $*" >&2
    exit 1
}

bash -n "$CLI" "$DRIVER"

for provider in xai grok xai-fast grok-code; do
    expected="provider=$provider canonical=xai model=$EXPECTED_MODEL endpoint=$EXPECTED_ENDPOINT"
    observed="$(XAI_GROK_MODEL=grok-4.3 "$CLI" --resolve-provider "$provider")"
    [[ "$observed" == "$expected" ]] || \
        fail "provider=$provider expected='$expected' observed='$observed'"
done

for provider in xai grok xai-fast grok-code; do
    observed_set="$("$CLI" --resolve-provider-set "$provider")"
    [[ "$observed_set" == "providers=xai" ]] || \
        fail "provider=$provider was not canonicalized to one xAI leg: $observed_set"
done

if "$CLI" --resolve-provider-set xai grok xai-fast grok-code >/dev/null 2>&1; then
    fail "duplicate Grok aliases were accepted as independent review legs"
fi
if "$CLI" --resolve-provider-set zai glm zhipu >/dev/null 2>&1; then
    fail "duplicate Z.AI aliases were accepted as independent review legs"
fi
if "$CLI" --resolve-provider-set unknown-provider >/dev/null 2>&1; then
    fail "unknown provider was accepted"
fi

if grep -En 'grok-4\.3|grok-4-1-fast-reasoning|grok-code-fast-1' "$DRIVER"; then
    fail "legacy Grok model identifier remains in the active driver"
fi

model_declarations="$(grep -Ec '^readonly XAI_GROK_MODEL="grok-4\.6"$' "$DRIVER" || true)"
[[ "$model_declarations" == "1" ]] || \
    fail "fixed Grok model declaration is missing or duplicated"

grep -Fq 'xai (grok-4.6, fixed)' "$ROUTING" || \
    fail "routing documentation does not name the fixed Grok 4.6 identity"
grep -Fq 'the fixed `grok-4.6` model' "$POLICY" || \
    fail "offload policy does not require the fixed Grok 4.6 identity"

if [[ "${1:-}" == "--live" ]]; then
    live_output="$(
        printf '%s\n' 'Return exactly: GROK_4_6_IDENTITY_OK. Do not add explanation.' |
            OFFLOAD_MAX_TOKENS=64 OFFLOAD_TIMEOUT="${OFFLOAD_TIMEOUT:-180}" \
            "$CLI" -t review -p xai
    )"
    grep -Fq 'Sending to Grok 4.6 (grok-4.6' <<< "$live_output" || \
        fail "live request did not declare the fixed Grok 4.6 identity"
    grep -Fxq 'GROK_4_6_IDENTITY_OK' <<< "$live_output" || \
        fail "live Grok 4.6 identity response was absent"
elif [[ $# -gt 0 ]]; then
    fail "unknown argument: $1"
fi

echo "LLM_OFFLOAD_GROK_IDENTITY_SELFTEST_PASS model=$EXPECTED_MODEL aliases=4 duplicate_alias_sets=denied override=denied live=${1:-not_run}"
