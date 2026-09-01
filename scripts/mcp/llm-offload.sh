#!/usr/bin/env bash
# llm-offload — Fan-out a prompt to multiple LLM providers
# Usage: ./scripts/mcp/llm-offload.sh <prompt-file> [providers...]
# If no providers specified, fans out to the default 5 (diverse consensus set).
#
# Available providers:
#   deepseek     — DeepSeek V4 Pro (reasoning; 'deepseek-coder' silently
#                  resolved to the weaker v4-flash and is no longer a listed model)
#   xai|grok     — Grok 4.6 (fixed primary adversarial math/review lane)
#   xai-fast     — compatibility alias; resolves to the same fixed Grok 4.6
#   zai|glm      — Z.AI GLM-5.2 direct (independent math/review provider)
#   local        — a LOCAL OpenAI-compatible endpoint (Ollama/vLLM/llama.cpp/LM Studio):
#                  set LOCAL_LLM_URL (with the /v1 prefix) and LOCAL_LLM_MODEL
#   local2       — a second local endpoint (LOCAL2_LLM_URL / LOCAL2_LLM_MODEL), so a
#                  two-local fan-out can satisfy the two-provider review policy
#   grok-code    — compatibility alias; resolves to the same fixed Grok 4.6
#   groq         — Llama 3.3 70B on Groq (fast inference)
#   gemini       — Gemini 2.5 Pro via OpenRouter (1M ctx, best long-context)
#   qwen         — Qwen 3 235B via OpenRouter (strong math/code, Chinese perspective)
#   mistral      — Mistral Large via OpenRouter (formal methods, European)
#   llama        — Llama 4 Maverick via OpenRouter (diverse training)
#   cohere       — Command R+ via OpenRouter (structured analysis, lit review)
#   openrouter   — OpenRouter Auto (auto-routes to best model)
#   minimax      — MiniMax M2.7 (Anthropic-compat, long context)
#   all          — all distinct configured providers
#
# Keys read from env vars (set in ~/.sounio-keys.env):
#   DEEPSEEK_API_KEY, XAI_API_KEY, ZAI_API_KEY or ZHIPU_API_KEY,
#   GROQ_API_KEY, OPENROUTER_API_KEY, MINIMAX_API_KEY

set -euo pipefail

readonly XAI_GROK_BASE_URL="https://api.x.ai/v1"
readonly XAI_GROK_MODEL="grok-4.6"
readonly XAI_GROK_NAME="Grok 4.6"
readonly -a XAI_GROK_ALIASES=(xai grok xai-fast grok-code)

canonical_provider() {
    local alias
    for alias in "${XAI_GROK_ALIASES[@]}"; do
        if [[ "$1" == "$alias" ]]; then
            echo "xai"
            return
        fi
    done
    case "$1" in
        zai|glm|zhipu) echo "zai" ;;
        *)             echo "$1" ;;
    esac
}

provider_is_known() {
    case "$1" in
        deepseek|xai|zai|local|local1|local2|groq|gemini|qwen|mistral|llama|cohere|openrouter|minimax)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

normalize_providers() {
    local provider canonical existing duplicate
    local -a expanded=() normalized=()

    for provider in "$@"; do
        if [[ "$provider" == "all" ]]; then
            expanded+=(deepseek xai zai groq gemini qwen mistral llama cohere openrouter minimax)
        else
            expanded+=("$provider")
        fi
    done

    for provider in "${expanded[@]}"; do
        canonical="$(canonical_provider "$provider")"
        if ! provider_is_known "$canonical"; then
            echo "ERROR: unknown provider '$provider'" >&2
            return 2
        fi
        duplicate=false
        for existing in "${normalized[@]}"; do
            if [[ "$existing" == "$canonical" ]]; then
                duplicate=true
                break
            fi
        done
        if [[ "$duplicate" == true ]]; then
            echo "ERROR: provider '$provider' duplicates canonical review leg '$canonical'" >&2
            return 2
        fi
        normalized+=("$canonical")
    done

    echo "${normalized[*]}"
}

if [[ "${1:-}" == "--resolve-provider" ]]; then
    provider="${2:-}"
    canonical="$(canonical_provider "$provider")"
    if [[ "$canonical" != "xai" ]]; then
        echo "ERROR: no fixed resolution is exposed for provider '$provider'" >&2
        exit 2
    fi
    printf 'provider=%s canonical=%s model=%s endpoint=%s\n' \
        "$provider" "$canonical" "$XAI_GROK_MODEL" "$XAI_GROK_BASE_URL"
    exit 0
fi

if [[ "${1:-}" == "--resolve-provider-set" ]]; then
    shift
    [[ $# -gt 0 ]] || {
        echo "ERROR: --resolve-provider-set requires at least one provider" >&2
        exit 2
    }
    normalized_providers="$(normalize_providers "$@")" || exit $?
    printf 'providers=%s\n' "$normalized_providers"
    exit 0
fi

PROMPT_FILE="${1:?Usage: llm-offload.sh <prompt-file> [provider1 provider2 ...]}"
shift
PROVIDERS=("$@")

OUTDIR="$(mktemp -d /tmp/llm-offload-XXXXXX)"
echo "=== LLM Offload Fan-Out ==="
echo "Output dir: $OUTDIR"

# Load keys. Search order: explicit override, per-agent HOME, then the shared
# openvscode-server home (canonical location where all Sounio keys live).
for _keyfile in "${SOUNIO_KEYS_ENV:-}" "$HOME/.sounio-keys.env" "/workspace/.home/openvscode-server/.sounio-keys.env"; do
    if [[ -n "$_keyfile" && -f "$_keyfile" ]]; then
        source "$_keyfile"
        break
    fi
done

PROMPT="$(cat "$PROMPT_FILE")"

call_openai_compat() {
    local name="$1" url="$2" key="$3" model="$4" outfile="$5"
    local max_tok="${OFFLOAD_MAX_TOKENS:-8192}"
    # Expensive models get fewer tokens to stay within credits (unless overridden).
    if [[ -z "${OFFLOAD_MAX_TOKENS:-}" ]]; then
        case "$model" in
            google/gemini-2.5-pro*) max_tok=2048 ;;
            mistralai/mistral-large*) max_tok=3000 ;;
            cohere/command-a*) max_tok=2048 ;;
            glm-5*|glm-4.7*|*reasoning*) max_tok=8192 ;;  # raise explicitly with OFFLOAD_MAX_TOKENS for deep audits
            *) max_tok=8192 ;;
        esac
    fi
    echo "  -> Sending to $name ($model, max=$max_tok)..."
    # Local reasoning models are slow; give them room rather than losing the leg.
    local _tmo="${OFFLOAD_TIMEOUT:-180}"
    case "$name" in Local*) _tmo="${OFFLOAD_TIMEOUT:-600}" ;; esac
    curl -s -m "$_tmo" "$url/chat/completions" \
        -H "Authorization: Bearer $key" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg model "$model" --arg prompt "$PROMPT" --argjson maxtok "$max_tok" '{
            model: $model,
            messages: [{role: "user", content: $prompt}],
            max_tokens: $maxtok,
            temperature: 0.7
        }')" > "$outfile" 2>&1 || true
    # `|| true` is load-bearing under `set -e`: curl exits non-zero on a TIMEOUT or a
    # connection failure (unlike an HTTP error, where it exits 0 with a JSON body), and
    # without it the whole background subshell dies right here — no "<- name: ERROR" line,
    # no mention of the provider at all.  The fan-out then prints a clean Results section
    # and exits 0 having silently lost a leg.  Measured 2026-08-24 with a local reasoning
    # model that needed longer than the 180 s cap.

    # Prefer .content; fall back to .reasoning_content for reasoning models
    # (e.g. Z.AI GLM-5.x) that leave .content empty. Treat empty output as error.
    if jq -e '(.choices[0].message.content // "") != "" or (.choices[0].message.reasoning_content // "") != ""' "$outfile" > /dev/null 2>&1; then
        jq -r 'if (.choices[0].message.content // "") != "" then .choices[0].message.content else .choices[0].message.reasoning_content end' "$outfile" > "${outfile%.json}.md"
        echo "  <- $name: DONE ($(wc -c < "${outfile%.json}.md") bytes)"
    else
        if [[ ! -s "$outfile" ]]; then
            echo "  <- $name: EMPTY after ${_tmo}s — timeout or unreachable endpoint (raise OFFLOAD_TIMEOUT)"
        else
            echo "  <- $name: ERROR (see $outfile)"
        fi
    fi
}

run_provider() {
    local p
    p="$(canonical_provider "$1")"
    case "$p" in
        deepseek)
            [[ -n "${DEEPSEEK_API_KEY:-}" ]] && \
            call_openai_compat "DeepSeek" "https://api.deepseek.com" "$DEEPSEEK_API_KEY" "deepseek-v4-pro" "$OUTDIR/deepseek.json"
            ;;
        xai|grok)
            [[ -n "${XAI_API_KEY:-}" ]] && \
            call_openai_compat "$XAI_GROK_NAME" "$XAI_GROK_BASE_URL" "$XAI_API_KEY" "$XAI_GROK_MODEL" "$OUTDIR/grok.json"
            ;;
        zai|glm|zhipu)
            # Z.AI (Zhipu) direct, OpenAI-compatible. Accepts ZAI_API_KEY or ZHIPU_API_KEY.
            # Uses the Coding Plan endpoint (/api/coding/paas/v4) — subscription plans return
            # 1113 "insufficient balance" on the pay-as-you-go /api/paas/v4 path. Override the
            # base URL with ZAI_BASE_URL if your account uses pay-as-you-go credits instead.
            # Falls back to OpenRouter (z-ai/glm-4.6) when only an OpenRouter key is present.
            if [[ -n "${ZAI_API_KEY:-${ZHIPU_API_KEY:-}}" ]]; then
                call_openai_compat "Z.AI GLM-5.2 (coding plan)" "${ZAI_BASE_URL:-https://api.z.ai/api/coding/paas/v4}" "${ZAI_API_KEY:-$ZHIPU_API_KEY}" "${ZAI_MODEL:-glm-5.2}" "$OUTDIR/zai.json"
            elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
                call_openai_compat "Z.AI GLM-4.6 (via OpenRouter)" "https://openrouter.ai/api/v1" "$OPENROUTER_API_KEY" "z-ai/glm-4.6" "$OUTDIR/zai.json"
            else
                echo "  <- Z.AI: SKIPPED (set ZAI_API_KEY or ZHIPU_API_KEY, or fund OPENROUTER_API_KEY)"
            fi
            ;;
        groq)
            [[ -n "${GROQ_API_KEY:-}" ]] && \
            call_openai_compat "Groq/Llama" "https://api.groq.com/openai/v1" "$GROQ_API_KEY" "llama-3.3-70b-versatile" "$OUTDIR/groq.json"
            ;;
        gemini)
            [[ -n "${OPENROUTER_API_KEY:-}" ]] && \
            call_openai_compat "Gemini 2.5 Pro" "https://openrouter.ai/api/v1" "$OPENROUTER_API_KEY" "google/gemini-2.5-pro" "$OUTDIR/gemini.json"
            ;;
        qwen)
            [[ -n "${OPENROUTER_API_KEY:-}" ]] && \
            call_openai_compat "Qwen 3 235B" "https://openrouter.ai/api/v1" "$OPENROUTER_API_KEY" "qwen/qwen3-235b-a22b" "$OUTDIR/qwen.json"
            ;;
        mistral)
            [[ -n "${OPENROUTER_API_KEY:-}" ]] && \
            call_openai_compat "Mistral Large" "https://openrouter.ai/api/v1" "$OPENROUTER_API_KEY" "mistralai/mistral-large" "$OUTDIR/mistral.json"
            ;;
        llama)
            [[ -n "${OPENROUTER_API_KEY:-}" ]] && \
            call_openai_compat "Llama 4 Maverick" "https://openrouter.ai/api/v1" "$OPENROUTER_API_KEY" "meta-llama/llama-4-maverick" "$OUTDIR/llama.json"
            ;;
        cohere)
            [[ -n "${OPENROUTER_API_KEY:-}" ]] && \
            call_openai_compat "Command A" "https://openrouter.ai/api/v1" "$OPENROUTER_API_KEY" "cohere/command-a" "$OUTDIR/cohere.json"
            ;;
        openrouter)
            [[ -n "${OPENROUTER_API_KEY:-}" ]] && \
            call_openai_compat "OpenRouter Auto" "https://openrouter.ai/api/v1" "$OPENROUTER_API_KEY" "openrouter/auto" "$OUTDIR/openrouter.json"
            ;;
        minimax)
            [[ -n "${MINIMAX_API_KEY:-}" ]] && \
            call_openai_compat "MiniMax M2.7" "https://api.minimax.io/v1" "$MINIMAX_API_KEY" "MiniMax-M2.7" "$OUTDIR/minimax.json"
            ;;
        local|local1|local2)
            # LOCAL endpoints (Ollama / vLLM / llama.cpp / LM Studio — all OpenAI-compatible).
            # Configure with LOCAL_LLM_URL (must end in the OpenAI-compatible prefix, e.g.
            # http://host:11434/v1) and LOCAL_LLM_MODEL.  LOCAL_LLM_KEY is optional; most local
            # servers ignore it but curl still needs a bearer, so it defaults to "local".
            # A SECOND endpoint can be given as LOCAL2_LLM_URL / LOCAL2_LLM_MODEL, so that a
            # fan-out of two independent local models satisfies the two-provider review policy.
            local _u _m _k _tag
            if [[ "$p" == "local2" ]]; then
                _u="${LOCAL2_LLM_URL:-}"; _m="${LOCAL2_LLM_MODEL:-}"; _k="${LOCAL2_LLM_KEY:-local}"; _tag="local2"
            else
                _u="${LOCAL_LLM_URL:-}"; _m="${LOCAL_LLM_MODEL:-}"; _k="${LOCAL_LLM_KEY:-local}"; _tag="local"
            fi
            if [[ -n "$_u" && -n "$_m" ]]; then
                call_openai_compat "Local $_m" "$_u" "$_k" "$_m" "$OUTDIR/$_tag.json"
            else
                echo "  <- Local ($_tag): SKIPPED (set ${_tag^^}_LLM_URL and ${_tag^^}_LLM_MODEL; URL must include the /v1 prefix)"
            fi
            ;;
        *)
            echo "  ?? Unknown provider: $p" >&2
            return 2
            ;;
    esac
}

# Default: diverse consensus set (5 models, geographic diversity)
if [[ ${#PROVIDERS[@]} -eq 0 ]]; then
    PROVIDERS=(deepseek xai gemini qwen mistral)
fi

# Canonicalize compatibility aliases before fan-out. Multiple names for Grok 4.6
# are one review leg, so a request that repeats that leg fails closed.
normalized_providers="$(normalize_providers "${PROVIDERS[@]}")" || exit $?
read -ra PROVIDERS <<< "$normalized_providers"

echo "Providers: ${PROVIDERS[*]}"
echo ""

# Fan out in parallel
for p in "${PROVIDERS[@]}"; do
    run_provider "$p" &
done

wait
echo ""
echo "=== Results ==="
for f in "$OUTDIR"/*.md; do
    [[ -f "$f" ]] || continue
    name="$(basename "$f" .md)"
    echo ""
    echo "━━━ $name ━━━"
    cat "$f"
    echo ""
done

echo "Raw JSON: $OUTDIR/"
