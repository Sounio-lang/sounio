#!/usr/bin/env bash
# llm-offload — Fan-out a prompt to multiple LLM providers
# Usage: ./scripts/llm-offload.sh <prompt-file> [providers...]
# If no providers specified, fans out to the default 5 (diverse consensus set).
#
# Available providers:
#   deepseek     — DeepSeek V4 Pro (reasoning; 'deepseek-coder' silently
#                  resolved to the weaker v4-flash and is no longer a listed model)
#   xai|grok     — Grok 4.3 (primary adversarial math/review lane)
#   xai-fast     — Grok 4.1 Fast Reasoning (lower-latency fallback)
#   zai|glm      — Z.AI GLM-5.2 direct (independent math/review provider)
#   grok-code    — Grok Code Fast 1 (fast code tasks)
#   groq         — Llama 3.3 70B on Groq (fast inference)
#   gemini       — Gemini 2.5 Pro via OpenRouter (1M ctx, best long-context)
#   qwen         — Qwen 3 235B via OpenRouter (strong math/code, Chinese perspective)
#   mistral      — Mistral Large via OpenRouter (formal methods, European)
#   llama        — Llama 4 Maverick via OpenRouter (diverse training)
#   cohere       — Command R+ via OpenRouter (structured analysis, lit review)
#   openrouter   — OpenRouter Auto (auto-routes to best model)
#   minimax      — MiniMax M2.7 (Anthropic-compat, long context)
#   all          — ALL providers (13 models)
#
# Keys read from env vars (set in ~/.sounio-keys.env):
#   DEEPSEEK_API_KEY, XAI_API_KEY, ZAI_API_KEY or ZHIPU_API_KEY,
#   GROQ_API_KEY, OPENROUTER_API_KEY, MINIMAX_API_KEY

set -euo pipefail

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
    curl -s -m 180 "$url/chat/completions" \
        -H "Authorization: Bearer $key" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg model "$model" --arg prompt "$PROMPT" --argjson maxtok "$max_tok" '{
            model: $model,
            messages: [{role: "user", content: $prompt}],
            max_tokens: $maxtok,
            temperature: 0.7
        }')" > "$outfile" 2>&1

    # Prefer .content; fall back to .reasoning_content for reasoning models
    # (e.g. Z.AI GLM-5.x) that leave .content empty. Treat empty output as error.
    if jq -e '(.choices[0].message.content // "") != "" or (.choices[0].message.reasoning_content // "") != ""' "$outfile" > /dev/null 2>&1; then
        jq -r 'if (.choices[0].message.content // "") != "" then .choices[0].message.content else .choices[0].message.reasoning_content end' "$outfile" > "${outfile%.json}.md"
        echo "  <- $name: DONE ($(wc -c < "${outfile%.json}.md") bytes)"
    else
        echo "  <- $name: ERROR (see $outfile)"
    fi
}

run_provider() {
    local p="$1"
    case "$p" in
        deepseek)
            [[ -n "${DEEPSEEK_API_KEY:-}" ]] && \
            call_openai_compat "DeepSeek" "https://api.deepseek.com" "$DEEPSEEK_API_KEY" "deepseek-v4-pro" "$OUTDIR/deepseek.json"
            ;;
        xai|grok)
            [[ -n "${XAI_API_KEY:-}" ]] && \
            call_openai_compat "Grok 4.3" "https://api.x.ai/v1" "$XAI_API_KEY" "grok-4.3" "$OUTDIR/grok.json"
            ;;
        xai-fast)
            [[ -n "${XAI_API_KEY:-}" ]] && \
            call_openai_compat "Grok 4.1 fast" "https://api.x.ai/v1" "$XAI_API_KEY" "grok-4-1-fast-reasoning" "$OUTDIR/grok-fast.json"
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
        grok-code)
            [[ -n "${XAI_API_KEY:-}" ]] && \
            call_openai_compat "Grok Code" "https://api.x.ai/v1" "$XAI_API_KEY" "grok-code-fast-1" "$OUTDIR/grok-code.json"
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
        *)
            echo "  ?? Unknown provider: $p"
            ;;
    esac
}

# Expand "all" keyword
expand_providers() {
    local result=()
    for p in "$@"; do
        if [[ "$p" == "all" ]]; then
            result+=(deepseek xai zai grok-code groq gemini qwen mistral llama cohere openrouter minimax)
        else
            result+=("$p")
        fi
    done
    echo "${result[@]}"
}

# Default: diverse consensus set (5 models, geographic diversity)
if [[ ${#PROVIDERS[@]} -eq 0 ]]; then
    PROVIDERS=(deepseek xai gemini qwen mistral)
fi

# Expand "all"
PROVIDERS=($(expand_providers "${PROVIDERS[@]}"))

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
