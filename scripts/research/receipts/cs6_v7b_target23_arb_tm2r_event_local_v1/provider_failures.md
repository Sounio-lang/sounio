# Provider failures

- Z.AI GLM-5.2 was attempted three times, including the default dual-provider
  invocation and a focused single-provider retry.  Each attempt produced an
  empty `zai.json`; no mathematical opinion was returned.
- Qwen via OpenRouter returned HTTP 402 because the account has no credits.
- DeepSeek returned `Insufficient Balance`.
- Groq returned `Invalid API Key`.

The pre-execution review is therefore single-provider degraded and must be
re-reviewed independently when a second provider is restored.
