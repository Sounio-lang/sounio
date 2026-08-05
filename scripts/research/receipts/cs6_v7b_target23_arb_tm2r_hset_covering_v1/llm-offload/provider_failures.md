# Independent-provider failure receipt

The mandatory default fan-out was attempted on 2026-08-05. The xAI leg
completed and is retained in `math_review_xai.md`. Every available independent
fallback failed at the provider boundary before returning a review:

```text
zai/GLM-5.2:
  code 1308: Usage limit reached for 5 hour.
  raw directory: /tmp/llm-offload-KlberH

qwen/Qwen 3 235B:
  HTTP 402: Insufficient credits.
  raw directory: /tmp/llm-offload-3pxZfM

deepseek/deepseek-coder:
  Insufficient Balance.
  raw directory: /tmp/llm-offload-YR5M5s

groq/Llama 3.3 70B:
  Invalid API Key.
  raw directory: /tmp/llm-offload-a89LGN
```

Disposition: `PASS_SINGLE_PROVIDER_DEGRADED`. The artifact is explicitly
flagged for independent re-review in the next session with restored provider
availability. No unavailable provider is counted as a mathematical pass.
