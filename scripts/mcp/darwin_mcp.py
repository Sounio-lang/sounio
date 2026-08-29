#!/usr/bin/env python3
"""
darwin_mcp.py — MCP server exposing DARWIN local models + triangle partners.

Tools:
  darwin_scout      — Qwen 2.5 7B on RTX 4000 Ada (port 30011), fast extraction
  darwin_code       — Qwen 2.5 Coder 14B on L4 (port 30010), mechanical code work
  triangle_refute   — GPT-5 via LiteLLM gateway (port 4000), IC pass 2 critique
  triangle_formalise — Gemini 2.5 Pro via LiteLLM gateway (port 4000), IC pass 3 polish

Prerequisites:
  pip install "mcp[cli]" httpx
  DARWIN models running via scripts/mcp/darwin-sglang-launch.sh
  LiteLLM running via: litellm --config scripts/mcp/litellm-triangle.yaml --port 4000
"""

import asyncio
import os
import httpx
from mcp.server.fastmcp import FastMCP

DARWIN_HOST = os.environ.get("DARWIN_HOST", "darwin.internal")
SGLANG_CODER_PORT = int(os.environ.get("DARWIN_CODER_PORT", "30010"))
SGLANG_SCOUT_PORT = int(os.environ.get("DARWIN_SCOUT_PORT", "30011"))
LITELLM_HOST = os.environ.get("LITELLM_HOST", "localhost")
LITELLM_PORT = int(os.environ.get("LITELLM_PORT", "4000"))

mcp = FastMCP("darwin")


async def call_openai_compat(host: str, port: int, model: str, prompt: str, timeout: float = 120.0) -> str:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except httpx.ConnectError:
        return f"[darwin_mcp] ERROR: Could not connect to {host}:{port}. Is the service running?"
    except httpx.TimeoutException:
        return f"[darwin_mcp] ERROR: Timeout after {timeout}s calling {host}:{port}."
    except Exception as e:
        return f"[darwin_mcp] ERROR: {type(e).__name__}: {e}"


@mcp.tool()
async def darwin_scout(prompt: str) -> str:
    """Fast extraction, enumeration, classification, summarisation using local
    Qwen 2.5 7B on DARWIN (RTX 4000 Ada, port 30011). Zero Anthropic quota.

    ROUTING: Use for Rule 1 tasks — signature extraction, file enumeration,
    symbol search, caller lookup, bibliographic triage. Do NOT use for tasks
    requiring Sounio semantic reasoning or proof work.
    """
    return await call_openai_compat(DARWIN_HOST, SGLANG_SCOUT_PORT, "darwin-scout", prompt)


@mcp.tool()
async def darwin_code(prompt: str) -> str:
    """Mechanical code work using local Qwen 2.5 Coder 14B on DARWIN (L4 24GB,
    port 30010). Zero Anthropic quota.

    ROUTING: Use for Rule 2 tasks — docstring generation, style lint, mechanical
    refactor to an already-specified diff, test stub generation, bulk rename.
    Do NOT use for tasks requiring novel Sounio semantic reasoning.
    """
    return await call_openai_compat(DARWIN_HOST, SGLANG_CODER_PORT, "darwin-coder", prompt)


@mcp.tool()
async def triangle_refute(draft: str) -> str:
    """Submit a draft to the Refuter (GPT-5, low temperature) for structured
    critique. Use exactly at Iterative Convergence pass 2.

    Returns a structured critique: (1) logical gaps, (2) unsupported claims,
    (3) alternative framings. Do not use for IC pass 1 or 3.
    """
    return await call_openai_compat(LITELLM_HOST, LITELLM_PORT, "refuter", draft, timeout=180.0)


@mcp.tool()
async def triangle_formalise(draft: str) -> str:
    """Submit a draft to the Formaliser (Gemini 2.5 Pro, medium temperature)
    for mechanical polish. Use exactly at Iterative Convergence pass 3.

    Returns: polished prose, LaTeX-clean notation, tightened argument flow.
    Do not use for pass 1 or 2.
    """
    return await call_openai_compat(LITELLM_HOST, LITELLM_PORT, "formaliser", draft, timeout=180.0)


if __name__ == "__main__":
    mcp.run()
