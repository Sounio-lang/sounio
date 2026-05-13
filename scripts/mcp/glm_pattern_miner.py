#!/usr/bin/env python3
"""
LLM Pattern Miner for Sounio

Analyzes Sounio source files using one or more LLM providers to discover
optimization patterns. Results are stored as JSONL for human review and
eventual encoding into deterministic compiler heuristics.

Usage:
    python scripts/glm_pattern_miner.py [--providers glm,openai,deepseek] [--input-dir DIR] [--output-dir DIR]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

# Configuration
PROVIDER_CONFIG: Dict[str, Dict[str, str]] = {
    "glm": {
        "api_url": "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions",
        "default_model": "glm-4.7",
        "api_key_env": "GLM_API_KEY",
        "model_env": "GLM_PATTERN_MODEL",
        "reasoning_field": "reasoning_content",
    },
    "openai": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_PATTERN_MODEL",
        "reasoning_field": "",
    },
    "deepseek": {
        "api_url": "https://api.deepseek.com/chat/completions",
        "default_model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_PATTERN_MODEL",
        "reasoning_field": "reasoning_content",
    },
}
MAX_TOKENS = 1500
TEMPERATURE = 0.1


def parse_providers(raw_value: str) -> List[str]:
    """Parse provider selection from CLI input."""
    if raw_value.strip().lower() == "auto":
        return [
            provider
            for provider, cfg in PROVIDER_CONFIG.items()
            if os.environ.get(cfg["api_key_env"])
        ]

    providers = [provider.strip().lower() for provider in raw_value.split(",") if provider.strip()]
    unknown = [provider for provider in providers if provider not in PROVIDER_CONFIG]
    if unknown:
        supported = ", ".join(sorted(PROVIDER_CONFIG.keys()))
        raise ValueError(f"Unknown providers: {unknown}. Supported providers: {supported}")
    ordered_unique: List[str] = []
    for provider in providers:
        if provider not in ordered_unique:
            ordered_unique.append(provider)
    return ordered_unique


def get_api_key(provider: str) -> str:
    """Get provider API key from environment."""
    config = PROVIDER_CONFIG[provider]
    key_env = config["api_key_env"]
    key = os.environ.get(key_env)
    if not key:
        raise ValueError(f"{key_env} environment variable not set for provider '{provider}'")
    return key


def get_model(provider: str) -> str:
    """Get provider model from environment override or default."""
    config = PROVIDER_CONFIG[provider]
    return os.environ.get(config["model_env"], config["default_model"])


def parse_sounio_file(filepath: Path) -> Dict[str, Any]:
    """Parse a Sounio file and extract basic features."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')

    # Basic feature extraction
    features = {
        "filepath": str(filepath),
        "line_count": len(lines),
        "function_count": content.count("fn "),
        "struct_count": content.count("struct "),
        "knowledge_types": content.count("Knowledge<"),
        "loop_count": content.count("for ") + content.count("while "),
        "branch_count": content.count("if ") + content.count("match "),
        "has_effects": "with " in content,
        "has_units": any(unit in content for unit in ["mg", "kg", "ml", "mol", "Hz", "Pa"]),
        "has_refinement": "{" in content and "|" in content,
        "content": content[:2000],  # First 2000 chars for context
    }

    return features


def query_provider(
    provider: str,
    model: str,
    code_content: str,
    features: Dict[str, Any],
    api_key: str,
) -> Optional[Dict[str, Any]]:
    """Query selected provider for optimization suggestions."""

    prompt = f"""Analyze this Sounio code and suggest compiler optimizations.

Code:
```sio
{code_content[:1500]}
```

Code Features:
- Functions: {features['function_count']}
- Structs: {features['struct_count']}
- Knowledge<T> types: {features['knowledge_types']}
- Loops: {features['loop_count']}
- Branches: {features['branch_count']}
- Has effects: {features['has_effects']}
- Has units: {features['has_units']}
- Has refinement types: {features['has_refinement']}

Suggest 2-3 specific compiler optimizations. Focus on:
1. Traditional optimizations (constant folding, DCE, CSE, inlining)
2. Sounio-specific optimizations (epistemic type handling, uncertainty propagation)

Respond ONLY with JSON:
{{"suggestions": [{{"type": "...", "confidence": 0.0-1.0, "target": "...", "reasoning": "..."}}]}}"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    config = PROVIDER_CONFIG[provider]
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert compiler optimization assistant for Sounio, a scientific computing language with epistemic types. Be concise. Output JSON only."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }

    try:
        response = requests.post(config["api_url"], headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        reasoning = ""
        reasoning_field = config.get("reasoning_field", "")
        if reasoning_field:
            reasoning = result.get("choices", [{}])[0].get("message", {}).get(reasoning_field, "")

        # Extract JSON from content
        if "{" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            json_str = content[json_start:json_end]
            suggestions = json.loads(json_str)

            return {
                "suggestions": suggestions.get("suggestions", []),
                "reasoning_trace": reasoning,
                "usage": result.get("usage", {})
            }

        return None

    except Exception as e:
        print(f"Error querying provider '{provider}' (model={model}): {e}", file=sys.stderr)
        return None


def mine_patterns(input_dir: Path, output_dir: Path, providers: List[str]) -> List[Dict[str, Any]]:
    """Mine optimization patterns from all Sounio files in directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    provider_keys = {provider: get_api_key(provider) for provider in providers}
    provider_models = {provider: get_model(provider) for provider in providers}

    # Find all .sio files
    sio_files = list(input_dir.rglob("*.sio"))
    provider_list = ", ".join(f"{provider}:{provider_models[provider]}" for provider in providers)
    print(f"Found {len(sio_files)} Sounio files to analyze with providers [{provider_list}]")

    results = []

    for i, filepath in enumerate(sio_files):
        print(f"[{i+1}/{len(sio_files)}] Analyzing {filepath.name}...")

        try:
            features = parse_sounio_file(filepath)

            # Skip very small files
            if features["line_count"] < 5:
                print(f"  Skipping (too small)")
                continue

            for provider in providers:
                provider_model = provider_models[provider]
                print(f"  [{provider}] Querying model={provider_model}...")
                provider_result = query_provider(
                    provider,
                    provider_model,
                    features["content"],
                    features,
                    provider_keys[provider],
                )

                if provider_result and provider_result.get("suggestions"):
                    result = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "provider": provider,
                        "model": provider_model,
                        "filepath": str(filepath.relative_to(input_dir)),
                        "features": {k: v for k, v in features.items() if k != "content"},
                        "suggestions": provider_result["suggestions"],
                        "reasoning_trace": provider_result.get("reasoning_trace", ""),
                        "tokens_used": provider_result.get("usage", {}).get("total_tokens", 0),
                    }
                    results.append(result)

                    # Print summary
                    for suggestion in provider_result["suggestions"]:
                        print(
                            f"  [{provider}] → {suggestion.get('type', 'Unknown')}: {suggestion.get('confidence', 0):.2f}"
                        )
                else:
                    print(f"  [{provider}] No suggestions")

                # Rate limit between provider queries
                time.sleep(1)

        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)

    return results


def save_results(results: List[Dict[str, Any]], output_dir: Path):
    """Save results to JSONL file."""

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"patterns_{timestamp}.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} patterns to {output_file}")

    # Also save summary
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "files_analyzed": len({r["filepath"] for r in results}),
        "result_entries": len(results),
        "providers": sorted({r.get("provider", "glm") for r in results}),
        "total_suggestions": sum(len(r["suggestions"]) for r in results),
        "suggestion_types": {},
        "suggestion_types_by_provider": {},
        "avg_confidence": 0,
    }

    all_suggestions = [s for r in results for s in r["suggestions"]]
    if all_suggestions:
        for s in all_suggestions:
            t = s.get("type", "Unknown")
            summary["suggestion_types"][t] = summary["suggestion_types"].get(t, 0) + 1

        for result in results:
            provider = result.get("provider", "glm")
            provider_map = summary["suggestion_types_by_provider"].setdefault(provider, {})
            for suggestion in result["suggestions"]:
                suggestion_type = suggestion.get("type", "Unknown")
                provider_map[suggestion_type] = provider_map.get(suggestion_type, 0) + 1

        confidences = [s.get("confidence", 0) for s in all_suggestions]
        summary["avg_confidence"] = sum(confidences) / len(confidences)

    summary_file = output_dir / f"summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="LLM Pattern Miner for Sounio")
    parser.add_argument(
        "--providers",
        type=str,
        default="auto",
        help="Comma-separated provider list (glm,openai,deepseek) or 'auto' to use all configured keys",
    )
    parser.add_argument("--input-dir", type=Path, default=Path("examples"),
                        help="Directory containing Sounio files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/optimization_patterns"),
                        help="Directory to store mined patterns")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without querying providers")

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        sio_files = list(args.input_dir.rglob("*.sio"))
        print(f"Would analyze {len(sio_files)} files:")
        for f in sio_files[:20]:
            print(f"  - {f}")
        if len(sio_files) > 20:
            print(f"  ... and {len(sio_files) - 20} more")
        return

    # Resolve providers and required API keys
    try:
        providers = parse_providers(args.providers)
        if not providers:
            supported = ", ".join(sorted(PROVIDER_CONFIG.keys()))
            print(
                f"Error: no providers enabled. Set API keys and pass --providers (<{supported}>), or use --providers auto",
                file=sys.stderr,
            )
            sys.exit(1)
        for provider in providers:
            # Validate required API key is set
            get_api_key(provider)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Mine patterns
    print(f"Mining patterns from {args.input_dir}")
    results = mine_patterns(args.input_dir, args.output_dir, providers)

    # Save results
    if results:
        summary = save_results(results, args.output_dir)

        print("\n=== Summary ===")
        print(f"Files analyzed: {summary['files_analyzed']}")
        print(f"Total suggestions: {summary['total_suggestions']}")
        print(f"Average confidence: {summary['avg_confidence']:.2f}")
        print("\nSuggestion types:")
        for t, count in sorted(summary["suggestion_types"].items(), key=lambda x: -x[1]):
            print(f"  {t}: {count}")
    else:
        print("No patterns found")


if __name__ == "__main__":
    main()
