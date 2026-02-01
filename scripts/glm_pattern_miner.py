#!/usr/bin/env python3
"""
GLM-4.7 Pattern Miner for Sounio

Analyzes Sounio source files using GLM-4.7 to discover optimization patterns.
Results are stored as JSONL for human review and eventual encoding into
deterministic compiler heuristics.

Usage:
    python scripts/glm_pattern_miner.py [--input-dir DIR] [--output-dir DIR]
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
GLM_API_URL = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
GLM_MODEL = "glm-4.7"
MAX_TOKENS = 1500
TEMPERATURE = 0.1


def get_api_key() -> str:
    """Get GLM API key from environment."""
    key = os.environ.get("GLM_API_KEY")
    if not key:
        raise ValueError("GLM_API_KEY environment variable not set")
    return key


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


def query_glm(code_content: str, features: Dict[str, Any], api_key: str) -> Optional[Dict[str, Any]]:
    """Query GLM-4.7 for optimization suggestions."""

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

    payload = {
        "model": GLM_MODEL,
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
        response = requests.post(GLM_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        reasoning = result.get("choices", [{}])[0].get("message", {}).get("reasoning_content", "")

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
        print(f"Error querying GLM: {e}", file=sys.stderr)
        return None


def mine_patterns(input_dir: Path, output_dir: Path, api_key: str) -> List[Dict[str, Any]]:
    """Mine optimization patterns from all Sounio files in directory."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .sio files
    sio_files = list(input_dir.rglob("*.sio"))
    print(f"Found {len(sio_files)} Sounio files to analyze")

    results = []

    for i, filepath in enumerate(sio_files):
        print(f"[{i+1}/{len(sio_files)}] Analyzing {filepath.name}...")

        try:
            features = parse_sounio_file(filepath)

            # Skip very small files
            if features["line_count"] < 5:
                print(f"  Skipping (too small)")
                continue

            # Query GLM
            glm_result = query_glm(features["content"], features, api_key)

            if glm_result and glm_result.get("suggestions"):
                result = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "filepath": str(filepath.relative_to(input_dir)),
                    "features": {k: v for k, v in features.items() if k != "content"},
                    "suggestions": glm_result["suggestions"],
                    "reasoning_trace": glm_result.get("reasoning_trace", ""),
                    "tokens_used": glm_result.get("usage", {}).get("total_tokens", 0)
                }
                results.append(result)

                # Print summary
                for suggestion in glm_result["suggestions"]:
                    print(f"  → {suggestion.get('type', 'Unknown')}: {suggestion.get('confidence', 0):.2f}")
            else:
                print(f"  No suggestions")

            # Rate limit
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
        "files_analyzed": len(results),
        "total_suggestions": sum(len(r["suggestions"]) for r in results),
        "suggestion_types": {},
        "avg_confidence": 0,
    }

    all_suggestions = [s for r in results for s in r["suggestions"]]
    if all_suggestions:
        for s in all_suggestions:
            t = s.get("type", "Unknown")
            summary["suggestion_types"][t] = summary["suggestion_types"].get(t, 0) + 1

        confidences = [s.get("confidence", 0) for s in all_suggestions]
        summary["avg_confidence"] = sum(confidences) / len(confidences)

    summary_file = output_dir / f"summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="GLM-4.7 Pattern Miner for Sounio")
    parser.add_argument("--input-dir", type=Path, default=Path("examples"),
                        help="Directory containing Sounio files")
    parser.add_argument("--output-dir", type=Path, default=Path("data/optimization_patterns"),
                        help="Directory to store mined patterns")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without querying GLM")

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

    # Get API key
    try:
        api_key = get_api_key()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Mine patterns
    print(f"Mining patterns from {args.input_dir}")
    results = mine_patterns(args.input_dir, args.output_dir, api_key)

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
