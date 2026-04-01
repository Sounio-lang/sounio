#!/usr/bin/env python3
"""
Extract LoRA fine-tuning pairs from Sounio codebase.

Generates Alpaca-style JSONL for fine-tuning CodeLlama-7B or DeepSeek-Coder-6.7B.
Based on 6-model consensus (artifacts/research/llm_seeding_fanout/SYNTHESIS.md):
  - Contrastive Rust→Sounio examples (deprogramming Rust patterns)
  - Instruction-response pairs from existing verified code
  - Error injection + correction pairs

Output: artifacts/llm_training/sounio_lora_pairs.jsonl

Usage:
    python3 scripts/extract_lora_pairs.py
    # Then fine-tune on Colab/RunPod — no Claude Code needed.
"""

import glob
import json
import os
import re
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "artifacts", "llm_training")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sounio_lora_pairs.jsonl")

# Sounio system prompt (prepended to all instructions)
SYSTEM_PROMPT = (
    "You are an expert Sounio programmer. Sounio is NOT Rust. "
    "You NEVER use semicolons, `let mut`, `&mut`, `assert!()`, or `println!()`. "
    "You ALWAYS use `var` for mutable bindings, `&!` for exclusive refs, "
    "`assert()` without `!`, and declare effects with `with IO, Mut, Div, Panic`."
)

# Top 5 Rust→Sounio error pairs (from SYNTHESIS.md consensus)
RUST_ERRORS = [
    {
        "rust": "let mut x = 5;",
        "sounio": "var x = 5",
        "rule": "Use `var` for mutable bindings, no semicolons"
    },
    {
        "rust": "fn foo(x: &mut i32) { *x = 42; }",
        "sounio": "fn foo(x: &!i32) with Mut { *x = 42 }",
        "rule": "Use `&!` for exclusive refs, declare Mut effect"
    },
    {
        "rust": 'assert!(x > 0, "must be positive");',
        "sounio": "assert(x > 0)",
        "rule": "No Rust macros — use assert() without !"
    },
    {
        "rust": "let v: Vec<i32> = Vec::new();\nv.push(42);",
        "sounio": "var arr: [i32; 64] = [0; 64]\narr[0] = 42",
        "rule": "Fixed-size arrays only, no Vec<T>"
    },
    {
        "rust": 'println!("Hello, {}!", name);',
        "sounio": 'println("Hello"); println(name)',
        "rule": "No format macros — use println() for each value"
    },
]


def extract_sio_files(patterns):
    """Collect .sio files matching glob patterns."""
    files = []
    for pat in patterns:
        full = os.path.join(REPO_ROOT, pat)
        files.extend(glob.glob(full, recursive=True))
    return sorted(set(files))


def extract_docstring(code):
    """Extract //! or //@ docstring from top of file."""
    lines = code.split("\n")
    doc = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("//!") or stripped.startswith("//@"):
            doc.append(stripped[3:].strip())
        elif stripped.startswith("//"):
            continue
        elif stripped == "":
            continue
        else:
            break
    return " ".join(doc) if doc else None


def extract_functions(code):
    """Extract individual function definitions."""
    funcs = []
    lines = code.split("\n")
    i = 0
    while i < len(lines):
        match = re.match(r'^(pub\s+)?fn\s+(\w+)', lines[i])
        if match:
            fn_name = match.group(2)
            start = i
            brace_depth = 0
            found_open = False
            while i < len(lines):
                brace_depth += lines[i].count("{") - lines[i].count("}")
                if "{" in lines[i]:
                    found_open = True
                if found_open and brace_depth <= 0:
                    funcs.append({
                        "name": fn_name,
                        "code": "\n".join(lines[start:i+1]),
                        "line": start + 1
                    })
                    break
                i += 1
        i += 1
    return funcs


def make_pair(instruction, output, input_ctx=""):
    """Create an Alpaca-style training pair."""
    return {
        "instruction": f"{SYSTEM_PROMPT}\n\n{instruction}",
        "input": input_ctx,
        "output": output
    }


def generate_file_pairs(filepath):
    """Generate pairs from a .sio file."""
    pairs = []
    rel_path = os.path.relpath(filepath, REPO_ROOT)

    with open(filepath, "r") as f:
        code = f.read()

    doc = extract_docstring(code)

    # Pair 1: Full file generation
    if doc and len(code) < 3000:
        pairs.append(make_pair(
            f"Write a Sounio program: {doc}",
            code
        ))

    # Pair 2: Individual functions
    funcs = extract_functions(code)
    for fn in funcs:
        if len(fn["code"]) > 50 and fn["name"] != "main":
            pairs.append(make_pair(
                f"Write a Sounio function named `{fn['name']}` that implements the following logic:",
                fn["code"],
                f"// From {rel_path}"
            ))

    return pairs


def generate_contrastive_pairs():
    """Generate Rust→Sounio contrastive pairs (deprogramming)."""
    pairs = []
    for err in RUST_ERRORS:
        # Positive: correct Sounio
        pairs.append(make_pair(
            f"Fix this Rust-style code to be valid Sounio. Rule: {err['rule']}",
            err["sounio"],
            err["rust"]
        ))
        # Negative awareness
        pairs.append(make_pair(
            "Is this valid Sounio code? If not, fix it.",
            f"No, this is Rust syntax. In Sounio:\n{err['sounio']}",
            err["rust"]
        ))
    return pairs


def generate_error_injection_pairs(filepath):
    """Inject common errors into working code, create fix pairs."""
    pairs = []
    with open(filepath, "r") as f:
        code = f.read()

    # Only process small files
    if len(code) > 2000:
        return pairs

    funcs = extract_functions(code)
    for fn in funcs[:3]:  # limit per file
        original = fn["code"]

        # Inject semicolons
        broken = original.replace("\n", ";\n")
        if broken != original:
            pairs.append(make_pair(
                "Fix this broken Sounio code (hint: Sounio does not use semicolons):",
                original,
                broken
            ))

        # Inject &mut
        if "&!" in original:
            broken = original.replace("&!", "&mut")
            pairs.append(make_pair(
                "Fix this broken Sounio code (hint: Sounio uses &! not &mut):",
                original,
                broken
            ))

        # Inject assert!
        if "assert(" in original and "assert!(" not in original:
            broken = original.replace("assert(", "assert!(")
            pairs.append(make_pair(
                "Fix this broken Sounio code (hint: Sounio uses assert() not assert!()):",
                original,
                broken
            ))

    return pairs


def generate_effect_pairs():
    """Generate pairs teaching the effect system."""
    examples = [
        ("Write a Sounio function that prints a greeting",
         'fn greet(name: &str) with IO {\n    println("Hello")\n    println(name)\n}'),
        ("Write a Sounio function that mutates a reference",
         "fn increment(x: &!i64) with Mut {\n    *x = *x + 1\n}"),
        ("Write a Sounio function that divides two numbers safely",
         "fn safe_div(a: f64, b: f64) -> f64 with Div, Panic {\n    a / b\n}"),
        ("Write a pure Sounio function (no effects)",
         "fn add(a: i64, b: i64) -> i64 {\n    a + b\n}"),
        ("Write a Sounio main function",
         "fn main() with IO, Mut, Div, Panic {\n    println(\"running\")\n}"),
    ]
    return [make_pair(inst, out) for inst, out in examples]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_pairs = []

    # 1. Contrastive Rust→Sounio pairs
    print("Generating contrastive pairs...")
    all_pairs.extend(generate_contrastive_pairs())
    print(f"  {len(all_pairs)} contrastive pairs")

    # 2. Effect system teaching pairs
    print("Generating effect system pairs...")
    effect_pairs = generate_effect_pairs()
    all_pairs.extend(effect_pairs)
    print(f"  {len(effect_pairs)} effect pairs")

    # 3. File-based pairs from examples/
    print("Extracting from examples/*.sio...")
    example_files = extract_sio_files(["examples/*.sio", "examples/**/*.sio"])
    for f in example_files:
        all_pairs.extend(generate_file_pairs(f))
    print(f"  {len(all_pairs)} total after examples")

    # 4. File-based pairs from tests/run-pass/
    print("Extracting from tests/run-pass/*.sio...")
    test_files = extract_sio_files(["tests/run-pass/*.sio"])
    for f in test_files:
        all_pairs.extend(generate_file_pairs(f))
    print(f"  {len(all_pairs)} total after run-pass tests")

    # 5. Error injection pairs
    print("Generating error injection pairs...")
    inject_files = extract_sio_files(["examples/*.sio", "tests/run-pass/*.sio"])
    random.seed(42)
    random.shuffle(inject_files)
    for f in inject_files[:50]:  # limit
        all_pairs.extend(generate_error_injection_pairs(f))
    print(f"  {len(all_pairs)} total after error injection")

    # 6. Stdlib function pairs (select key modules)
    print("Extracting from stdlib highlights...")
    stdlib_files = extract_sio_files([
        "stdlib/epistemic/knowledge.sio",
        "stdlib/epistemic/gum.sio",
        "stdlib/epistemic/confidence_gate.sio",
        "stdlib/epistemic/budget.sio",
        "stdlib/math/sedenion64.sio",
        "stdlib/autodiff/tape.sio",
        "stdlib/autodiff/grad.sio",
    ])
    for f in stdlib_files:
        all_pairs.extend(generate_file_pairs(f))
    print(f"  {len(all_pairs)} total after stdlib")

    # Deduplicate
    seen = set()
    unique_pairs = []
    for p in all_pairs:
        key = p["output"][:200]
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    # Write JSONL
    with open(OUTPUT_FILE, "w") as f:
        for pair in unique_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nWrote {len(unique_pairs)} unique pairs to {OUTPUT_FILE}")
    print(f"Breakdown:")
    print(f"  Contrastive:    {len(generate_contrastive_pairs())}")
    print(f"  Effects:        {len(effect_pairs)}")
    print(f"  Examples:       {len(example_files)} files")
    print(f"  Run-pass:       {len(test_files)} files")
    print(f"  Stdlib:         {len(stdlib_files)} files")
    print(f"\nNext: Upload to Colab, fine-tune with LoRA on CodeLlama-7B or DeepSeek-Coder-6.7B")


if __name__ == "__main__":
    main()
