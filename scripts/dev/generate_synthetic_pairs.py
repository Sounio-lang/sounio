#!/usr/bin/env python3
"""
Generate synthetic Rust-wrong → Sounio-correct contrastive pairs
by injecting classic Rust mistakes into verified .sio source files.

Output: datasets/sounio-contrastive/synthetic.jsonl
Format: {"instruction": "...", "input": broken_code, "completion": correct_code, "rule": rule_name}
"""
import os
import re
import json
import random
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "datasets/sounio-contrastive/synthetic.jsonl"

INSTRUCTION_TEMPLATES = [
    "Fix this Sounio code that contains Rust syntax errors:",
    "The following Sounio code has Rust syntax mistakes. Correct it:",
    "This code mixes Rust and Sounio syntax. Rewrite it as valid Sounio:",
    "Identify and fix the Rust hallucination in this Sounio code:",
    "Convert this invalid Sounio (Rust-style) into correct Sounio:",
]


# ---------------------------------------------------------------------------
# Injection rules: each returns (broken_code, rule_name) or None if no match
# ---------------------------------------------------------------------------

def inject_semicolons(code: str):
    """Add semicolons to line ends (Rust style)."""
    lines = code.splitlines()
    injected = []
    changed = 0
    for line in lines:
        stripped = line.rstrip()
        # Only add to non-empty, non-comment, non-brace, non-annotation lines
        if (stripped and
                not stripped.startswith("//") and
                not stripped.startswith("#") and
                not stripped.startswith("@") and
                not stripped.endswith("{") and
                not stripped.endswith("}") and
                not stripped.endswith(",") and
                not stripped.endswith(";") and
                not stripped.endswith("(") and
                not stripped.endswith("\\") and
                len(stripped) > 4):
            injected.append(stripped + ";")
            changed += 1
        else:
            injected.append(line)
    if changed < 1:
        return None
    return "\n".join(injected), "no_semicolons"


def inject_let_mut(code: str):
    """Replace `var` with `let mut` (Rust style)."""
    if "var " not in code:
        return None
    broken = re.sub(r'\bvar\b', 'let mut', code)
    if broken == code:
        return None
    return broken, "var_not_let_mut"


def inject_mut_ref(code: str):
    """Replace `&!` with `&mut ` (Rust style)."""
    if "&!" not in code:
        return None
    # Add space: &!foo → &mut foo
    broken = re.sub(r'&!(\w)', r'&mut \1', code)
    # Also handle bare &! (e.g. &!T, &!arr)
    broken = broken.replace("&!", "&mut ")
    if broken == code:
        return None
    return broken, "exclusive_ref"


def inject_assert_macro(code: str):
    """Replace `assert(` with `assert!(` (Rust style)."""
    if "assert(" not in code:
        return None
    broken = re.sub(r'\bassert\(', 'assert!(', code)
    if broken == code:
        return None
    return broken, "no_macros"


def inject_println_macro(code: str):
    """Replace `println(` with `println!(` (Rust style)."""
    if "println(" not in code:
        return None
    broken = re.sub(r'\bprintln\(', 'println!(', code)
    if broken == code:
        return None
    return broken, "no_macros"


def inject_unary_minus(code: str):
    """Replace `0 - N` patterns with `-N` (Rust style unary minus)."""
    broken = re.sub(r'\b0\s*-\s*(\d+\.?\d*)\b', r'-\1', code)
    if broken == code:
        return None
    return broken, "no_unary_minus"


def inject_missing_effect(code: str):
    """Remove `with IO` from fn main declarations."""
    if "fn main() with IO" not in code and "fn main() with " not in code:
        return None
    broken = re.sub(r'fn main\(\) with \w+(\s*,\s*\w+)*', 'fn main()', code)
    if broken == code:
        return None
    return broken, "effects_mandatory"


def inject_let_immutable(code: str):
    """Nothing to inject — let is already correct Sounio; skip."""
    return None


INJECTORS = [
    inject_semicolons,
    inject_let_mut,
    inject_mut_ref,
    inject_assert_macro,
    inject_println_macro,
    inject_unary_minus,
    inject_missing_effect,
]


def should_skip(path: Path) -> bool:
    text = path.read_text(errors="replace")
    # Skip compile-fail, ignore, empty, or pure comment files
    if "//@ compile-fail" in text:
        return True
    if "//@ ignore" in text:
        return True
    if len(text.strip()) < 30:
        return True
    return False


def strip_annotations(code: str) -> str:
    """Remove //@ run-pass style annotations (keep for completion, strip from input display)."""
    return code


def generate_pairs(sources: list[Path], max_per_file: int = 3) -> list[dict]:
    pairs = []
    for path in sources:
        if should_skip(path):
            continue
        try:
            correct = path.read_text(errors="replace")
        except Exception:
            continue

        # Try each injector; collect up to max_per_file distinct pairs
        file_pairs = []
        random.shuffle(INJECTORS)
        for injector in INJECTORS:
            result = injector(correct)
            if result is None:
                continue
            broken, rule = result
            if broken == correct:
                continue
            instr = random.choice(INSTRUCTION_TEMPLATES)
            file_pairs.append({
                "instruction": instr,
                "input": broken,
                "completion": correct,
                "rule": rule,
                "source_path": str(path.relative_to(REPO_ROOT)),
            })
            if len(file_pairs) >= max_per_file:
                break
        pairs.extend(file_pairs)
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic contrastive Sounio pairs")
    parser.add_argument("--max-per-file", type=int, default=2,
                        help="Max injections per source file (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    random.seed(args.seed)

    sources = (
        sorted((REPO_ROOT / "tests/run-pass").glob("*.sio")) +
        sorted((REPO_ROOT / "examples").glob("*.sio"))
    )
    print(f"Found {len(sources)} source files")

    pairs = generate_pairs(sources, max_per_file=args.max_per_file)
    print(f"Generated {len(pairs)} pairs")

    # Rule distribution
    from collections import Counter
    dist = Counter(p["rule"] for p in pairs)
    for rule, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {rule}: {count}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"\nWrote {len(pairs)} pairs → {out_path}")


if __name__ == "__main__":
    main()
