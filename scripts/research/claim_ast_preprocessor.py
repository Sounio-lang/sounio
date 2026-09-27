#!/usr/bin/env python3
"""
AST-native claims preprocessor.

Converts `claim name { field = value, ... }` blocks into
`const name: Claim = Claim { field: value, ... }` literals.

Companion to:
  docs/research/ast_native_claims_spec_2026-07-25.md
  docs/research/ast_native_claims_falsifiers_2026-07-25.md

Pure Python; no external dependencies.
"""

import re
import sys


def preprocess(text: str) -> str:
    """Convert claim blocks to const Claim literals."""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^\s*claim\s+(\w+)\s*\{\s*$', line)
        if not m:
            out.append(line)
            i += 1
            continue
        name = m.group(1)
        i += 1
        fields = {}
        while i < len(lines):
            line = lines[i].strip()
            if line == '}':
                i += 1
                break
            m2 = re.match(r'^(\w+)\s*=\s*(.*?)\s*,?\s*$', line)
            if m2:
                fields[m2.group(1)] = m2.group(2)
            i += 1
        # emit const literal
        out.append(f"const {name}: Claim = Claim {{")
        for key, value in fields.items():
            out.append(f"    {key}: {value},")
        out.append("}")
    return '\n'.join(out)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # simple round-trip test
        src = """claim test_claim {
    hypothesis = "test",
    falsifier = "none",
    evidence = Evidence::InstrumentControlled,
    harness = "h",
    gate = "g",
    verdict = Verdict::Negative,
}
"""
        result = preprocess(src)
        expected_fields = ['hypothesis', 'falsifier', 'evidence', 'harness', 'gate', 'verdict']
        ok = all(f in result for f in expected_fields) and 'const test_claim: Claim = Claim {' in result
        print(f"ROUNDTRIP {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(result)
        return 0 if ok else 1
    # read stdin, write stdout
    text = sys.stdin.read()
    sys.stdout.write(preprocess(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
