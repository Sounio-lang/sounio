#!/usr/bin/env python3
"""
MultiPL-E translator: Python HumanEval -> Sounio (.sio)

Translates function signatures, docstrings, type annotations, and test assertions
from Python HumanEval problems to idiomatic Sounio syntax.

Follows the MultiPL-E convention (https://github.com/nuprl/MultiPL-E):
  - Each translator is a Python class with a translate_prompt() entry point
  - Output is a valid source file in the target language
  - Tests use assertions that halt on failure

Sounio-specific translation rules:
  - No semicolons
  - var for mutable bindings, let for immutable
  - &! for mutable references (not &mut)
  - Effects required: with IO, Mut, Panic, Div as needed
  - No unary minus: 0 - x instead of -x
  - No closure literals: named fn refs only
  - Fixed-size arrays: [i64; 256] with explicit length
  - assert() not assert!()
  - No string interpolation

Usage:
    python humaneval_to_sounio.py                       # translate all problems
    python humaneval_to_sounio.py --problem HumanEval/0 # translate one problem
    python humaneval_to_sounio.py --input problems.jsonl --outdir generated/

Example:
    >>> t = SounioTranslator()
    >>> sig = t.translate_signature("add", [("a", "int"), ("b", "int")], "int")
    >>> print(sig)
    fn add(a: i64, b: i64) -> i64 with Mut, Panic, Div
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Type mapping: Python -> Sounio
# ---------------------------------------------------------------------------

TYPE_MAP: dict[str, str] = {
    "int": "i64",
    "float": "f64",
    "bool": "bool",
    "str": "[i8; 256]",
    "List[int]": "[i64; 256]",
    "List[float]": "[f64; 256]",
    "List[bool]": "[bool; 256]",
    "List[str]": "[[i8; 256]; 64]",
    "List[List[int]]": "[[i64; 256]; 64]",
    "List[List[float]]": "[[f64; 256]; 64]",
    "Optional[int]": "(i64, bool)",      # (value, is_some)
    "Optional[float]": "(f64, bool)",
    "Optional[bool]": "(bool, bool)",
    "Optional[str]": "([i8; 256], bool)",
    "Tuple[int, int]": "(i64, i64)",
    "Tuple[int, int, int]": "(i64, i64, i64)",
    "Tuple[float, float]": "(f64, f64)",
    "Dict[str, int]": "[[i64; 2]; 256]",  # key-value pairs as flat array
}

# Default effects added to translated functions.  Most HumanEval functions
# touch arrays / do division / print, so we add a generous set and let the
# Sounio type-checker prune if the user tightens them later.
DEFAULT_EFFECTS = "Mut, Panic, Div"


# ---------------------------------------------------------------------------
# Value translation helpers
# ---------------------------------------------------------------------------

_PYTHON_BOOL_RE = re.compile(r"\bTrue\b")
_PYTHON_BOOL_FALSE_RE = re.compile(r"\bFalse\b")
_PYTHON_NONE_RE = re.compile(r"\bNone\b")
# Matches negative integer literals like -42 but not expressions like x-42
_NEG_INT_RE = re.compile(r"(?<![a-zA-Z0-9_)}\]])\s*-(\d+)")
# Matches Python list literals [1, 2, 3]
_LIST_LITERAL_RE = re.compile(r"\[([^\[\]]*)\]")


def _translate_neg_int(m: re.Match) -> str:
    """Convert -N to (0 - N) for Sounio (no unary minus)."""
    return f"(0 - {m.group(1)})"


def translate_value(py_value: str) -> str:
    """Convert a Python literal/expression to its Sounio equivalent.

    Handles:
      - None -> 0
      - True/False -> true/false
      - -42 -> (0 - 42)
      - [1,2,3] -> array initialization comment (manual)
      - "string" -> "string" (unchanged, Sounio string literals match)

    Examples:
        >>> translate_value("True")
        'true'
        >>> translate_value("None")
        '0'
        >>> translate_value("-42")
        '(0 - 42)'
        >>> translate_value("[1, 2, 3]")
        '[1, 2, 3]'
    """
    result = py_value.strip()

    # Boolean literals
    result = _PYTHON_BOOL_RE.sub("true", result)
    result = _PYTHON_BOOL_FALSE_RE.sub("false", result)

    # None -> 0
    result = _PYTHON_NONE_RE.sub("0", result)

    # Negative integers: -N -> (0 - N)
    result = _NEG_INT_RE.sub(_translate_neg_int, result)

    return result


# ---------------------------------------------------------------------------
# Assertion translation
# ---------------------------------------------------------------------------

def translate_assert(assertion: str) -> str:
    """Convert a Python assert statement to Sounio assert().

    Handles:
      - assert f(x) == y        -> assert(f(x) == y)
      - assert f(x) == True     -> assert(f(x) == true)
      - assertEqual(a, b)       -> assert(a == b)
      - assertAlmostEqual(a,b)  -> assert(abs_diff(a, b) < 0.0001)
      - assert f(x) is None     -> assert(f(x) == 0)

    Examples:
        >>> translate_assert("assert has_close_elements([1.0, 2.0], 0.5) == True")
        'assert(has_close_elements([1.0, 2.0], 0.5) == true)'
    """
    line = assertion.strip()

    # assertEqual(a, b) -> assert(a == b)
    m = re.match(r"assert\w*Equal\s*\((.+),\s*(.+)\)\s*$", line)
    if m:
        lhs = translate_value(m.group(1).strip())
        rhs = translate_value(m.group(2).strip())
        return f"assert({lhs} == {rhs})"

    # assertAlmostEqual(a, b) -> assert(check_near(a, b))
    m = re.match(r"assertAlmostEqual\s*\((.+),\s*(.+)\)\s*$", line)
    if m:
        lhs = translate_value(m.group(1).strip())
        rhs = translate_value(m.group(2).strip())
        return f"assert(check_near({lhs}, {rhs}, 0.0001))"

    # assert ... is None -> assert(... == 0)
    if " is None" in line:
        line = line.replace(" is None", " == 0")

    # assert ... is not None -> assert(... != 0)
    if " is not None" in line:
        line = line.replace(" is not None", " != 0")

    # Standard assert expr -> assert(expr)
    m = re.match(r"assert\s+(.+)$", line)
    if m:
        expr = translate_value(m.group(1).strip())
        # Remove trailing comma if present (Python allows it)
        expr = expr.rstrip(",")
        return f"assert({expr})"

    # If nothing matched, return as comment
    return f"// TODO: translate assertion: {line}"


# ---------------------------------------------------------------------------
# Type translation
# ---------------------------------------------------------------------------

def translate_type(py_type: str) -> str:
    """Convert a Python type annotation string to a Sounio type.

    Uses the TYPE_MAP for known types, falls back to i64 for unknown types
    (with a comment).

    Examples:
        >>> translate_type("int")
        'i64'
        >>> translate_type("float")
        'f64'
        >>> translate_type("List[int]")
        '[i64; 256]'
        >>> translate_type("Optional[int]")
        '(i64, bool)'
    """
    cleaned = py_type.strip()
    if cleaned in TYPE_MAP:
        return TYPE_MAP[cleaned]

    # Handle generic List[T] where T is not in the map
    m = re.match(r"List\[(.+)\]", cleaned)
    if m:
        inner = translate_type(m.group(1))
        return f"[{inner}; 256]"

    # Handle Optional[T]
    m = re.match(r"Optional\[(.+)\]", cleaned)
    if m:
        inner = translate_type(m.group(1))
        return f"({inner}, bool)"

    # Handle Tuple[T, ...]
    m = re.match(r"Tuple\[(.+)\]", cleaned)
    if m:
        parts = [translate_type(t.strip()) for t in m.group(1).split(",")]
        return "(" + ", ".join(parts) + ")"

    # Fallback: unknown type -> i64 with comment
    return f"i64 /* TODO: unknown Python type '{cleaned}' */"


# ---------------------------------------------------------------------------
# Signature translation
# ---------------------------------------------------------------------------

def translate_signature(
    name: str,
    params: list[tuple[str, str]],
    return_type: str,
    effects: str = DEFAULT_EFFECTS,
) -> str:
    """Convert a function signature to Sounio syntax.

    Args:
        name: Function name.
        params: List of (param_name, python_type) tuples.
        return_type: Python return type string.
        effects: Sounio effects clause (without 'with').

    Returns:
        A Sounio function signature string (no body).

    Examples:
        >>> translate_signature("add", [("a", "int"), ("b", "int")], "int")
        'fn add(a: i64, b: i64) -> i64 with Mut, Panic, Div'
        >>> translate_signature("greet", [], "str", effects="IO")
        'fn greet() -> [i8; 256] with IO'
    """
    sio_params = []
    for pname, ptype in params:
        sio_type = translate_type(ptype)
        # List parameters are passed by reference in Sounio
        if sio_type.startswith("[") and not sio_type.startswith("[("):
            sio_params.append(f"{pname}: &{sio_type}")
        else:
            sio_params.append(f"{pname}: {sio_type}")

    param_str = ", ".join(sio_params)
    ret = translate_type(return_type)

    sig = f"fn {name}({param_str}) -> {ret}"
    if effects:
        sig += f" with {effects}"
    return sig


# ---------------------------------------------------------------------------
# Docstring translation
# ---------------------------------------------------------------------------

def translate_docstring(docstring: str) -> str:
    """Convert a Python docstring to Sounio line comments.

    Examples:
        >>> print(translate_docstring("Add two numbers.\\nReturns their sum."))
        // Add two numbers.
        // Returns their sum.
    """
    lines = docstring.strip().splitlines()
    return "\n".join(f"// {line.strip()}" if line.strip() else "//" for line in lines)


# ---------------------------------------------------------------------------
# Test wrapper
# ---------------------------------------------------------------------------

def wrap_test(func_code: str, test_code: str, problem_id: str = "") -> str:
    """Wrap function code and test assertions in a runnable .sio file.

    Generates a complete Sounio program with:
      - //@ run-pass header
      - Problem identification comment
      - Helper functions (check_near for float comparisons)
      - The candidate function
      - A main() that runs all test assertions

    Args:
        func_code: The translated function body.
        test_code: Translated assertion lines (one per line).
        problem_id: Optional HumanEval problem identifier.

    Returns:
        Complete .sio file content.
    """
    header = "//@ run-pass\n"
    if problem_id:
        header += f"// MultiPL-E HumanEval: {problem_id}\n"
    header += "// Auto-translated from Python by humaneval_to_sounio.py\n\n"

    # Float comparison helper (used by assertAlmostEqual translations)
    helpers = ""
    if "check_near" in test_code:
        helpers = textwrap.dedent("""\
            fn sio_abs(x: f64) -> f64 {
                if x < 0.0 { 0.0 - x } else { x }
            }

            fn check_near(a: f64, b: f64, tol: f64) -> bool {
                sio_abs(a - b) < tol
            }

        """)

    # Indent test code for main body
    test_lines = test_code.strip().splitlines()
    indented_tests = "\n".join(f"    {line}" for line in test_lines)

    main_block = textwrap.dedent(f"""\
        fn main() -> i64 with IO, Mut, Panic, Div {{
        {indented_tests}

            println("{problem_id or 'test'}: ALL TESTS PASSED")
            0
        }}
    """)

    return header + helpers + func_code.rstrip() + "\n\n" + main_block


# ---------------------------------------------------------------------------
# Problem-level translator
# ---------------------------------------------------------------------------

@dataclass
class SounioTranslator:
    """MultiPL-E translator from Python HumanEval to Sounio.

    Attributes:
        type_map: Mapping from Python type strings to Sounio types.
        default_effects: Default effects clause for generated functions.
    """

    type_map: dict[str, str] = field(default_factory=lambda: dict(TYPE_MAP))
    default_effects: str = DEFAULT_EFFECTS

    def translate_type(self, py_type: str) -> str:
        """Convert Python type annotation to Sounio type."""
        return translate_type(py_type)

    def translate_signature(
        self, name: str, params: list[tuple[str, str]], return_type: str
    ) -> str:
        """Convert function signature, adding effects."""
        return translate_signature(name, params, return_type, self.default_effects)

    def translate_assert(self, assertion: str) -> str:
        """Convert assert statement."""
        return translate_assert(assertion)

    def translate_value(self, py_value: str) -> str:
        """Convert Python literal to Sounio."""
        return translate_value(py_value)

    def translate_docstring(self, docstring: str) -> str:
        """Convert docstring to Sounio comments."""
        return translate_docstring(docstring)

    def wrap_test(self, func_code: str, test_code: str) -> str:
        """Wrap in main() with effects and return 0."""
        return wrap_test(func_code, test_code)

    def _parse_params_from_prompt(self, prompt: str) -> list[tuple[str, str]]:
        """Extract parameter names and types from a Python function definition.

        Parses lines like:
            def foo(a: int, b: List[int], c: float = 0.0) -> int:

        Returns list of (name, type_string) tuples.
        """
        # Find the def line
        m = re.search(r"def\s+\w+\s*\(([^)]*)\)", prompt)
        if not m:
            return []

        param_str = m.group(1).strip()
        if not param_str:
            return []

        params: list[tuple[str, str]] = []
        # Split on commas, but respect brackets (for List[int] etc.)
        depth = 0
        current = ""
        for ch in param_str + ",":
            if ch in "([":
                depth += 1
                current += ch
            elif ch in ")]":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                part = current.strip()
                if part:
                    # Remove default value
                    part = re.sub(r"\s*=\s*[^,]*$", "", part)
                    # Split name: type
                    if ":" in part:
                        pname, ptype = part.split(":", 1)
                        params.append((pname.strip(), ptype.strip()))
                    else:
                        params.append((part.strip(), "int"))
                current = ""
            else:
                current += ch

        return params

    def _parse_return_type(self, prompt: str) -> str:
        """Extract return type from a Python function definition."""
        m = re.search(r"\)\s*->\s*([^:]+):", prompt)
        if m:
            return m.group(1).strip()
        return "int"

    def _parse_function_name(self, prompt: str) -> str:
        """Extract function name from a Python function definition."""
        m = re.search(r"def\s+(\w+)\s*\(", prompt)
        if m:
            return m.group(1)
        return "solution"

    def _translate_body_stub(self, prompt: str, return_type: str) -> str:
        """Generate a placeholder body for the function.

        The body is a comment with the original docstring and a placeholder
        return statement matching the Sounio return type.
        """
        # Extract docstring
        docstring_lines = []
        in_doc = False
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_doc:
                    # End of docstring
                    if stripped not in ('"""', "'''"):
                        docstring_lines.append(
                            stripped.rstrip('"').rstrip("'").strip()
                        )
                    break
                else:
                    in_doc = True
                    rest = stripped[3:]
                    if rest.endswith('"""') or rest.endswith("'''"):
                        docstring_lines.append(rest[:-3].strip())
                        break
                    if rest.strip():
                        docstring_lines.append(rest.strip())
            elif in_doc:
                docstring_lines.append(stripped)

        doc_comment = ""
        if docstring_lines:
            doc_comment = "\n".join(
                f"    // {line}" if line else "    //" for line in docstring_lines
            )
            doc_comment += "\n"

        sio_ret = self.translate_type(return_type)

        # Generate appropriate zero/default value for the return type
        if sio_ret == "i64":
            default_return = "0"
        elif sio_ret == "f64":
            default_return = "0.0"
        elif sio_ret == "bool":
            default_return = "false"
        elif sio_ret.startswith("["):
            # Array type: return zero-initialized array
            m_arr = re.match(r"\[(.+);\s*(\d+)\]", sio_ret)
            if m_arr:
                elem_type = m_arr.group(1).strip()
                size = m_arr.group(2)
                if elem_type == "f64":
                    default_return = f"[0.0; {size}]"
                else:
                    default_return = f"[0; {size}]"
            else:
                default_return = "0"
        elif sio_ret.startswith("("):
            # Tuple type: zero-fill
            inner = sio_ret[1:-1]
            parts = []
            for t in inner.split(","):
                t = t.strip()
                if t == "f64":
                    parts.append("0.0")
                elif t == "bool":
                    parts.append("false")
                else:
                    parts.append("0")
            default_return = "(" + ", ".join(parts) + ")"
        else:
            default_return = "0"

        body = f"{doc_comment}    // TODO: implement\n    {default_return}"
        return body

    def translate_problem(self, problem: dict[str, Any]) -> str:
        """Translate a full HumanEval problem dict to a .sio file.

        Expected problem dict keys (from HumanEval JSONL):
          - task_id: str          e.g. "HumanEval/0"
          - prompt: str           Python function signature + docstring
          - canonical_solution: str  (not used, but available)
          - test: str             Python test code with assertions
          - entry_point: str      Function name

        Returns:
            Complete .sio file content as a string.
        """
        task_id = problem.get("task_id", "unknown")
        prompt = problem.get("prompt", "")
        test_code = problem.get("test", "")
        entry_point = problem.get("entry_point", "")

        # Parse the Python signature
        func_name = entry_point or self._parse_function_name(prompt)
        params = self._parse_params_from_prompt(prompt)
        return_type = self._parse_return_type(prompt)

        # Build Sounio function
        sig = self.translate_signature(func_name, params, return_type)
        body = self._translate_body_stub(prompt, return_type)
        func_code = f"{sig} {{\n{body}\n}}"

        # Translate test assertions
        translated_tests = []
        for line in test_code.splitlines():
            stripped = line.strip()
            if stripped.startswith("assert"):
                translated_tests.append(self.translate_assert(stripped))
            elif stripped.startswith("candidate = "):
                # Skip the `candidate = entry_point` binding
                continue
            elif stripped.startswith("def check"):
                # Skip test function definition wrapper
                continue
            elif stripped.startswith("check("):
                continue
            elif stripped and not stripped.startswith("#") and not stripped.startswith("def "):
                # Include other test lines as comments
                translated_tests.append(f"// {stripped}")

        test_str = "\n".join(translated_tests) if translated_tests else "// No tests translated"

        return wrap_test(func_code, test_str, problem_id=task_id)

    def translate_file(self, input_path: str, output_dir: str) -> list[str]:
        """Translate all problems in a JSONL file to .sio files.

        Args:
            input_path: Path to HumanEval JSONL file.
            output_dir: Directory to write .sio files.

        Returns:
            List of output file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_files = []

        with open(input_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    problem = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"  [WARN] line {line_num}: invalid JSON: {e}",
                        file=sys.stderr,
                    )
                    continue

                task_id = problem.get("task_id", f"problem_{line_num}")
                # Generate filename: HumanEval/123 -> humaneval_123.sio
                safe_name = task_id.replace("/", "_").replace(" ", "_").lower()
                out_path = os.path.join(output_dir, f"{safe_name}.sio")

                sio_code = self.translate_problem(problem)
                with open(out_path, "w") as out_f:
                    out_f.write(sio_code)

                output_files.append(out_path)
                print(f"  [{line_num}] {task_id} -> {out_path}")

        return output_files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MultiPL-E translator: Python HumanEval -> Sounio (.sio)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Translate all problems from a JSONL file
              python humaneval_to_sounio.py --input HumanEval.jsonl --outdir generated/

              # Translate a single problem from stdin JSON
              echo '{"task_id":"test/0","prompt":"def add(a:int,b:int)->int:\\n    ...","test":"assert add(1,2)==3","entry_point":"add"}' \\
                | python humaneval_to_sounio.py --stdin

              # Show translation of a single type
              python humaneval_to_sounio.py --show-types
        """),
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to HumanEval JSONL file",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="generated",
        help="Output directory for .sio files (default: generated/)",
    )
    parser.add_argument(
        "--problem",
        type=str,
        help="Translate only this problem ID (e.g., HumanEval/0)",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read a single problem JSON from stdin",
    )
    parser.add_argument(
        "--show-types",
        action="store_true",
        help="Print the type mapping table and exit",
    )
    parser.add_argument(
        "--effects",
        type=str,
        default=DEFAULT_EFFECTS,
        help=f"Default effects clause (default: {DEFAULT_EFFECTS})",
    )

    args = parser.parse_args()

    translator = SounioTranslator(default_effects=args.effects)

    if args.show_types:
        print("Python Type -> Sounio Type")
        print("-" * 50)
        for py_t, sio_t in sorted(TYPE_MAP.items()):
            print(f"  {py_t:30s} -> {sio_t}")
        return

    if args.stdin:
        raw = sys.stdin.read().strip()
        problem = json.loads(raw)
        print(translator.translate_problem(problem))
        return

    if args.input:
        print(f"Translating {args.input} -> {args.outdir}/")
        if args.problem:
            # Filter to a single problem
            with open(args.input, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    prob = json.loads(line)
                    if prob.get("task_id") == args.problem:
                        print(translator.translate_problem(prob))
                        return
            print(f"Problem {args.problem} not found in {args.input}", file=sys.stderr)
            sys.exit(1)
        else:
            files = translator.translate_file(args.input, args.outdir)
            print(f"\nTranslated {len(files)} problems to {args.outdir}/")
        return

    # No input specified: print usage
    parser.print_help()


if __name__ == "__main__":
    main()
