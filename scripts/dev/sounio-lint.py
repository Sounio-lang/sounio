#!/usr/bin/env python3
"""
sounio-lint — Grammar enforcer for Sounio code.

Catches the mistakes LLMs make when writing Sounio by confusing it with Rust.
Returns specific, compiler-like error messages with exact fixes.

Usage:
    python3 scripts/dev/sounio-lint.py file.sio
    cat file.sio | python3 scripts/dev/sounio-lint.py
    python3 scripts/dev/sounio-lint.py --json file.sio   # machine-readable output
    python3 scripts/dev/sounio-lint.py --fix file.sio    # apply automatic fixes

Exit codes:
    0  No issues found
    1  Issues found
    2  Usage error
"""

import sys
import re
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional

# ─── Data Types ──────────────────────────────────────────────────────────────

@dataclass
class Diagnostic:
    rule: str
    severity: str          # "error" | "warning" | "hint"
    line: int
    col: int
    message: str
    fix: Optional[str]     # suggested fix (the corrected line)
    explanation: str

@dataclass
class LintResult:
    file: str
    diagnostics: list = field(default_factory=list)

    @property
    def errors(self):
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def warnings(self):
        return [d for d in self.diagnostics if d.severity == "warning"]

# ─── Rules ───────────────────────────────────────────────────────────────────

def check_line(line: str, lineno: int, source: str) -> list[Diagnostic]:
    """Run all per-line checks. Returns a list of diagnostics."""
    diags = []
    stripped = line.rstrip()
    content = stripped.lstrip()

    # Skip comment lines entirely
    if content.startswith("//"):
        return []

    # ── Rule 1: No semicolons ──────────────────────────────────────────────
    # Allow semicolons inside string literals
    # Strip string content before checking
    code_no_strings = re.sub(r'"[^"]*"', '""', stripped)
    if ";" in code_no_strings:
        # Exception: inside extern "C" { } blocks, inside //@ annotations
        if not content.startswith("//"):
            diags.append(Diagnostic(
                rule="no_semicolons",
                severity="error",
                line=lineno, col=stripped.index(";") + 1,
                message="Semicolons are not valid in Sounio.",
                fix=stripped.replace(";", ""),
                explanation="Remove all semicolons. Sounio has no statement terminators. "
                            "The last expression in a block is the return value.",
            ))

    # ── Rule 2: `let mut` → `var` ──────────────────────────────────────────
    m = re.search(r'\blet\s+mut\b', stripped)
    if m:
        fix = re.sub(r'\blet\s+mut\b', 'var', stripped)
        diags.append(Diagnostic(
            rule="var_not_let_mut",
            severity="error",
            line=lineno, col=m.start() + 1,
            message="`let mut` is Rust syntax. Use `var` for mutable bindings.",
            fix=fix,
            explanation="In Sounio: `let` = immutable, `var` = mutable. "
                        "There is no `let mut`.",
        ))

    # ── Rule 3: `&mut` → `&!` ─────────────────────────────────────────────
    m = re.search(r'&\s*mut\b', stripped)
    if m:
        fix = re.sub(r'&\s*mut\b', '&!', stripped)
        diags.append(Diagnostic(
            rule="exclusive_ref",
            severity="error",
            line=lineno, col=m.start() + 1,
            message="`&mut` is Rust syntax. Use `&!` for exclusive (mutable) references.",
            fix=fix,
            explanation="`&T` = shared (read-only), `&!T` = exclusive (mutable). "
                        "Call sites also use `&!`: `process(&!buffer)`.",
        ))

    # ── Rule 4: Rust macros (`assert!`, `println!`, `vec!`, etc.) ─────────
    macro_m = re.search(r'\b(\w+)![\(\[]', stripped)
    if macro_m and not content.startswith("//"):
        name = macro_m.group(1)
        replacements = {
            "assert": "assert",
            "assert_eq": None,   # special case
            "assert_ne": None,
            "println": "println",
            "print": "print",
            "eprintln": "println",
            "eprint": "print",
            "panic": None,
            "format": None,
            "vec": None,
            "todo": None,
            "unimplemented": None,
        }
        if name in replacements:
            repl = replacements[name]
            if repl:
                fix = re.sub(r'\b' + name + r'!\s*\(', name + "(", stripped)
                diags.append(Diagnostic(
                    rule="no_macros",
                    severity="error",
                    line=lineno, col=macro_m.start() + 1,
                    message=f"`{name}!()` is a Rust macro. Use `{repl}()` (no `!`).",
                    fix=fix,
                    explanation="Sounio has no macro syntax. Functions are called without `!`.",
                ))
            elif name == "assert_eq":
                m2 = re.search(r'assert_eq!\s*\(([^,]+),\s*([^)]+)\)', stripped)
                if m2:
                    fix = f"assert({m2.group(1).strip()} == {m2.group(2).strip()})"
                    diags.append(Diagnostic(
                        rule="no_macros",
                        severity="error",
                        line=lineno, col=macro_m.start() + 1,
                        message="`assert_eq!` is Rust. Use `assert(a == b)`.",
                        fix=fix,
                        explanation="No `assert_eq!`. Write `assert(a == b)` instead.",
                    ))
            elif name == "vec":
                fix = re.sub(r'vec!\s*\[([^\]]*)\]', r'[\1]', stripped)
                diags.append(Diagnostic(
                    rule="no_macros",
                    severity="error",
                    line=lineno, col=macro_m.start() + 1,
                    message="`vec![]` is Rust. Use array literal `[...]` instead.",
                    fix=fix,
                    explanation="No `vec!`. Arrays are written as `[1, 2, 3]`. "
                                "Size must be a compile-time literal.",
                ))
            else:
                diags.append(Diagnostic(
                    rule="no_macros",
                    severity="error",
                    line=lineno, col=macro_m.start() + 1,
                    message=f"`{name}!` is a Rust macro with no direct equivalent.",
                    fix=None,
                    explanation="Sounio has no macros. Restructure using functions and error codes.",
                ))

    # ── Rule 5: Unary minus literal (-42, -3.14) ──────────────────────────
    # Only flag: assignment/return context with a bare negative literal
    # Exclude: "0 - N" pattern (already correct), inside arithmetic expressions
    # Pattern: = -N or return -N or ( -N at start of expression
    unary_m = re.search(r'(?:=|return|\(|,)\s*(-\s*\d+\.?\d*)(?!\s*[\-+])', stripped)
    if unary_m:
        full_match = unary_m.group(1).strip()
        # Exclude if preceded by a digit (e.g., `0 - 42` → the `- 42` part)
        pre_char_pos = unary_m.start(1) - 1
        pre_char = stripped[pre_char_pos:pre_char_pos+1].strip() if pre_char_pos >= 0 else ""
        if pre_char not in "0123456789":
            num = re.sub(r'^-\s*', '', full_match)
            typ = "0.0" if "." in num else "0"
            fix_expr = f"{typ} - {num}"
            diags.append(Diagnostic(
                rule="no_unary_minus",
                severity="error",
                line=lineno, col=unary_m.start(1) + 1,
                message=f"Unary minus `{full_match}` is not valid. "
                        f"Use `{fix_expr}` instead.",
                fix=None,
                explanation="Sounio has no unary minus operator. "
                            "Write `0 - 42` not `-42`, `0.0 - 3.14` not `-3.14`.",
            ))

    # ── Rule 6: `Vec<T>` ──────────────────────────────────────────────────
    if re.search(r'\bVec\s*<', stripped):
        diags.append(Diagnostic(
            rule="fixed_arrays_only",
            severity="error",
            line=lineno, col=stripped.index("Vec") + 1,
            message="`Vec<T>` is Rust. Sounio uses fixed-size arrays: `[T; N]`.",
            fix=None,
            explanation="No dynamic Vec. Declare `var buf: [T; N] = [default; N]` "
                        "and track length manually as a separate `var len: i64 = 0`.",
        ))

    # ── Rule 7: `&[T]` slice type ─────────────────────────────────────────
    if re.search(r':\s*&\s*\[', stripped):
        diags.append(Diagnostic(
            rule="fixed_arrays_only",
            severity="error",
            line=lineno, col=1,
            message="`&[T]` slice type is Rust. Use `[T; N]` fixed-size arrays.",
            fix=None,
            explanation="No slices in Sounio. Pass a fixed-size array plus a length parameter: "
                        "`fn foo(data: [f64; 64], n: i64)`.",
        ))

    # ── Rule 8: Generic type parameters <T> ───────────────────────────────
    gen_m = re.search(r'\bfn\s+\w+\s*<[A-Z][A-Za-z0-9]*', stripped)
    if gen_m:
        diags.append(Diagnostic(
            rule="no_generics",
            severity="error",
            line=lineno, col=gen_m.start() + 1,
            message="Generic type parameters `<T>` are not allowed in user code.",
            fix=None,
            explanation="No generics. Write separate monomorphic functions per type. "
                        "Exception: `Knowledge<T>` is a built-in generic.",
        ))

    # ── Rule 9: Attributes #[...] ─────────────────────────────────────────
    if re.match(r'\s*#\[', stripped):
        attr_m = re.search(r'#\[(\w+)', stripped)
        attr_name = attr_m.group(1) if attr_m else "..."
        if attr_name == "test":
            msg = "`#[test]` is Rust. Use `//@ run-pass` file header instead."
            expl = "Test files use `//@ run-pass` as the first line. Tests live in `main()`."
        elif attr_name in ("derive", "allow", "warn", "deny", "cfg"):
            msg = f"`#[{attr_name}]` is a Rust attribute. Sounio has no attributes."
            expl = "No attributes. No `#[derive]`, `#[cfg]`, `#[allow]`, etc."
        else:
            msg = f"`#[{attr_name}]` is a Rust attribute. Sounio has no attributes."
            expl = "Sounio has no attribute syntax."
        diags.append(Diagnostic(
            rule="no_attributes",
            severity="error",
            line=lineno, col=1,
            message=msg,
            fix=None,
            explanation=expl,
        ))

    # ── Rule 10: Compound assignment operators (+=, -=, *=, /=) ──────────
    comp_m = re.search(r'\b\w+\s*(\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=)', stripped)
    if comp_m:
        op = comp_m.group(1)
        diags.append(Diagnostic(
            rule="no_compound_assign",
            severity="error",
            line=lineno, col=comp_m.start() + 1,
            message=f"Compound assignment `{op}` is not valid. Spell out fully.",
            fix=None,
            explanation=f"No `+=`, `-=`, `*=` etc. Write `x = x {op[0]} expr` instead.",
        ))

    # ── Rule 11: Bit shift without u8 suffix ──────────────────────────────
    shift_m = re.search(r'>>\s*\d+(?!u8)\b|<<\s*\d+(?!u8)\b', stripped)
    if shift_m:
        diags.append(Diagnostic(
            rule="shift_operand_u8",
            severity="error",
            line=lineno, col=shift_m.start() + 1,
            message="Shift amount must be `u8`. Add `u8` suffix to the shift operand.",
            fix=re.sub(r'(>>|<<)\s*(\d+)(?!u8)', r'\1 \2u8', stripped),
            explanation="Example: `x >> 4` → `x >> 4u8`. "
                        "For variable shift amounts: `x >> (n as u8)`.",
        ))

    # ── Rule 12: `self: &mut` instead of `self: &!` ───────────────────────
    m = re.search(r'self\s*:\s*&\s*mut\b', stripped)
    if m:
        fix = re.sub(r'(self\s*:\s*)&\s*mut\b', r'\1&!', stripped)
        diags.append(Diagnostic(
            rule="explicit_self",
            severity="error",
            line=lineno, col=m.start() + 1,
            message="`self: &mut Type` is Rust. Use `self: &!Type` for mutable self.",
            fix=fix,
            explanation="Sounio uses `&!T` not `&mut T`. Add `with Mut` to the function signature.",
        ))

    # ── Rule 13: `use std::` prefix ───────────────────────────────────────
    if re.match(r'\s*use\s+std::', stripped):
        diags.append(Diagnostic(
            rule="use_imports",
            severity="error",
            line=lineno, col=1,
            message="`use std::` is Rust. Sounio stdlib paths have no `std::` prefix.",
            fix=re.sub(r'use\s+std::', 'use ', stripped),
            explanation="Example: `use std::collections::HashMap` → `use collections::hash_map::{HashMap}`.",
        ))

    # ── Rule 14: `format!` or `format_args!` ─────────────────────────────
    if re.search(r'\bformat(_args)?!\s*\(', stripped):
        diags.append(Diagnostic(
            rule="no_string_format",
            severity="error",
            line=lineno, col=1,
            message="`format!()` is Rust. Sounio has no string formatting macros.",
            fix=None,
            explanation="Use `print()` + `print_i64()` / `print_f64()` for each part separately.",
        ))

    # ── Rule 15: `Result<` and `Option<` (non-Knowledge) ─────────────────
    if re.search(r'\bResult\s*<(?!.*Knowledge)', stripped):
        diags.append(Diagnostic(
            rule="no_error_propagation",
            severity="warning",
            line=lineno, col=1,
            message="`Result<T, E>` is Rust. Use `(T, i32)` error code tuples.",
            fix=None,
            explanation="Return `(value, 0)` for success, `(default, error_code)` for failure. "
                        "Check error codes explicitly.",
        ))
    if re.search(r'\bOption\s*<(?!.*Knowledge)', stripped):
        diags.append(Diagnostic(
            rule="no_option_result",
            severity="warning",
            line=lineno, col=1,
            message="`Option<T>` is Rust. Use sentinel values or `(T, bool)` tuples.",
            fix=None,
            explanation="Return `0 - 1` for not-found indices, or `(value, false)` for absent values.",
        ))

    # ── Rule 16: `?` operator ─────────────────────────────────────────────
    if re.search(r'\w\?(?:\s*;|\s*\n|\s*$)', stripped) or re.search(r'\?\s*$', stripped.rstrip()):
        # Only flag if it looks like error propagation, not ternary
        if re.search(r'[)}\w]\s*\?\s*[;,\n]?$', stripped):
            diags.append(Diagnostic(
                rule="no_error_propagation",
                severity="error",
                line=lineno, col=1,
                message="`?` error propagation operator is Rust. Handle errors explicitly.",
                fix=None,
                explanation="No `?` operator. Use `if err != 0 { return (default, err) }` patterns.",
            ))

    # ── Rule 17: Lifetime annotations 'a ──────────────────────────────────
    if re.search(r"'\s*[a-z]\b", stripped):
        if not "//'" in stripped:  # not in comment
            diags.append(Diagnostic(
                rule="no_lifetime",
                severity="error",
                line=lineno, col=1,
                message="Lifetime annotations `'a` are Rust. Sounio has no lifetime syntax.",
                fix=None,
                explanation="References are valid for their lexical scope. No lifetime annotations needed.",
            ))

    # ── Rule 18: `impl Trait` / `dyn Trait` ──────────────────────────────
    if re.search(r'\b(impl|dyn)\s+[A-Z]\w*', stripped):
        found = re.search(r'\b(impl|dyn)\s+[A-Z]\w*', stripped)
        keyword = found.group(1)
        diags.append(Diagnostic(
            rule="no_traits",
            severity="error",
            line=lineno, col=found.start() + 1,
            message=f"`{keyword} Trait` is Rust. Sounio has no trait objects.",
            fix=None,
            explanation="No traits, no dynamic dispatch. Use concrete types in function signatures.",
        ))

    # ── Rule 19: `Box<T>` ─────────────────────────────────────────────────
    if re.search(r'\bBox\s*<', stripped):
        diags.append(Diagnostic(
            rule="no_box_heap",
            severity="error",
            line=lineno, col=1,
            message="`Box<T>` is Rust. Sounio uses stack allocation by default.",
            fix=None,
            explanation="Stack-allocate structs directly. Use `Alloc` effect only for rare heap needs.",
        ))

    # ── Rule 20: `#[derive]` on struct ────────────────────────────────────
    # Covered by rule 9

    return diags


def check_file_level(lines: list[str]) -> list[Diagnostic]:
    """File-level checks (require seeing multiple lines)."""
    diags = []
    source = "\n".join(lines)

    # Check: fn main should return i32 and declare effects
    main_match = re.search(r'\bfn\s+main\s*\(\s*\)', source)
    if main_match:
        main_line = source[:main_match.start()].count("\n") + 1
        main_sig = source[main_match.start():main_match.start() + 100]
        if "with" not in main_sig.split("{")[0]:
            diags.append(Diagnostic(
                rule="effects_mandatory",
                severity="warning",
                line=main_line, col=1,
                message="`fn main()` has no effects declared. Should it have `with IO, Mut, Div, Panic`?",
                fix=None,
                explanation="main() must declare all effects used anywhere in its call tree. "
                            "Safe default: `fn main() -> i32 with IO, Mut, Div, Panic`.",
            ))

    # Check: division without Div effect in function signature
    # (Simplified: flag functions with / operator but no `with` clause)
    fn_blocks = re.findall(r'fn\s+\w+[^{]*\{[^}]*/', source)
    for block in fn_blocks:
        if " with " not in block.split("{")[0]:
            fn_line = source[:source.index(block)].count("\n") + 1
            diags.append(Diagnostic(
                rule="effects_mandatory",
                severity="warning",
                line=fn_line, col=1,
                message="Function contains `/` but may not declare `Div` effect.",
                fix=None,
                explanation="Division requires `with Div, Panic` in the function signature.",
            ))
            break  # Only report once per file for this heuristic

    return diags


# ─── Formatting ──────────────────────────────────────────────────────────────

SEVERITY_ICONS = {"error": "✗", "warning": "⚠", "hint": "→"}
SEVERITY_COLORS = {
    "error":   "\033[31m",  # red
    "warning": "\033[33m",  # yellow
    "hint":    "\033[36m",  # cyan
}
RESET = "\033[0m"

def format_diagnostic(d: Diagnostic, line_content: str, use_color: bool = True) -> str:
    color = SEVERITY_COLORS[d.severity] if use_color else ""
    reset = RESET if use_color else ""
    icon = SEVERITY_ICONS[d.severity]

    lines = [
        f"{color}{icon} [{d.rule}] line {d.line}:{d.col} — {d.message}{reset}",
        f"  {line_content.rstrip()}",
    ]
    if d.fix and d.fix.strip() != line_content.strip():
        lines.append(f"  Fix: {d.fix.strip()}")
    lines.append(f"  {d.explanation}")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def lint(source: str, filename: str = "<stdin>") -> LintResult:
    result = LintResult(file=filename)
    lines = source.splitlines()

    for i, line in enumerate(lines, start=1):
        result.diagnostics.extend(check_line(line, i, source))

    result.diagnostics.extend(check_file_level(lines))
    result.diagnostics.sort(key=lambda d: (d.line, d.col))
    return result


def apply_fixes(source: str) -> tuple[str, int]:
    """Apply all auto-fixable transformations. Returns (fixed_source, n_fixes)."""
    lines = source.splitlines(keepends=True)
    n_fixes = 0

    simple_rules = [
        (r';(?=\s*$)', '', "remove semicolons"),
        (r'\blet\s+mut\b', 'var', "let mut → var"),
        (r'&\s*mut\b', '&!', "&mut → &!"),
        (r'\b(assert|println|print|eprintln|eprint)!\s*\(', r'\1(', "macro → fn"),
        (r'(>>|<<)\s*(\d+)(?!u8)', r'\1 \2u8', "shift → u8"),
        (r'use\s+std::', 'use ', "remove std::"),
        (r'assert_eq!\s*\(([^,]+),\s*([^)]+)\)', r'assert(\1 == \2)', "assert_eq → assert"),
        (r'vec!\s*\[', '[', "vec! → ["),
    ]

    new_lines = []
    for line in lines:
        new_line = line
        for pattern, replacement, _ in simple_rules:
            if re.search(r'"[^"]*"', new_line):
                # Apply only outside strings
                parts = re.split(r'("(?:[^"\\]|\\.)*")', new_line)
                fixed_parts = []
                for j, part in enumerate(parts):
                    if j % 2 == 0:  # outside string
                        fixed_part = re.sub(pattern, replacement, part)
                        if fixed_part != part:
                            n_fixes += 1
                        fixed_parts.append(fixed_part)
                    else:
                        fixed_parts.append(part)
                new_line = "".join(fixed_parts)
            else:
                fixed = re.sub(pattern, replacement, new_line)
                if fixed != new_line:
                    n_fixes += 1
                new_line = fixed
        new_lines.append(new_line)

    return "".join(new_lines), n_fixes


def main():
    parser = argparse.ArgumentParser(
        description="Sounio grammar enforcer — catches LLM Rust hallucinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("file", nargs="?", help="Input .sio file (or stdin if omitted)")
    parser.add_argument("--json", action="store_true", help="JSON output (machine-readable)")
    parser.add_argument("--fix", action="store_true", help="Print auto-fixed code to stdout")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    parser.add_argument("--errors-only", action="store_true", help="Only show errors, not warnings")
    args = parser.parse_args()

    # Read input
    if args.file:
        try:
            with open(args.file) as f:
                source = f.read()
            filename = args.file
        except FileNotFoundError:
            print(f"error: file not found: {args.file}", file=sys.stderr)
            sys.exit(2)
    else:
        source = sys.stdin.read()
        filename = "<stdin>"

    # Auto-fix mode
    if args.fix:
        fixed, n = apply_fixes(source)
        sys.stdout.write(fixed)
        if n > 0:
            print(f"# sounio-lint: applied {n} automatic fix(es)", file=sys.stderr)
        sys.exit(0)

    result = lint(source, filename)
    lines = source.splitlines()

    diags = result.diagnostics
    if args.errors_only:
        diags = result.errors

    if args.json:
        output = {
            "file": filename,
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
            "diagnostics": [
                {
                    "rule": d.rule,
                    "severity": d.severity,
                    "line": d.line,
                    "col": d.col,
                    "message": d.message,
                    "fix": d.fix,
                    "explanation": d.explanation,
                }
                for d in diags
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        use_color = not args.no_color and sys.stdout.isatty()
        if diags:
            print(f"sounio-lint: {filename}")
            print()
            for d in diags:
                line_content = lines[d.line - 1] if d.line <= len(lines) else ""
                print(format_diagnostic(d, line_content, use_color=use_color))
                print()
            summary_color = SEVERITY_COLORS["error"] if result.errors else SEVERITY_COLORS["warning"]
            reset = RESET if use_color else ""
            print(
                f"{summary_color if use_color else ''}"
                f"{len(result.errors)} error(s), {len(result.warnings)} warning(s)"
                f"{reset if use_color else ''}"
            )
        else:
            print(f"sounio-lint: {filename} — OK")

    sys.exit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
