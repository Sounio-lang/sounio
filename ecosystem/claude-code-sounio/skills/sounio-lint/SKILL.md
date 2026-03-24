---
name: sounio-lint
description: Scan a .sio file for Rust-isms and suggest Sounio-native fixes
user-invocable: true
allowed-tools: Bash, Read, Edit, Glob, Grep
---

# Sounio Lint — Rust-ism Detector

Scan Sounio source files for common Rust syntax that doesn't belong.

## Instructions

1. If a file argument is given, lint that file. Otherwise, lint all `.sio` files in the current directory.

2. Run the lint script:
   ```bash
   bash scripts/sounio-lint.sh <file>
   ```

3. If the script is not available, perform manual grep-based detection for these patterns:

   | Pattern | Rule | Fix |
   |---------|------|-----|
   | `println!` or `assert!` | rust-macro | Remove `!` → `println()`, `assert()` |
   | `&mut` | rust-borrow | Replace with `&!` |
   | `let mut` | rust-let-mut | Replace with `var` |
   | `#[` | rust-attr | Remove attribute, use `//@ annotation` if testing |
   | Trailing `;` on statements | semicolons | Remove the semicolon |
   | `\|x\|` lambda syntax | rust-closure | Extract to named function + fn ref |

4. Report findings with line numbers in `file:line: error: rule: description` format

5. If `--fix` is requested (or user asks to fix), apply corrections using the Edit tool:
   - `println!(...)` → `println(...)`
   - `assert!(...)` → `assert(...)`
   - `&mut` → `&!`
   - `let mut` → `var`
   - Remove trailing semicolons
   - Remove `#[...]` attribute lines

6. After fixing, run `/sounio-check` to verify the fixes compile
