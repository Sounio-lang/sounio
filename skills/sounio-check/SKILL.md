---
name: sounio-check
description: Type-check a Sounio .sio file using the souc compiler
user-invocable: true
allowed-tools: Bash, Read, Glob
---

# Sounio Type-Check

Type-check the specified Sounio source file.

## Instructions

1. Resolve the souc compiler binary:
   - Use `$SOUC_BIN` or `$SOUC` environment variable if set
   - Otherwise try `souc` on PATH
   - Otherwise try `./artifacts/omega/souc-bin/souc-linux-x86_64-jit`
   - If none found, tell the user to set `SOUC_BIN`

2. If no file argument given, find the most recently modified `.sio` file in the working directory

3. Run type-check:
   ```bash
   $SOUC check <file>
   ```

4. If the user asked for `--show-ast` or `--show-types`, pass those flags through

5. Parse the output:
   - If clean (exit 0, no stderr), report "Check passed"
   - If errors, show them with file:line references and suggest fixes
   - Common fixes: missing `with` effects, `&mut` → `&!`, semicolons, missing type annotations

6. If the error mentions a missing effect, suggest the correct `with` clause based on what the function body does (IO for print, Mut for mutation, Div for division, Panic for indexing/assert)
