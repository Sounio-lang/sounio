---
name: sounio-run
description: JIT-execute a Sounio .sio file and show output
user-invocable: true
allowed-tools: Bash, Read, Glob
---

# Sounio Run

JIT-execute the specified Sounio source file.

## Instructions

1. Resolve the souc compiler binary:
   - Use `$SOUC_BIN` or `$SOUC` environment variable if set
   - Otherwise try `souc` on PATH
   - Otherwise try `./artifacts/omega/souc-bin/souc-linux-x86_64-jit`

2. If no file argument given, find the most recently modified `.sio` file in the working directory

3. Set stdlib path if importing stdlib modules:
   ```bash
   export SOUNIO_STDLIB_PATH=./stdlib
   ```

4. Run the file:
   ```bash
   $SOUC run <file>
   ```

5. Show stdout output to the user. If there are errors, run `$SOUC check <file>` first to get diagnostic details, then suggest fixes.
