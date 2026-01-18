# Format Sounio source code

Format Sounio source files according to the standard style guidelines.

## Arguments
- `<path>` - File or directory to format (default: current directory)
- `--check` - Check formatting without modifying files
- `--diff` - Show diff of changes that would be made
- `--max-width <n>` - Maximum line width (default: 100)

## Examples
- `/sounio-fmt` - Format all .sio files in current directory
- `/sounio-fmt examples/` - Format all files in examples directory
- `/sounio-fmt src/module.sio --check` - Check if file is formatted
- `/sounio-fmt --check --diff` - Show formatting differences

$ARGUMENTS

Execute from the `compiler/` directory:

1. Determine target path (default to current directory if not specified)

2. Construct the format command:
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo run -- fmt <path> [--check] [--diff]
   ```

3. Argument handling:
   - `--check` - Only check, don't modify (exit code 1 if unformatted)
   - `--diff` - Show unified diff of changes
   - `--max-width` - Pass to formatter for line width control

4. Report results:
   - If formatting: List files modified
   - If check mode: List files that need formatting
   - If diff mode: Show the diff output

5. For the Rust compiler code itself, also mention `cargo fmt` can be used
