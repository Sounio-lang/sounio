# Build Sounio compiler or source files with backend and feature selection

Build the Sounio compiler or compile specific .sio files with various backend options and feature flags.

## Arguments
- `--backend <native|llvm|cranelift|gpu>` - Select compilation backend (default: native)
- `--features <list>` - Comma-separated feature flags (jit, llvm, lsp, smt, gpu, ontology, pkg, full)
- `--release` - Build in release mode with optimizations
- `[file]` - Optional specific .sio file to build

## Examples
- `/sounio-build` - Build with default settings
- `/sounio-build --release` - Release build
- `/sounio-build --features jit,smt` - Build with JIT and SMT features
- `/sounio-build --backend llvm --release` - LLVM backend release build
- `/sounio-build examples/hello.sio` - Build specific file

$ARGUMENTS

Execute from the `compiler/` directory:

1. Parse arguments to determine:
   - Backend selection (maps to feature flags)
   - Additional features requested
   - Release vs debug mode
   - Target file (if any)

2. Construct and run the cargo command:
   ```bash
   cd /home/demetrios/sounio-1/compiler && cargo build [--features FEATURES] [--release]
   ```

3. Feature flag mappings:
   - `--backend llvm` → `--features llvm`
   - `--backend cranelift` or `--backend jit` → `--features jit`
   - `--backend gpu` → `--features gpu`
   - `--features full` → all features enabled

4. If a specific .sio file is provided, use `cargo run -- build <file>` instead

5. Report build success/failure with any warnings or errors
