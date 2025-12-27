# Sounio Debugger Integration

This directory contains debugger integration tools for Sounio programs.

## Overview

Sounio programs compiled with debug info (`-g` flag) can be debugged with standard debuggers. The tools in this directory provide:

- **Pretty printers** for Sounio's core types (Knowledge, Quantity, Effects, etc.)
- **Custom commands** for inspecting epistemic values
- **VS Code integration** templates for a seamless debugging experience

## GDB Support

The GDB pretty printers are located at `../gdb/sounio_printers.py`.

### Setup

Add to your `~/.gdbinit`:

```python
python
import sys
sys.path.insert(0, '/path/to/sounio/tools/gdb')
import sounio_printers
sounio_printers.register_printers(None)
end
```

Or load manually in a GDB session:

```gdb
(gdb) source /path/to/sounio/tools/gdb/sounio_printers.py
```

### Features

The GDB integration provides:

- **Knowledge<T> pretty printing**: Shows value, confidence, source, and ontology
- **Quantity<T, U> pretty printing**: Shows value with unit symbol
- **Effect type display**: Shows active effects (IO, Mut, Alloc, etc.)
- **Linear/Affine type display**: Indicates linearity constraints

### Custom Commands

- `sounio-info <expr>` - Display detailed Sounio-specific information about a value
- `sounio-confidence` - List all Knowledge values in scope with their confidence levels

### Example Session

```gdb
(gdb) print measurement
Knowledge<f64> { value: 3.14159, confidence: 95.0%, source: Measurement, ontology: PATO:mass }

(gdb) print dosage
500.0 mg

(gdb) sounio-confidence
Knowledge values and confidence levels:
--------------------------------------------------
  measurement: 95.0%
  estimate: 80%
  prior: uncertain
```

## LLDB Support

The LLDB formatters are in `sounio_lldb.py`.

### Setup

Add to your `~/.lldbinit`:

```
command script import /path/to/sounio/tools/debug/sounio_lldb.py
```

Or load manually in an LLDB session:

```lldb
(lldb) command script import /path/to/sounio/tools/debug/sounio_lldb.py
```

### Features

Same features as GDB:

- Type summary providers for Knowledge, Quantity, Effect, Linear/Affine types
- Custom commands: `sounio-info`, `sounio-confidence`

### Example Session

```lldb
(lldb) p measurement
(Knowledge<f64>) Knowledge<f64> { value: 3.14159, confidence: 95.0%, source: Measurement }

(lldb) p dosage
(Quantity<f64, Milligram>) 500.0 mg

(lldb) sounio-confidence
Knowledge values and confidence levels:
--------------------------------------------------
  measurement: 95.0%
  estimate: 80%
```

## VS Code Integration

Copy the template files to your project's `.vscode/` directory:

```bash
mkdir -p .vscode
cp /path/to/sounio/tools/debug/launch.json.template .vscode/launch.json
cp /path/to/sounio/tools/debug/tasks.json.template .vscode/tasks.json
```

Then edit the files to match your project structure.

### Debug Configurations

The launch.json template includes:

1. **Debug Sounio Program (GDB)** - Debug a compiled binary with GDB
2. **Debug Sounio Program (LLDB)** - Debug a compiled binary with LLDB
3. **Debug Sounio Program (CodeLLDB)** - For the CodeLLDB VS Code extension
4. **Debug Current Sounio File (JIT)** - Run current file with JIT compiler under debugger
5. **Attach to Running Sounio Process** - Attach debugger to an already running process
6. **Debug Sounio Tests** - Debug test runner

### Tasks

The tasks.json template includes:

- Build tasks (debug, release)
- Check/typecheck task
- Run and JIT run tasks
- Format and lint tasks
- Test tasks
- Documentation generation

## Compiling with Debug Info

To enable debugging, compile with the `-g` flag:

```bash
# Using the CLI
souc build myprogram.sio -g

# Or with Cargo (for compiler development)
cargo run -- build myprogram.sio -g
```

This generates DWARF debug information including:

- Source file and line mappings
- Variable names and types
- Sounio-specific attributes (units, effects, epistemic state)

## Sounio-Specific DWARF Extensions

Sounio uses custom DWARF attributes to encode language-specific type information:

| Attribute | Value | Description |
|-----------|-------|-------------|
| `DW_AT_SOUNIO_EFFECTS` | 0x3000 | Effect set for function |
| `DW_AT_SOUNIO_UNIT` | 0x3001 | Unit type for Quantity |
| `DW_AT_SOUNIO_EPSILON` | 0x3002 | Confidence/epsilon value |
| `DW_AT_SOUNIO_PROVENANCE` | 0x3003 | Provenance source |
| `DW_AT_SOUNIO_ONTOLOGY` | 0x3004 | Ontology binding (CURIE) |
| `DW_AT_SOUNIO_LINEAR` | 0x3005 | Linearity kind |

The pretty printers use these to provide rich type information during debugging.

## Troubleshooting

### Pretty printers not loading

1. Check Python version (requires Python 3.6+)
2. Verify the path in your gdbinit/lldbinit
3. Try loading manually and check for errors

### Missing debug info

Ensure you compiled with `-g`:

```bash
souc build myprogram.sio -g --show-types
```

### Type information incomplete

The DWARF output may not capture all Sounio type features yet. Check the codegen debug module for the latest supported attributes.

## Development

To extend the pretty printers:

1. Add new type patterns to `SounioPrinterLookup.__call__` (GDB) or `__lldb_init_module` (LLDB)
2. Create a new printer class with `to_string()` method
3. Test with a sample program

For DWARF extensions, see `compiler/src/codegen/debug/`.
