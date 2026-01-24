# Phase 3c: Stdlib Module Organization

## Overview

Phase 3c establishes the stdlib module system for Sounio, enabling imports of standard library items via `use` statements. This phase organizes stdlib modules with proper entry points and public function declarations.

## What Changed

### Module Entry Points Created

For each major stdlib module, we created lib.sio files to serve as module entry points:

- **math/lib.sio** - Re-exports from core, provides sin, cos, sqrt, PI, E, etc.
- **io/lib.sio** - Full I/O module with file operations, environment access
- **collections/lib.sio** - Vec, HashMap, HashSet, Deque
- **linalg/lib.sio** - Linear algebra operations
- **geometry/lib.sio** - Geometric algorithms
- **stats/lib.sio** - Statistical functions
- **causal/lib.sio** - Causal inference
- **epistemic/lib.sio** - Epistemic/uncertainty operations

### Function Visibility

All public functions in stdlib modules are now marked with `pub` keyword:

```sio
// math/core.sio
pub fn sin(x: f64) -> f64 { panic("sin builtin") }
pub fn cos(x: f64) -> f64 { panic("cos builtin") }
pub const PI: f64 = 3.141592653589793

// io/lib.sio
pub fn read_file(path: &str) -> Result<String, IoError> with IO { ... }
pub fn println(s: &str) with IO { ... }
pub mod env {
    pub fn args() -> Vec<String> with IO { ... }
}
```

### Module Re-export Pattern

Each module entry point (lib.sio) uses a standard re-export pattern:

```sio
// math/lib.sio
pub use core::*;
```

This allows:
```sio
use math
use math::sin
use math::PI
```

## Current State

### What Works

1. **Module loading**: The module loader successfully finds and loads lib.sio and mod.sio files
2. **Visibility enforcement**: Functions marked `pub` are accessible from outside modules
3. **Re-exports**: pub use statements properly re-export items

### Known Issues

1. **Builtin duplicates**: Math functions are defined both as:
   - Builtins in the resolver (register_builtins)
   - Function stubs in math/core.sio

   This causes duplicate definition errors when importing math. Resolution strategy:
   - Option 1: Remove builtins and rely on module system
   - Option 2: Prevent module-loaded functions from conflicting with builtins
   - Option 3: Give module definitions priority over builtins

2. **@extern syntax not yet working**: The @extern("name") syntax for FFI is not yet implemented in the parser. Current workaround is using panic stubs in math functions.

## Next Steps for Phase 3d

### Remove Builtin Math Functions

Update `compiler/src/resolve/resolver.rs` `register_builtins()` to:
- Remove math function registrations (sin, cos, sqrt, etc.)
- Keep dual number operations (dual_sin, dual_cos) if not yet exposed via modules
- Keep other builtins (print, panic, etc.)

### Implement @extern Syntax

Add parser support for `@extern("name")` in function bodies:
```sio
pub fn sin(x: f64) -> f64 { @extern("sin") }
```

This enables proper FFI bindings without requiring builtin registrations.

### Complete Other Stdlib Modules

- Update all remaining stdlib modules with lib.sio entry points
- Mark all public functions and types with `pub`
- Create sub-modules (e.g., random::distributions)

### Test Suite

Create `tests/module_imports/` with:
- Import of individual functions: `use math::sin`
- Glob imports: `use math::*`
- Qualified paths: `math::sin(0.0)`
- Constants from modules: `use math::PI`
- Re-exports: `pub use math::*`
- Nested modules: `use io::env::args`

## Architecture

### Module Loading Flow

```
use statement
  ↓
Resolver detects import
  ↓
Module loader finds stdlib module
  ↓
Loads math/lib.sio (or math/mod.sio)
  ↓
Parses function declarations
  ↓
Registers items in symbol table
  ↓
Returns to resolver with definitions
  ↓
Name resolution succeeds
```

### File Structure Pattern

```
stdlib/module_name/
├── lib.sio           # Module entry point (public API)
├── mod.sio           # Alternative entry point (falls back to lib.sio)
├── core.sio          # Core implementations
├── submodule1.sio    # Optional submodules
└── submodule2.sio
```

## Migration Path

### Phase 3c (Current)
- Organize stdlib with lib.sio entry points
- Mark functions as pub
- Module system loads them

### Phase 3d (Next)
- Remove duplicate builtins
- Fix @extern parser support
- Complete all stdlib modules

### Phase 3e (Follow-up)
- Package manager integration
- Versioning for stdlib modules
- Namespace/organization hierarchy

## Impact

This phase enables:
- **Cleaner APIs**: Functions imported from modules rather than global scope
- **Better organization**: Related functions grouped in modules
- **Clearer dependencies**: Use statements show what modules are needed
- **Extensibility**: Users can create their own modules following same pattern
- **Phase 3a+3b integration**: Visibility from type checker now properly constrains module imports

## References

- CLAUDE.md: Project working principles
- docs/MINIMUM_VIABLE_SOUNIO.md: What works today
- compiler/src/module_loader.rs: Module file discovery
- compiler/src/resolve/mod.rs: Name resolution with modules
