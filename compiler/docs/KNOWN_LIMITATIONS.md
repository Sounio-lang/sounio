# Known Language Limitations

This document tracks current limitations in the Sounio language implementation. These are planned for future releases.

## Syntax Limitations

### Module System
- **Status**: Not implemented
- **Issue**: `module` and `use` statements are not supported
- **Workaround**: Define all code in a single file
- **Priority**: High

### Visibility Modifiers
- **Status**: Not implemented
- **Issue**: `pub` keyword is not recognized
- **Workaround**: Remove `pub` prefix from all declarations
- **Priority**: Medium

### Logical Operators
- **Status**: Partial
- **Issue**: `&&` (logical AND) operator not supported
- **Workaround**: Use nested `if` statements
```d
// Instead of:
if a > 0 && b > 0 { ... }

// Use:
if a > 0 {
    if b > 0 {
        ...
    }
}
```
- **Priority**: High

### Documentation Comments
- **Status**: Not implemented
- **Issue**: `///` doc comments are not parsed
- **Workaround**: Use regular `//` comments
- **Priority**: Low

### Numeric Literals
- **Status**: Partial
- **Issue**: Scientific notation (e.g., `1e10`) not supported
- **Workaround**: Use full decimal form (e.g., `10000000000.0`)
- **Priority**: Medium

## Type System Limitations

### Type Aliases
- **Status**: Not implemented
- **Issue**: `type` aliases are not supported
- **Workaround**: Use structs directly or duplicate type annotations
- **Priority**: Medium

### Unit Definitions
- **Status**: Not implemented
- **Issue**: Custom unit type definitions not supported
- **Workaround**: Use structs with fields to represent units
```d
// Instead of:
unit kg;

// Use:
struct Mass {
    value: f64
}
```
- **Priority**: Medium

## Reserved Keywords

The following identifiers are reserved and cannot be used as variable names:
- `var`
- `effect`
- `type`
- `module`
- `use`
- `pub`

## Scoping Behavior

### Variable Shadowing
- **Issue**: Reusing variable names in nested scopes can have unexpected behavior
- **Workaround**: Use unique variable names, especially in test code
- **Priority**: Medium

### Forward Declarations
- **Issue**: Functions must be defined before they are called
- **Workaround**: Order function definitions with dependencies first (helpers at top)
- **Priority**: High

## Planned Features

These limitations will be addressed in future releases:

| Feature | Target Version | Notes |
|---------|---------------|-------|
| Module system | 0.70.0 | Basic `module`/`use` support |
| `&&` / `\|\|` operators | 0.66.0 | Logical operators |
| `pub` visibility | 0.70.0 | With module system |
| Scientific notation | 0.67.0 | `1e10`, `1.5e-3` |
| Type aliases | 0.68.0 | `type Name = Type;` |
| Doc comments | 0.69.0 | `///` to documentation |

## Reporting Issues

If you encounter additional limitations not listed here, please report them at:
https://github.com/Chiuratto-AI/sounio/issues
