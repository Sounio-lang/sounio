# Sounio Error Catalog

This catalog documents all compiler error codes produced by the Sounio compiler (`souc`). Each error includes a description, common causes, and solutions.

## Error Code Organization

Sounio uses a structured error code system with the following prefixes:

| Code Range | Category | Description |
|------------|----------|-------------|
| E0001-E0099 | Syntax Errors | Lexical and parsing errors |
| E0100-E0199 | Type Errors | Type checking and inference errors |
| E0200-E0299 | Effect Errors | Effect system violations |
| E0300-E0399 | Ownership Errors | Ownership, borrowing, and linearity errors |
| E0400-E0499 | Unit Errors | Unit of measure errors |
| E0500-E0599 | Semantic Errors | Ontology and semantic type errors |

Additionally, parser-specific errors use a `P` prefix (P0001-P0099).

## How to Read Error Messages

Sounio error messages follow this structure:

```
error[E0100]: Type mismatch: expected `i32`, found `string`
  --> src/main.sio:15:10
   |
15 |     let x: i32 = "hello"
   |            ^^^   ^^^^^^^ expected `i32`
   |            |
   |            type annotation here
   |
   = help: convert the string to an integer with `parse()`
```

### Components

1. **Error Level and Code**: `error[E0100]` indicates this is an error with code E0100
2. **Message**: Brief description of the problem
3. **Location**: File path, line number, and column
4. **Source Context**: The relevant source code with annotations
5. **Labels**: Arrows (`^^^`) pointing to specific parts of the code
6. **Notes**: Additional context prefixed with `= note:`
7. **Help**: Suggestions for fixing the error prefixed with `= help:`

### Severity Levels

- **error**: Compilation will fail; must be fixed
- **warning**: Compilation proceeds but the code may have issues
- **info**: Informational message about the code
- **hint**: Suggestions for improvement

## Error Categories

### [Syntax Errors (E0001-E0099)](./E0001-E0099.md)

Errors that occur during lexical analysis and parsing:

- Unexpected tokens
- Unclosed delimiters
- Invalid characters
- Unterminated strings
- Invalid number literals

### [Type Errors (E0100-E0199)](./E0100-E0199.md)

Errors related to the type system:

- Type mismatches
- Cannot infer type
- Missing type annotations
- Invalid operations on types
- Generic constraint violations

### [Effect Errors (E0200-E0299)](./E0200-E0299.md)

Errors related to the algebraic effect system:

- Undeclared effects
- Unhandled effects
- Effect mismatches
- Effects in pure context

### [Ownership Errors (E0300-E0399)](./E0300-E0399.md)

Errors related to ownership, borrowing, and linear types:

- Use of moved value
- Double mutable borrow
- Linear value not consumed
- Affine value used multiple times

## Using the Error Index

When you encounter an error, you can:

1. **Look up the error code** in the appropriate section
2. **Read the description** to understand what went wrong
3. **Check the common causes** to identify your situation
4. **Apply the suggested solution**

### Example Lookup

If you see `error[E0300]: Use of moved value`, navigate to:

1. [Ownership Errors (E0300-E0399)](./E0300-E0399.md)
2. Find section for E0300
3. Read the description and solutions

## Getting Help

If an error message is unclear:

1. Use `souc check --explain E0100` to get detailed information about a specific error code
2. Check the [LLM Programming Guide](../../LLM_PROGRAMMING_GUIDE.md) for correct syntax
3. Review the [Grammar Reference](../grammar.md) for formal syntax rules

## Reporting Unclear Errors

If you find an error message confusing or unhelpful, please report it. Good error messages are a priority for Sounio. Include:

- The full error message
- The code that triggered it
- What you expected to happen
- What you think would make the message clearer
