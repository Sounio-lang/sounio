# Explain compiler error codes

Get detailed explanations for Sounio compiler error codes.

## Arguments
- `<code>` - Error code to explain (e.g., E0001, T0042, L0003)

## Examples
- `/sounio-explain E0001` - Explain error E0001
- `/sounio-explain T0042` - Explain type error T0042
- `/sounio-explain L0003` - Explain linearity error L0003

$ARGUMENTS

Execute from the `compiler/` directory:

```bash
cd /home/demetrios/sounio-1/compiler && cargo run -- explain <code>
```

## Error Code Categories

**E#### - General Errors**
- E0001: Syntax error
- E0002: Unresolved name
- E0003: Duplicate definition
- E0004: Invalid expression

**T#### - Type Errors**
- T0001: Type mismatch
- T0002: Cannot infer type
- T0003: Invalid type annotation
- T0004: Generic constraint not satisfied
- T0010: Unit dimension mismatch
- T0020: Effect not declared

**L#### - Linearity Errors**
- L0001: Linear value used multiple times
- L0002: Linear value not consumed
- L0003: Cannot copy linear type
- L0004: Linear value dropped without consuming

**O#### - Ownership Errors**
- O0001: Use after move
- O0002: Cannot borrow as mutable
- O0003: Borrow conflicts with existing borrow
- O0004: Value does not live long enough

**F#### - Effect Errors**
- F0001: Effect not handled
- F0002: Effect not declared in signature
- F0003: Invalid effect handler
- F0004: Effect escape

**R#### - Refinement Errors**
- R0001: Refinement predicate not satisfied
- R0002: Cannot prove refinement
- R0003: SMT solver timeout

## Error Message Format

Sounio error messages include:
1. Error code and brief description
2. Source location (file:line:column)
3. Code snippet with highlighting
4. Suggestions for fixing the error
5. Related notes and hints

For complex errors, the explanation includes:
- Detailed description of what went wrong
- Common causes
- Examples of correct code
- Links to relevant documentation
