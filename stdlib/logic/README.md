# stdlib/logic

Logic programming and boolean operations.

## Key Types
- `Expr`: Logical expression (literal, variable, not, and, or, implies)
- `ExprType`: Expression type enum
- `KnowledgeBase`: Collection of logical expressions

## Key Functions
- `expr_new_literal(value)`: Create boolean literal
- `expr_new_var(name)`: Create variable expression
- `expr_new_not(expr)`: Create NOT expression
- `expr_new_and(left, right)`: Create AND expression
- `expr_new_or(left, right)`: Create OR expression
- `expr_new_implies(left, right)`: Create IMPLIES expression
- `knowledge_base_new()`: Create empty KB
- `knowledge_base_add(kb, expr)`: Add expression to KB
- `knowledge_base_size(kb)`: Get KB size

## Test Status
5/5 tests passing.