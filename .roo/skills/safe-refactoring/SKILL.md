---
name: safe-refactoring
description: Perform systematic, safety‑preserving refactoring of code in any language, using dependency analysis, invariant detection, and verification techniques to avoid introducing bugs.
---

# Safe Refactoring

## When to use this skill

Use this skill when you need to:

- Restructure code while preserving its external behavior
- Improve readability, maintainability, or performance without breaking functionality
- Migrate between API versions or library dependencies
- Split large functions or modules into smaller, cohesive units
- Rename identifiers consistently across a codebase

## When NOT to use this skill

- For adding new features (use the appropriate development skill)
- For fixing bugs that are not related to code structure
- For performance optimizations that change algorithmic complexity

## Inputs required

- The code to be refactored (file paths, repository location)
- The specific refactoring operation (rename, extract function, split module, etc.)
- Any invariants that must be preserved (e.g., “function X must always return a positive integer”)
- Existing test suite that can be used to validate correctness

## Workflow

1. **Understand the code and its dependencies**
   - Read the relevant source files and identify data flows
   - Map callers and callees of the functions/modules being changed
   - Note any implicit invariants (e.g., global state, side effects)

2. **Plan the refactoring**
   - Choose the refactoring steps and their order
   - Determine which tests will be affected and need updating
   - Decide whether to use automated tools (e.g., semantic grep, AST transformers)

3. **Execute the changes**
   - Apply one refactoring step at a time
   - After each step, run the test suite to ensure nothing broke
   - Use version control to create checkpoints (commits) for easy rollback

4. **Validate the result**
   - Run the full test suite (unit, integration, regression)
   - If available, run static analysis tools to detect new warnings
   - Verify that the refactored code meets the original invariants

5. **Update documentation**
   - Adjust comments, READMEs, and API references that refer to the changed code
   - Ensure any examples in documentation still work

## Examples

### Renaming a widely‑used function

```python
# Before
def compute_total(items):
    return sum(item.price for item in items)

# After
def calculate_total_cost(items):
    return sum(item.price for item in items)
```

Steps:
1. Find all call sites of `compute_total` (grep, AST search)
2. Update each call site to `calculate_total_cost`
3. Update the function definition
4. Run tests to ensure no missing references

### Extracting a helper function

```javascript
// Before
function processOrder(order) {
    // ... 30 lines of validation ...
    // ... 20 lines of pricing ...
    // ... 10 lines of inventory update ...
}

// After
function processOrder(order) {
    validateOrder(order);
    const total = calculateOrderTotal(order);
    updateInventory(order);
}
```

Steps:
1. Identify a cohesive block of code that can become a function
2. Extract it, choosing a descriptive name and parameters
3. Replace the original block with a call to the new function
4. Ensure any captured variables are passed as arguments

## Troubleshooting / edge cases

- **Hidden dependencies**: indirect dependencies (e.g., via global variables) may be missed; use dynamic analysis or careful reading.
- **Test coverage gaps**: if tests are lacking, consider writing additional validation before refactoring.
- **Language‑specific quirks**: some languages have subtle semantics (e.g., macro expansions, reflection) that can break after refactoring; consult language experts.

## Related concepts

- **Semantic grep**: pattern matching that understands code structure
- **AST‑based transformations**: using a language’s abstract syntax tree to automate changes
- **Invariant detection**: static or dynamic analysis to find properties that must hold

## Next steps

After completing a refactoring, consider:

- Running a broader test suite (integration, system tests) to catch cross‑module issues
- Performing a code review to ensure the new structure is clear and maintainable
- Updating any continuous‑integration configurations that depend on the old structure