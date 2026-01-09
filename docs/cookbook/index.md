# Sounio Cookbook

This cookbook provides practical, working recipes for common tasks in Sounio. Each recipe is self-contained and can be adapted to your specific use case.

## How to Use This Cookbook

Each recipe follows a consistent format:

1. **Problem**: What you're trying to accomplish
2. **Solution**: Working Sounio code
3. **Discussion**: Why it works and important considerations
4. **See Also**: Related recipes and documentation

## Recipe Categories

### [Uncertainty Recipes](uncertainty-recipes.md)

Working with `Knowledge<T>` and epistemic values:

- Combining measurements from multiple sources
- Checking if a value is significantly above a threshold
- Propagating uncertainty through custom functions
- Handling correlated uncertainties
- Monte Carlo uncertainty for complex calculations

### [Pharmacokinetics Recipes](pk-recipes.md)

Building and using PK models:

- One-compartment IV bolus model
- Two-compartment model with oral absorption
- Population PK with random effects
- Dosing optimization with uncertainty
- Parameter sensitivity analysis

### [Data Loading](data-loading.md)

Getting data into Sounio:

- Loading CSV files
- Loading JSON data
- Handling missing values with uncertainty
- Converting raw data to Knowledge types

### [Error Handling](error-handling.md)

Robust error handling patterns:

- Result<T, E> patterns
- Effect-based error handling
- Combining errors from multiple sources
- Confidence-based fallbacks

## Quick Reference

### Creating Epistemic Values

```sio
// With standard uncertainty
let mass = epistemic_std(75.0, 0.5, 0.95)

// With interval bounds
let range = epistemic_interval(10.0, 20.0, 0.90)

// Exact value (no uncertainty)
let constant = epistemic_exact(3.14159, 1.0)
```

### Basic Propagation

```sio
// Arithmetic operations propagate uncertainty automatically
let sum = add_epistemic(a, b)
let product = mul_epistemic(a, b)
let ratio = div_epistemic(a, b)

// Confidence is min(a.conf, b.conf)
// Uncertainty grows via quadrature (GUM)
```

### Units of Measure

```sio
// Literals with units
let dose = 500.0_mg
let volume = 10.0_mL
let time = 24.0_h

// Compound units
let concentration: mg/mL = dose / volume  // Type-safe!
```

### Effect Annotations

```sio
// Declare effects in function signature
fn read_data(path: string) -> Data with IO { ... }
fn simulate() -> f64 with Prob, Alloc { ... }
fn process(x: &!f64) with Mut { ... }
```

## Code Style Conventions

### Use `&!` for Mutable References

```sio
// CORRECT: Sounio syntax
fn increment(x: &!i32) {
    *x = *x + 1
}

// WRONG: Rust syntax (does not work in Sounio)
// fn increment(x: &mut i32) { ... }
```

### Use `var` for Mutable Bindings

```sio
// CORRECT: Sounio syntax
var counter = 0
counter = counter + 1

// WRONG: Rust syntax (does not work in Sounio)
// let mut counter = 0
```

### Explicit Return Preferred

```sio
fn calculate(x: f64) -> f64 {
    let result = x * x + 2.0 * x + 1.0
    return result  // Explicit return preferred
}
```

## Contributing Recipes

When contributing new recipes:

1. Follow the Problem/Solution/Discussion format
2. Test all code examples
3. Use proper Sounio syntax (`&!`, `var`, no Rust macros)
4. Include effect annotations where appropriate
5. Demonstrate epistemic features when relevant
