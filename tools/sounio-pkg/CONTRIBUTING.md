# Contributing to sounio-pkg

Thank you for your interest in contributing to sounio-pkg, the package manager for the Sounio programming language! This guide will help you get started with development.

## 🎯 Development Philosophy

sounio-pkg follows these core principles:

1. **Epistemic First** - All operations consider uncertainty and provenance
2. **Scientific Rigor** - Code should be scientifically valid and reproducible
3. **User Experience** - Simple for beginners, powerful for experts
4. **Integration** - Seamless with existing Sounio ecosystem

## 🚀 Getting Started

### Prerequisites

- Sounio compiler (`souc`) version 0.5.0 or higher
- Git
- Basic command-line skills

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/sounio-pkg.git
cd sounio-pkg

# 2. Set up development environment
./scripts/setup-dev.sh

# 3. Build in development mode
./scripts/build-dev.sh

# 4. Run tests to verify setup
./scripts/test.sh
```

### Project Structure

```
sounio-pkg/
├── src/                    # Source code
│   ├── main.sio           # CLI entry point
│   ├── lib.sio            # Core library
│   ├── commands/          # Command implementations
│   ├── registry/          # Package registry client
│   ├── resolver/          # Dependency resolver
│   └── utils/             # Utilities
├── tests/                 # Test files
├── examples/              # Example packages
├── scripts/               # Build and development scripts
├── docs/                  # Documentation
└── sounio.toml           # Package manifest
```

## 🔧 Development Workflow

### 1. Choose an Issue

Check the [issue tracker](https://github.com/sounio-lang/sounio-pkg/issues) for:
- **Good first issues** - Labeled for new contributors
- **Bug fixes** - Issues with `bug` label
- **Feature requests** - Issues with `enhancement` label

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Changes

Follow the coding standards:

```sounio
// Good example
fn calculate_mean(data: Knowledge[f64]) -> Knowledge[f64] with Div {
    // Clear purpose
    let sum = array_reduce(data, fn(acc, x) { acc + x })
    let count = array_len(data) as f64
    
    // Propagate uncertainty
    Knowledge(
        value: sum.value / count
        ε: sum.ε / count
        prov: "mean_calculation"
    )
}

// Document public functions
/// Calculate the mean of epistemic data
/// 
/// # Arguments
/// * `data` - Array of Knowledge[f64] values
/// 
/// # Returns
/// Mean with propagated uncertainty
/// 
/// # Effects
/// Requires Div effect for division
pub fn mean(data: Knowledge[f64]) -> Knowledge[f64] with Div {
    calculate_mean(data)
}
```

### 4. Write Tests

```sounio
// Test file example
fn test_mean_basic() with IO {
    let data = [
        Knowledge(value: 1.0, ε: 0.1, prov: "test"),
        Knowledge(value: 2.0, ε: 0.1, prov: "test"),
        Knowledge(value: 3.0, ε: 0.1, prov: "test"),
    ]
    
    let result = mean(data)
    
    // Test value
    assert(abs(result.value - 2.0) < 0.001)
    
    // Test uncertainty propagation
    assert(result.ε > 0.0 && result.ε < 0.1)
    
    // Test provenance
    assert(str_contains(result.prov, "mean"))
    
    println("✅ test_mean_basic passed")
}
```

### 5. Run Tests

```bash
# Run all tests
./scripts/test.sh

# Run specific test
souc test --test test_mean_basic

# Run with verbose output
./scripts/test.sh --verbose
```

### 6. Update Documentation

- Update function documentation in source code
- Update README.md if needed
- Add examples for new features
- Update CHANGELOG.md

### 7. Submit Pull Request

1. Push your branch: `git push origin feature/your-feature`
2. Create PR on GitHub
3. Fill out PR template
4. Request review from maintainers

## 📝 Coding Standards

### Sounio Code Style

```sounio
// Use descriptive names
fn calculate_standard_deviation(data: Knowledge[f64]) -> Knowledge[f64]
// Not: fn calc_std_dev(d: Knowledge[f64]) -> Knowledge[f64]

// Declare effects explicitly
fn process_data(data: string) -> string with IO, Panic {
    // ...
}

// Use Knowledge types for uncertain values
let measurement = Knowledge(value: 42.0, ε: 0.5, prov: "sensor_reading")

// Handle errors gracefully
fn safe_divide(a: f64, b: f64) -> Knowledge[f64] with Div {
    if b == 0.0 {
        return Knowledge(value: 0.0/0.0, ε: 1.0, prov: "division_by_zero")
    }
    Knowledge(value: a / b, ε: 0.01, prov: "division")
}
```

### File Organization

- One logical concept per file
- Group related functions together
- Place public API in `lib.sio`
- Keep file sizes under 500 lines

### Documentation

```sounio
/// Brief description of function
/// 
/// Detailed explanation including:
/// - What the function does
/// - How uncertainty is handled
/// - Any side effects
/// - Example usage
/// 
/// # Arguments
/// * `param1` - Description with units if applicable
/// * `param2` - Description with confidence level
/// 
/// # Returns
/// Return type with uncertainty characteristics
/// 
/// # Effects
/// List of required effects
/// 
/// # Examples
/// ```sounio
/// let result = function_name(arg1, arg2)
/// println("Result: " + str(result.value) + " ± " + str(result.ε))
/// ```
/// 
/// # Errors
/// Conditions that may cause errors or panic
pub fn function_name(param1: Type1, param2: Type2) -> ReturnType with Effects {
    // Implementation
}
```

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests** - Test individual functions
2. **Integration Tests** - Test command-line interface
3. **Epistemic Tests** - Test uncertainty propagation
4. **Performance Tests** - Benchmark critical paths

### Test Structure

```sounio
// tests/unit/test_statistics.sio

// Import what you're testing
use sounio_pkg::statistics::{mean, variance}

// Test group
fn test_descriptive_statistics() with IO {
    println("🧪 Testing Descriptive Statistics")
    
    test_mean_basic()
    test_variance_uncertainty()
    test_edge_cases()
    
    println("✅ All descriptive statistics tests passed")
}

// Individual test
fn test_mean_basic() {
    // Arrange
    let data = create_test_data()
    
    // Act
    let result = mean(data)
    
    // Assert
    assert_value_within(result.value, 2.0, 0.001)
    assert_uncertainty_bounds(result.ε, 0.0, 0.1)
    assert_provenance_contains(result.prov, "mean")
}

// Helper functions
fn create_test_data() -> Knowledge[f64] {
    [
        Knowledge(value: 1.0, ε: 0.1, prov: "test"),
        Knowledge(value: 2.0, ε: 0.1, prov: "test"),
        Knowledge(value: 3.0, ε: 0.1, prov: "test"),
    ]
}
```

### Running Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration

# Run tests with coverage
make test-coverage

# Run specific test file
souc test tests/unit/test_statistics.sio
```

## 🔍 Code Review Process

### What We Look For

1. **Correctness** - Does the code work correctly?
2. **Uncertainty Handling** - Is uncertainty properly propagated?
3. **Performance** - Is the code efficient?
4. **Readability** - Is the code easy to understand?
5. **Testing** - Are there adequate tests?
6. **Documentation** - Is the code well-documented?

### Review Checklist

- [ ] Code follows Sounio conventions
- [ ] Uncertainty is properly handled with Knowledge types
- [ ] Effects are explicitly declared
- [ ] Tests exist and pass
- [ ] Documentation is updated
- [ ] No unnecessary dependencies added
- [ ] Performance considerations addressed
- [ ] Error handling is appropriate

## 🚀 Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** - Breaking changes
- **MINOR** - New features (backward compatible)
- **PATCH** - Bug fixes

### Release Steps

1. **Prepare Release Branch**
   ```bash
   git checkout -b release/v1.2.0
   ```

2. **Update Version**
   - Update `sounio.toml` version
   - Update `CHANGELOG.md`
   - Update any version references

3. **Run Full Test Suite**
   ```bash
   make test-all
   make integration-test
   make performance-test
   ```

4. **Build Release Artifacts**
   ```bash
   make release
   ```

5. **Create Release on GitHub**
   - Tag the release: `git tag v1.2.0`
   - Push tags: `git push --tags`
   - Create GitHub release with release notes

6. **Update Documentation**
   - Update website documentation
   - Update examples if needed
   - Announce on community channels

## 🐛 Bug Reports

### Reporting Bugs

Use the GitHub issue template and include:

1. **Description** - What happened vs what you expected
2. **Reproduction Steps** - Step-by-step how to reproduce
3. **Environment** - OS, Sounio version, sounio-pkg version
4. **Logs** - Any error messages or logs
5. **Uncertainty Context** - How uncertainty was involved

### Example Bug Report

```markdown
## Bug: Uncertainty not propagated in dependency resolution

**Description**: When resolving dependencies with low confidence, 
uncertainty is not properly propagated to the final resolution.

**Expected**: Resolution confidence should be product of dependency confidences
**Actual**: Resolution confidence is always 1.0

**Reproduction**:
1. Create package with low-confidence dependency
2. Run `sounio-pkg resolve`
3. Check output confidence

**Environment**:
- OS: Ubuntu 22.04
- Sounio: 0.5.0
- sounio-pkg: 0.1.0

**Logs**:
```
Resolving dependencies...
Confidence: 1.0 (should be 0.76)
```
```

## 💡 Feature Requests

### Suggesting Features

1. Check if feature already exists or is planned
2. Describe the problem you're solving
3. Explain how it fits with epistemic programming
4. Provide examples of usage
5. Consider implementation complexity

### Good Feature Request

```markdown
## Feature: Epistemic version constraints

**Problem**: Current version constraints don't consider uncertainty

**Solution**: Allow version constraints with confidence levels

**Example**:
```toml
[dependencies]
epistemic-stats = { version = "^1.0", min_confidence = 0.8 }
```

**Benefits**:
- Users can specify confidence requirements
- Better uncertainty propagation
- More scientific rigor

**Implementation Notes**:
- Extend dependency resolver
- Add confidence field to Dependency struct
- Update resolution algorithm
```

## 🏆 Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- GitHub contributors page
- Release notes
- Project documentation

## ❓ Getting Help

- **Discord**: Join our [Discord server](https://discord.gg/sounio)
- **GitHub Issues**: For bugs and feature requests
- **Documentation**: Check [docs.sounio.dev](https://docs.sounio.dev)
- **Email**: dev@sounio.dev

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 license.

---

Thank you for contributing to sounio-pkg and helping advance epistemic programming! 🎯
