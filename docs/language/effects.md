---
title: Algebraic Effects
description: First-class effect system for tracking and handling computational side effects
prerequisites:
  - /docs/getting-started.md
  - /docs/language/functions.md
reading_time: 15 minutes
---

# Algebraic Effects

Sounio has a first-class algebraic effect system for tracking computational side effects. Effects make it explicit when functions perform operations like I/O, mutation, allocation, or probabilistic sampling. This enables the compiler to enforce effect safety at compile time and allows programmers to write effect handlers that customize how effects are interpreted.

## Why Effects Matter

In traditional programming languages, side effects are invisible. A function signature `fn process(data: string) -> string` tells you nothing about whether it reads files, makes network calls, allocates memory, or might panic. This invisibility leads to:

- **Hidden dependencies**: Functions may fail in unexpected ways
- **Difficult testing**: Mocking requires complex dependency injection
- **Uncertain resource usage**: Memory and I/O behavior is opaque
- **Unpredictable failure modes**: Errors can propagate silently

Sounio's effect system makes all of this explicit in the type system:

```sio
// This function's effects are documented in its signature
fn read_config(path: string) -> Config with IO, Panic {
    let content = read_file(path)
    return parse_config(content)
}
```

## Built-in Effects

Sounio provides several built-in effects for common computational patterns.

### IO - Input/Output Operations

The `IO` effect tracks operations that interact with the external world: file I/O, network calls, console output, and environment access.

```sio
fn greet(name: string) with IO {
    println("Hello, " ++ name ++ "!")
}

fn save_results(data: &[f64], path: string) with IO {
    let content = format_csv(data)
    write_file(path, content)
}

fn fetch_data(url: string) -> Response with IO {
    return http_get(url)
}
```

### Mut - Mutable State

The `Mut` effect tracks operations that modify mutable state. This includes writing to mutable variables and modifying data through exclusive references (`&!`).

```sio
fn increment_counter(counter: &!i32) with Mut {
    *counter = *counter + 1
}

fn shuffle(arr: &![i32]) with Mut {
    // Fisher-Yates shuffle modifies array in place
    var i = len(arr) - 1
    while i > 0 {
        let j = random_int(0, i)
        let temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp
        i = i - 1
    }
}
```

### Alloc - Memory Allocation

The `Alloc` effect tracks heap allocation. This is important for embedded systems, real-time applications, and understanding memory behavior.

```sio
fn create_buffer(size: usize) -> Vec<u8> with Alloc {
    return Vec::with_capacity(size)
}

fn clone_data<T: Clone>(data: &[T]) -> Vec<T> with Alloc {
    var result = Vec::new()
    for item in data {
        result.push(item.clone())
    }
    return result
}
```

### Panic - Non-local Failure

The `Panic` effect indicates that a function may abort execution. Operations like division, array indexing, and assertion failures can panic.

```sio
fn divide(a: i32, b: i32) -> i32 with Panic {
    if b == 0 {
        panic("division by zero")
    }
    return a / b
}

fn get_first<T>(arr: &[T]) -> &T with Panic {
    // Indexing may panic if array is empty
    return &arr[0]
}

fn assert_positive(x: f64) with Panic {
    if x <= 0.0 {
        panic("expected positive value")
    }
}
```

### Async - Asynchronous Computation

The `Async` effect marks operations that may suspend and resume later. This includes network requests, file I/O with async runtimes, and concurrent operations.

```sio
async fn fetch_user(id: i64) -> User with Async, IO {
    let response = http_get(format("/api/users/{}", id)).await
    return parse_json(response.body)
}

async fn process_batch(items: Vec<Item>) -> Vec<Result> with Async {
    var results = Vec::new()
    for item in items {
        let result = process_item(item).await
        results.push(result)
    }
    return results
}
```

### GPU - GPU Device Access

The `GPU` effect tracks operations that execute on a GPU device.

```sio
fn gpu_compute(data: &[f32]) -> Vec<f32> with GPU, Alloc {
    let n = len(data)
    let result = gpu.alloc::<f32>(n)

    let grid = ((n + 255) / 256, 1, 1)
    let block = (256, 1, 1)

    perform GPU.launch(process_kernel, grid, block)(data, result, n)
    perform GPU.sync()

    return result.to_vec()
}

kernel fn process_kernel(input: &[f32], output: &![f32], n: u32) {
    let i = gpu.thread_id.x + gpu.block_id.x * gpu.block_dim.x
    if i < n {
        output[i] = input[i] * 2.0
    }
}
```

### Prob - Probabilistic Computation

The `Prob` effect marks operations involving randomness or probabilistic inference.

```sio
fn sample_distribution() -> f64 with Prob {
    return sample(Normal(0.0, 1.0))
}

fn bayesian_inference(data: &[f64]) -> f64 with Prob {
    // Prior
    let theta = sample(Beta(1.0, 1.0))

    // Likelihood
    for observation in data {
        observe(Bernoulli(theta), observation)
    }

    return theta
}

fn monte_carlo_pi(samples: i32) -> f64 with Prob {
    var inside = 0
    for _ in 0..samples {
        let x = sample(Uniform(0.0, 1.0))
        let y = sample(Uniform(0.0, 1.0))
        if x*x + y*y <= 1.0 {
            inside = inside + 1
        }
    }
    return 4.0 * (inside as f64) / (samples as f64)
}
```

### Div - Division (Can Fail)

The `Div` effect specifically tracks division operations that might divide by zero. This is distinct from `Panic` to allow more granular effect handling.

```sio
fn safe_ratio(a: f64, b: f64) -> f64 with Div {
    return a / b
}

fn normalize(values: &[f64]) -> Vec<f64> with Div, Alloc {
    let total: f64 = values.iter().sum()
    return values.iter().map(|v| v / total).collect()
}
```

## Effect Annotations

Effects are declared using the `with` keyword after the return type:

```sio
// Single effect
fn read_file(path: string) -> string with IO {
    // ...
}

// Multiple effects
fn process_data(path: string) -> Result<Data, Error> with IO, Panic, Alloc {
    // ...
}

// Pure function (no effects) - no `with` clause needed
fn add(a: i32, b: i32) -> i32 {
    return a + b
}
```

### Effect Propagation

When a function calls another function with effects, those effects propagate upward:

```sio
fn helper() with IO {
    println("helper called")
}

fn main_process() with IO {
    // Calling helper() propagates its IO effect
    helper()
    println("done")
}

// ERROR: missing IO effect
fn broken() {
    helper()  // Compile error: helper has IO effect not declared here
}
```

## Custom Effects

You can define custom effects for domain-specific side effects:

```sio
// Define a custom effect
effect State<T> {
    fn get() -> T;
    fn put(value: T);
}

// Use the effect
fn counter() -> i32 with State<i32> {
    let current = perform State.get()
    perform State.put(current + 1)
    return current
}

// Define a logging effect
effect Log {
    fn log(level: LogLevel, message: string);
}

fn audited_operation(x: i32) -> i32 with Log {
    perform Log.log(LogLevel::Info, "Starting operation")
    let result = x * 2
    perform Log.log(LogLevel::Info, "Operation complete")
    return result
}
```

## Effect Handlers

Effect handlers define how effects are interpreted. This enables powerful patterns like testing, mocking, and alternative interpretations of side effects.

### Basic Handler Syntax

```sio
handler IntState for State<i32> {
    get() => resume(self.value),
    put(v) => {
        self.value = v
        resume(())
    }
}

fn main() {
    let result = handle {
        counter() + counter() + counter()
    } with IntState { value: 0 }
    // result = 0 + 1 + 2 = 3
}
```

### Exception-like Handlers

Effects can implement exception-like behavior with custom handling:

```sio
effect Exn {
    fn throw(message: string) -> !;
}

// Handler that converts to Option
fn safe_divide(a: i32, b: i32) -> Option<i32> {
    handle {
        if b == 0 {
            perform Exn.throw("division by zero")
        }
        a / b
    } with {
        throw(msg) => None,
        return(v) => Some(v),
    }
}

fn example() {
    let result1 = safe_divide(10, 2)  // Some(5)
    let result2 = safe_divide(10, 0)  // None
}
```

### IO Mocking for Testing

```sio
// Production IO handler
handler RealIO for IO {
    read_file(path) => {
        let content = system_read_file(path)
        resume(content)
    },
    write_file(path, content) => {
        system_write_file(path, content)
        resume(())
    },
}

// Test IO handler that uses in-memory storage
handler MockIO for IO {
    files: HashMap<string, string>,

    read_file(path) => {
        match self.files.get(path) {
            Some(content) => resume(content.clone()),
            None => resume(""),
        }
    },
    write_file(path, content) => {
        self.files.insert(path, content)
        resume(())
    },
}

// Use mock handler in tests
fn test_config_save() {
    let mock = MockIO { files: HashMap::new() }

    handle {
        save_config("/tmp/test.json", my_config)
        let loaded = load_config("/tmp/test.json")
        assert(loaded == my_config)
    } with mock
}
```

## Row Polymorphism for Effects

Sounio supports row polymorphism, allowing functions to be generic over their effects:

```sio
// map is polymorphic in the effect E
fn map<T, U, E>(f: fn(T) -> U with E, xs: &[T]) -> Vec<U> with E, Alloc {
    var result = Vec::new()
    for x in xs {
        result.push(f(x))
    }
    return result
}

// Works with pure functions
fn double(x: i32) -> i32 { return x * 2 }
let doubled = map(double, &[1, 2, 3])  // No effects other than Alloc

// Works with effectful functions
fn print_and_double(x: i32) -> i32 with IO {
    println(x.to_string())
    return x * 2
}
let printed = map(print_and_double, &[1, 2, 3])  // Has IO effect
```

This allows writing generic code that works with any effect combination.

## Pure Functions

Functions without any effect annotations are pure - they cannot perform side effects:

```sio
// Pure function: no IO, no mutation, no allocation
fn fibonacci(n: i32) -> i32 {
    if n <= 1 {
        return n
    }
    return fibonacci(n - 1) + fibonacci(n - 2)
}

// Pure functions have valuable properties:
// - Referentially transparent (same inputs = same outputs)
// - Safe to memoize
// - Safe to parallelize
// - Easy to test
```

Attempting to call effectful operations from a pure function is a compile-time error:

```sio
fn broken_pure(x: i32) -> i32 {
    println(x.to_string())  // ERROR: IO effect not declared
    return x * 2
}
```

## Effect Inference

The Sounio compiler infers effects from function bodies and checks them against declared signatures:

```sio
// Compiler infers that this function needs IO
fn needs_io() with IO {
    println("hello")  // IO inferred from println
}

// Compiler catches missing effect declarations
fn forgot_io() {
    println("oops")  // ERROR: undeclared IO effect
}

// Compiler infers multiple effects
fn complex_operation(x: i32) with IO, Panic {
    println("processing...")  // requires IO
    let y = 100 / x           // requires Panic (division)
    println("result: " ++ y.to_string())
}
```

## Best Practices

### 1. Minimize Effect Scope

Keep effectful code separated from pure logic:

```sio
// Good: pure computation separate from effects
fn compute(data: &[f64]) -> f64 {
    return data.iter().sum() / len(data) as f64
}

fn process_file(path: string) -> f64 with IO, Panic {
    let data = read_file(path)
    let values = parse_numbers(data)
    return compute(&values)  // Pure computation
}
```

### 2. Use Specific Effects

Prefer specific effects over broad ones:

```sio
// Less precise
fn risky(x: i32) with Panic { ... }

// More precise - documents the specific risk
fn divide_by(x: i32, y: i32) with Div { return x / y }
```

### 3. Document Effect Sources

When a function has multiple effects, consider commenting why:

```sio
fn analyze_data(path: string) -> Analysis with IO, Panic, Alloc {
    let content = read_file(path)       // IO: file reading
    let data = parse_csv(content)       // Panic: parse errors, Alloc: data structures
    return compute_statistics(&data)    // Alloc: result allocation
}
```

## See Also

- [Async Programming](/docs/language/async.md) - Async effects in detail
- [GPU Programming](/docs/language/gpu.md) - GPU effects and kernels
- [Probabilistic Programming](/docs/language/probabilistic.md) - Prob effects
- [Error Handling](/docs/language/errors.md) - Panic and error handling
- [LLM Programming Guide](/docs/LLM_PROGRAMMING_GUIDE.md) - Complete syntax reference
