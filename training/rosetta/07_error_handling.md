# Error Handling

## Python
```python
def safe_divide(a: float, b: float) -> float:
    if b == 0.0:
        raise ValueError("Division by zero")
    return a / b

def safe_sqrt(x: float) -> float:
    if x < 0:
        raise ValueError("Negative input")
    return x ** 0.5

try:
    result = safe_divide(10.0, 3.0)
    print(f"10/3 = {result}")
except ValueError as e:
    print(f"Error: {e}")

try:
    result = safe_divide(10.0, 0.0)
except ValueError as e:
    print(f"Error: {e}")  # Division by zero
```

## Sounio
```sio
// Error codes instead of exceptions
// Convention: 0 = success, nonzero = error
fn safe_divide(a: f64, b: f64) -> (f64, i32) with Div, Panic {
    if b == 0.0 { return (0.0, 1) }   // error code 1: division by zero
    (a / b, 0)                          // success
}

fn safe_sqrt(x: f64) -> (f64, i32) with Mut, Panic, Div {
    if x < 0.0 { return (0.0, 2) }    // error code 2: negative input
    var guess = x
    if x > 1.0 { guess = x * 0.5 }
    var i: i64 = 0
    while i < 50 {
        guess = 0.5 * (guess + x / guess)
        i = i + 1
    }
    (guess, 0)
}

fn main() -> i32 with IO, Mut, Panic, Div {
    // Success case
    let (result, err) = safe_divide(10.0, 3.0)
    if err == 0 {
        print("10/3 = ")
        print(result)
        println("")
    } else {
        println("Error: division by zero")
    }

    // Error case
    let (result2, err2) = safe_divide(10.0, 0.0)
    if err2 == 0 {
        print(result2)
    } else {
        println("Error: division by zero")
    }

    0
}
```

## Key Differences
- **No exceptions** — return `(value, error_code)` tuple
- **No try/catch** — check error code with `if err != 0`
- **Tuple destructuring** works: `let (result, err) = safe_divide(...)`
- **Error codes are integers** — convention: 0 = success
- This is similar to Go's error handling pattern
- Stdlib also provides `IntResult`/`FloatResult` structs for richer error info
