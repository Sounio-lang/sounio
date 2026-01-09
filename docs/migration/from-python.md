# Migrating from Python to Sounio

This guide helps Python and NumPy developers understand Sounio and adopt idiomatic patterns for scientific computing.

## Quick Reference

| Python/NumPy | Sounio | Notes |
|--------------|--------|-------|
| `x = 5` | `let x = 5` or `var x = 5` | Explicit mutability |
| `def f(x):` | `fn f(x: i32) -> i32 {` | Types required |
| `# comment` | `// comment` | Different comment syntax |
| `np.array([1,2,3])` | `vec![1, 2, 3]` | Vectors are typed |
| `None` | `None` (in Option<T>) | Must be wrapped |
| `try/except` | `Result<T, E>` | Explicit error types |
| `with open()` | Effect system | Resource management |
| Dynamic typing | Static typing | All types known at compile time |

## Static Typing

The biggest difference: Sounio requires explicit types that are checked at compile time.

### Python

```python
def add(a, b):
    return a + b

x = add(1, 2)       # Works
y = add("a", "b")   # Also works
z = add([1], [2])   # Also works
```

### Sounio

```sio
fn add(a: i32, b: i32) -> i32 {
    return a + b
}

let x = add(1, 2)       // Works
// add("a", "b")        // Compile error: expected i32
// add(1.0, 2.0)        // Compile error: expected i32

// For different types, use separate functions or generics
fn add_f64(a: f64, b: f64) -> f64 {
    return a + b
}
```

### Type Inference

Sounio can infer types in many cases:

```sio
let x = 42          // Inferred as i32
let y = 3.14        // Inferred as f64
let z = "hello"     // Inferred as string
let v = vec![1, 2]  // Inferred as Vec<i32>
```

## Variables and Mutability

Python variables are always mutable. Sounio distinguishes immutable and mutable bindings.

### Python

```python
x = 5
x = 10  # Always allowed
```

### Sounio

```sio
let x = 5
// x = 10  // Compile error: cannot assign to immutable binding

var y = 5
y = 10  // OK: var creates mutable binding

const PI = 3.14159  // Compile-time constant
```

## Functions

### Python

```python
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

result = greet("Alice")
result = greet("Bob", greeting="Hi")
```

### Sounio

```sio
fn greet(name: string, greeting: string) -> string {
    return greeting ++ ", " ++ name ++ "!"
}

let result = greet("Alice", "Hello")

// For default arguments, use overloading or Option
fn greet_default(name: string) -> string {
    return greet(name, "Hello")
}
```

## Arrays and Vectors

### NumPy Arrays vs Sounio Vectors

```python
# NumPy
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = a * 2           # Element-wise: [2, 4, 6, 8, 10]
c = a + b           # Element-wise: [3, 6, 9, 12, 15]
d = np.sum(a)       # 15
e = a[1:4]          # Slicing: [2, 3, 4]
f = a[a > 2]        # Boolean indexing: [3, 4, 5]
```

```sio
// Sounio
let a = vec![1, 2, 3, 4, 5]

// Element-wise operations (explicit)
var b: Vec<i32> = vec![]
for x in a {
    b.push(x * 2)
}

// Or using map-style (when available)
let c = a.map(|x| x + b[x])

// Sum
var sum = 0
for x in a {
    sum = sum + x
}

// Slicing
let slice = a[1..4]  // [2, 3, 4]

// Filtering
var filtered: Vec<i32> = vec![]
for x in a {
    if x > 2 {
        filtered.push(x)
    }
}
```

### Matrix Operations

```python
# NumPy
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B  # Matrix multiplication
D = A.T    # Transpose
```

```sio
// Sounio (using stdlib matrix module)
use math::matrix::*

let A = Matrix::from_rows(vec![
    vec![1.0, 2.0],
    vec![3.0, 4.0],
])
let B = Matrix::from_rows(vec![
    vec![5.0, 6.0],
    vec![7.0, 8.0],
])
let C = matmul(A, B)    // Matrix multiplication
let D = transpose(A)    // Transpose
```

## None and Optional Values

### Python

```python
def find(items, target):
    for i, item in enumerate(items):
        if item == target:
            return i
    return None  # Not found

result = find([1, 2, 3], 2)
if result is not None:
    print(f"Found at index {result}")
```

### Sounio

```sio
fn find(items: &Vec<i32>, target: i32) -> Option<i64> {
    for i in 0..items.len() {
        if items[i] == target {
            return Some(i as i64)
        }
    }
    return None
}

let result = find(&vec![1, 2, 3], 2)
match result {
    Some(idx) => println("Found at index " ++ idx.to_string()),
    None => println("Not found"),
}
```

## Error Handling

### Python

```python
def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

try:
    result = divide(10, 0)
except ValueError as e:
    print(f"Error: {e}")
```

### Sounio

```sio
fn divide(a: f64, b: f64) -> Result<f64, string> {
    if b == 0.0 {
        return Err("Division by zero")
    }
    return Ok(a / b)
}

match divide(10.0, 0.0) {
    Ok(result) => println("Result: " ++ result.to_string()),
    Err(msg) => println("Error: " ++ msg),
}
```

## Classes vs Structs

### Python

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

p1 = Point(0, 0)
p2 = Point(3, 4)
d = p1.distance(p2)  # 5.0
```

### Sounio

```sio
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Point {
        return Point { x: x, y: y }
    }

    fn distance(self: &Point, other: &Point) -> f64 {
        let dx = self.x - other.x
        let dy = self.y - other.y
        return sqrt_f64(dx * dx + dy * dy)
    }

    fn to_string(self: &Point) -> string {
        return "Point(" ++ self.x.to_string() ++ ", " ++ self.y.to_string() ++ ")"
    }
}

let p1 = Point::new(0.0, 0.0)
let p2 = Point::new(3.0, 4.0)
let d = p1.distance(&p2)  // 5.0
```

## Loops

### Python

```python
# For loop
for i in range(10):
    print(i)

# For with enumerate
for i, item in enumerate(items):
    print(f"{i}: {item}")

# While loop
while condition:
    do_something()

# List comprehension
squares = [x**2 for x in range(10)]
```

### Sounio

```sio
// For loop
for i in 0..10 {
    println(i)
}

// For with index
for i in 0..items.len() {
    println(i.to_string() ++ ": " ++ items[i].to_string())
}

// While loop
while condition {
    do_something()
}

// No comprehensions - use explicit loop
var squares: Vec<i32> = vec![]
for x in 0..10 {
    squares.push(x * x)
}
```

## Uncertainty Propagation

This is where Sounio truly shines compared to Python. Python requires manual uncertainty handling; Sounio makes it automatic.

### Python (manual)

```python
import numpy as np
from uncertainties import ufloat

# Using uncertainties package
a = ufloat(10.0, 0.5)   # 10.0 +/- 0.5
b = ufloat(5.0, 0.2)    # 5.0 +/- 0.2
c = a * b               # Uncertainty propagates
print(c)                # 50.0+/-2.9
```

### Sounio (native)

```sio
use epistemic::core::*

// Epistemic values with uncertainty and confidence
let a = epistemic_std(10.0, 0.5, 0.95)  // value, std, confidence
let b = epistemic_std(5.0, 0.2, 0.90)

// Uncertainty propagates automatically through all operations
let c = mul_epistemic(a, b)

// Access components
println("Value: " ++ c.value.to_string())
println("Uncertainty: " ++ get_std_uncertainty(c).to_string())
println("Confidence: " ++ c.conf.to_string())

// Confidence intervals
let lo = get_interval_lo(c)
let hi = get_interval_hi(c)
println("95% CI: [" ++ lo.to_string() ++ ", " ++ hi.to_string() ++ "]")
```

### Key Differences

| Python (uncertainties) | Sounio |
|----------------------|--------|
| External package | Built-in |
| Uncertainty only | Uncertainty + confidence + provenance |
| Linear propagation | Multiple methods (GUM, Monte Carlo) |
| No confidence tracking | Confidence tracked through all operations |

## Units of Measure

### Python (pint)

```python
import pint
ureg = pint.UnitRegistry()

dose = 500 * ureg.mg
volume = 10 * ureg.mL
concentration = dose / volume
print(concentration.to('mg/mL'))  # 50.0 mg/mL
```

### Sounio (native)

```sio
let dose: mg = 500.0
let volume: mL = 10.0
let concentration: mg/mL = dose / volume  // Type-checked at compile time!

// Compile error if units don't match:
// let wrong: kg = dose  // Error: expected kg, got mg
```

## File I/O

### Python

```python
with open('data.csv', 'r') as f:
    content = f.read()

# Or
lines = open('data.csv').readlines()
```

### Sounio

```sio
// Effect annotation declares I/O side effect
fn load_file(path: string) -> string with IO {
    let content = read_file(path)
    return content
}

fn load_lines(path: string) -> Vec<string> with IO {
    let content = read_file(path)
    return content.lines().collect()
}
```

## Dictionary/HashMap

### Python

```python
data = {
    'name': 'Alice',
    'age': 30,
}
data['city'] = 'Boston'
print(data.get('name', 'Unknown'))
```

### Sounio

```sio
use std::collections::HashMap

var data: HashMap<string, string> = HashMap::new()
data.insert("name", "Alice")
data.insert("age", "30")  // Note: values must be same type
data.insert("city", "Boston")

match data.get("name") {
    Some(name) => println(name),
    None => println("Unknown"),
}
```

## Performance Characteristics

| Aspect | Python | Sounio |
|--------|--------|--------|
| Execution | Interpreted | Compiled (native/JIT) |
| Type checking | Runtime | Compile time |
| Memory management | GC | Ownership/RAII |
| Numeric loops | Slow (need NumPy) | Fast (native) |
| FFI | ctypes, Cython | Direct C ABI |
| Parallelism | GIL limits threads | No GIL |

### When Sounio is Faster

- Tight loops without NumPy
- Custom numeric algorithms
- Memory-intensive operations
- Multi-threaded code

### When Python is Easier

- Quick prototyping
- Using specialized libraries (pandas, scikit-learn)
- Interactive exploration

## Equivalent Patterns

### List Comprehension Alternative

```python
# Python
result = [x**2 for x in data if x > 0]
```

```sio
// Sounio
var result: Vec<f64> = vec![]
for x in data {
    if x > 0.0 {
        result.push(x * x)
    }
}
```

### Lambda Functions

```python
# Python
f = lambda x: x * 2
result = list(map(f, data))
```

```sio
// Sounio closures
let f = |x: f64| x * 2.0
var result: Vec<f64> = vec![]
for x in data {
    result.push(f(x))
}
```

### Context Managers

```python
# Python
with open('file.txt') as f:
    data = f.read()
# f is automatically closed
```

```sio
// Sounio - use linear types
fn process_file(path: string) with IO {
    let handle = open_file(path)  // Linear type
    let data = read_all(handle)
    close_file(handle)  // Must be called (enforced by type system)
}
```

## Migration Strategy

1. **Start with types**: Add type annotations to your Python code first
2. **Identify effects**: Note which functions do I/O, mutation, etc.
3. **Convert data structures**: Translate numpy arrays to Sounio vectors
4. **Add uncertainty**: Replace `float` with epistemic types where appropriate
5. **Add units**: Use dimensional types for physical quantities
6. **Test thoroughly**: Sounio's type system catches many bugs at compile time

## Further Reading

- [Sounio Language Guide](../language/index.md)
- [Epistemic Types](../epistemic/core.md)
- [Uncertainty Recipes](../cookbook/uncertainty-recipes.md)
- [Data Loading](../cookbook/data-loading.md)
