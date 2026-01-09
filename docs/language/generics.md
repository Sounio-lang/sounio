---
title: Generics in Sounio
description: Generic programming with type parameters, trait bounds, and effect polymorphism
prerequisites:
  - ./syntax-reference.md
  - ./type-system.md
  - ./ownership-borrowing.md
reading_time: 18 minutes
---

# Generics in Sounio

Generics enable writing code that works with multiple types while maintaining type safety. Sounio supports type parameters, trait bounds, where clauses, and effect polymorphism.

## Generic Functions

### Basic Generic Functions

```sio
// Single type parameter
fn identity<T>(x: T) -> T {
    return x
}

// Multiple type parameters
fn pair<A, B>(first: A, second: B) -> (A, B) {
    return (first, second)
}

// Usage - types are inferred
let n = identity(42)           // T = i32
let s = identity("hello")      // T = string
let p = pair(1, "one")         // A = i32, B = string
```

### Generic Functions with References

**Remember: Sounio uses `&!T` for exclusive/mutable references, NOT `&mut T`.**

```sio
// Swap two values
fn swap<T>(a: &!T, b: &!T) {
    let temp = *a
    *a = *b
    *b = temp
}

// Usage
var x = 10
var y = 20
swap(&!x, &!y)
// x = 20, y = 10
```

### Generic Functions with Slices

```sio
// Find first element matching predicate
fn find<T>(arr: &[T], pred: fn(&T) -> bool) -> Option<&T> {
    for item in arr {
        if pred(item) {
            return Some(item)
        }
    }
    return None
}

// Sum all elements
fn sum<T: Add>(arr: &[T]) -> T {
    var total = T::zero()
    for item in arr {
        total = total + *item
    }
    return total
}
```

## Generic Structs

### Basic Generic Structs

```sio
// Single type parameter
struct Container<T> {
    value: T,
}

// Multiple type parameters
struct Pair<A, B> {
    first: A,
    second: B,
}

// Usage
let int_container: Container<i32> = Container { value: 42 }
let pair: Pair<string, f64> = Pair { first: "pi", second: 3.14159 }
```

### Generic Structs with Methods

```sio
struct Stack<T> {
    items: Vec<T>,
}

impl<T> Stack<T> {
    fn new() -> Self {
        return Stack { items: Vec::new() }
    }

    fn push(&!self, item: T) {
        self.items.push(item)
    }

    fn pop(&!self) -> Option<T> {
        return self.items.pop()
    }

    fn peek(&self) -> Option<&T> {
        if self.items.is_empty() {
            return None
        }
        return Some(&self.items[self.items.len() - 1])
    }

    fn is_empty(&self) -> bool {
        return self.items.is_empty()
    }
}
```

## Generic Enums

```sio
// Option type
enum Option<T> {
    Some(T),
    None,
}

// Result type with two parameters
enum Result<T, E> {
    Ok(T),
    Err(E),
}

// Usage
let maybe: Option<i32> = Option::Some(42)
let result: Result<string, Error> = Result::Ok("success")
```

## Trait Bounds

Trait bounds constrain type parameters to types that implement specific traits.

### Basic Trait Bounds

```sio
// T must implement Display
fn print<T: Display>(x: T) {
    println(x.to_string())
}

// T must implement both Clone and Debug
fn log_and_clone<T: Clone + Debug>(x: T) -> T {
    println(x.debug())
    return x.clone()
}
```

### Common Traits

| Trait | Purpose |
|-------|---------|
| `Clone` | Create a copy of a value |
| `Copy` | Implicit copy on assignment |
| `Debug` | Debug formatting |
| `Display` | User-facing formatting |
| `Eq` | Equality comparison |
| `Ord` | Ordering comparison |
| `Default` | Default value |
| `Hash` | Hash computation |
| `Add`, `Sub`, `Mul`, `Div` | Arithmetic operations |

### Examples with Trait Bounds

```sio
// Generic comparison
fn max<T: Ord>(a: T, b: T) -> T {
    if a > b { return a }
    return b
}

// Generic default
fn or_default<T: Default>(opt: Option<T>) -> T {
    match opt {
        Some(v) => v,
        None => T::default(),
    }
}

// Multiple bounds
fn process<T: Clone + Display>(items: &[T]) {
    for item in items {
        let copy = item.clone()
        println(copy.to_string())
    }
}
```

## Where Clauses

For complex bounds, use `where` clauses for clarity.

### Basic Where Clauses

```sio
fn complex<T, U>(x: T, y: U) -> T
where
    T: Clone + Debug,
    U: Into<T>,
{
    println(x.debug())
    return y.into()
}
```

### Where Clauses with Multiple Constraints

```sio
fn transform<T, U, V>(input: T, converter: fn(T) -> U, finalizer: fn(U) -> V) -> V
where
    T: Clone,
    U: Debug,
    V: Default,
{
    let cloned = input.clone()
    let intermediate = converter(cloned)
    println(intermediate.debug())
    return finalizer(intermediate)
}
```

### Where Clauses on Impl Blocks

```sio
struct Wrapper<T> {
    value: T,
}

// Methods available for all T
impl<T> Wrapper<T> {
    fn new(value: T) -> Self {
        return Wrapper { value: value }
    }

    fn unwrap(self) -> T {
        return self.value
    }
}

// Methods only available when T: Clone
impl<T> Wrapper<T>
where
    T: Clone,
{
    fn clone_value(&self) -> T {
        return self.value.clone()
    }
}

// Methods only available when T: Display
impl<T> Wrapper<T>
where
    T: Display,
{
    fn print(&self) {
        println(self.value.to_string())
    }
}
```

## Associated Types

Traits can have associated types that are determined by implementations.

```sio
trait Iterator {
    type Item;

    fn next(&!self) -> Option<Self::Item>;
}

struct Counter {
    current: i32,
    max: i32,
}

impl Iterator for Counter {
    type Item = i32;

    fn next(&!self) -> Option<i32> {
        if self.current < self.max {
            let value = self.current
            self.current = self.current + 1
            return Some(value)
        }
        return None
    }
}
```

## Const Generics

Type parameters can also be compile-time constants.

```sio
// Array with compile-time known size
struct ArrayWrapper<T, const N: usize> {
    data: [T; N],
}

impl<T, const N: usize> ArrayWrapper<T, N> {
    fn len(&self) -> usize {
        return N
    }

    fn get(&self, index: usize) -> Option<&T> {
        if index < N {
            return Some(&self.data[index])
        }
        return None
    }
}

// Usage
let wrapper: ArrayWrapper<i32, 5> = ArrayWrapper { data: [1, 2, 3, 4, 5] }
let size = wrapper.len()  // 5
```

## Effect Polymorphism

Sounio supports polymorphism over effects, enabling functions to be generic over the effects they perform.

### Generic Effect Parameters

```sio
// Function polymorphic over effect E
fn map<T, U, effect E>(f: fn(T) -> U with E, xs: [T]) -> [U] with E {
    var result: Vec<U> = Vec::new()
    for x in xs {
        result.push(f(x))
    }
    return result
}

// Usage with different effects
fn pure_double(x: i32) -> i32 {
    return x * 2
}

fn logged_double(x: i32) -> i32 with IO {
    println("Doubling " + x.to_string())
    return x * 2
}

let data = [1, 2, 3, 4, 5]

// E = {} (pure)
let doubled = map(pure_double, data)

// E = {IO}
let logged = map(logged_double, data)
```

### Effect Rows

```sio
// Function with multiple effects
fn process<effect E>(input: string) -> Result<Data, Error> with E + IO {
    // IO effect is explicit, E represents additional effects
    let content = read_file(input)
    return parse(content)
}
```

## Instantiation and Monomorphization

Generic code is monomorphized at compile time, creating specialized versions for each concrete type used.

```sio
fn identity<T>(x: T) -> T { return x }

// These create separate compiled functions:
identity(42)      // Generates: identity_i32
identity("hi")    // Generates: identity_string
identity(3.14)    // Generates: identity_f64
```

## Generic Type Inference

The compiler infers type parameters when possible.

```sio
// Type parameter inferred from argument
let x = identity(42)  // T = i32

// Type parameter inferred from return type
let v: Vec<string> = Vec::new()  // T = string

// Sometimes explicit annotation needed
let empty = Vec::new()  // ERROR: cannot infer T
let empty: Vec<i32> = Vec::new()  // OK
```

### Turbofish Syntax

Explicitly specify type parameters using `::<>`.

```sio
// Explicit type parameter
let v = Vec::<i32>::new()

// Multiple type parameters
let result = parse::<i32>("42")

// Generic method call
let items = collection.into::<Vec<string>>()
```

## Common Generic Patterns

### Builder Pattern with Generics

```sio
struct QueryBuilder<T> {
    table: string,
    conditions: Vec<string>,
    _marker: PhantomData<T>,
}

impl<T> QueryBuilder<T> {
    fn new(table: string) -> Self {
        return QueryBuilder {
            table: table,
            conditions: Vec::new(),
            _marker: PhantomData,
        }
    }

    fn where_eq(&!self, field: string, value: string) -> &!Self {
        self.conditions.push(field + " = '" + value + "'")
        return self
    }

    fn build(&self) -> Query<T> {
        // Build and return query
    }
}
```

### Generic Containers

```sio
struct Cache<K, V>
where
    K: Hash + Eq,
{
    data: HashMap<K, V>,
    capacity: usize,
}

impl<K, V> Cache<K, V>
where
    K: Hash + Eq,
{
    fn new(capacity: usize) -> Self {
        return Cache {
            data: HashMap::new(),
            capacity: capacity,
        }
    }

    fn get(&self, key: &K) -> Option<&V> {
        return self.data.get(key)
    }

    fn insert(&!self, key: K, value: V) {
        if self.data.len() >= self.capacity {
            // Evict oldest entry
            self.evict_oldest()
        }
        self.data.insert(key, value)
    }
}
```

### Generic Algorithms

```sio
// Binary search
fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    var low: usize = 0
    var high = arr.len()

    while low < high {
        let mid = low + (high - low) / 2
        match arr[mid].cmp(target) {
            Ordering::Less => low = mid + 1,
            Ordering::Greater => high = mid,
            Ordering::Equal => return Some(mid),
        }
    }
    return None
}

// Merge sort
fn merge_sort<T: Ord + Clone>(arr: &![T]) {
    if arr.len() <= 1 {
        return
    }

    let mid = arr.len() / 2
    let left = arr[..mid].to_vec()
    let right = arr[mid..].to_vec()

    merge_sort(&!left)
    merge_sort(&!right)
    merge(&left, &right, arr)
}
```

## Limitations

### No Higher-Kinded Types (Yet)

Sounio does not currently support higher-kinded types.

```sio
// NOT supported
fn map_functor<F<_>, A, B>(f: fn(A) -> B, fa: F<A>) -> F<B>
```

### Trait Object Limitations

Trait objects have some limitations compared to static generics.

```sio
// Limited trait object support
let dynamic: Box<dyn Display> = Box::new(42)
```

## Best Practices

1. **Start concrete, then generalize** - Write specific code first, then add generics when you need reuse

2. **Use trait bounds sparingly** - Only add bounds that you actually need

3. **Prefer where clauses for complex bounds** - They're more readable

4. **Document type parameters** - Explain what constraints and semantics each parameter has

5. **Remember `&!T` not `&mut T`** - Sounio's exclusive reference syntax

```sio
/// Sorts a slice in place.
///
/// # Type Parameters
/// - `T`: The element type, must implement `Ord` for comparison
///
/// # Arguments
/// - `arr`: Exclusive reference to the slice to sort
fn sort<T: Ord>(arr: &![T]) {
    // Implementation
}
```

## See Also

- [Syntax Reference](./syntax-reference.md) - Core language syntax
- [Type System](./type-system.md) - Type system reference
- [Ownership and Borrowing](./ownership-borrowing.md) - Memory safety model
