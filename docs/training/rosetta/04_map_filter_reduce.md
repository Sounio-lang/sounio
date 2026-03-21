# Map, Filter, Reduce (Higher-Order Functions)

## Python
```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8]
doubled = list(map(lambda x: x * 2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
total = sum(numbers)  # reduce
print(doubled)  # [2, 4, 6, 8, 10, 12, 14, 16]
print(evens)    # [2, 4, 6, 8]
print(total)    # 36
```

## Sounio
```sio
// Named functions instead of lambdas
fn double(x: i64) -> i64 { x * 2 }
fn is_even(x: i64) -> i64 with Div, Panic { if x % 2 == 0 { 1 } else { 0 } }
fn add(a: i64, b: i64) -> i64 { a + b }

// Higher-order: map
fn map8(arr: [i64; 8], f: fn(i64) -> i64) -> [i64; 8] with Mut, Panic, Div {
    var out: [i64; 8] = [0; 8]
    var i: i64 = 0
    while i < 8 { out[i] = f(arr[i]); i = i + 1 }
    out
}

// Higher-order: filter (returns count of matches)
fn filter8(arr: [i64; 8], pred: fn(i64) -> i64 with Div, Panic, out: &![i64; 8]) -> i64 with Mut, Panic, Div {
    var count: i64 = 0
    var i: i64 = 0
    while i < 8 {
        if pred(arr[i]) != 0 {
            (*out)[count] = arr[i]
            count = count + 1
        }
        i = i + 1
    }
    count
}

// Higher-order: fold (reduce)
fn fold8(arr: [i64; 8], init: i64, f: fn(i64, i64) -> i64) -> i64 with Mut, Panic, Div {
    var acc = init
    var i: i64 = 0
    while i < 8 { acc = f(acc, arr[i]); i = i + 1 }
    acc
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let data: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8]

    // Map: double each
    let doubled = map8(data, double)
    // doubled = [2, 4, 6, 8, 10, 12, 14, 16]

    // Filter: keep evens
    var evens: [i64; 8] = [0; 8]
    let even_count = filter8(data, is_even, &! evens)
    // evens[0..4] = [2, 4, 6, 8], even_count = 4

    // Reduce: sum
    let total = fold8(data, 0, add)
    // total = 36

    print("Sum: ")
    print(total)
    println("")
    assert(total == 36)
    0
}
```

## Key Differences
- **No lambdas** — define named functions (`double`, `is_even`, `add`)
- **Function references** work as first-class values: `map8(data, double)`
- Fixed-size arrays require fixed-size operations (`map8`, `fold8`)
- Filter returns count + fills output array via `&!` mutable reference
- `is_even` returns `i64` (0/1) instead of `bool` for predicate compatibility
