# Fibonacci Sequence

## Python
```python
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

for i in range(10):
    print(fibonacci(i))
```

## Sounio
```sio
fn fibonacci(n: i64) -> i64 {
    if n <= 1 { return n }
    var a: i64 = 0
    var b: i64 = 1
    var i: i64 = 2
    while i <= n {
        let temp = b
        b = a + b
        a = temp
        i = i + 1
    }
    b
}

fn main() with IO, Mut, Panic, Div {
    var i: i64 = 0
    while i < 10 {
        print(fibonacci(i))
        print(" ")
        i = i + 1
    }
    println("")
}
```

## Key Differences
- No semicolons in Sounio
- `var` instead of reassignable variables
- `while` loop instead of `for _ in range()`
- No tuple unpacking `a, b = b, a + b` — use temp variable
- Effects: `with IO` for printing, `Mut` for variable mutation
