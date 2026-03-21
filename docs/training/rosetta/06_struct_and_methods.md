# Structs and Methods

## Python
```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx * dx + dy * dy) ** 0.5

    def translate(self, dx: float, dy: float) -> 'Point':
        return Point(self.x + dx, self.y + dy)

p1 = Point(3.0, 4.0)
p2 = Point(0.0, 0.0)
print(f"Distance: {p1.distance_to(p2)}")  # 5.0
p3 = p1.translate(1.0, -1.0)
print(f"Translated: ({p3.x}, {p3.y})")    # (4.0, 3.0)
```

## Sounio
```sio
struct Point { x: f64, y: f64 }

fn pt_sqrt(x: f64) -> f64 with Mut, Panic, Div {
    if x <= 0.0 { return 0.0 }
    var guess = x
    if x > 1.0 { guess = x * 0.5 }
    var i: i64 = 0
    while i < 50 {
        guess = 0.5 * (guess + x / guess)
        i = i + 1
    }
    guess
}

fn distance_to(a: Point, b: Point) -> f64 with Mut, Panic, Div {
    let dx = a.x - b.x
    let dy = a.y - b.y
    pt_sqrt(dx * dx + dy * dy)
}

fn translate(p: Point, dx: f64, dy: f64) -> Point {
    Point { x: p.x + dx, y: p.y + dy }
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let p1 = Point { x: 3.0, y: 4.0 }
    let p2 = Point { x: 0.0, y: 0.0 }

    let dist = distance_to(p1, p2)
    print("Distance: ")
    print(dist)
    println("")

    let p3 = translate(p1, 1.0, 0.0 - 1.0)
    print("Translated: (")
    print(p3.x)
    print(", ")
    print(p3.y)
    println(")")
    0
}
```

## Key Differences
- **Structs** instead of `@dataclass`
- **Free functions** instead of methods — `distance_to(a, b)` not `a.distance_to(b)`
  - (Sounio also supports `impl` blocks with explicit `self`, but free functions are simpler)
- **By-value return** for `translate` — returns new `Point`, no mutation
- **No `**0.5`** — write your own `sqrt` via Newton's method
- **No f-strings** — use multiple `print()` calls
- **No unary minus** — `0.0 - 1.0` instead of `-1.0`
