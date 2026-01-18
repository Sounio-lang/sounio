# Quick Sounio syntax reference

Display Sounio syntax reference for specific topics or the full language guide.

## Arguments
- `<topic>` - Topic to show: variables, functions, types, effects, units, gpu, epistemic, ffi, async, linear, pattern-matching
- `--all` - Show complete syntax reference
- `--examples` - Include code examples

## Examples
- `/sounio-syntax variables` - Variable declaration syntax
- `/sounio-syntax effects` - Effect system syntax
- `/sounio-syntax gpu --examples` - GPU kernel syntax with examples
- `/sounio-syntax --all` - Full reference

$ARGUMENTS

Provide Sounio syntax reference from the LLM Programming Guide.

**IMPORTANT: Sounio is NOT Rust. Key differences:**
- `var` for mutable variables (not `let mut`)
- `&!T` for mutable references (not `&mut T`)
- No Rust macros (`assert!`, `println!`, etc.)
- Effects are explicit with `with` keyword

Reference content by topic:

**variables:**
```sio
let x = 5              // immutable binding
var y = 10             // mutable binding
let z: i32 = 42        // explicit type annotation
```

**functions:**
```sio
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn greet(name: string) -> string with IO {
    print("Hello, " ++ name)
    name
}
```

**types:**
```sio
struct Point { x: f64, y: f64 }
enum Option<T> { Some(T), None }
type Alias = i32
linear struct FileHandle { fd: i32 }  // linear type
```

**effects:**
```sio
fn read_file(path: string) -> string with IO { ... }
fn mutate(x: &!i32) with Mut { ... }
fn allocate<T>() -> Box<T> with Alloc { ... }
```

**units:**
```sio
let dose: mg = 500.0
let volume: mL = 10.0
let concentration: mg/mL = dose / volume
```

**gpu:**
```sio
kernel fn vector_add(a: &[f32], b: &[f32], c: &![f32]) {
    let i = gpu.thread_id.x
    c[i] = a[i] + b[i]
}
```

**epistemic:**
```sio
let measurement: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
let confidence = measurement.confidence()
let provenance = measurement.provenance()
```

**linear:**
```sio
linear struct File { handle: i32 }
fn open(path: string) -> File with IO { ... }
fn close(file: File) with IO { ... }  // must be called exactly once
```

For complete reference, read: docs/LLM_PROGRAMMING_GUIDE.md
