# Effect system reference

Display reference for Sounio's algebraic effect system.

## Arguments
- `<effect>` - Specific effect: IO, Mut, Alloc, Panic, GPU, Prob, Async, Div
- `--handlers` - Show effect handler patterns
- `--custom` - Show custom effect definition syntax
- `--all` - Show all effects reference

## Examples
- `/sounio-effects IO` - IO effect documentation
- `/sounio-effects GPU` - GPU effect documentation
- `/sounio-effects --handlers` - Effect handler patterns
- `/sounio-effects --custom` - Define custom effects

$ARGUMENTS

Provide Sounio effect system reference:

## Built-in Effects

**IO** - Input/Output operations
```sio
fn read_file(path: string) -> string with IO {
    // File system access, network, console I/O
}

fn print(msg: string) with IO {
    // Console output
}
```

**Mut** - Mutable state
```sio
fn increment(counter: &!i32) with Mut {
    *counter = *counter + 1
}
```

**Alloc** - Memory allocation
```sio
fn create_buffer<T>(size: usize) -> Vec<T> with Alloc {
    Vec::with_capacity(size)
}
```

**Panic** - Recoverable failures
```sio
fn divide(a: i32, b: i32) -> i32 with Panic {
    if b == 0 {
        panic("division by zero")
    }
    a / b
}
```

**GPU** - GPU compute operations
```sio
kernel fn matmul(a: &[f32], b: &[f32], c: &![f32]) with GPU {
    // GPU kernel execution
}
```

**Prob** - Probabilistic computations
```sio
fn sample_normal(mean: f64, std: f64) -> f64 with Prob {
    // Probabilistic sampling
}
```

**Async** - Asynchronous operations
```sio
async fn fetch_data(url: string) -> Response with Async, IO {
    // Non-blocking I/O
}
```

**Div** - Divergence (non-termination)
```sio
fn loop_forever() -> ! with Div {
    loop { }
}
```

## Effect Handlers

```sio
handle expr with {
    IO.print(msg) => { /* custom print handling */ resume(()) }
    IO.read() => { resume("mocked input") }
}
```

## Custom Effects

```sio
effect Logger {
    fn log(level: LogLevel, msg: string) -> ()
    fn get_level() -> LogLevel
}

fn use_logger() with Logger {
    Logger.log(Info, "message")
}
```

## Effect Composition

Effects compose with comma separation:
```sio
fn complex_op() -> Result with IO, Mut, Alloc {
    // Can use all three effects
}
```
