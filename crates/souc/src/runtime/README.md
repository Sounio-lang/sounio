# Sounio Runtime Library

This directory contains the runtime support for compiled Sounio programs, including effect handler dispatch and continuation management.

## Building the Runtime Library

The runtime is compiled as a static library (`libsounio.a`) that can be linked with native backend executables:

```bash
cargo build --lib --release
```

This produces `target/release/libsounio.a` (approximately 139MB including debug symbols).

## Exported Runtime Functions

All runtime functions use the `#[no_mangle]` attribute and `extern "C"` calling convention for C-compatible linking.

### Handler Stack Management

```c
void __sounio_push_handler_io(void);
void __sounio_push_handler_mut(void);
void __sounio_push_handler_div(void);
void __sounio_push_handler_prob(void);
void __sounio_push_handler_alloc(void);
void __sounio_push_handler_panic(void);
void __sounio_push_handler_async(void);
void __sounio_push_handler_gpu(void);
void __sounio_push_handler_grad(void);
void __sounio_push_handler_network(void);
void __sounio_push_handler_sensor(void);
void __sounio_push_handler_exn(void);
void __sounio_push_handler_causal(void);

void __sounio_pop_handler(void);
int64_t __sounio_handler_depth(void);
```

### Effect Dispatch Functions

#### IO Effect

```c
double __sounio_dispatch_io_print(double value);
double __sounio_dispatch_io_println(double value);
double __sounio_dispatch_io_read(void);
```

#### Mut Effect

```c
double __sounio_dispatch_mut_get(double key);
double __sounio_dispatch_mut_set(double key, double value);
double __sounio_dispatch_mut_modify(double key, double delta);
```

#### Div Effect

```c
double __sounio_dispatch_div_div(double a, double b);
```

#### Prob Effect

```c
double __sounio_dispatch_prob_sample(double low, double high);
double __sounio_dispatch_prob_observe(double mean, double std, double value);
```

#### Alloc Effect

```c
double __sounio_dispatch_alloc_alloc(double size);
double __sounio_dispatch_alloc_dealloc(double ptr);
```

#### Panic Effect

```c
double __sounio_dispatch_panic_panic(void);
```

#### GPU Effect

```c
double __sounio_dispatch_gpu_sync(void);
double __sounio_dispatch_gpu_alloc(double size);
double __sounio_dispatch_gpu_free(double buffer_id);
double __sounio_dispatch_gpu_copy_htod(double buffer_id, double host_ptr, double size);
double __sounio_dispatch_gpu_copy_dtoh(double buffer_id, double host_ptr, double size);
double __sounio_dispatch_gpu_load_ptx(double name_ptr, double ptx_ptr);
double __sounio_dispatch_gpu_launch(double kernel_id, double grid_x, double grid_y,
                                     double grid_z, double block_x, double block_y,
                                     double block_z, double args_ptr);
```

#### Async Effect

```c
double __sounio_dispatch_async_spawn(double task_id);
double __sounio_dispatch_async_yield(void);
double __sounio_dispatch_async_await(double task_id);
double __sounio_dispatch_async_join(double task_count);
double __sounio_dispatch_async_select(double task_count);
```

#### Grad Effect

```c
double __sounio_dispatch_grad_forward(double value, double derivative);
double __sounio_dispatch_grad_reverse(double value, double adjoint);
```

#### Causal Effect

```c
double __sounio_dispatch_causal_do(double value);
double __sounio_dispatch_causal_observe(double value);
```

### Generic Dispatch

```c
double __sounio_dispatch_generic(const uint8_t* effect_ptr,
                                  const uint8_t* op_ptr,
                                  const double* args_ptr,
                                  size_t args_len);
```

## Usage with Native Backend

The native backend (AArch64/x86-64) generates calls to these runtime functions and creates relocations that are resolved during linking:

### 1. Generate Code with Runtime Calls

```rust
use sounio::backend::native::aarch64::AArch64Emitter;

let mut emitter = AArch64Emitter::new();

// Push IO handler
emitter.bl_external("__sounio_push_handler_io");

// Call print with value in V0
emitter.bl_external("__sounio_dispatch_io_print");

// Pop handler
emitter.bl_external("__sounio_pop_handler");

let code = emitter.finish()?;
let relocations = emitter.relocations();
```

### 2. Create ELF Object File

```rust
use sounio::backend::native::elf::{ElfWriter, ElfConfig};

let mut elf = ElfWriter::new(ElfConfig::default());
elf.add_text_section(&code);
elf.add_function("main", 0, code.len() as u64, true);

// Add undefined symbols
for symbol in emitter.external_symbols().keys() {
    elf.add_undefined_symbol(symbol);
}

let obj_data = elf.finish()?;
std::fs::write("program.o", obj_data)?;
```

### 3. Link with Runtime Library

```rust
use sounio::backend::native::linker::{Linker, LinkerConfig, LinkMode};

let config = LinkerConfig::executable()
    .with_runtime("target/release/libsounio.a");

let linker = Linker::new(config)?;
linker.link(&["program.o"], "program", LinkMode::Executable)?;
```

The resulting executable will have all runtime functions statically linked.

## Architecture

### Handler Stack

The runtime maintains a thread-local handler stack implemented in [handler_stack.rs](handler_stack.rs):

- **Global Stack**: A `Mutex<RuntimeHandlerStack>` accessed via `get_handler_stack()`
- **Push/Pop**: Handlers are pushed onto the stack when entering a `handle` block
- **Dispatch**: Effect operations search the stack from top to bottom
- **Fallback**: Default behavior if no handler found

### Continuation Support

Basic continuation support is available via:

```c
void* __sounio_capture_continuation(void);
void __sounio_resume_continuation(void* cont_ptr);
```

Full continuation support with multi-shot and delimited continuations is planned for Phase A (CPS Transformation).

## Testing

Runtime linking tests are in `tests/runtime_linking_test.rs`:

```bash
cargo test --test runtime_linking_test
```

Tests verify:
- Static library exists and has reasonable size
- Runtime symbols are exported correctly
- Code generation with runtime calls works
- Symbol registration and relocation tracking

## Future Work

- **Smaller Runtime**: Extract runtime-only code into a separate lightweight library
- **Platform Support**: Add x86-64, RISC-V, WebAssembly targets
- **Optimization**: LTO, dead code elimination, size optimization
- **Continuations**: Full CPS transformation with delimited continuations
- **JIT Integration**: Dynamic effect handler registration
