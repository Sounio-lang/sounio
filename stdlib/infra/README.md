# stdlib/infra

Umbrella for infrastructure modules with pure implementations and FFI stubs.

| Module | Pure layer | FFI |
|--------|------------|-----|
| `queue` | 256-cap FIFO | RabbitMQ/Kafka stubs |
| `wasm` | Module bytecode buffer | Wasmtime stubs |
| `database` | In-memory tables | SQLite/libpq stubs |
| `mesh` | Fixed-capacity mesh | OpenGL/Vulkan stubs |
| `cache` | 64-entry KV cache | Redis stubs |
| `serial` | RX/TX ring buffers | libftdi stubs |
| `simulation` | Time series stats | GPU MC stubs |

## Tests

`tests/stdlib/{queue,wasm,database,mesh,cache,serial,simulation}/test_*_core.sio` (check-only)