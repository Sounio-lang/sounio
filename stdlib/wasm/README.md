# stdlib/wasm

WebAssembly runtime support and memory management.

## Pure API (`wasm::pure::types`)

- `WasmModule` — fixed bytecode buffer + export name table
- `wasm_module_new`, `wasm_module_load`, `wasm_module_add_export`

## FFI stubs

- `wasm::ffi::bindings::wasmtime_available()` → `false`
- `wasm::ffi::wrapper::wasm_runtime_ready()` → `false`

## Tests

- `tests/stdlib/wasm/test_wasm_core.sio` (check-only)