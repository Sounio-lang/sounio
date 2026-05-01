# stdlib/wasm

WebAssembly runtime support and memory management.

## Key Types
- `WasmMemory`: WASM linear memory
- `WasmTable`: WASM function table
- `WasmGlobal`: WASM global variable

## Key Functions
- `wasm_memory_new()`: Create new memory
- `wasm_memory_grow(mem, pages)`: Grow memory by pages
- `wasm_memory_size(mem)`: Get current size in pages
- `wasm_table_new(size)`: Create function table
- `wasm_table_size(tbl)`: Get table size
- `wasm_global_new(value, mutable)`: Create global

## Test Status
2/2 tests passing.