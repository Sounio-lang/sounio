// Minimal C-ABI exports for Julia/Python interop.
//
// Build a shared library (requires LLVM backend):
//   cd compiler
//   cargo build --release --features llvm
//   ./target/release/dc build --cdylib ../examples/ffi_exports.d -O2
//
// Julia:
//   lib = "path/to/libffi_exports.so"  # .dylib on macOS, .dll on Windows
//   ccall((:add_i64, lib), Int64, (Int64, Int64), 2, 3) == 5
//
// Python (ctypes):
//   import ctypes
//   lib = ctypes.CDLL("path/to/libffi_exports.so")
//   lib.add_i64.argtypes = (ctypes.c_longlong, ctypes.c_longlong)
//   lib.add_i64.restype = ctypes.c_longlong
//   assert lib.add_i64(2, 3) == 5

// Export a simple pure function
pub extern "C" fn add_i64(a: i64, b: i64) -> i64 {
    a + b
}

// Export a function that writes to an output buffer (little-endian u64).
// Caller must provide at least 8 bytes.
pub extern "C" fn write_u64_le(out: *mut u8, value: u64) {
    var v: u64 = value;
    var i: usize = 0;
    while i < 8 {
        out[i as i64] = (v % 256) as u8;
        v = v / 256;
        i = i + 1;
    }
}

