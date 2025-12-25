import os
import ctypes

lib_path = os.environ.get("DEMETRIOS_LIB")
if not lib_path:
    raise SystemExit(
        "Set DEMETRIOS_LIB to the built shared library path (e.g., libffi_exports.so/.dylib/.dll)"
    )

lib = ctypes.CDLL(lib_path)

lib.add_i64.argtypes = (ctypes.c_longlong, ctypes.c_longlong)
lib.add_i64.restype = ctypes.c_longlong
assert lib.add_i64(2, 3) == 5

lib.write_u64_le.argtypes = (ctypes.POINTER(ctypes.c_ubyte), ctypes.c_ulonglong)
lib.write_u64_le.restype = None

buf = (ctypes.c_ubyte * 8)(*([0xAA] * 8))
lib.write_u64_le(buf, 4)
assert list(buf) == [4, 0, 0, 0, 0, 0, 0, 0]

print("OK")

