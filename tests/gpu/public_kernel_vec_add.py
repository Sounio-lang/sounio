#!/usr/bin/env python3
"""Execute generated vec_add PTX; require exact results and intact guards.

Requires an allocated NVIDIA GPU and libcuda. Missing hardware is an error,
never a skip. The PTX ABI is (i64 n, u64 a, u64 b, u64 c), one block of 128.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ptx", type=Path)
    args = parser.parse_args()
    ptx = args.ptx.read_bytes()
    cuda = ct.CDLL("libcuda.so.1")
    pointer = ct.c_uint64
    handle = ct.c_void_p

    def bind(name, types):
        fn = getattr(cuda, name)
        fn.argtypes = types
        fn.restype = ct.c_int

        def checked(*values):
            rc = fn(*values)
            if rc:
                raise RuntimeError(f"{name} failed: CUDA error {rc}")
        return checked

    init = bind("cuInit", [ct.c_uint])
    get_device = bind("cuDeviceGet", [ct.POINTER(ct.c_int), ct.c_int])
    get_name = bind("cuDeviceGetName", [handle, ct.c_int, ct.c_int])
    get_version = bind("cuDriverGetVersion", [ct.POINTER(ct.c_int)])
    create = bind("cuCtxCreate_v2", [ct.POINTER(handle), ct.c_uint, ct.c_int])
    destroy = bind("cuCtxDestroy_v2", [handle])
    load = bind("cuModuleLoadData", [ct.POINTER(handle), handle])
    unload = bind("cuModuleUnload", [handle])
    function = bind("cuModuleGetFunction", [ct.POINTER(handle), handle, ct.c_char_p])
    alloc = bind("cuMemAlloc_v2", [ct.POINTER(pointer), ct.c_size_t])
    free = bind("cuMemFree_v2", [pointer])
    to_device = bind("cuMemcpyHtoD_v2", [pointer, handle, ct.c_size_t])
    to_host = bind("cuMemcpyDtoH_v2", [handle, pointer, ct.c_size_t])
    launch = bind("cuLaunchKernel", [
        handle, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint, ct.c_uint,
        ct.c_uint, ct.c_uint, handle, ct.POINTER(handle), ct.POINTER(handle),
    ])
    sync = bind("cuCtxSynchronize", [])
    init(0)
    device, version = ct.c_int(), ct.c_int()
    get_device(ct.byref(device), 0)
    get_version(ct.byref(version))
    name = ct.create_string_buffer(256)
    get_name(name, len(name), device)
    context, module = handle(), handle()
    create(ct.byref(context), 0, device)
    allocations = []
    loaded = False
    try:
        image = ct.create_string_buffer(ptx)
        load(ct.byref(module), image)
        loaded = True
        kernel = handle()
        function(ct.byref(kernel), module, b"vec_add")
        # Guard all inactive threads, plus a prefix and a suffix.
        count = 130
        array = ct.c_double * count
        for _ in range(3):
            ptr = pointer()
            alloc(ct.byref(ptr), ct.sizeof(array))
            allocations.append(ptr)
        cases = []
        for n in (1, 31, 32, 33, 127):
            a = array(*(i * 0.25 - 8.0 for i in range(count)))
            b = array(*(16.0 - i * 0.5 for i in range(count)))
            c = array(*([-4096.0] * count))
            originals = [bytes(a), bytes(b)]
            expected = list(c)
            for i in range(n):
                expected[i + 1] = a[i + 1] + b[i + 1]
            for ptr, host in zip(allocations, (a, b, c)):
                to_device(ptr, host, ct.sizeof(array))
            # Offset each pointer past its prefix canary.
            values = [ct.c_int64(n)] + [
                pointer(ptr.value + ct.sizeof(ct.c_double))
                for ptr in allocations
            ]
            params = (handle * 4)(*[
                ct.cast(ct.byref(value), handle) for value in values
            ])
            launch(kernel, 1, 1, 1, 128, 1, 1, 0, None, params, None)
            sync()
            for ptr, host in zip(allocations, (a, b, c)):
                to_host(host, ptr, ct.sizeof(array))
            if [bytes(a), bytes(b)] != originals:
                raise AssertionError(f"n={n}: input modified")
            for i, (actual, wanted) in enumerate(zip(c, expected)):
                if actual != wanted:
                    raise AssertionError(
                        f"n={n} offset={i}: got {actual}, expected {wanted}"
                    )
            cases.append(n)
        print(json.dumps({
            "status": "pass", "sizes": cases, "block_threads": 128,
            "device": name.value.decode(), "driver_version": version.value,
            "ptx_sha256": hashlib.sha256(ptx).hexdigest(),
            "comparison": "exact f64; all outputs, inputs and guards",
        }, sort_keys=True))
    finally:
        for ptr in allocations:
            free(ptr)
        if loaded:
            unload(module)
        destroy(context)


if __name__ == "__main__":
    main()
