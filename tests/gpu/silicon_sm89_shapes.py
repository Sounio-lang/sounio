#!/usr/bin/env python3
"""Slurm-only sm89 SASS image gate against exact CPU and PTX controls."""

import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SHAPES = [
    (1, 1), (1, 2), (2, 1), (2, 7), (31, 5), (31, 64),
    (32, 65), (33, 128), (63, 17), (64, 256), (65, 576),
    (128, 7), (129, 1024), (576, 576), (1023, 511),
    (1024, 2048), (1024, 4096),
]


def compile_and_run(source, output):
    env = os.environ.copy()
    env["SOUNIO_STDLIB_PATH"] = str(ROOT / "stdlib")
    subprocess.run(
        [str(ROOT / "bin/souc"), "check", str(source)],
        cwd=ROOT, env=env, check=True,
    )
    subprocess.run(
        [str(ROOT / "bin/souc"), "compile", str(source), "-o", str(output)],
        cwd=ROOT, env=env, check=True,
    )
    return subprocess.run([str(output)], capture_output=True, check=True, timeout=60).stdout


def emitted_images(output):
    lines = output.decode().splitlines()
    images = {}
    pos = 0
    while pos < len(lines):
        fields = lines[pos].split()
        if len(fields) != 3 or fields[0] != "SHAPE":
            raise RuntimeError(f"unexpected image header: {lines[pos]}")
        shape = (int(fields[1]), int(fields[2]))
        words = lines[pos + 1:pos + 364]
        if len(words) != 363 or shape in images:
            raise RuntimeError(f"incomplete or duplicate image: {shape}")
        images[shape] = b"".join(
            struct.pack("<Q", int(word) & ((1 << 64) - 1)) for word in words
        )
        pos += 364
    if set(images) != set(SHAPES):
        raise RuntimeError("unexpected set of image shapes")
    return images


def cuda_driver():
    lib = C.CDLL("libcuda.so.1")
    p, u64, u32, i32, size = C.c_void_p, C.c_uint64, C.c_uint, C.c_int, C.c_size_t
    signatures = {
        "cuInit": [u32],
        "cuDeviceGet": [C.POINTER(i32), i32],
        "cuDevicePrimaryCtxRetain": [C.POINTER(p), i32],
        "cuCtxSetCurrent": [p],
        "cuMemAlloc_v2": [C.POINTER(u64), size],
        "cuMemFree_v2": [u64],
        "cuMemcpyHtoD_v2": [u64, p, size],
        "cuMemcpyDtoH_v2": [p, u64, size],
        "cuModuleLoadData": [C.POINTER(p), p],
        "cuModuleGetFunction": [C.POINTER(p), p, C.c_char_p],
        "cuLaunchKernel": [p, u32, u32, u32, u32, u32, u32, u32, p, C.POINTER(p), p],
        "cuCtxSynchronize": [],
        "cuModuleUnload": [p],
    }
    for name, args in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = i32

    def call(name, *args):
        status = getattr(lib, name)(*args)
        if status:
            raise RuntimeError(f"{name} returned CUDA error {status}")

    call("cuInit", 0)
    device = i32()
    call("cuDeviceGet", C.byref(device), 0)
    context = p()
    call("cuDevicePrimaryCtxRetain", C.byref(context), device)
    call("cuCtxSetCurrent", context)
    return call


def load_module(call, image, name):
    data = C.create_string_buffer(image)
    module, function = C.c_void_p(), C.c_void_p()
    call("cuModuleLoadData", C.byref(module), C.cast(data, C.c_void_p))
    call("cuModuleGetFunction", C.byref(function), module, name)
    return module, function


def run_case(call, sass, ptx, n, k, seed, fractional, grid_image=False):
    p, u64, i32 = C.c_void_p, C.c_uint64, C.c_int
    guard = 256
    span = max(n, k)
    payload = (2 * n + span + n * k) * 4
    total = payload + 2 * guard
    allocation = u64()
    call("cuMemAlloc_v2", C.byref(allocation), total)
    try:
        base = allocation.value + guard
        if base >> 32 != (base + payload + 4 * n - 1) >> 32:
            raise RuntimeError("allocation crosses a 4-GiB address boundary")
        initial = bytearray([0xA5]) * total
        for col in range(n):
            struct.pack_into("<f", initial, guard + 4 * col, 1234567.0)
            struct.pack_into("<I", initial, guard + 4 * n + 4 * col, 0xDEADBEEF)
        scale_x, scale_w = (8, 16) if fractional else (1, 1)
        vector = [((row * 3 + seed) % 7 - 3) / scale_x for row in range(k)]
        for row, x in enumerate(vector):
            struct.pack_into("<f", initial, guard + 8 * n + 4 * row, x)
        expected = [0.0] * n
        for row, x in enumerate(vector):
            for col in range(n):
                weight = ((row * 7 + col * 11 + seed) % 9 - 4) / scale_w
                struct.pack_into("<f", initial, guard + 4 * (2 * n + span + row * n + col), weight)
                expected[col] += x * weight
        host = C.create_string_buffer(bytes(initial), total)
        outputs = []
        for kind, function in (("sass", sass), ("ptx", ptx)):
            call("cuMemcpyHtoD_v2", allocation, C.cast(host, p), total)
            if kind == "sass":
                args = (u64(base),)
                block, grid = (128, (n + 127) // 128) if grid_image else (n, 1)
            else:
                args = (u64(base + 8 * n), u64(base + 4 * (2 * n + span)),
                        u64(base), i32(n), i32(k))
                block, grid = 128, (n + 127) // 128
            params = (p * len(args))(*(C.cast(C.byref(arg), p) for arg in args))
            call("cuLaunchKernel", function, grid, 1, 1, block, 1, 1, 0, None, params, None)
            call("cuCtxSynchronize")
            result = C.create_string_buffer(total)
            call("cuMemcpyDtoH_v2", C.cast(result, p), allocation, total)
            actual = result.raw
            dots = [struct.unpack_from("<f", actual, guard + col * 4)[0] for col in range(n)]
            tids = [struct.unpack_from("<I", actual, guard + 4 * n + 4 * col)[0] for col in range(n)]
            if dots != expected:
                bad = [(i, dots[i], expected[i]) for i in range(n) if dots[i] != expected[i]]
                raise RuntimeError(f"{kind} {n}x{k} seed={seed} fractional={fractional}: {bad[:8]}")
            if kind == "sass" and tids != list(range(n)):
                raise RuntimeError(f"{kind} {n}x{k}: incorrect thread IDs")
            if kind == "ptx" and tids != [0xDEADBEEF] * n:
                raise RuntimeError(f"{kind} {n}x{k}: unexpected TID buffer write")
            if (actual[:guard] != initial[:guard]
                    or actual[guard + payload:] != initial[guard + payload:]
                    or actual[guard + 8 * n:guard + payload] != initial[guard + 8 * n:guard + payload]):
                raise RuntimeError(f"{kind} {n}x{k}: canary or input modified")
            outputs.append(dots)
        if outputs[0] != outputs[1]:
            raise RuntimeError(f"SASS/PTX mismatch: {n}x{k}")
    finally:
        call("cuMemFree_v2", allocation)


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU gate must run inside an allocated Slurm job")
    with tempfile.TemporaryDirectory(prefix="sounio-sm89-shapes-") as tmp:
        tmp = Path(tmp)
        check = compile_and_run(ROOT / "examples/silicon_shape_probe.sio", tmp / "check")
        if b"SILICON SHAPE PASS" not in check:
            raise RuntimeError("structural/refusal check did not pass")
        images = emitted_images(
            compile_and_run(ROOT / "examples/silicon_shape_dump.sio", tmp / "dump")
        )
        ptx = (ROOT / "stdlib/hetero/kernels/gemv_f32.ptx").read_bytes()
        print("PTX_SHA256", hashlib.sha256(ptx).hexdigest(), flush=True)
        call = cuda_driver()
        ptx_module, ptx_function = load_module(call, ptx, b"gemv_f32")
        try:
            for n, k in SHAPES:
                image = images[(n, k)]
                module, function = load_module(call, image, b"stamp")
                try:
                    for fractional in (False, True):
                        for seed in (0, 1):
                            run_case(call, function, ptx_function, n, k, seed, fractional)
                    print(json.dumps({
                        "n": n, "k": k, "cases": 4,
                        "cubin_sha256": hashlib.sha256(image).hexdigest(),
                    }), flush=True)
                finally:
                    call("cuModuleUnload", module)
        finally:
            call("cuModuleUnload", ptx_module)
    print("SILICON SM89 SHAPES PASS", flush=True)


if __name__ == "__main__":
    main()
