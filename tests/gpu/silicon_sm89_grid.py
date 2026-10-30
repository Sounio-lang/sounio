#!/usr/bin/env python3
"""Slurm-only sm89 grid regression; checked Sounio launch plus CPU/PTX oracle."""

import hashlib
import os
from pathlib import Path
import struct
import subprocess
import tempfile

from silicon_sm89_shapes import cuda_driver, load_module, run_case


ROOT = Path(__file__).resolve().parents[2]
SHAPES = [
    (1, 1), (1, 4096), (31, 7), (127, 17), (128, 17), (129, 17),
    (255, 31), (256, 31), (257, 31), (576, 64), (1023, 7),
    (1024, 7), (1025, 7), (2049, 7), (4096, 7), (65535, 1),
]


def compile_source(source, output):
    env = os.environ.copy()
    env["SOUNIO_STDLIB_PATH"] = str(ROOT / "stdlib")
    for args in (["check", str(source)], ["compile", str(source), "-o", str(output)]):
        subprocess.run([str(ROOT / "bin/souc"), *args], cwd=ROOT, env=env, check=True)


def run_checked_probe(tmp):
    executable = tmp / "checked-probe"
    compile_source(ROOT / "examples/silicon_grid_probe.sio", executable)
    for repeat in range(2):
        result = subprocess.run(
            ["timeout", "--signal=TERM", "--kill-after=5s", "120s", str(executable)],
            check=False, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"checked launcher returned {result.returncode}: "
                f"{result.stdout}\n{result.stderr}"
            )
        lines = result.stdout.splitlines()
        expected = {
            f"GRID CASE {n} {k} {fractional}"
            for n, k in SHAPES
            for fractional in (0, 1)
        }
        cases = [line for line in lines if line.startswith("GRID CASE ")]
        if "GRID STRUCTURAL PASS" not in lines or "GRID CHECKED PASS" not in lines:
            raise RuntimeError(f"checked launcher did not complete: {result.stdout}")
        if set(cases) != expected or len(cases) != len(expected):
            raise RuntimeError(f"unexpected checked-launch cases: {cases}")
        print(f"CHECKED_LAUNCH_PASS repeat={repeat} cases={len(cases)}", flush=True)


def emit_grid_images(tmp):
    ns = ", ".join(str(n) for n, _ in SHAPES)
    ks = ", ".join(str(k) for _, k in SHAPES)
    source = tmp / "grid_dump.sio"
    source.write_text(
        "use hetero::silicon::image::{si_gemv_grid_sm89}\n"
        "fn main() with IO, Alloc, Mut, Div, Panic {\n"
        f" let ns: [i64; {len(SHAPES)}] = [{ns}]\n"
        f" let ks: [i64; {len(SHAPES)}] = [{ks}]\n"
        " var j: i64 = 0\n"
        f" while j < {len(SHAPES)} {{\n"
        "  let image = si_gemv_grid_sm89(ns[j], ks[j])\n"
        "  if !image.ok || image.nbytes != 3112 { exit(1) }\n"
        '  print("SHAPE "); print_int(ns[j]); print(" "); print_int(ks[j]); print("\\n")\n'
        "  var i: i64 = 0\n"
        '  while i < 389 { print_int(read_i64(image.ptr as *mut i64, i)); print("\\n"); i = i + 1 }\n'
        "  heap_free(image.ptr as *mut i8)\n"
        "  j = j + 1\n"
        " }\n"
        "}\n"
    )
    executable = tmp / "grid-dump"
    compile_source(source, executable)
    lines = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=60,
    ).stdout.splitlines()
    images = {}
    for pos in range(0, len(lines), 390):
        fields = lines[pos].split()
        if len(fields) != 3 or fields[0] != "SHAPE":
            raise RuntimeError(f"unexpected image header: {lines[pos]}")
        shape = (int(fields[1]), int(fields[2]))
        words = lines[pos + 1:pos + 390]
        if len(words) != 389 or shape in images:
            raise RuntimeError(f"incomplete or duplicate image: {shape}")
        images[shape] = b"".join(
            struct.pack("<Q", int(word) & ((1 << 64) - 1)) for word in words
        )
    if set(images) != set(SHAPES):
        raise RuntimeError("unexpected set of grid images")
    return images


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU gate must run inside an allocated Slurm job")
    with tempfile.TemporaryDirectory(prefix="sounio-sm89-grid-") as directory:
        tmp = Path(directory)
        run_checked_probe(tmp)
        images = emit_grid_images(tmp)
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
                            run_case(call, function, ptx_function, n, k, seed, fractional, True)
                    print(
                        f"GRID_ORACLE_PASS n={n} k={k} cases=4 "
                        f"sha256={hashlib.sha256(image).hexdigest()}",
                        flush=True,
                    )
                finally:
                    call("cuModuleUnload", module)
        finally:
            call("cuModuleUnload", ptx_module)
    print("SILICON SM89 GRID PASS", flush=True)


if __name__ == "__main__":
    main()
