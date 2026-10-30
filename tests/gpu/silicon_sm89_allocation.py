#!/usr/bin/env python3
"""Linux CPU-only allocation-failure regression for the sm89 interfaces."""

import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE = r'''
use hetero::cuda_exec::{CudaExec}
use hetero::probe_cuda::{pb_cstr}
use hetero::silicon::gemv::{sg_launch_sm89}
use hetero::silicon::image::{si_own, si_gemv_sm89, si_gemv_grid_sm89}

extern "C" {
    fn dlopen(filename: *mut u8, flags: i32) -> *mut u8
    fn dlsym(handle: *mut u8, symbol: *mut u8) -> *mut u8
}

fn main() with IO, Alloc, Mut, Div, Panic {
    let ok = si_gemv_sm89(1, 1)
    if !ok.ok || (ok.ptr as i64) == 0 || ok.nbytes != 2904 { exit(10) }
    heap_free(ok.ptr as *mut i8)
    let handle = dlopen(0 as *mut u8, 1)
    if (handle as i64) == 0 { exit(11) }
    let missing = CudaExec { cuda: handle, ok: true }
    if sg_launch_sm89(&missing, 4096, 1, 1) != 3 { exit(17) }
    let get_name = pb_cstr("getrlimit")
    let set_name = pb_cstr("setrlimit")
    let raw_get = dlsym(handle, get_name)
    let raw_set = dlsym(handle, set_name)
    if (raw_get as i64) == 0 || (raw_set as i64) == 0 { exit(12) }
    heap_free(get_name as *mut i8)
    heap_free(set_name as *mut i8)
    let limits = heap_alloc(16)
    if (limits as i64) == 0 { exit(13) }
    let get_limit = raw_get as fn(i32, *mut u8) -> i32
    let set_limit = raw_set as fn(i32, *mut u8) -> i32
    if get_limit(9, limits) != 0 { exit(14) }
    write_i64(limits as *mut i64, 0, 0)
    if set_limit(9, limits) != 0 { exit(15) }
    if (heap_alloc(8) as i64) != 0 { exit(16) }
    TEST_BODY
    print("ALLOCATION REFUSAL PASS\n")
}
'''

CASES = {
    "single": '''
    let image = si_gemv_sm89(1, 1)
    if image.ok || (image.ptr as i64) != 0 || image.nbytes != 0 { exit(20) }
''',
    "own": '''
    let image = si_own()
    if image.ok || (image.ptr as i64) != 0 || image.nbytes != 0 { exit(21) }
''',
    "grid": '''
    let image = si_gemv_grid_sm89(1, 1)
    if image.ok || (image.ptr as i64) != 0 || image.nbytes != 0 { exit(22) }
''',
    "launcher": '''
    let exec = CudaExec { cuda: handle, ok: true }
    if sg_launch_sm89(&exec, 4096, 1, 1) != 2 { exit(23) }
''',
}


def main():
    env = os.environ.copy()
    env["SOUNIO_STDLIB_PATH"] = str(ROOT / "stdlib")
    with tempfile.TemporaryDirectory(prefix="sounio-sm89-allocation-") as directory:
        tmp = Path(directory)
        for name, body in CASES.items():
            source = tmp / f"{name}.sio"
            executable = tmp / name
            source.write_text(SOURCE.replace("TEST_BODY", body))
            for args in (["check", str(source)], ["compile", str(source), "-o", str(executable)]):
                subprocess.run([str(ROOT / "bin/souc"), *args], cwd=ROOT, env=env, check=True)
            result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
            if result.returncode != 0 or "ALLOCATION REFUSAL PASS" not in result.stdout:
                raise RuntimeError(
                    f"{name}: exit={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}"
                )
            print(f"{name}: PASS", flush=True)


if __name__ == "__main__":
    main()
