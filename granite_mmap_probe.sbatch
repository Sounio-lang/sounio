#!/usr/bin/env bash
#SBATCH --job-name=granite-mmap-probe
#SBATCH --comment=lane:llm-granite
#SBATCH --no-requeue
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --gres=none
#SBATCH --time=00:05:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/granite-mmap-probe-%j.log
set -uo pipefail
w=/tmp/granite-mmap-probe-$SLURM_JOB_ID
mkdir -p "$w"
rec() { srun --ntasks=1 --cpus-per-task=1 --mem=0 --gres=none --job-name="$(printf '%s' "$1" | tr '\n\t' '  ' | cut -c1-120)" true </dev/null >/dev/null 2>&1 || true; }
compiler=/tmp/granite-tail-29222/madaros
export SOUNIO_STDLIB_PATH=/tmp/granite-tail-29222/src/stdlib
cd "$w"
cat > probe.sio <<'SIO'
extern "C" {
    fn dlopen(filename: *mut u8, flags: i32) -> *mut u8
    fn dlsym(handle: *mut u8, symbol: *mut u8) -> *mut u8
    fn dlclose(handle: *mut u8) -> i32
}
fn get_sym(h: *mut u8, s: i64) -> *mut u8 with Alloc, Mut {
    let ptr = heap_alloc(8) as *mut i64
    (*ptr) = s
    let sym = dlsym(h, ptr as *mut u8)
    heap_free(ptr as *mut u8)
    sym
}
fn main() -> i64 with IO, Alloc, Mut, Panic {
    let h = dlopen(0 as *mut u8, 1)
    if h as i64 == 0 { return 1 }
    println("open-ok")
    let map_raw = get_sym(h, 0x70616d6d)
    let unmap_raw = get_sym(h, 0x70616d6e756d)
    let open_raw = get_sym(h, 0x6e65706f)
    let close_raw = get_sym(h, 0x65736f6c63)
    if map_raw as i64 == 0 || unmap_raw as i64 == 0 || open_raw as i64 == 0 || close_raw as i64 == 0 { return 2 }
    println("symbols-ok")
    let f_open = open_raw as fn(i64, i64, i64) -> i64
    let f_close = close_raw as fn(i64) -> i64
    let f_map = map_raw as fn(i64, i64, i64, i64, i64, i64) -> i64
    let f_unmap = unmap_raw as fn(i64, i64) -> i64
    let path = heap_alloc(16) as *mut i64
    write_i64(path, 0, 0x72657a2f7665642f)
    write_i64(path, 1, 0x6f)
    let fd = f_open(path as i64, 2, 0)
    heap_free(path as *mut u8)
    if fd < 0 { return 3 }
    println("fd-ok")
    let addr = f_map(0, 16384, 3, 2, fd, 0)
    println("map-returned")
    let _c = f_close(fd)
    if addr <= 0 { return 4 }
    let p = addr as *mut i64
    (*p) = 42
    if *p != 42 { return 5 }
    if f_unmap(addr, 16384) != 0 { return 6 }
    let _d = dlclose(h)
    println("MMAP_DYNAMIC_PASS")
    0
}
SIO
"$compiler" check probe.sio > check.log 2>&1
rec "mmap-stack-check-rc-$?"
"$compiler" build probe.sio -o probe > build.log 2>&1
rec "mmap-stack-build-rc-$?"
if [[ -f probe ]]; then chmod +x probe; ./probe > run.log 2>&1; rec "mmap-stack-run-rc-$?"; cat run.log | while IFS= read -r l; do rec "$l"; done; fi
grep -E 'error\[|= name|failed' check.log build.log | head -10 | while IFS= read -r l; do rec "$l"; done
exit 0
__CA_BUNDLE__
