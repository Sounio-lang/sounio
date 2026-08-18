#!/usr/bin/env python3
"""rc182_diagnose.py — runtime-context-base diagnostic for rc=182 failures.

Reads the runtime context DIRECTLY from the runtime mmap base (NOT by magic
scan). The runtime mmap is the 2 GiB anonymous rw-p region the entry
trampoline allocates via sys_mmap; the runtime context lives at the front of
it, and the handle table lives at runtime_base + 0xC000000.

Why base-relative and not magic-scan:
    Earlier versions found the 2^22 magic by scanning /proc/<pid>/mem and
    then read handle_count relative to the magic's address. That heuristic
    broke in two ways: (a) the .data section of the binary contains many
    static 4194304 occurrences (descriptor table entries with default
    counts) that polluted the candidate list, and (b) my earlier filter
    `hc < hcap` discarded the real context the moment handle_count climbed
    near capacity — which is precisely when the diagnostic matters most.

Reading from the runtime mmap base avoids both: the runtime context is a
single struct at a known offset (0..247), there is exactly one of it, and
its values stay coherent regardless of how close handle_count gets to
capacity.

Offsets used (per self-hosted/native/runtime_context.sio):
    runtime_base +   0  : heap_base
    runtime_base +   8  : heap_cursor
    runtime_base +  16  : heap_limit
    runtime_base +  24  : handle_table_base
    runtime_base +  32  : handle_count
    runtime_base +  40  : handle_capacity  (the wall, 4194304 = 2^22)
    runtime_base +  72  : pin_count        (LIVE proxy)
    runtime_base + 120  : gc_state_ptr     (pointer to gc_state, in .data)

Usage:
    python3 rc182_diagnose.py /path/to/binary.out
    SOUNIO_STDLIB_PATH=... python3 rc182_diagnose.py a.out
"""
import struct, subprocess, time, sys, os

HANDLE_CAPACITY_DEFAULT = 4194304  # 2^22 per gc.sio:64
RUNTIME_MMAP_MIN_BYTES  = 1024 * 1024 * 1024  # only consider mmap >= 1 GiB
SANE_COUNT_BOUND        = HANDLE_CAPACITY_DEFAULT * 4  # anything larger is race garbage


def find_runtime_mmap(pid):
    """Find the runtime mmap base: largest anonymous rw-p region."""
    with open(f"/proc/{pid}/maps") as f:
        maps = f.read()
    candidates = []
    for line in maps.splitlines():
        parts = line.split()
        if 'rw' not in parts[1]:
            continue
        # Skip named files (the binary's .data/.bss)
        if len(parts) >= 6 and parts[-1] not in ("anon", ""):
            continue
        start, end = [int(x, 16) for x in parts[0].split('-')]
        size = end - start
        if size < RUNTIME_MMAP_MIN_BYTES:
            continue
        candidates.append((start, end, size))
    if not candidates:
        return None
    # The runtime mmap is the 2 GiB region (the only one >= 1 GiB that is anon)
    candidates.sort(key=lambda c: -c[2])
    return candidates[0]


def read_u64(pid, addr):
    try:
        with open(f"/proc/{pid}/mem", "rb") as f:
            f.seek(addr)
            return struct.unpack("<Q", f.read(8))[0]
    except (OSError, ValueError):
        return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    cmd = sys.argv[1:]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pid = p.pid
    # Wait for the entry trampoline to mmap
    for _ in range(50):
        time.sleep(0.01)
        m = find_runtime_mmap(pid)
        if m is not None:
            break
    if m is None:
        print(f"# rc182_diagnose: no 2 GiB mmap found for pid={pid} (not a Sounio native ELF?)", file=sys.stderr)
        try: p.kill()
        except: pass
        p.wait()
        sys.exit(0)

    runtime_base = m[0]
    handle_table_base_offset = 24  # runtime_context_field_handle_table_base()
    handle_count_offset      = 32
    handle_capacity_offset   = 40
    pin_count_offset         = 72

    t_start = time.time()
    peak_handle = 0
    peak_pin = 0
    last_handle = 0
    last_pin = 0
    last_capacity = HANDLE_CAPACITY_DEFAULT
    wall_crossed = False
    wall_cross_t = None

    while p.poll() is None:
        hc = read_u64(pid, runtime_base + handle_count_offset)
        pc = read_u64(pid, runtime_base + pin_count_offset)
        hcap = read_u64(pid, runtime_base + handle_capacity_offset)
        if hc is not None and 0 <= hc < SANE_COUNT_BOUND:
            peak_handle = max(peak_handle, hc)
            last_handle = hc
        if pc is not None and 0 <= pc < SANE_COUNT_BOUND:
            peak_pin = max(peak_pin, pc)
            last_pin = pc
        if hcap is not None and hcap > 0:
            last_capacity = hcap
        if not wall_crossed and hc is not None and hcap is not None and hc >= hcap:
            wall_crossed = True
            wall_cross_t = time.time() - t_start
        time.sleep(0.005)

    rc = p.returncode
    t_total = time.time() - t_start

    if rc != 182:
        print(f"# rc182_diagnose: process exited rc={rc}, t={t_total:.3f}s, peak_handle={peak_handle}, peak_pin={peak_pin}")
        sys.exit(0)

    handle_capacity = last_capacity
    delta = handle_capacity - peak_handle
    pct_used = peak_handle * 100.0 / handle_capacity if handle_capacity else 0.0
    pct_pin  = peak_pin * 100.0 / handle_capacity if handle_capacity else 0.0

    print(f"")
    print(f"=== rc=182 DIAGNOSTIC ===")
    print(f"")
    print(f"  binary              : {cmd[0]}")
    print(f"  runtime             : t_total={t_total:.3f}s  wall_crossed_at={wall_cross_t}")
    print(f"  runtime_mmap        : 0x{runtime_base:x}  (read from base, no magic scan)")
    print(f"  handle_capacity     : {handle_capacity}  (2^22 default)")
    print(f"  peak_handle_count   : {peak_handle}  (last observed: {last_handle})")
    print(f"  peak_pin_count      : {peak_pin}  (last observed: {last_pin})  [LIVE proxy]")
    print(f"  delta_to_wall       : {delta}  ({pct_used:.4f}% of capacity used at peak)")
    print(f"  peak_pin_fraction   : {pct_pin:.4f}% of capacity")
    print(f"")
    print(f"=== INTERPRETATION ===")
    print(f"")
    if peak_pin == 0:
        print(f"  peak_pin = 0 means the runtime did not pin handles in this code path.")
        print(f"  peak_handle ({peak_handle}) hit the wall by ALLOCATION, not retention.")
        print(f"  → Reclamation of unpinned handles cannot help (there are none).")
        print(f"    The wall was hit by init footprint alone; the program needs more")
        print(f"    capacity OR less init footprint, not reclamation.")
    elif peak_pin < handle_capacity * 0.1:
        print(f"  peak_pin ({peak_pin}) is {peak_pin*100.0/peak_handle:.4f}% of peak_handle ({peak_handle}).")
        print(f"  → Reclamation WOULD HELP. Most allocated handles are unpinned.")
    elif peak_pin < handle_capacity:
        print(f"  peak_pin ({peak_pin}) is {peak_pin*100.0/peak_handle:.4f}% of peak_handle ({peak_handle}).")
        print(f"  → Reclamation WOULD HELP partially. live=peak_pin still under the wall.")
    else:
        print(f"  peak_pin ({peak_pin}) >= handle_capacity ({handle_capacity}).")
        print(f"  → Reclamation ALONE is not enough. Live set itself exceeds the table.")
    print(f"")


if __name__ == "__main__":
    main()
