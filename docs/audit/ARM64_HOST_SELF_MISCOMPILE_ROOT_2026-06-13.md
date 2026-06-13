# The arm64-host self-miscompilation root (2026-06-13)

## Unifying finding

The macOS arm64 runtime-proof gap (Mac: 33/95 runtime ≈ 35%, vs Linux 100/130 ≈
77% with the **x86** host) is **not** ~23 independent emitted-binary bugs. It is
**one root**: the self-hosted **arm64 compiler binary miscompiles its own
emission code**, specifically the **f64 / array type-tracking** paths. The x86
host compiles the identical source correctly; the arm64 host does not.

This is the same root as the deferred "x86_64-from-arm64-host E200 flood"
(`X86_64_FROM_ARM64_HOST_BUG_2026-06-13.md`) — there the arm64 host mis-reads
token spans (empty identifiers); here it mis-tracks f64/array types. Both are the
arm64 host executing subtly-wrong machine code that the a64 backend produced for
the compiler itself.

## Why it was invisible on Linux

All prior verification used the **x86 host** to emit `--target aarch64-linux`
binaries and ran them under `qemu-aarch64-static`. The x86 host emits correct
code, so the tests passed — masking the arm64-host bug entirely. The Mac runs the
**arm64 host**, which emits the broken code.

## Faithful Linux reproduction (no Mac needed)

Build the compiler **as an arm64-linux ELF** and run *it* under qemu to emit a
test; the output is the broken code the Mac sees.

```bash
scripts/dev/souc-build-lock.sh <x86-souc> self-hosted/compiler/lean_single.sio \
  /tmp/souc-arm64linux --target aarch64-linux
chmod +x /tmp/souc-arm64linux

cat > /tmp/f64cmp.sio <<'EOF'
fn main() -> i64 with IO {
    let a: f64 = 1.0
    let b: f64 = 2.0
    if a < b { println(111) } else { println(222) }
    var arr: [f64; 2] = [3.0, 4.0]
    arr[0] = 9.0
    println(arr[0] as i64)
    0
}
EOF

# x86 host  → emits correct code → runs "111 9"
<x86-souc> /tmp/f64cmp.sio /tmp/x.elf --target aarch64-linux && qemu-aarch64-static /tmp/x.elf
# arm64 host → emits BROKEN code → runs "222 0"
qemu-aarch64-static /tmp/souc-arm64linux /tmp/f64cmp.sio /tmp/a.elf --target aarch64-linux
qemu-aarch64-static /tmp/a.elf
```

## Symptoms (all from the same root)

Diffing x86-host-emitted vs arm64-host-emitted for `array_return_local_f64_42`
and `f64cmp` (same source, same `aarch64-linux` target):

- **f64 comparison → integer compare.** The arm64 host emits `cmp x1,x0; cset gt`
  where the x86 host emits `fmov d0,x1; fmov d1,x0; fcmp d0,d1; cset hi`. It loses
  `EXPR_IS_F64`/`left_f64`. (arm64-host emits ~15 fcmp vs x86-host's 41 for the
  same program — partial, conditional loss.)
- **f64 array store/subscript → `mov x0,#0`.** The arm64 host replaces the whole
  `str x10,[x0,x1,lsl#3]` (and the matching load) with a single `mov x0,#0`,
  i.e. it fails to recognize the f64-array op and takes a zero/undefined fallback.
  The downstream array-return memcpy then reads from NULL → SIGSEGV (the Mac's
  `ldr x0,[x0,x10,lsl#3]` x0=0 crash family).

These two explain the Mac's null-pointer SIGSEGV cluster, the wrong-value exits
(e.g. `array_sum_42 → 0`), and the f64-heavy assertion failures, because every
affected program's correctness hinges on f64/array type tracking inside the
arm64 host's codegen.

## Mechanism pinned to: local-variable LOAD → `mov x0,#0`

Further narrowing (all via the arm64-linux host under qemu):

- `println(5)` (no local) → arm64 host **correct** (`5`).
- `let a: i64 = 5; println(a)` → arm64 host wrong (`0`); **not f64-specific** —
  any *use* of a local var breaks; an *unused* local is fine.
- Diffing x86-host vs arm64-host emission for `let a:i64=5; println(a)`: the
  arm64 host emits **`mov x0, #0`** at every site where the x86 host emits
  `sub x9, x29, #off; ldr x0, [x9]` (a local-variable load). The store side is
  correct; only the **load** is replaced by a zero. f64 comparisons likewise
  degrade to integer `cmp` (EXPR_IS_F64 lost), and f64 array store/subscript to
  `mov x0,#0`.

So the arm64 host's **variable-load lowering takes a zero/undefined branch**.
`var_find` and `name_eq`'s var path are **byte-identical to a5ab**, so this is
HEAD's a64 backend miscompiling the (stable) variable-load function — not a
source change to the lookup itself. The exact a64 defect (why `var_find`/the
load decision yields the zero branch at runtime in the arm64 host) is not yet
pinned; it needs symbol-level gdb on the arm64-linux host compiling the
`let a=5; println(a)` repro.

## Status / next step

Root **localized to a class** (arm64 host mis-tracks f64/array types), **not yet
pinned to the exact a64-codegen miscompilation**. The bug is now fully
Linux-reproducible and debuggable (arm64-linux host under qemu + gdb-multiarch),
which removes the need for Mac round-trips to iterate. The fix is a single a64
backend defect (or small family) in how the compiler's own type-tracking
functions are lowered; pinning it should cascade to a large fraction of the Mac
gap at once. The Mac-side lldb dumps (families 1–3) are consistent with this and
point at the same `mov x0,#0`-where-a-value-belongs micro-pattern.
