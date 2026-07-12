# Proposal: a heap-allocation primitive for the native (`souc run`) ELF

**Status:** proposal · **Date:** 2026-07-12 · **Motivating work:** PR #828 (pure-Sounio PNG encoder, `image::pure::png`)

## Ask (one line)

Give user code a **working heap allocator under the `lean_single` native ELF**, so
buffers can be sized to the data at run time instead of to a compile-time maximum.

## Why

`image::pure::png` (PR #828) is a complete, dependency-free PNG encoder — dynamic
Huffman DEFLATE + adaptive scanline filters, no libpng, no Python. Its only real
limitation is **image size**. Every buffer is a fixed array sized to the worst
case (`[i8; 800000]`, `[i64; 800000]`, `[u32; 262144]`), so:

- The maximum image is a compile-time constant (currently 512×512).
- Every encode allocates the maximum footprint (~11 MB at 512²) regardless of the
  actual image — a 32×32 icon pays the same as a 512×512 fractal.

The correct fix is to allocate `raw`, the hash chain, the pixel buffer and the
output proportionally to `width × height` on the heap. That needs a heap.

## What was measured (2026-07-12, `SOUNIO_SOUC_ENGINE=lean_single`)

| Probe | Result |
|---|---|
| `use mem::box::{heap_alloc}` → `heap_alloc(64)` → `write_i8`/`read_i8` | **`souc check` passes**, but the native ELF exits **1 at startup** — the very first `print("START")` never runs |
| Same, direct run (no pipe) | `REAL_EXIT=1`, zero output |
| A plain `print` program (no heap) | runs, `EXIT=0` |
| `syscall(9, …)` (raw mmap attempt) | **typecheck fails** — no `syscall` intrinsic for user code |
| Fixed arrays `[i8;3.2M]+[i64;3.2M]+[u32;1.05M]` (~46 MB, a 1024² footprint) | **runs, `EXIT=0`**, all prints, 0.24 s |

**Conclusion.** `heap_alloc` in `stdlib/mem/box.sio` is `extern "C" { calloc … }`.
The lean_single native ELF is **not linked against libc**, so those symbols are
unresolved and the process dies before `main` executes. There is **no
`syscall`/`mmap` intrinsic** to allocate memory another way. The Sounio **arena**
(what fixed arrays live in) *does* handle tens of MB — but it only grows fixed
arrays, which are compile-time-sized.

So the blocker is not byte addressing or file output — it is that **there is no
way to obtain a run-time-sized block of memory in the native path.**

## The one primitive that unblocks everything

Any *one* of these makes the heap-backed rewrite possible. Ranked by
preference:

1. **Link libc (or crt) into the native ELF**, so the existing
   `extern "C" { malloc/calloc/free }` in `stdlib/mem/box.sio` resolve. Smallest
   surface — `mem::box`, `mem::pool`, `display::window` already assume this and
   would start working. `write_i64`/`read_i64` (pointer load/store) are already
   intrinsics and already used by `display::window`.
2. **A raw `mmap`/`munmap` syscall intrinsic** (or a `brk`) exposed to user code,
   e.g. `fn os_mmap(len: i64) -> *mut u8`. No libc dependency; the native codegen
   already emits `syscall` (see `self-hosted/native/encode.sio::emit_syscall`) —
   this just surfaces it behind a checked builtin.
3. **A builtin bump/arena allocator returning a run-time pointer**, e.g.
   `fn heap_bytes(n: i64) -> *mut u8` implemented in the runtime (same allocator
   the arena uses, but callable with a dynamic size).

Byte addressing is *not* a blocker: `write_i8`/`read_i8` and `ptr[i]` on
`*mut u8` type-check today; they simply could not be runtime-verified because the
pointer they need comes from the missing allocator.

## Secondary ask (for *unbounded* output, not just input)

`write_file(path, data: [i8; N], len)` takes a **fixed** array, so the final PNG
must live in a compile-time-sized buffer. For compressible images this is fine
(the compressed PNG is small and fits a modest fixed buffer even when the image
is large), so **the primary ask alone already enables large *compressible*
images**. Truly unbounded output (incompressible data at arbitrary size) also
needs a pointer/`fd`-based writer, e.g. `fn write_file_ptr(path, ptr: *mut u8,
len: i64) -> i64`, or an `open`/`write`/`close` syscall trio.

## The rewrite this unblocks (design sketch)

With a heap allocator, `png_write` becomes size-agnostic:

```
let px_bytes = w * h * 4          // Image pixels
let raw_len  = h * (1 + w*3)
let pix  = heap_bytes(px_bytes)   // or reuse the caller's heap raster
let raw  = heap_bytes(raw_len)    // filtered scanlines
let prev = heap_i64(raw_len)      // LZ hash chain, one i64 per position
let out  = heap_bytes(raw_len)    // deflate body (≤ raw_len via stored fallback)
… run the existing filter + dynamic-Huffman pipeline over pointers via
   read_i8/write_i8 / read_i64/write_i64 …
write_file_ptr(path, png_ptr, png_len)   // secondary ask
free(pix); free(raw); free(prev); free(out)
```

Memory then scales with the image and is released after each call — no
compile-time cap, no worst-case tax on small images. The algorithm is unchanged
and already validated byte-exact against an independent zlib decoder.

## Available today (stopgap, no compiler change)

Fixed-array arena buffers, bounded by the arena (~46 MB verified → up to
1024×1024). Current cap is **512×512** (~11 MB/call) as the balance point; raising
it to 1024×1024 is a one-line change to the buffer constants in
`stdlib/image/pure/png.sio` and `types.sio`, at the cost of ~46 MB per encode
regardless of image size.

## Recommendation

Adopt option **1 (link libc into the native ELF)** — it is the smallest change,
unblocks `mem::box` / `mem::pool` / `display` simultaneously, and needs no new
surface since the pointer load/store intrinsics already exist. Track the
secondary `write_file_ptr` ask separately; it is only needed for unbounded
*incompressible* output.
