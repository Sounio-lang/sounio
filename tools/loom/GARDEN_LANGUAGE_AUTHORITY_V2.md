# Garden: Language Authority V2 (entrypoint input robustness)

Stage: GARDEN. Producer: Sounio. Parity open: no. Claim ready: no.

## Direction

`tools/loom/language_authority_main.sio` reads its frame with the `read_line()`
intrinsic, which performs a single `read(0, buf, 4095)`. When a producer writes a
frame in more than one `write()` (or the pipe delivers it in fragments), the
adapter sees a truncated frame and refuses or emits nothing. Measured on the v1
executable: the frozen 584-byte freeze frame, delivered in two halves 50 ms
apart, produces empty output instead of `SOUNIO_LANGUAGE_AUTHORITY_ALLOW`.

## Change

Replace the single read with a loop over the `read_byte()` intrinsic that stops at
LF, EOF, or 4095 bytes. The semantic module `stdlib/coordination/loom_language_authority.sio`
does not change; the decision table and the 33 selftest cases do not change.

## Freeze discipline

V2 is append-only: `language_authority.freeze.v1` stays byte-identical and remains
the parent pinned by routing, lane-health and custody-transfer until those are
re-frozen. V2 keeps the frozen v1 toolchain, named by a new `toolchain_commit`
field, so every consumer that reconstructs the compiler from Git keeps building
with the same compiler bytes.
