# Garden: Garden: Native Hook Generation Drain V2 (entrypoint input robustness)

Stage: GARDEN. Producer: Sounio. Parity open: no. Claim ready: no.

`tools/loom/native_hook_generation_drain_authority_main.sio` reads its frame with the `read_line()` intrinsic, a single
`read(0, buf, 4095)`, so a frame delivered in fragments is truncated. V2 replaces
that read with a `read_byte()` loop until LF, EOF or 4095 bytes. The semantic module
`stdlib/coordination/loom_native_hook_generation_drain_authority.sio` does not change and the frozen decisions do not change. `tools/loom/native_hook_generation_drain.freeze.v1` and its first
receipt stay byte-identical as predecessors. Records already written under v1
(generation pins, activation heads, quarantine receipts) stay valid: consumers accept
both the v1 and the v2 freeze during and after the transition.
