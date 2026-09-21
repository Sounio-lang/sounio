# Garden: Garden: LOOM ExecIntent Envelope V2 (entrypoint input robustness)

Stage: GARDEN. Producer: Sounio. Parity open: no. Claim ready: no.

`tools/loom/exec_intent_envelope_authority_main.sio` reads its frame with the `read_line()` intrinsic, a single
`read(0, buf, 4095)`, so a frame delivered in fragments is truncated. V2 replaces
that read with a `read_byte()` loop until LF, EOF or 4095 bytes. The semantic module
`stdlib/coordination/loom_exec_intent_envelope_authority.sio` does not change and the frozen decisions do not change. `tools/loom/exec_intent_envelope.freeze.v1` stays
byte-identical as predecessor.
