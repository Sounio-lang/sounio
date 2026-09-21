# Garden: Execution Authority V3 (entrypoint input robustness)

Stage: GARDEN. Producer: Sounio. Parity open: no. Claim ready: no.

`tools/loom/execution_authority_main.sio` reads its frame 9021 with the
`read_line()` intrinsic, a single `read(0, buf, 4095)`, so a frame delivered in
fragments is truncated. V3 replaces that read with a `read_byte()` loop until LF,
EOF or 4095 bytes. `stdlib/coordination/loom_execution_authority.sio` does not
change; the 32 selftest cases do not change. `execution_authority.freeze.v2` stays
byte-identical as the predecessor and remains the parent pinned by the kernel
exec-grant cell and subprocess membrane freezes until those are re-frozen. V3
keeps the v2 compiler bytes through `toolchain_commit` and is admitted by
language authority v2.
