# Garden: GARDEN: LOOM Kernel PrincipalCapsule Authority V2 (entrypoint input robustness)

Stage: GARDEN. Producer: Sounio. Parity open: no. Claim ready: no.

`tools/loom/kernel_principal_capsule_authority_main.sio` reads its frame with the `read_line()` intrinsic, a single
`read(0, buf, 4095)`, so a frame delivered in fragments is truncated. V2 replaces
that read with a `read_byte()` loop until LF, EOF or 4095 bytes. The semantic module
`stdlib/coordination/loom_kernel_principal_capsule_authority.sio` does not change and the frozen decisions do not change. `tools/loom/kernel_principal_capsule_authority.freeze.v1` stays
byte-identical as predecessor. This authority serves the host principal broker over a
Unix socket, where a fragmented read is the real hazard.
