# Garden: Garden Seed: Sovereign Material Change V3 (entrypoint input robustness)

Stage: GARDEN. Producer: Sounio. Parity open: no. Claim ready: no.

`tools/loom/sovereign_material_change_authority_main.sio` reads its frame with the `read_line()` intrinsic, a single
`read(0, buf, 4095)`, so a frame delivered in fragments is truncated. V3 replaces
that read with a `read_byte()` loop until LF, EOF or 4095 bytes. The semantic module
`stdlib/coordination/loom_sovereign_material_change_authority.sio` does not change and the frozen decisions do not change. `tools/loom/sovereign_material_change.freeze.v2` and its first
receipt stay byte-identical as predecessors.

