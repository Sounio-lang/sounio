# Garden: Garden: Sovereign Change Kernel V2 (entrypoint input robustness)

Stage: GARDEN. Producer: Sounio. Parity open: no. Claim ready: no.

`tools/loom/sovereign_change_kernel_authority_main.sio` reads its frame with the `read_line()` intrinsic, a single
`read(0, buf, 4095)`, so a frame delivered in fragments is truncated. V2 replaces
that read with a `read_byte()` loop until LF, EOF or 4095 bytes. The semantic module
`stdlib/coordination/loom_sovereign_change_kernel_authority.sio` does not change and the frozen decisions do not change. `tools/loom/sovereign_change_kernel.freeze.v1` and its first
receipt stay byte-identical as predecessors.
Shared concept documents drifted after v1 was frozen ( concept_registry ); v1 still passes with the documents of its first receipt commit 8dbcbaf7382286dd07dc600c51137ca900316066, and v2 records their current hashes.
