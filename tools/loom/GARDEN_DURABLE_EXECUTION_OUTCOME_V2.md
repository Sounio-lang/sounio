# Garden: Durable Execution Outcome V2 (entrypoint input robustness)

Stage: GARDEN. Producer: Sounio. Parity open: no. Claim ready: no.

`tools/loom/execution_outcome_main.sio` reads its frame 9022 with the `read_line()`
intrinsic, a single `read(0, buf, 4095)`, so a frame delivered in fragments is
truncated. V2 replaces that read with a `read_byte()` loop until LF, EOF or 4095
bytes. `stdlib/coordination/loom_execution_outcome_authority.sio` does not change;
the 28 selftest cases do not change. Its parent becomes execution authority v3.
`execution_outcome.freeze.v1` stays byte-identical as the predecessor.
