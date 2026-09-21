# Garden: Truthful Lane Health V2 (entrypoint input robustness)

Stage: GARDEN. Producer: Sounio. Parity open: no. Claim ready: no.

`tools/loom/lane_health_main.sio` reads its frame 9030 with the `read_line()`
intrinsic, a single `read(0, buf, 4095)`, so a frame delivered in fragments is
truncated. V2 replaces that read with a `read_byte()` loop until LF, EOF or 4095
bytes. `stdlib/coordination/loom_lane_health.sio` does not change; the 28 selftest
cases and the exhaustive 8,388,608-case decision stream do not change. The OCaml
realization is re-bound to the v2 semantics in a new receipt,
`lane_health.ocaml.v2`. `lane_health.freeze.v1` and `lane_health.ocaml.v1` stay
byte-identical as predecessors.
