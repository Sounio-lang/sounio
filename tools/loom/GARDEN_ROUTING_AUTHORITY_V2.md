# Garden: Routing Authority V2 (entrypoint input robustness)

Stage: GARDEN. Producer: Sounio. Parity open: no. Claim ready: no.

`tools/loom/routing_authority_main.sio` reads its frame with the `read_line()`
intrinsic, a single `read(0, buf, 4095)`. A frame delivered in more than one write
is truncated. V2 replaces that read with a `read_byte()` loop until LF, EOF or
4095 bytes. The semantic module `stdlib/coordination/loom_routing_authority.sio`
does not change; action 9032, the receipt fields and the 29 selftest cases do not
change. `routing_authority.freeze.v1` stays byte-identical as the predecessor.
