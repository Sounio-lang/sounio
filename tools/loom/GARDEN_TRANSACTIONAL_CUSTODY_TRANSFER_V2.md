# Garden: Transactional Custody Transfer V2 (entrypoint input robustness)

Stage: GARDEN. Producer: Sounio. Parity open: no. Claim ready: no.

`tools/loom/custody_transfer_main.sio` reads its frame 9040 with the `read_line()`
intrinsic, a single `read(0, buf, 4095)`, so a frame delivered in fragments is
truncated. V2 replaces that read with a `read_byte()` loop until LF, EOF or 4095
bytes. `stdlib/coordination/loom_custody_transfer.sio` does not change; the 30
selftest cases do not change. `custody_transfer.freeze.v1` stays byte-identical
as the predecessor. The freeze is admitted by language authority v2.
