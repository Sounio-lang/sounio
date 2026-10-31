# Bounded retirement of one OSD0 replica

At 2026-09-07T05:45:11Z, PG5.b finished moving its T560 replica from OSD0
to OSD1. The retained mapping is `5.b 11 23 0 1`; it preserves the earlier
OSD11 capacity correction. Final up/acting were [1,23,20], active+clean,
with zero degraded, unfound or misplaced objects. The bounded run took
1579.60 seconds and restored the previously enabled balancer.

The moved PG contained 127,687,353,654 logical bytes. Ceph retired the old
replica asynchronously and issued managed discards; the observed cumulative
discard volume later exceeded 137 GiB. No raw-device trim, OSD restart,
warning mute or threshold change occurred.

This did not repair OSD0 latency. V4/V5 monitoring reproduced new slow
operations. The source PostgreSQL remained online and authoritative.
The full-device evacuation simulation was also rejected: it projected OSD1
at about 88.85% and OSD26 at about 82.90%, before operational margin.
It was a simulation only; OSD0 was not marked out.

Evidence: before.json, progress.jsonl, after.json, balancer-final.json and
the normalized review copy of run.py. additional-custody.json binds the
executed root script and review copy. Do not remove either upmap pair
without inspecting capacity and replica placement again.
