# Prospective journal diagnostic protocol

This packet binds source 342b3f4e36c78c48d70b3ca27b7ef69eb659001e and preserves the eight-request workload, full cache 6144, SWA 896, context 16384, concurrency 1, 33 GiB early-stop and 32 GiB floor. The only runtime-byte change from source-v2 is offline_generate.py using exclusive per-rank journals. Parent job 11987 remains completed inference with rejected aggregate probe records.

host_decode_journal_attempt.context reuses the pinned parent launcher with the new protocol and pinned orchestration inventory. collect preserves base custody plus separate journal custody under a composed manifest. inspect requires the full base source/freeze/runtime/input/worker verification before journal runtime/lifecycle/order verification. Source-v2's partial raw stdout is not reused or repaired.

Materialization and all host-decode controls passed without a model allocation. Source acceptance remains pending: the required checks must succeed for exactly 342b3f4e36, not its parent or a later head. The private freeze path and digest are in materialization.json. No waiter or model attempt has been started for this packet.

A complete probe window does not establish successful inference, memory causality, timing acceptance or pilot acceptance. Those claims remain separate. Before execution, use a pinned one-shot driver in remote tmux with fresh exclusive-pair readiness; preserve terminal failure and do not retry automatically.
