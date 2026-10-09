# Diagnostic execution and partial custody

ops/host_decode_attempt.py materializes and verifies the exact runtime and input inventory, checks the published source CI, and launches one diagnostic arm through the established exclusive Spark transport. No observed arm or automatic retry is accepted. Its collect command preserves terminal failure and missing records. Its inspect command checks immutable collection bytes, launcher command, helper identities, worker and boot custody, runtime bytes, and probe ordering/timestamps.

A partial BEGIN/END prefix remains partial; missing metrics remain unknown. Even a complete probe window returns loaded_model_qualified=false and timing_eligible=false. Fifteen host-decode tests passed locally and are discovered by the CI pattern test_host_decode_*.py. The materialized packet and helper hashes are recorded in execution-packet.json.

No job was submitted. Exact-source CI and a fresh exclusive-pair preflight remain required. The runtime source is a1cd764b9c; the orchestration identity is separately hash-bound in the packet.
