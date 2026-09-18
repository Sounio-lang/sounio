# Baseline 11976 versus 11983: preserved-record comparison

Both collections and every declared collected file were hash-verified before comparison. Runtime inventories, input bundle, launch command and worker identities match; orchestration/source receipt identities differ. No external v3 observer ran in either baseline.

All deltas below are 11983 minus 11976, in MiB.

| Rank | Available at MODEL_READY | Available at DECODE_ENTRY | Process PSS at DECODE_ENTRY | Host Cached at DECODE_ENTRY |
|---|---:|---:|---:|---:|
| 0 | +63.70 | -217.63 | -51.43 | -344.13 |
| 1 | +103.79 | -281.60 | -51.75 | -160.75 |

At EXTEND_ENTRY, DECODE_ENTRY and the first DECODE_SAMPLE, all four recorded CUDA allocation/reservation and peak counters match for both ranks. The failing baseline therefore does not show larger recorded CUDA allocator counters or larger target PSS at these matched hooks.

11983 ends before the token-16 lifecycle sample and has no complete response. The available records cannot partition host MemAvailable into every process, kernel allocation, reclaimable page or driver allocation. Lower host Cached is an observation, not an established cause. The pair yields no observer-overhead estimate and does not isolate a causal mechanism.

Next diagnostic obligation: inspect the frozen runtime's host-memory and CUDA allocation paths against these phase deltas, including memory outside the reported allocator/PSS scopes, before proposing a new resource profile. Do not retry 11983 or loosen the guardian threshold.

Reproduce with python3 -B tools/pireus/continuity/validation/external-overhead-v3-preparation-20260910/compare_baselines.py. This reads archives only; it submits no workload.
