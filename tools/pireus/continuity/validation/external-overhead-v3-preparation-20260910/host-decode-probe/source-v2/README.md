# Prospective source v2

This packet requires exact source 7c0c15d445 and CI run 34435050417. It preserves all runtime and input hashes from source v1; only source acceptance and separately pinned orchestration identity are revised. The source v1 failed run and terminal waiter remain unchanged.

Thirteen source-v2 controls passed, including all inherited launch/partial-custody controls, old-check rejection and protocol corruption. Materialization and driver --check passed without allocation. The prospective waiter invokes its pinned driver at most once after successful exact-run CI, with fresh driver checks and exclusive Spark preflight. This is not inference qualification or automatic retry of a model run.

Private packet: /workspace/.cache/pireus-continuity/host-decode-diagnostic-source-v2-freeze-20260910
Private run: /workspace/.cache/pireus-continuity/host-decode-diagnostic-source-v2-run-20260910
