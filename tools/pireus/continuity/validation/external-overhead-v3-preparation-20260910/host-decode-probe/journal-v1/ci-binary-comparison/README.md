# Identical compiler: successful versus timeout source qualification

The complete 95,062,953-byte Madaros executable is identical in artifacts 10138844087 (successful run 34439721975) and 10149325534 (timeout run 34463834489). SHA-256: 859d37dd115c9f51a3182ff3fcb1a6c72efcd5b65547fe609d49fa2b97cb67a7. The checker logs are byte-identical too.

The entire timeout gen2 log is an exact prefix of the successful gen2 log. The only additional lines in the successful log report merged IR and gen2 artifact emission. This excludes different compiler executable bytes as the explanation between these attempts; it does not prove identical runner conditions, memory causality, or qualification of the timed-out source.

The independently pinned Git source comparison already records equal compiler and stdlib trees. The new resource samples show little host memory and swap headroom in the timeout attempt; the successful attempt did not collect corresponding resource samples. Accordingly, this is not a paired memory experiment.

Replay compare.py with the two ZIPs retained in /workspace/.cache/pireus-continuity/ci-resource-0655747507-20260910 (successful-artifact.zip, artifact.zip). It rejects mismatched archive digests. Two executions produced byte-identical comparison.json. Both archive digests were verified against GitHub artifact metadata.

Next measurement should freeze this exact compiler, the exact source/import closure and flags, and collect process CPU time, wall time, peak RSS, page faults and host/cgroup counters on an independently declared CPU resource envelope. It must preserve the source-CI failure and remain separate from Spark inference and pilot qualification. Compare a baseline and changed resource envelope only if both can be held to the same host/runtime identity; a different CPU machine alone cannot establish the effect of memory.
