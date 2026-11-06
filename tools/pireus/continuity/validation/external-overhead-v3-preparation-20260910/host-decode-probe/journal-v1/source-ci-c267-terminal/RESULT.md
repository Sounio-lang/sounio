# Published c267 source qualification: terminal failure

Run 34456330295, attempt 1, source c2671c4721bb50eebaea0e3be48fa62fbfd0d73a completed with failure. Current-Source job 102803907553 started at 08:40:53 UTC and completed at 09:54:52 UTC on 2026-09-10. Its self-compilation step failed at 09:54:50 UTC.

The raw log reports a runner shutdown signal and exit 143 after observing Merged IR: 13269 functions. It has no completed gen2 observer receipt. The API annotations report exit 143, without an explicit timeout or OOM annotation. Proximity to the configured 75-minute job budget is a diagnostic clue, not proof of the signal sender or root cause.

This is a separate published-source qualification, not a third attempt on source 342. It does not qualify the old frozen diagnostic or authorize model execution. No retry is scheduled by this archive.

Next: obtain termination evidence and compare the compiler resource/timing envelope before selecting a repair. Preserve memory thresholds, prior frozen inputs, and negative pilot evidence.
