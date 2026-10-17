# In-allocation runtime binding supplement

The frozen host inspector checks source bytes and process binding but does not inspect the archived in-allocation runtime receipt fields. This supplemental audit verifies its job, rank, worker UID, boot ID, runtime/input hashes and temporal precedence over every available lifecycle record. It must run alongside the frozen inspection before interpreting the diagnostic. It does not replace full custody or qualify model execution. Missing lifecycle remains explicitly absent.

Four local tests passed, including seven identity/hash mutations and four timestamp mutations. Replay against the pinned negative collection 11983 passed on both ranks; this validates the supplemental check against archived receipts, not the new diagnostic, which has not run. Frozen runtime, helpers, packet and driver are unchanged.
