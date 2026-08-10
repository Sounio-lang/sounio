# `call_trace.json` — what was dropped, and why

The per-entry `pairs` arrays are **not** in the checked-in artefact.

They were 9 873 411 bytes of this file's ~10 MB, and **3 491 125 of PR #1560's
3 588 784 added lines — 97.3 % of that pull request**. A 221-commit PR was
unreviewable because of one generated file, and the file's bulk was a field no
consumer reads.

`self_falsifying_compilation_line_r14_contract.py` uses exactly four fields:
`contract`, `by_bits`, `verdict`, `error`. A census over the whole repository
found no other reader of `pairs`.

Every entry keeps `pairs_dropped`: the number of pair records that were there, so
the count remains checkable even though the records do not ship. 698 135 in total.

**Regenerating the full trace:** `python3 scripts/research/r14/trace.py`. The raw
arrays also remain in git history — the last commit carrying them is the parent of
the one that removed them.

Nothing in R14's argument rests on the dropped rows. The claims C1, C2 and C3 are
computed from `by_bits`, `verdict` and `error`, and the gate's assertions are
unchanged: `C1_CONTROL_INERT 30/30`, `C2_LOAD_BEARING_MEASURED 536 cells`,
`C3_VACUITY_REFUTED`.
