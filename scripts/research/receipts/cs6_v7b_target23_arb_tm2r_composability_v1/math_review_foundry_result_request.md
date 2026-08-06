# Result math review: Foundry composability execution

Review the final Foundry result in `result_report.md`,
`foundry_execution_context.txt`, and the two exact XLEL refusal logs in this
directory. The relevant verified facts are:

- The base profile failed XLEL at exact split depth 8 with
  `EVENT_SLAB_UNRESOLVED` and `PREDICTOR_ESCAPED` for all slab radii `2^-18`
  through `2^-7`.
- The audited retry changed only the split search budget to depth 12 and 255
  nodes. It closed one depth-10 sibling but the remaining child reached depth
  12 with the same all-radii `PREDICTOR_ESCAPED` refusal.
- No tile emitted its atomic JSON receipt. The other jobs were timed out or
  cancelled after XLEL made full support impossible.
- Therefore the analyzer for h-set C was not run and all exit, entry, degree,
  determinant, local covering, recurrence, chaos, and open-problem flags remain
  false.

Check that this is a correct fail-closed conclusion and that the proposed next
falsifier, event-local QR reanchoring or an equivalent Taylor-model event chart
that removes the persistent predictor offset while preserving six symbolic
variables, follows from the evidence. Flag any claim that should be narrowed.
