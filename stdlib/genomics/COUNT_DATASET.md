# Checked count dataset preparation

`genomics::count_dataset` connects the caller's integer count-table columns to
metadata/design rows without changing raw counts or allocating hidden storage.
It is preparation for existing normalization and count-model kernels, not a
filesystem/header/string-ID reader, biological QC certificate or DE pipeline.
Integer keys are arbitrary exact i64 values, including both extremes. The caller
declares physical sample identity; the API cannot detect falsified valid keys.

`count_dataset_align(count_keys,metadata_keys,comparison_budget,permutation,
count_order,metadata_order,stage)` requires equal positive lengths S and three
caller integer scratch buffers of length S. Two index heapsorts and a merge
match exact keys in O(S log S) comparisons, followed by transactional commit.
Inputs are never reordered. `permutation[metadata_row]=count_table_column`.
Duplicates in either list and unequal key sets fail explicitly. Ordering uses
no f64 cast or key subtraction. Each comparison charges the explicit budget;
exhaustion returns6 with actual work. Public permutation remains unchanged on
failure; scratch may change. Budget0 cannot even complete a singleton match.
Shape errors precede budget exhaustion and the first encountered discrepancy.

`count_dataset_gene(counts,G,S,g,permutation,max_count,row,marks)` validates a
packed G-by-S input, selected gene, whole permutation and all selected counts
before writing the length-S row. Integer marks of length S validate permutation
in O(S), avoiding quadratic per-gene work. max_count selects an explicit inclusive
limit, such as2^53 or i64MAX. No factor division or f64 conversion occurs.
Failure preserves row; marks are scratch. Other rows are validated separately
by normalization or the caller's complete dataset policy.

`count_dataset_offsets(log_factors,log_exposure,permutation,offsets,stage,marks)`
uses factors in count-column order and supplied log-exposures in metadata/design
order. It stages `log_factors[permutation[i]]+log_exposure[i]`, requires finite
operands/sums and commits all offsets together. It neither reconstructs positive
exposures nor certifies units. Finite rounding/cancellation follows binary64;
no exact-sum or relative-error guarantee is made. Factors enter the log-link
model as offsets while its response remains raw integer counts. Ordinary linear
borrows require distinct mutable output/scratch owners.

Statuses:0 success,1 invalid shape/owner/configuration,2 duplicate key,
3 mismatched key sets,4 invalid permutation/count/range,5 nonfinite/unavailable
offset arithmetic,6 explicit budget exhausted. Only alignment returns a record
`CountAlignment{status,samples,comparisons}`. Other preparation returns scalar
status, avoiding per-gene rich-result handles.

`count_dataset_budget_new(max_genes,max_evaluations)` creates private scalar
reservation state. `count_dataset_reserve_gene(&!budget,evaluation_cap)` charges
one gene and its entire positive declared profile-evaluation cap BEFORE fitting.
An insufficient reservation changes nothing and returns6. Failed profiles are
not refunded. This bounds declared calls, not authenticated future execution
or inner IRLS work; each profile's own iteration/search controls remain required.
`count_dataset_budget_info` is a checkpoint snapshot, not hot telemetry.

No shrinkage, automatic filtering, FDR or calibration is performed. Per-gene
profile status, Poisson selection, censoring and curvature availability remain
explicit. Failed genes must not become available alpha0 or silently vanish
from a multiple-testing denominator.

Fixtures cover transactional failures, duplicates/missing keys, budget exhaustion,
negative counts, nonfinite offsets,512 samples and exact extreme keys.
`tests/reference/deep_count_dataset_reference.cpp` uses std::map independently
of production heapsort and checks EVERY512 native permutation/count/offset row
plus the terminal marker.

The central current-source focused check07 completed the status/reference
fixtures and the actual count-workflow example with typecheck, native compilation,
execution and completion markers all exit0. Both the exact512-row dataset
concordance and the workflow C++ structural/status concordance exited0. The
workflow checks six per-gene reports, separate Poisson/positive-edge/failure
availability and the actual rectangle-mark document grammar. This is focused
executable evidence, not independent profile-inference calibration, full-suite
acceptance or human inspection of a physical figure. The later full02 gate is
still running as of this update; it is not inferred from focused success.

The coordinator maintains artifact identities and receipt scope in the
[evidence ledger](../../docs/evidence/stdlib-inference-document-2026-10-23/README.md).
Focused SVG structure/concordance does not certify Canvas pixel parity or a
complete RNA-seq analysis pipeline.
