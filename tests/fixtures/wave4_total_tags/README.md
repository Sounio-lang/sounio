# Total integer-tag dispatch in dependent and epistemic helpers

The final source-tracking producer rejected four non-Unit functions in the real root05 build because integer-tag matches had no wildcard/default. Integer tags are not a closed enum domain. The existing provenance trust helper also documents why identifier patterns must not be substituted for explicit constant comparisons in Madaros.

This change replaces only those four dispatches with explicit comparisons:

- `dep_check_single_constraint`: preserves normalization and all seven comparison branches; unknown kinds return false.
- `provenance_kind_name`: preserves the six supported names; unsupported kinds return `unknown`.
- `provenance_kind_needs_verification`: preserves known values; unsupported kinds require verification.
- `validity_is_expired`: preserves kinds0..3 and the exact `current_time > until_timestamp` comparison; UNSATISFIABLE and unsupported kinds return true.

The unknown-tag policy is an explicit root-approved extension, not an assertion that the old valid-tag behavior was wrong. In particular this does not claim that every invalid floating timestamp is rejected: the known-kind floating comparison is unchanged. The dependent API's existing -1 invalid-value sentinel behavior is also preserved, rather than redesigned here.

The published focused gate passed Slurm19808, COMPLETED0:0 in5 seconds. It uses the actual full dependent source and its normalization/arena/term helpers, plus literal epistemic function/Validity/UNSAT/subsumption declarations and all12 tag constants, with per-declaration hashes. Before/final-seed compilation refuses exactly1 dependent and3 epistemic missing-return paths. Those refused artifacts are not executed. The source patch compiles and runs with both old9cf and finalccfa producers; an independent C++23 oracle checks69 dependent rows and57 epistemic rows. Wrong numeric value, duplicate row and missing final marker are rejected with exact exit codes1/69/70. The gate captures the oracle process status directly and never infers success from empty stderr.

The old9cf before programs pass their69/52 observed controls. The baseline excludes dereferencing an undefined string return for unknown provenance; it therefore makes no unsupported string-path guarantee. Old observed controls and the new explicit unknown policy are distinct claims. Source totality is demonstrated by the strict producer's before rejection and after acceptance, not by relabeling an old native output as a source proof.

The raw04 checks are deliberately an old-compiler control on the candidate component sources. No new Madaros/raw05 qualification or main bootstrap follows from this gate. The root05 failed ELF remains unqualified and was never executed. Historical producer104 and coherence34 matrices were preserved and not replaced by these focused controls.
