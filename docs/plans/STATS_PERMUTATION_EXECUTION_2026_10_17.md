<!-- docs:meta
topic_id: repo.docs.plans.stats-permutation-execution-2026-10-17
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-permutation-execution-2026-10-17
-->

# Checked permutation inference — 2026-10-17

## Native acceptance

Run `stats-viz-permutation-20261017-final2`, Slurm **14974**.
COMPLETED, exit **0:0**, elapsed **00:01:38**.
JUnit **32 tests, 0 failures, 0 skips, 0 errors**.
Snapshot:
`59ae19d5dd3faf6eb8c9fc5fbd6242b16df8ace0ee4bb5ea9335c9928fd6bb74`.
The archived manifest matches the fully prepared source package.

Resources: 2 CPUs, 8192 MiB, GPU off, five-minute limit, clean-v1,
UID/GID 1000/1000. Compiler: pinned committed Madaros v0.80, decompressed ELF
SHA256 `125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
This is not a compiler rebuilt from current sources. The canonical workspace
currently resolves a separate local compiler build; that was not used here.

Worktree `/workspace/sounio-sprint-stats-viz`, original remote-main baseline
`f5b2c5850be9f3d3995719ffcade44a6d851658a`.
Remote main finally rechecked as `c4f1d3ce73987de9e2694af393711180f6a57fb1`.
Its advance from `106f7a988d5b007472d0eead724f58ac50f92200` touches only site/docs;
no stdlib/compiler/dependency changes. Earlier stdlib changes from the original
baseline remain outside this focused closure.
Canonical checkout left untouched. No commit, push, PR, merge or CI acceptance.

## Implemented

- New checked `permutation_means` result with status, method, observed signed
  difference, two-sided p-value, hits, draws, seed and Monte Carlo diagnostics.
- Exhaustive labeled-subset enumeration with bounded binomial counting, at most
  65536 assignments. Exhaustive p=hits/draws, no add-one.
- Monte Carlo: 1..100000 draws, fresh partial Fisher-Yates index selection,
  integer MINSTD and rejection mapping; p=(hits+1)/(draws+1).
- Anchor/scaled data, separately compensated group sums, relative 64-epsilon tie
  convention replacing a fixed absolute raw-unit tolerance.
- Nominal Wilson 95% bounds and plug-in SE describe the uncorrected Monte Carlo
  exceedance probability, not the effect or probability of the null.
- Scalar compatibility surface delegates to the checked implementation; 1.0
  remains its ambiguous invalid/unavailable sentinel. The existing example passes.
- A real Canvas/SVG consumer compares B=256 and B=2048, explicitly labeling
  raw tail probability q, Wilson bounds and budgets.
- The enumerator reuses the selected prefix of its index vector as the cursor:
  one normalized-value buffer and one index buffer, each capacity 256.
  No heap, stored sampling distribution or per-draw sample copy. Source work
  O(draws*n), fixed storage O(n). No measured speedup, peak stack or RSS claim.
- Packaging preserves all previous 27 fixtures plus the five current fixtures.

## Independent C++23 reference

`tests/reference/permutation_reference.cpp`, g++ 14.2.0 with
`-std=c++23 -O2 -Wall -Wextra -pedantic`.
Reference uses long-double accumulation and independent bit-mask label enumeration
with std::next_permutation, versus Sounio's lexicographic selected-index cursor.
Monte Carlo specifies the same integer stream and partial Fisher-Yates draws for
pathwise count agreement. It does not constitute a random-generator quality test.

Fourteen cases check:
- Exhaustive separated and tied/unequal-size groups.
- Exactly 2/20 tail labels for [1,2,3] against [7,8,9].
- MC 195/2048 hits in ordinary and 1e-20 units; 20/256 hits for the figure.
- Constants, negative seeds, i64-min seed and seed equal to the RNG modulus.
- One-draw zero-hit run: corrected p=0.5, nonzero Wilson width.
- Maximum pooled capacity, both 1-by-255 and 255-by-1 group orientations:
  196/256 tail labelings.
- Finite values around 1e308 and small differences on a 1e15 offset.
- Full 9-by-9 enumeration: 2 hits out of 48620 assignments.

The native fixture compares exact integer hits/draws, method, normalized seed,
availability flags, signed normalized effect, p-value, MC SE and Wilson endpoints.
Probability and MC fields use absolute tolerance 3e-12. The observed raw
mean difference uses relative tolerance 3e-12 against the independently computed
effect itself, with an exact-zero check when the expected effect is zero. This
prevents a large common offset from masking a wrong small effect.
Separate status fixtures check invalid sizes, combined capacity, draw budget,
NaN/inf, exact-capacity refusal, nonrepresentable raw effect, compatibility and
unavailable MC fields. Representative caller entries are checked unchanged.

Reference source SHA256:
`bc91d82a58e2c398ac7fe514bdb02b85f0259543b183a798ff20cdd2549a67b1`.
Reference executable SHA256:
`69403f3f3bc663f0b16df15b618d4460759ea5af00c5f25b04657da22caa44e7`.
Generated native fixture SHA256:
`dc0478ff671ae932feabca6f23af22e692956eff028f67343aacbcb82bc392e8`.

## Before / after

Control `stats-viz-permutation-20261017-before`, Slurm **14961**:
28 tests, 27 pass, one runtime failure, no skips/errors; exit 1:0, 00:01:28.
Snapshot:
`b7865bed7feefcab876b427ce2ae4723a219ee85ace73c27c71446359e85b245`.

The old permutation module is untouched. The identical
`stats_permutation_units.sio` fixture compiles then fails: its raw-unit
1e-12 tolerance counts every 1e-20-scale statistic as a tail hit, yielding p=1.
The accepted implementation returns the same corrected p-value for the ordinary
and tiny-unit data, with identical 195/2048 exceedance counts.

Initial job **14960** passed 31/31 in 00:01:38. The final candidate eliminates
one cursor buffer, extends C++ coverage from 12 to 14 cases and adds the existing
legacy example, then passes **32/32** in job 14962. Final job 14974 retains that implementation
and strengthens the observed-effect reference checks as described above.

## Scientific limits

[PERMUTATION.md](../../stdlib/stats/PERMUTATION.md) defines the full contract.
The null requires exchangeability of pooled labels or an applicable randomization
design. Equality of means alone, heteroscedasticity, pairing/clustering and
longitudinal dependence are not automatically justified. Assumptions are explicitly
unassessed. Exhaustive enumeration still uses floating-point statistics/ties.
MINSTD is a deterministic finite 31-bit linear generator; the Wilson interval is
nominal under ideal independent uniform draws, not a coverage guarantee for that
generator or an adaptive stopping/multiple-testing procedure.

Other legacy bootstrap/BCa methods remain separate and unvalidated. The five-domain
stdlib sprint and execution/memory/artifact optimization remain in progress.
