<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-168.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-168.readme
-->

# Conditional exact signed-rank inference: SIO-168

This increment replaces the fixed-size Wilcoxon façade's default normal approximation with conditional sign-pattern counting, corrects its treatment of tiny nonzero differences, and adds caller-owned dynamic ranking and inference. The explicitly named normal approximation remains available. Rank-biserial correlation and the older `z/sqrt(n_eff)` diagnostic have separate fields and availability flags.

The dependency is the checked Normal façade and canonical special-function work in PR [3102](https://github.com/Sounio-lang/sounio/pull/3102), parent `1a2e6493205ffd8dd140fdb5ed8a0143f0b0ee7a`. This branch also incorporates remote main `7ce5eddeef78c76c7e744150ed9c169c53a3d3b0`. The difference from compiler-source main `c21b0e139830ca4949f80cba62887a8922a972d7` is a workflow and changelog fragment; the compiler and numerical source trees are identical. Final science code is commit `21ad57dbd333ca78f8898c9c889c1830d4dce440`, gate hardening is `3b24811d428337747770bfb9ddeba7695bbdbe16`, and the additive central registration is `88dca283427a8932d79445e782b4d319b0edd420`.

## Statistical contract

`stdlib/stats/checked_wilcoxon.sio` exposes `wilcoxon_rank_workspace`, `wilcoxon_rank_checked`, `wilcoxon_exact_workspace`, `wilcoxon_exact_ranked_checked`, and `wilcoxon_normal_approx_checked`. Ranking accepts finite differences. Zero policy 0 removes exact binary64 zeros before ranking (Wilcox); policy 1 ranks zeros and then excludes their deterministic signs (Pratt). Equal absolute binary64 values receive doubled integer midranks. Applications that require measurement-resolution rounding must apply that policy before calling; paired subtraction can affect floating-point ties.

Inference assumes independent fair signs conditional on the observed nonzero magnitudes; a symmetric null distribution is sufficient. A zero median under arbitrary asymmetry is insufficient. The distinction between symmetry, zeros, ties, and exact conditional inference is consistent with the [R statistics manual](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/wilcox.test.html); no R execution is used as a numerical reference here.

Alternative 0 sums the inclusive absolute-deviation tail: after GCD normalization, a state `j` is included when `j <= min(observed,S-observed)` or `j >= S-min(observed,S-observed)`. The center is counted once. Alternatives 1 and 2 are inclusive less and greater tails. This documents the chosen discrete two-sided convention rather than asserting that every convention gives the same p-value.

The coefficient polynomial is the product of `(1+t^weight)` over nonzero doubled ranks divided by their GCD. Descending in-place dynamic programming stores its integer coefficients in base `2^62`, with explicit carries. The numerator is a sum of selected coefficients; the denominator is exactly `2^n_eff`. There is no 62-observation cutoff. Successful results expose the caller's exact numerator limbs and denominator exponent, independently of scalar probability availability. All-zero input is a conditional point mass with p=1; rank-biserial and standardized effect availability are false.

`rank_biserial` is `(W_plus-W_minus)/(W_plus+W_minus)`. The legacy `effect_r` retains `z/sqrt(n_eff)` and is not renamed into rank-biserial. Ranking records whether continuity correction was requested. Its legacy diagnostic z uses a correction toward the center. The explicit normal approximation derives one-sided corrections from the uncorrected rank deviation: +0.5 for less, -0.5 for greater, and a correction toward zero for the two-sided convention. Normal kernel iterations and exact-count operations are separate counters. There is no automatic fallback from exact counting to a normal approximation.

## Work, memory, and compatibility

`stdlib/collections/numeric_index_sort.sio` provides a finite-input checked indexed heapsort. It orders keys ascending and original indices within ties, including signed zero. Scratch is reusable and no per-observation clone is made. Ranking supports up to 1,000,000 observations, with 24*n bytes of active payload for magnitudes, sort indices, and signed ranks, excluding the 8*n-byte input differences.

The exact-workspace query accepts signed nonzero integer weights of magnitude at most 2,000,000. It computes GCD support, limb count, a conservative operation bound, and count-plus-numerator payload before allocation. The caller supplies all buffers; the core has no allocation effect. Practical support and operation caps may refuse a problem far below the observation ceiling. Payload means active buffer data, not allocator capacity, page accounting, or process RSS.

Status 0 reports algorithm completion. Status 1 is invalid shape/options, 2 nonfinite input, 3 work budget, 4 arithmetic failure, 5 unavailable work region, and 6 payload cap. A complete operation-bound query has its own availability flag. Preflight refusals preserve caller buffers. After computation begins, scratch may change on refusal; the numerator is committed only on success. Sort-budget exhaustion may leave a partial heap. These distinctions are exercised by the native smoke fixture.

Exact dyadic probabilities use integer leading bits, sticky bits, and round-to-nearest/ties-to-even conversion, including subnormals. Scalar and logarithmic availability are independent. A positive probability can underflow while its logarithm remains available. Near unit probability, the exact complementary numerator avoids cancellation. When that complement underflows, logarithmic availability is conservative, with a specifically qualified `2^-1075` midpoint case: `-log(1-q)>q` rounds the logarithm to negative minimum subnormal. The preserved development failure and 4096-bit C++ control are arithmetic evidence, not a Lean proof or a universal transcendental error certificate.

The legacy `stdlib/stats/wilcoxon.sio` preserves its array arguments and original seven result fields, adds explicit method/availability and rank-biserial fields, and now requires `Alloc`. Its exact scratch policy is bounded at 8 MB and 50 million operations; unsupported input fails closed. Explicit `_normal_approx` entry points retain a named approximation. Both tracked external import callers, `examples/stats/full_analysis_report.sio` and `examples/stats/epistemic_suite_demo.sio`, declare the additional effect and were compiled and executed under both engines. Expanded struct layout and the added effect are documented in the breaking changelog fragment. This scoped tracked import search is not a claim about external downstream consumers.

## Independent references and retained failures

The canonical executable references are C++23 and Sounio. `tests/reference/wilcoxon_checked_reference.cpp` independently enumerates every sign pattern for the small cohorts, including ties and both zero policies. Larger all-tied controls use `boost::multiprecision::cpp_int` binomial coefficients, reaching n=2048. Every actual native numerator limb, denominator exponent, support, GCD, and relevant work/availability field is checked. Binary probability rounding and logarithms use a 4096-bit floating reference; normal comparisons use a 113-bit reference with extended exponent range. The qualified log envelope is 3e-12 absolute with a 2e-15 scale term, not a proven all-input bound.

There are 28 cohorts, two zero policies, and three alternatives. Nonempty balanced input explicitly checks two-sided normal p=1 and log p=0 with valid probability bounds; this is separate from the all-zero shortcut. A budget refusal after a Normal evaluation verifies that performed kernel work is retained.

`legacy/` preserves original and corrected executable output for 40 legacy rows under both engines. Against the new exact target, the old default fails 28 rows and the corrected default fails none. Some differences are deliberate method changes: the classic eight-observation example changes from approximately 0.14148219736025935 to the exact 0.1484375. A separate real input defect is demonstrated by tiny nonzero differences: the original implementation reports n=0 and p=1 where n=3 and conditional p=0.75 are required. The original module hash is `e82a62ba52a4dc7011de79206dea34a3e8ab8d3f274a78946e5f71bb0182e6aa`.

`midpoint_before/` retains the development counterexample at n=1075: the integer numerator and scalar value passed, but logarithmic availability failed in two zero-policy rows. Its original protocol predates later metadata additions and has not been rewritten. The final fixture passes those same cases.

`oracle_cutoff/` retains an independent-reference failure during the million-row stress test. Installed Boost 1.83's 113-bit erfc implementation returns zero above argument 110, despite the floating type's exponent range; see the [primary Boost source](https://raw.githubusercontent.com/boostorg/math/boost-1.83.0/include/boost/math/special_functions/erf.hpp). The native Sounio logarithm was finite while that oracle incorrectly produced negative infinity.

The repaired reference uses the positive scaled Mills integral `SF(z)=phi(z)/z * integral_0^infinity exp(-u-u*u/(2*z*z)) du`, following the Gaussian integral definition in [DLMF 7.2.2](https://dlmf.nist.gov/7.2.E2). Independent GK61 evaluations at tolerances 1e-24 and 1e-29 must agree, with estimated errors at most 1e-24; the exp(-128) remainder is checked relative to the positive integral. Overlap controls z=1,3,8,30,100 agree with the pre-cutoff Boost path within 1e-23. `mills/` verifies frozen failing-run native output without changing Sounio or relaxing its tolerance. The million-row Normal log error is 4.4965675242953619e-12.

This million-row z≈200 case is a numerical Gaussian-log stress test, not calibration of the conditional binomial tail by a Gaussian approximation. Exact counting was explicitly refused by its payload cap. A specialized binomial path and distributional calibration are future work; the base implementation does not silently substitute one.

## Final native qualification

Slurm job **17314** completed with exit **0:0** in **00:03:20**, using an immutable source packet and a 524288 KiB stack. Its `final/results/` receipts contain strict committed-seed checks, dual-engine compilation and actual execution, all reference comparisons, source/ELF/compiler hashes, resource measurements, and eight deliberate output corruptions. The exact gate result is `WILCOXON_CHECKED_GATE_PASS`.

The compiler artifacts are separately pinned:

- Source-built c21 Madaros: `a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3`.
- Committed seed: `8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc`.

Both engines pass all 28×2×3 exact rows, 40 legacy rows, the million-row rank/Normal stress test, indexed sorting through one million keys, the public module tests, atomic/budget/nonfinite smoke cases, intentional legacy refusals, and both example callers. Each of eight corrupted-output controls must fail with its expected exit and a specific numerical/metadata diagnostic; arbitrary nonzero failures are rejected. A second gate invocation on the completed output directory exits 65 with `fresh output directory required`; hashes demonstrate that prior receipts remain unchanged.

Observed native resource measurements on the allocated worker are:

| Fixture | Committed seed elapsed / max RSS (KiB) | Madaros elapsed / max RSS (KiB) |
| --- | --- | --- |
| Million-row ranking and explicit normal evaluation | 0:03.82 / 34368 | 0:10.82 / 31232 |
| Million-key indexed sorting | 0:03.84 / 24088 | 0:10.85 / 23040 |
| Exact 28-cohort inference fixture | 0:24.23 / 3616 | 1:20.02 / 1628 |

The million-row rank scratch payload is 24,000,000 bytes, plus 8,000,000 input bytes. Its exact-DP query requests 129,040,516,160 bytes and is refused at a 64 MB cap without allocation. These are observed workload-specific values, not universal performance promises. Batch RSS includes compilation and is distinct from the per-executable values above.

`wilcoxon_checked_gate.sh` requires a fresh output directory and refuses a raw compiler override that would bypass engine routing. The central deep-science runner registers this focused gate with a distinct child output path. The full shared runner and hosted CI are separate integration evidence; this receipt does not assert their latest state.

Generated million-row fixtures use loops rather than million-literal source files. Text evidence is committed here, with `FILES.sha256`; executables remain in retained scratch rather than Git. Snapshot overlay files were unlinked before replacement to preserve historical hardlinked inputs. Packet manifests describe the tested input projection rather than claiming a full repository archive. No shared quota was increased and no historical artifacts were removed.
