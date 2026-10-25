<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-166.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-166.readme
-->

# SIO-166 numerical implementation and custody

Initial source base: remote main a2a44df195bb8caf3fb07cbc9517275b29325e94. All numerical reference work uses C++23 with binary128; production is Sounio. The focused gate is scripts/stdlib/ode_bdf_checked_gate.sh and requires a fresh absolute output directory and an identified MADAROS_RAW_BIN. SEED_RAW_BIN additionally checks bootstrap compatibility. Supply a preserved pre-change bdf64.sio as the second argument for before/after regression controls.

Slurm17148 (source08/result08) passed all17 checked cases, binary128 concordance, independent Robertson RK4 refinements, legacy repairs,128-state analytic comparisons, malformed factor-layout atomic refusal and the existing DeepFactor generated native suite. Source and compiler integrity checks passed. These receipts concern the original compiler d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6; a replay against current main is recorded separately when complete.

The original four legacy regressions printed LEGACY 0 0 0 0 and exited1 before the fixes (17121), then passed after the fixes. They exercise singular elimination, relaxed Newton convergence, incomplete RHS writes and tiny endpoints. The additional bounded-trace and coupled-uncertainty controls compare actual native outputs.

Slurm17149 (perf08) used the same compiler, common libraries, fixed Linux affinity and AB/BA/AB ordering for three pairs. Only dynamic_factorize.sio differed. All six complete native outputs were identical. Baseline maximum RSS was18432KiB in each pair; scalar matrix reads used1620,1620,1612KiB. All six elapsed times were0.15s: no speed improvement is claimed. These are whole-fixture measurements for the128-state integration, with caller buffer payload402432bytes; they are not a general allocator bound. The malformed-layout fixture checks refusal before mutation after removing per-cell rich NumericValue snapshots from the LU copy path.

Retained failures:17128 bootstrap unsupported conversion/reserved name;17133 C++ warning-as-error formatting;17144 invalid Linux affinity derived from an abstract Slurm CPU identifier;17165 a shell control-insertion quoting defect after the positive native/oracle checks. Their packets remain under /scratch/sounio-stdlib-inference-scale-20261024/SIO166. No failed packet was overwritten or counted as a pass. Historical shared artifacts and the3TiB quota were left untouched.

Scope: checked first-order integration with finite numerical qualification. No higher-order variable-step BDF, general GUM covariance, compiler fixed-point proof or universal stability/performance guarantee is claimed.

## Current-main compiler replay and final norm correction

Slurm17194 passed the17-case gate with current-main CI compiler a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3. Slurm17203 repeated the complete focused gate after removing the remaining artificial1e-30 legacy error-scale floor and using scaled RMS accumulation. Native raw-bit controls match C++23 for a1e-100 error/tolerance ratio and finite1e200 RMS without squaring overflow. All checked cases, legacy regressions,128-state analytic rows, malformed-layout refusal, DeepFactor generated suite, hashes and negative controls passed again. Wrong status, wrong numeric value and missing completion were rejected with exact exits6,14,27. These focused replays use the preserved source snapshots and old bootstrap seed; current-main bootstrap validation is recorded separately if performed.

The compiler comes from main c21b0e139830ca4949f80cba62887a8922a972d7, GitHub run40144792765/Witness121024346990/artifact12580970584. ZIP digest e3eaa0099eb039094c47afdcc3f3e22a90a6bf8a2432d0b5d4e867c486feb211, ELF digest and all5295 exact-source input hashes/tracked paths were verified before replay.

Operational attempts17191 (compiler transfer not yet ready, preflight refusal) and17202 (archive transport with an invalid text prefix, missing gate) failed before numerical execution. Their receipts remain distinct from17203. Compact text custody excludes executable binaries; shared historical snapshots remain untouched.

After merging remote main into the feature branch (535a86a485a8), Slurm17205 passed the entire focused gate again using both current CI compiler a292 and current-main bootstrap seed8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc. The fresh snapshot source14 includes CI registration for this focused gate in Stdlib Deep Science. current-main-17205-text.tar.gz contains its native/oracle/control/hash/Slurm receipts; the bootstrap check exit was0. No whole-corpus or fixed-point claim follows from this focused check.

## Complete scientific-suite replay

Slurm17228 (source16/deep16) completed0:0 in17m50s. All497 recorded phases passed, including the two real scientific example imports, native execution, independent C++ concordance, the registered checked-BDF child gate, input hashes and compiler hashes. This snapshot combines current main c21 with this BDF/LU feature only; it does not qualify the other open feature branches. deep-science-17228-text.tar.gz contains textual results/resources/identities and accounting without executables. The additional runner/example snapshot hashes are explicitly post-run metadata; they are not mislabeled as pre-execution integrity checks.

The preceding full replay17209 failed four related phases because source14 omitted examples/viz_dynamic_count_workflow.sio and examples/viz_dynamic_ode_diagnostics.sio. Its checked-BDF child and all other executed phases passed. Source14 and its failed result remain unchanged. Fresh source16 added those required exact-source examples and reran the complete suite. Hosted PR CI and human review remain separate.
