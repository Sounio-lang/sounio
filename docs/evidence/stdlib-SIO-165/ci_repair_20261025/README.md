<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-165.ci-repair-20261025.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-165.ci-repair-20261025.readme
-->

# Seed compatibility repair and current compiler replay

The PR3103 hosted Full Test Suite failed at workflow run40143246567, job121020620880, step5. The checked and factor fixtures both reached nested-tuple comparison errors in `sr.sio` test_collision/test_boost. That job used source merge9a9b96aaf921fb7b9895ae25273a53c13e017617 (624fb3b3 merged into main a2a44) and downloaded native-compiler artifact12579164222, stage2 md5e351525c. Other Madaros Witness and Stdlib Deep Science jobs passed; they do not replace that seed/stage2 failure.

Before editing, Slurm job17280 froze the exact 624fb3b3 physics modules and fixtures and replayed them on the current committed seed8956946d and modular a292 compiler. Both fixtures still failed the seed checker. Besides nested-tuple tails/comparisons, current seed diagnostics expose direct borrowing of a function-call result as an undefined function identifier with spurious arity errors. The original sources and diagnostics remain immutable in the remote before01 source/artifact packet.

The repair binds fixture results to explicit PhysicsScalarResult/PhysicsPairResult locals before borrowing them and gives nested tuple elements/results explicit types in the two legacy pair wrappers and their existing self-tests. Called functions, argument order, equations and expected numerical/status records are unchanged. No legacy witness is skipped.

The committed seed has no supported bits_to_f64 inverse intrinsic. The original invalid-input fixture is retained before repair. Current fixtures construct the same NaN and positive-infinity classes with explicitly typed 0.0/0.0 and 1.0/0.0. They check actual f64_to_bits exponent and fraction classes before using those values and print SPECIAL_BITS receipts. The independent C++23 class verifier checks those rows on each compiler; NaN payload identity is not claimed.

The branch was serially rebased from its old a2a44 base onto verified remote main c21b0e139830ca4949f80cba62887a8922a972d7 under an exclusive 36-file incoming lease, released before editing. This replay uses the current main modular compiler `/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready`, SHA a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3, and committed current-main seed `/scratch/sounio-stdlib-inference-scale-20261024/SIO166/source14/bin/souc-lean-single-x86_64`, SHA8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc. The seed is committed, not freshly rebuilt. Source/compiler/ELF receipts and native/C++ qualification are separate from hosted CI.

## Qualified scope and preserved unsupported seed behavior

Slurm job17287 (`after02`) passes both checked and covariance-factor fixtures through actual check, compile, native execution and the unchanged C++23 decimal400 reference on BOTH current seed895 and modular a292. Both engines also pass the independent special-value bit-class verifier: the observed NaN has exponent all ones and nonzero fraction, and positive infinity has exponent all ones and zero fraction. Payload equality is not required. The full original 49-phase physics gate passes current modular a292, including eight legacy numerical rows, six corrupted/missing-output controls and five public invalid-call refusals.

Extra seed legacy replay17289 (`seed03`) deliberately went beyond those selected fixtures. Its legacy program compiled and exited0 with its marker, but the independent reference correctly rejected its nested-tuple energy/momentum outputs as stack-address-like garbage. The five public refusal witnesses passed; the standalone SR main still failed seed nested-tuple typing. A marker or exit0 is not numerical concordance.

Job17298 (`before_legacy05`) repeats the exact ORIGINAL624 physics modules and original legacy fixture on current seed895. It already exhibits the same wrong nested-tuple legacy outputs, while current modular passes the same unchanged C++ reference. Therefore this repair did not introduce that seed ABI failure and does not fix it. Both a typed-inner-tuple construction and a typed-scalar inline-return candidate were tested; the latter remained a sidecar and was not adopted. Tiny standalone nested-return variants in17299 (`tuple06`) pass modular native/C++ checks but are rejected by the seed checker with E001/E218. Their source and receipts are preserved for separate compiler work.

This PR's compatibility claim is consequently narrow: checked/factor numerical outputs pass both qualified compilers; the complete 49-phase suite and legacy numerical concordance pass current modular. Full seed legacy numerical support remains unsupported and is explicitly not claimed. No test tag, expected numerical value, nonfinite case or legacy witness was skipped or weakened.

Each `*.receipts.txt` packet preserves the original relative filenames, byte counts, SHA256 and full text of its worker logs/status/identity records, including expected failure witnesses. Executables and complete frozen source packets remain in the named immutable worker directories. This keeps Git custody compact and avoids copying bundled compiler archives.
