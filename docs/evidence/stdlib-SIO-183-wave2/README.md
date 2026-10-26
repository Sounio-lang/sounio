<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-183-wave2.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-183-wave2.readme
-->

# SIO-183: owned NN initialization and public rate dimensions

A newly declared EpistemicDenseLayer4 can now be initialized through the existing epistemic_dense_identity API and execute the original dense/MLP/ReLU tests in both qualified engines. The initializer constructs its owned nested weights and bias before accessing their contents. The internal MLP selftest calls the existing epistemic_mlp_4_4_4 constructor. units::lib also exposes dim_rate and dim_volume_rate with exact SI exponents for inverse time and volume per time. No magnitude conversion or clinical parameter change is introduced.

The final focused job 17655 completed 0:0 with 47 expected phase records. This follows preserved failed job 17642 and diagnostic job 17653. The original NN and PBPK run-pass fixtures are byte identical to the base and already cover the public initializer; no mirrored replacement test was added. This evidence qualifies the bounded library changes described below, without closing the rest of SIO183 or the compiler's observed invalid-baseline acceptance.

## Source and dependency pins

The source is base b22d3484c95b826955b3e232310b41a6018be751 plus the two-file changes.diff in the review packet. The base tree824d9b027d9298f5b70b57c036a3c2b0eb1c9be1 preserves remote-main c2916794e429aaca3cb1bd26d07b1d14f7ba9b84 compiler sources, canonical seed and paired receipt, and contains the reviewed earlier shape repair and Node workflow composition. The publication stack is Draft3200, devpass/stdlib-wave-main-20261026. No main merge, full-corpus replay, source-built current-main compiler or fixed-point qualification is claimed.

The actual engines are external raw a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3 and the committed canonical seed 06c90e5b408e3f42f7dbf5c0ef34817333381a770a0734dae4493e36a496c939, with paired receipt cd9b24558998696183243ebaabb01eb0bfef80a6c519e1504912ced1caa7ebec. The raw dependency was built from c21b0e139830ca4949f80cba62887a8922a972d7; its codegen source 46ba0abb531c258468aca6496514a2fa8f2bea690180b50998050de242cf1312 was read from that exact Git object. Historical seed 13003 qualifications remain separate and were not substituted into the tracked seed path.

The final qualified library hashes are:

- stdlib/nn/epistemic_layer.sio:3dfa498707c06cd765cfb51f0d19f5ff6185945f1d92a24681bf447e3d0a66b2.
- stdlib/units/lib.sio:68a10cff814d009922b971ec076e6179945488be3cd69a5ba819bbcedc7a3e44.
- Original NN fixture:ad1273590f917d54b848c354f3282eccc6fa338f372ef42286536594c6b8230f.
- Original PBPK fixture:388aa6ebe27301382c97567fac35980353226b5fc13c8de278cd1c2801678123.

## Failing before, diagnosis and passing after

| Evidence | Observed result | Meaning |
|---|---|---|
|17642 original baseline / raw | NN undeclared constructor and missing PBPK dimension constructors refused with E137/status 1 | Actual before type failures |
|17642 original baseline / seed 06c9 | PBPK refused E200/status 1; three NN callers emitted E200 but returned check/compile 0 | Six unexpected acceptances preserved; no baseline pass claim |
|17642 first candidate / raw | All checks/compiles0, but original NN and public NN unitrunner native 132/signal4 | Aggregate FAILED 1:0; initial identifier repair alone did not close NN runtime |
|17653 retained raw NN ELFs | Actual SIGILL 4 at RIP0x40ea31 and0x40a83f; zero descriptor-table handle branches to UD2 | Empty-handle resolver refusal before identity initializer returned |
|17653 separate diagnostic | Marked original stopped after IDENTITY_BEGIN; fully constructed factory dense and MLP passed original value/uncertainty/confidence predicates | Localized library initialization contract; original callers and predicates preserved |
|17655 final candidate / raw and seed | Five positive programs each check/compile/native0 | Original NN and PBPK, public NN unitrunner, NN GUM reference and PBPK dimension reference pass both engines |
|17655 wrong-oracle controls / both engines | Both check/compile0, native exactly1 | Active assertions reject NN first expected value+1 and wrong PBPK time exponent |
|17655 privacy controls | Instantiated private NN field rejected1/E259/no ELF in both engines; existing interval negative rejected1/E259/no ELF in raw | Privacy preserved; unsupported seed interval intrinsics are not relabeled as privacy evidence |

The actual native fault is bound to retained ELF hashes, register/frame captures and decoded executable-segment windows in the diagnostic receipt. Ordinary builtin assert/panic in the actual raw source emits exit1. The observed UD2 is not used as proof of a failed numerical assertion or an unsupported stub. No compiler source, original fixture, tolerance, skip, timeout or private field was changed.

## Numerical scope and controls

The unchanged C++23 reference generates two nontrivial 4x4 cases with uncertain weights, bias and inputs, using the disclosed first-order diagonal GUM propagation contract, identity/ReLU output rules and conservative minimum confidence. Sounio compares values, uncertainties and confidence with unchanged 2e-12 absolute plus2e-12 relative tolerance. This does not establish covariance-aware propagation or a broader neural-network estimator.

PBPK controls cover eight existing wrappers' stored magnitudes and uncertainties and all seven dimension channels, including both new constructors. Rate is (mass0,length0,time-1,temperature0,amount0,current0,luminosity0); volume rate has length3 and time-1. Existing clinical constants, observations and roundtrip/magnitude conventions remain unchanged. These are dimension-tag and existing-value checks, not a clinical model reestimate or additional unit rescaling.

Fresh generated consumers and REFERENCE.tsv are hashed before Sounio compilation and again at the footer. Exact one-assert mutants are checked against those fresh C++ outputs and the historical generated-source hashes. Counterfactual success requires native exactly1 after successful check/compile; crashes, signal132, timeouts or unknown-symbol failures do not count. The actual reference ELF8b0f20b8dfef608e2b43dde6813b14cb00dafbf4840d9251f9a4b609f7607aec and its loader/libraries are bound before numerical generation and at the footer.

## Resource and artifact custody

17655 used 1 requested CPU, 4 GiB and a 15 min limit, with body 780 s and a separate footer budget. Slurm allocated 2 CPUs. It completed in 18 s with MaxRSS 395024 KiB and effective admission headroom 95484432384 bytes. These are resource observations, not a comparative speed or memory benchmark. The initializer constructs two fixed-size owned nested objects per call; allocation throughput, repeated-reset retention and lifetime behavior are not certified here.

The full candidate archive193ee990d460878d2a640e749a3e4fdd166ef791caeab4743c8cc86bb7fe5a88 and source catalog7250ef9f8697634399e3f94c5c4681b2dfa3e49cdfb25c83dccdc54651011a0c cover 4284 files. Exact source censuses 2384 before/2386 after protect paths, types, modes, links, byte counts, hashes and extra entries at admission, initial check and footer. Fourteen native target ELF manifests and raw/seed compiler manifests are checked before first execution and recomputed/compared at the footer; the C++ generator has its own runtime closure. Toolchain and guard closures also pass. Three Bash-backed Python guard probes return 125 separately from an empty real-runtime log. No Python, Rust or Julia numerical reference is used.

root-review01-text.tar.gz is 1727863 bytes, SHA256ebccdb53add46275947a476fe2a9a1e3417ef3368b92826153c5a0d88c438719. Its 17 top-level manifest entries contain selected before/after sources, exact diff/provenance and three original compact receipt archives. Each nested receipt has its own FILES.sha256:

| Job | Compact bytes | SHA256 | Inner files / phase rows |
|---|---:|---|---|
|17642 FAILED |652926|3b9713e12f43a72926d5e1e122c06bd89c6f4eba0598b50bb89165bddec58f99|460 /65|
|17653 diagnostic |119977|cb8027c4d80e9b7b9c9c035f61fd5fa6edb9ca38fb0fcb195d2102a9a9406cfc|292 /31|
|17655 final |935404|08bbb2e468b2fbc08ce26d711b33b2bef28d89270bf19d598fdb1fe33b8b47a7|716 /47|

Extract the outer and each nested archive into separate fresh directories and verify every FILES.sha256. NUL diagnostic controls are losslessly encoded as base64 JSON with original byte counts and hashes. Full source snapshots, generated native ELFs and raw binary segments remain outside Git on the worker; the compact text packet retains commands, phases, resource logs, failures, manifests and selected review sources. Source/runtime closure is distinct from hosted CI, independent review, PR state, merge and full-corpus validation.

The independent root audit passed the outer and all three inner manifests, all phase ledgers, actual source identity, strict counterfactual/privacy outcomes and runtime/source closures. ROOT_INDEPENDENT_REVIEW.json and the exact ROOT_INDEPENDENT_VERIFY.mjs used for that read-only audit are retained alongside the packet. This review does not imply hosted CI success or merge.
