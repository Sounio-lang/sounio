<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-183.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-183.readme
-->

# SIO-183: fixed-array shape repair and bounded native evidence

The four shape fixes have strict failing-before/passing-after type evidence in Madaros, and the three statistics programs plus four-source Measurement control execute successfully in both engines with C++23 concordance. Aggregate qualification job17617 remains FAILED1:0. Seed13003 accepted invalid baseline shapes and the original particle core native returned139. Diagnostic17618 executes the exact retained before/after seed core ELFs and both return139, establishing preexisting signal139 on that seed; instruction-level root cause remains unresolved. No compiler change or whole-card closure is claimed.

## Source scope

Base0d7d349cc8d02bfc44fb8ded1507ff83ada8941b plus changes.diff: six substitutions in four files. Three Shapiro-only buffers256→100; two residual-only buffers256→100; Measurement initializer8→4 existing arguments. Original observations, OLS/Cook256 arrays, algorithms, all original tolerances/oracles remain identical. The original particle fixture remains byte-identical and retains its original check-only annotation. The tested four-source control is promoted byte-identical to tests/stdlib/particle_physics/test_measurement_four_sources.sio. Adding a future test does not relabel the historical627-case source or claim628 replay.

## Actual measured matrix

| Variant / engine | check | compile | native |
|---|---|---|---|
| Before / Madaros, all5 programs | 5×rc1 expected E009/E016 | 5×rc1 expected E009/E016, no ELF | not executed |
| Before / seed13003, all5 programs | 5×rc0 unexpected acceptance | 5×rc0 unexpected acceptance | not executed in17617 |
| After / Madaros, all5 programs | 5×rc0 | 5×rc0/nonempty ELF | 5×rc0 |
| After / seed13003, three stats +four-source control | 4×rc0 | 4×rc0/nonempty ELF | 4×rc0 |
| After / seed13003, original particle core | rc0 | rc0/nonempty ELF | rc139 unexpected |
| Independent C++23 | compiled rc0 | actual ELF pinned | concordance rc0 both engines |

NATIVE_PHASE_LEDGER.tsv is copied verbatim from17617:63 phase rows, including10 strict baseline unexpected acceptances,10 before-native NOT_RUN records, and the seed139. No expectation was relaxed. CORE_EXISTING_ELF_DIAGNOSTIC.tsv is the separate17618 two-run ledger; before49af0ec6… and after4846144f… both139 with original hashes verified before/after. Diagnostic COMPLETED0 means observation complete, explicitly NOT_A_SCIENTIFIC_PASS;17617 remainsFAILED.

## Numerical and runtime scope

C++ verifier source in qualified-controls is byte-identical to the actually compiled source. It reuses the original canonical reference functions whose exact Git blobs and SHA256 are in INPUTS_AND_REFERENCES.json: long-double independently computed Shapiro quantiles/W identity with shared Royston coefficients, and long-double QR/explicit case-deletion for exact fixture data. Its printed-value allowance2e-6*(1+abs(reference)) accommodates six-decimal output only; original Sounio1e-9/1e-12/etc assertions remain unchanged and passed. Measurement directly checks all four stored sources and total_error against sqrt(29), then C++ checks its printed total. No full physics/BLUE tightness, hardware IEEE, FDR, calibration or scale claim.

Actual reference ELF SHA1ed00d4fa95d26aabbac0a56d1dc08accc3b2c3b47b3d5e1858b8e175925c8e3. Initial/final ELF/loader/libstdc++/libgcc/libm/libc identities match. Raw a292 and seed13003 are previously built explicit dependencies, not builds from this branch. Node81925 closure, Bash guards, C++/GCC helper/binutils tools, and full before/after catalogs all close. Three Python basename probes return125 in separate logs; real runtime log0 bytes. No Python or Rust execution.

## Preserved jobs and limits

17615 FAILED1:0/2s, zero numerical phases, tool resolution wrongly treated Bash keyword time as file. Original failure receipts remain unchanged. Source03's only toolchain fix is command-v→type-P, plus requested metadata control and fresh paths.17617 FAILED1:0/25s, MaxRSS675704KiB, requested cpus-per-task1/4GiB/15min; actual Slurm AllocCPUS2.17618 COMPLETED0:0/<1s, MaxRSS10340KiB, requested1/4GiB/2min, actual2, only two already compiled ELF executions10s caps. Resource preflights require effective20GiB and<=7 preexisting jobs. No source/ELF payload snapshots in this text packet; all ELFs and frozen original source packets remain on worker. NUL logs are explicitly losslessly base64 encoded with original bytes/hash.

Notion183 stays Devpass/In progress; ten other acceptance cases and seed native/type-validation limits remain open. SIO181 was fetched and is a different E259/private-fields task; no duplicate card. Original Notion connector required authentication during diagnostic note, official alternate connection fetched samepage and successfully maintained subsequent notes. Root independently extracted and verified all602 packet hashes, both-engine target controls and the actual source/catalog identity. The independent result is retained in ROOT_INDEPENDENT_REVIEW.json. This is a bounded change for human review, not closure of the14-case task.

## Text custody and reconstruction

The retained [root-review01-text.tar.gz](root-review01-text.tar.gz) is1275913 bytes, SHA256848303dc08a903298461fe43a66717f49b6be1faa241304bab4594c637f4d782. Its602-entry FILES.sha256 is verified after fresh extraction. The README/matrix, native ledger, reference Git blob IDs, exact tested C++ verifier and Measurement source are inside that archive; their names above are archive-relative. The [independent root review](ROOT_INDEPENDENT_REVIEW.json) is also retained separately. FILES.sha256 next to this README checks the three documentation/custody files.

Full source archives and all native ELF payloads are excluded from Git. Original compiler/seed/runtime dependencies and frozen source03/ELFs remain on cpuops worker under /scratch/sounio-stdlib-inference-scale-20261024/SIO183-shapes-20261026-complement03;17615 under SIO183-shapes-20261026; the two-copy diagnostic under SIO183-core-native-diag01. Publication does not rerun the corpus or rebuild a compiler.

No new CI gate is registered. The strict historical before/after harness remainsFAILED and is preserved as evidence; it is never relabeled as a successful CI gate. Future incorporation of the new run-pass requires its own selection/source receipt and does not change old627-case measurements.
