# Literal producer preflight and durable custody

Proof component commit: e3d2ba3e903acaa17600476670569b69935490b9. Closed Storage component commit: ab55d0c6f9f6cbbce2e6813594aa020065f13d24. The latter is isolated and excluded from the Worker producer source.

Slurm19575 completed its private-helper modern check with verdict0. The legacy producer compiled a literal-path driver cleanly (no error[E], Error or field-initializer mismatch diagnostics), producing ELF SHA256 `8ab8ade80cad57f49ec1ae969b97089cf8279543a1f50489504c971ef241c3c6`. The program imports the actual module_frontend and native backend, enables Worker and lowering region support in that private driver, and attempts the unchanged direct and normal imported fixtures. No get_arg is used.

Both target checkers returned0, but both targets were refused before native emission: direct reported seven unresolved local identifiers p/task; normal reported44 p/task/kind identifiers and37 lowering errors. The shared failure is `IR lowering failed during merge: ir_bodies_failed`. Neither target published an ELF. The final Worker proof and its private full-instruction dump were not reached. This is a failed preflight of the legacy-produced driver; it neither qualifies real BFS callbacks nor establishes a closure-proof regression. No follow-up producer or full compiler bootstrap was submitted by this lane.

Slurm19570 is retained separately: its producer emitted explicit E001 because the private diagnostic passed &function.region to ir_region_slot_r, whose API takes the region by value. The harness refused that ELF before execution.19575 fixes the diagnostic and typechecks it before building. This private diagnostic is absent from production proof commit e3d2ba.

Slurm19579 is COMPLETED0 (1m43). Durable directory:

`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/worker-region-storage-export02`

The archive `worker-region-storage-sources-seeds-receipts-v1.tar.gz` has SHA256 `06f84802be20fc31f32f5b4b7d83c8bc92b2f151df801e960c6c690ab5dbaeb7`. It contains complete before/after sources, component ELFs, legacy seed, input/source/tool guards, accepted native receipts and rejected attempts, including the literal producers and the closed Storage classifiers. A separate durable copy `madaros-root-integrated02.raw` has SHA256 `8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932`.

The actual mount is /orangefs/training (Ceph).19578 rejected an incorrect mountpoint check for /orangefs before archiving; it is not a successful export. The original queued19577 was cancelled before execution when replacing the export recipe.

SDK custody is linked to the root's independently durable `root/integrated02-export01/integrated02.tar.gz`, SHA256 `d840afa3a675effab21316c8ee9a7a5ce22939dba7ef6c94b5024adc65ab817c`.19579 read back all relevant cases, stdlib files and runtime DSO from that archive and compared their hashes with TARGET.before/RUNTIME.before captured before driver.elf existed. Every SDK file matched. The scratch target/runtime guards also remained unchanged after the failed preflight.

SOURCE.final, RAW.final, ARCHIVE.final, SDK_ARCHIVE, SDK_MATCH, TARGET.final and RUNTIME.final verification logs each contain zero non-OK entries. EXPORT_STATUS is0. Full manifests/readback logs remain in that durable directory. Selected archive checksums, mount receipt, Slurm receipt and exact export recipe are committed beside this note.

Production Worker admission remains closed. Final CLI/BFS acceptance, positive pools and the canonical normal pipeline remain open. Storage has no public source recognizer, mint, facade or runtime authority, and does not certify HeapVec/HeapDeque.
