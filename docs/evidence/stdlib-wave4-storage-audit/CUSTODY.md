# Audit custody and attempt history

No production Storage file was edited. The worktree is based on the original584eedfe7d; the proposed guard change exists only as a patch artifact and in disposable qualification snapshots. The foundation's original durable source/seed/ELF archive remains `worker-region-storage-export02/worker-region-storage-sources-seeds-receipts-v1.tar.gz`, SHA256 `06f84802be20fc31f32f5b4b7d83c8bc92b2f151df801e960c6c690ab5dbaeb7` under the same compiler run root.

The audit used raw04 SHA256 `a4f5e8d655754202493bbb04d8fbb15890b10e8fc861ac3dc60438452704c333`, legacy9cf and the qualified private-v3 C++23 toolchain. Full executed tool and ELF hashes are in the component receipts and archives.

| Job | Result | Meaning |
| --- | --- | --- |
|19776|FAILED65,18s|The standalone compatibility probe omitted the real epistemic dependencies normally supplied by check.sio. Explicit E200 diagnostics caused the gate to reject the produced ELF; it was not run.|
|19778|FAILED65,21s|Importing the entire checker fragment also introduced Checker-dependent functions/noise hooks outside their owning scope. Explicit E200/E218 diagnostics caused rejection; no ELF execution.|
|19780|FAILED1,1s|The extraction preflight expected11 constants; the actual source has12, including ORBIT_BOUNDARY. No native compile was accepted.|
|19781|COMPLETED0,34s|Literal extraction of the needed actual epistemic declarations, all12 constants; original/candidate checks and legacy native probes; C++23 exact30-row capture.|
|19784|COMPLETED0,57s|Durable export of the first four attempts, complete source variants, compiler/seed and receipts.|
|19786|COMPLETED0,34s|Adds same-epoch i64→Ref classification/equality controls. Original/candidate checks, legacy native execution and C++23 exact32-row captures all pass.|
|19787|COMPLETED0,24s|Durable export of the final Ref controls, complete source variants, compiler/seed and receipts.|

The modern checks cover the unchanged/proposed production storage helper graphs, not an admitted public Storage program. The private probe uses synthetic nominal tables and stamps a parsed declaration explicitly. For compatibility dependencies,17 actual declarations (14functions and3records) are copied literally from production epistemic.sio and hashed individually. No semantic stubs or replacements are used. This preserves the limits of the component without importing unrelated Checker methods as standalone functions.

The first archive is:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/storage-closed-audit-export01/storage-closed-audit.tar.gz`

SHA256 `e679b6bce2db4aeda1b7117ad32df699d908f39e96b0fbe4800cd02b86183e8f`.

The final extended archive is:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/storage-closed-audit-export02/storage-closed-audit-ref.tar.gz`

SHA256 `f7869484643e9f30bbd46fc3a33c885addc01b132ac9e3ed58a684fc9aa5b9e2`.

Both exports verify the real OrangeFS mount, archive census/readback, tool/input hashes and identical before/after manifests. Because the paired recipe uses the same source path sequentially, the exports reconstruct each full source variant by substituting its exact saved injected helper and verify every file against that variant's original SOURCE manifest. Both reconstructed snapshots are retained in each archive. This is a verifiable reconstruction from saved exact source files, not inference from machine traces.

No whole-stdlib result, public admission, minter, runtime authority, source shutter, immutability guarantee, hidden-field compatibility closure or HeapVec/HeapDeque safety is promoted by this audit.
