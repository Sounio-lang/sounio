<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.compiler-record-owner.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.compiler-record-owner.readme
-->

# Nested-if record owner: native controls and qualification harness

This compact custody folder retains original text receipts, scripts, source hashes,
compiler patch, stdout and resources. Full ELFs, caches and source-tree snapshots
remain at the original scratch paths. No numerical calculation or native execution
was repeated to assemble this folder. FILES.sha256 covers the copied payload;
SOURCE_MAP.tsv.gz links the receipts to their original paths and
ORIGINAL_FILES.sha256 hashes those original files. Existing per-run input/output
manifests retain their original paths and can refer to larger artifacts not copied
here. STAGING_MANIFEST.original.txt records the earlier staging state only;
FILES.sha256 is the authoritative final copied-file manifest. Map target paths
precede optional gzip compression and the .sio-to-.sio.txt evidence renaming.
Gzip files are lossless text compression. Tiny Sounio/C++ source copies use
`.txt` extensions and are evidence, not additional corpus entries.

## Defect and narrow fix

An expression binding a record returned by nested `if` arms could lose its record
owner during lowering. Inline `.alpha` field projection then selected an unrelated
field index. A typed borrowed helper read the actual record correctly. This was
not evidence that the second fit mutated the underlying dispersion record.

The patch in build-16892/OWNER_FIX.patch extends expr_method_fn_id to recursively
join both if arms only when the registered record owner and tuple slot/name agree.
Unknown/incompatible owners and fixed-array returns remain conservatively rejected.
The existing direct-call, method and block cases are retained. The compiler source
hash and patch are preserved; fixed compiler SHA256 is
`92d2108ef595c5382d582aa363df6e95dab907a554f039d5f8f008073e535c21`.

## Evidence boundaries

| Job / folder | Actual observation |
| --- | --- |
| 16883 diagnostic-current | Current-source compiler reproduced incorrect inline alpha projection. |
| 16884 diagnostic-materialized | Materializing the already incorrect alpha value 0 allowed downstream assertions to pass. **This is not a valid fix or acceptance control**: it changed the numerical parameter consumed by later calls. |
| 16885 diagnostic-borrow | Typed borrowed projection read a stable nonzero alpha before and after the second fit while inline projection differed. This distinguishes field-owner lowering from record mutation. |
| 16888 witness-before | Identical import-free direct/simple controls passed on both old ee0 and fresh8c compilers; nested/annotated controls compiled but failed natively. The attempted disassembly route emitted a compiled executable instead; it is not disassembly evidence. |
| 16897 fixtures-before | New record fixture compiled then failed natively; tuple control passed; incompatible record arms were rejected by the checker. The negative annotation was subsequently aligned with the actual `incompatible types` diagnostic, without changing its body. |
| 16892 build | Fixed compiler built; recorded native guards and rejection controls passed. Consult artifacts/RESULTS.tsv, not scheduler exit alone. |
| 16900 after | The unchanged original witness sources and final record/tuple fixtures passed native execution; incompatible record arms remained rejected. Hash-after checks passed. |
| 16899 original | ORIGINAL traced and untraced full B2 scientific fixtures passed, including actual C++ concordance and off/on comparison. No scalar materialization workaround was used. This is B2 functional evidence, not a calibration campaign. |
| 16903 OpsTraced | Final operational harness passed traced B2, off/on result comparison, trace-cap0 fail-closed behavior, malformed input controls and rejection of a rebound old functional receipt. |
| 16908 summary-operational-failure | C++ compile, traced summaries and nine old generator identities passed; an added final nested-summary check used a wrongly expanded `/generated/...` path and exited2. The READY file predates that last failure and does not supersede STATUS2. |
| 16910 summary-final | Same final C++ source, corrected operational path: traced summaries passed; untraced nested summary and all nine legacy generated drivers/datasets were byte-identical. No new Sounio execution. |

The traced summary reports actual bootstrap work counters and separately recorded
callsite trace counters. Untraced nested output explicitly retains the limitation
that IRLS/QR callsite counters are absent. No work is inferred from B or RNG state.
Trace failure retains the measured partial prefix and makes the numerical aggregate
unavailable; summary code preserves the actual emitted status/p-value placeholders.

## Scope and composed compiler receipt

The two run-pass compiler fixtures require Madaros in the general suite; the
dedicated compiler gate compiles them unconditionally. No Stage2 compatibility
claim follows from these results. No full calibration campaign, broad numerical
accuracy certificate or performance superiority is claimed.

The first composed f64 gate, job16904, used `--native-compile` on import-free
sources and failed at invocation; root is preserving that attempt separately.
The corrected composed gate, job16913, completed with STATUS0 and compiler/input
hash verification passing. composed-pass-16913 retains its actual shared-ELF
dereference, global f64, direct/imported capacity, imported wide-call and nested
record-owner receipts. Large generated capacity sources and ELFs remain in scratch. No passing CENTRAL receipt was
created by this custody task.
