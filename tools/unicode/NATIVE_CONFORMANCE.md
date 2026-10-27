# Native Unicode conformance preparation

Generate with C++23 and the pinned ICU79.1 oracle under Slurm. `wave3_unicode_generator UCD_ROOT NEW_DIRECTORY sample|exhaustive` requires an explicit coverage mode. Both modes cover every official normalization and default grapheme/word/sentence test. Exhaustive mode covers every valid scalar value for all full case/fold operations, plus context controls. The manifest records chunk kinds and record counts; chunks are bounded to512 records and1MiB.

The committed current.bin is the91-record smoke bundle only. `unicode_conformance.sio` reads this file through checked native I/O, supports NUL, compares every expected output, closes every owned object, and prints the processed count plus UNICODE_CONFORMANCE_OK. A parser/type/compile diagnostic that overlays static_blob_bytes is not an executable acceptance result.

Compile the real Sounio harness once with the source-qualified blob compiler, then invoke `run_chunks ELF FIXTURE_DIRECTORY NEW_RUN_DIRECTORY PART PARTS`. Each selected chunk runs in a new process and directory. The runner copies oracle input rather than linking it, checks exit status, exact nonempty record count and success marker, and fails on empty selection, empty partition, timeout or any mismatch. Aggregate every partition against the generator manifest before claiming complete coverage.

`runner_control.cpp` is deliberately a header-print C++ program that tests the orchestrator. It does not implement Unicode and cannot qualify Sounio semantics. Its successful execution and rejection controls are recorded separately in receipts.

Slurm18187 generated the smoke/sample files twice byte-identically. Slurm18188 generated the exhaustive set. Slurm18196 passed runner positive/negative controls. Slurm18205 checked every code point for Script_Extensions and Age against ICU and exact negative Numeric_Value. Native Unicode execution remains pending in unicode-algorithms.json.


The runner now accepts kind4 collation-pair records. These contain root mode, strength, shifted/backwards settings, expected order and two raw codepoint sequences. The harness always exercises the direct IntegerBuffer entry; every valid-scalar operand is also checked against the strict UTF8 entry for identical keys. Surrogates never pass through UTF8, and adjacent surrogate values remain distinct codepoints. All1,176,850 original serialized rows were preserved exactly; deterministic additions bring the current set to1,180,750 rows in2307 chunks.

`collect_chunks FIXTURE_DIR EXPECTED_ELF_SHA256 NEW_SUMMARY_DIR PART_RUN_DIR...` requires complete manifest coverage, exactly one instance of each partition and fixture, matching kinds/counts, successful native status/marker and one declared ELF identity. Its C++ control tests reject missing/duplicate partitions, wrong identity/rows and empty parts. Those controls qualify orchestration only.

`scripts/stdlib/wave3_core_gate.sh OUT smoke` requires a pinned qualified compiler and platform .so. Chunk mode also requires the exact conformance ELF, prepared manifest and file-hash list identities; see the script's required environment variables. It verifies source/compiler/runtime/oracle-input identities before and after. A smoke result and one partition are never full-corpus acceptance. Full results require the complete collector plus the pinned generator's scope. Bundled locale-tailoring and BIDI acceptance remain separate tranches.

