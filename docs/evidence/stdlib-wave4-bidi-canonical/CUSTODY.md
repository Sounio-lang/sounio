# Compact receipt custody

The complete scientific/canonical archive remains the root export with SHA-256 1f0d5d5f1b8e1a72b5bb083e554cf29e0765d614a84d47e76e53fb10b28cd14d. Independent readback 19728 completed 0 in 1m13 and verified its hash, exact archive census, complete fixture artifact identities and result metadata. No numerical execution was repeated.

Compact collection/custody 19734 completed 0 in 33 seconds. Its 114 copied records preserve the original bytes or a lossless gzip representation verified against those bytes. A separate selection-only failure in collection 19730 is retained. The Git receipt contains 2.8 MiB of metadata, not another copy of the complete source/corpus archive.

- Compact archive: /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/bidi-canonical-readback-export13/bidi-canonical-readback.tar.gz
- SHA-256: 5f81ac8b5531ac4bb2c9b208ababc3bbf155bd3980c2f0438cb8323dfbb9aa75
- Bytes: 2019217
- COPY_STATUS: 0; READBACK.verify: OK.

The bundle includes readback/collector source, original JSON/phase/resource receipts, all 211 passing result rows, both negative controls, artifact/input/ELF manifests, compressed final guards and source manifests, historical failures and explicitly labeled session recipe copies. COMPACT_FILES.json separates archive members, export sidecars, new readback records and session copies; it records original and stored hashes. The source/corpus/production files were not changed. The document commit is based on root 12be30dd0b and is independently adoptable.

The Git receipt losslessly gzip-stores the archived readback source as readback/readback.cjs.gz, preserving its original trailing blank line and original SHA. Its storage row in COMPACT_FILES.json reflects that representation; the durable compact archive retains its original uncompressed representation. No original source byte was removed.
