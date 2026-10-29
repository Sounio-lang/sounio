# Protocol custody

Slurm 19649 completed with exit 0 in 13 seconds and archived the independent BIDI protocol evidence. The durable copy was read back and SHA-256 verified.

- Archive: /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/bidi-protocol-export10/bidi-runtime-protocol.tar.gz
- SHA-256: 388d19a9f698f19674dbf88c5dbee79f29368dc26c2f368c5d14e52b2e1fcd4e
- Bytes: 8277545
- COPY_STATUS: 0; READBACK.verify: archive OK.

The archive contains the official corpus once, its immutable producer inputs and upstream pins, the golden manifest, the before/final helpers, protocol source and complete receipts. Each mutated clone is represented by its changed bytes and references to unchanged golden files; symlink controls are recorded as metadata and are not followed. This preserves exact corruptions without duplicating the corpus for every case. Source, corpus, golden and helper before/after guards all passed.

The final protocol qualification is 19648 (22/22, final helper 8b35a59e4b63c2b0df0797ec5c57aea7f01c24e6067a23b1334a31ec2e1b4627). The original failing-before 19641 and intermediate 19643 remain historical evidence. This archive does not turn protocol controls into a canonical full scientific execution or a live cancellation qualification.
