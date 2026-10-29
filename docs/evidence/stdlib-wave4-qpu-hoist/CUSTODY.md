# Hoist qualification custody

Slurm 19719 completed 0 in 34 seconds. All selected content hashes and symlink targets were unchanged before/after archive creation; the durable copy passed readback verification.

- Archive: /orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/qpu-hoist-export11/qpu-hoist.tar.gz
- SHA-256: 0773ef9fdefed2ecdec7f84a687792f492c5848b7778ffff026a423998a90bbb
- Bytes: 154285142
- COPY_STATUS: 0; READBACK.verify: OK.

The archive retains the sealed source trees, C++ references, SDK/toolchain files, exact raw03/raw04 binaries, native ELFs, generated cases, complete logs and identity guards for19693,19702,19710,19713 and19718. It includes the original failures: FULL capture remains refused and the first historical recipe lacked its native runtime DSO. The source patch series in the archive ends at069804582e; the later documentation clarification610fdcb2ca is preserved alongside it as oracle-protocol-clarification.patch, with its own SHA/readback receipts. The clarification changes no source, oracle or qualification result.

The final source validator requires canonical hoist placement (3positive/7negative controls). All14 numerical cases and the three original fast imports passed on raw03 and raw04 as documented, while original exhaustive all2^32 scale agreement remains qualified in the historical dyadic gate. This custody does not claim successful imported FULL capture, target checkpoint removal, slow61B completion or physical QPU execution.
