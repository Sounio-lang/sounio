# Original slow diagnostic with loop regions disabled

Slurm 19576 completed 0. The unchanged original `tests/stdlib/qpu/test_qpu_chsh_overtake_slow.sio` checked 0, compiled 0 and executed 0, printing `Q_overtake: PASS` and `ALL_PASS`. It kept its original workload (62,914,560,000 simulated draws), seeds, assertions and 3000 second native deadline.

This execution used raw `8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932`, quantum library commit 328ce88 and the original qpu wide-counts body. It includes neither the dyadic scale change nor the p00 hoist. The sole intentional configuration difference from job 19533 was `SOUNIO_LOOP_REGIONS=0`. Loop regions remain opt-in.

GNU time measured the native process at 46:25.28 wall, 2781.03 seconds user, 0.32 seconds system, maximum RSS 1568 KiB. Slurm elapsed 46:30 includes checking and compilation. These are one observed execution, without a repeated best-of selection.

The final SOURCE, COMPILER, EXECUTABLE and WITNESS manifests all verified; NATIVE_STATUS and collector STATUS are 0. The historical mode-1 executions 19524 and 19533 still reached status 124 at 3000 seconds. This mode-0 result does not close mode 1, the full stdlib gate, or a physical-QPU claim.

Slurm 19604 sealed the complete original source, ELF, raw compiler, invocation and terminal receipts. Durable copy and readback:
`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/quantum-mode0-export07/quantum-original-mode0.tar.gz`

SHA-256 `a8045e9e5ca7d7b9690defd632b8ecb76c9c6d5f3ce4da56ce3ac38592982603`, 40066752 bytes. Custody STATUS 0. Readback verified both SHA-256 and length.
