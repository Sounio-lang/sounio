# CPU measurement 11986: successful boundary after 544 seconds

The frozen compiler and two input files were verified byte-for-byte on the DL380 worker. Slurm job 11986 completed 0:0. Raw compilation completed with rc 0 in 544.401 seconds; the emitted ELF executed and returned 7. The compiler accumulated 531.911 user and 11.510 system CPU seconds, with peak RSS 7,695,392 KiB and zero major page faults.

This run spent approximately 99.82% of wall time in CPU execution, across one process. It exceeded the CI wrapper default of 300 seconds while producing the expected output under the separately declared 600-second diagnostic ceiling. This supports a CPU-time budget explanation for this DL380 execution; it does not establish resource conditions on the GitHub runner or independently explain its timeout. It is not CI acceptance or PIREUS model qualification. No timeout was changed and no model was submitted.
