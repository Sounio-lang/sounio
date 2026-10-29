# Graph slice artifact custody

Export19607 completed0 in13s. The complete before/after SDK trees, original Graph program, public slice-size probe, C++ oracle source/ELF, all native executables and raw8abe are retained in the archive identified by SHA256SUMS. Readback checksum and complete tar listing validated0.

The exporter also rebuilt the oracle under an explicitly hashed GCC/cc1plus pair and compared its bytes with the originally executed oracle: identical. The rebuilt oracle accepted the original public API output; a wrong SIZE4=5 stream was refused1. These are post-execution reproduction controls, not a retroactive claim of a GCC manifest at the initial compile. Original source/input/raw/ELF manifests were reverified before archiving.
