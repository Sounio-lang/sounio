# First-15-decode host probe preparation

Source review found file-backed mmap embedding gathers and tiled LM-head projections. Their gather/project paths create owned CPU copies and request MADV_DONTNEED/POSIX_FADV_DONTNEED on file backing. The LM head creates GPU tiles per projection; the existing process and CUDA counters do not form a complete host-memory accounting. These paths are candidates for measurement, not an established failure cause.

The failed baseline records no token-16 sample. This separate artifact records host meminfo, vmstat and process status with native units, plus the existing CUDA allocator counters, before and after decode calls 1 through 15 of request 0. Each source has its own timestamp and duration; errors remain explicit. No new device synchronization, allocator flush, cache change, guardian change or output-budget reduction is introduced. Additional reads and JSON logging can affect execution; timings from this diagnostic are ineligible for performance claims.

The 11983 log reports roughly 0.08 GB full KV storage and 0.06 GB sliding-window KV storage per rank, with rounded component sizes. Those logs alone do not justify expecting a cache reduction to recover hundreds of MiB.

Five local tests passed: reversible transformation, changed-source refusal, overwrite refusal, explicit missing counters, and raw-unit/timestamp preservation. The new helper is included in the existing CI control step. This is preparation only: a separate frozen execution profile, launch/custody binding and independent qualification are required before hardware use. Neither failed baseline 11983 nor observed job 11977 is retried by this artifact.
