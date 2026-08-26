# Vendored nanoarrow

The generated C headers in this directory and the adjacent
`loom_nanoarrow.c`, `loom_nanoarrow_ipc.c`, and `loom_flatcc.c` sources are the
Apache Arrow nanoarrow amalgamation with IPC and flatcc support enabled.

- upstream: `https://github.com/apache/arrow-nanoarrow`
- release: `apache-arrow-nanoarrow-0.9.0`
- upstream commit: `9fad80292360d0f3978264c11ba865ccc94020d8`
- symbol namespace: `SounioLoom`
- license: Apache License 2.0

They were generated with upstream's `ci/scripts/bundle.py` using
`--with-ipc --with-flatcc --symbol-namespace=SounioLoom`. Python is used only
by that upstream source-generation step; Loom's build and runtime remain
native OCaml/C and do not depend on Python.
