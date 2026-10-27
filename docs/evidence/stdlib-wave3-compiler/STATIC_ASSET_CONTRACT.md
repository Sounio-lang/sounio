# Source-relative binary asset contract

Implementation candidate; no execution qualification is claimed by this record.

static_blob_bytes("literal-relative-path") returns (*const u8, i64), is pure at the Sounio call site, and embeds the exact binary byte sequence in native x86-64 Linux readonly .rodata. Embedded NUL and bytes128..255 are preserved. Zero length returns a stable readonly address and length0; dereferencing beyond the length is outside the raw-pointer contract.

The parser records the defining source file's parent at parse time. Literal paths must be relative, nonempty and free of NUL/control characters or backslashes. The joined lexical path is bounded to318bytes, with an explicit E275 refusal rather than truncation. Parent components remain relative to that defining file. In-memory parses without a file identity refuse this intrinsic.

Dedicated IrStaticBlob instructions serialize the path, a NUL separator, exact SHA-256 hex identity, and byte count. Payload bytes do not enter Name/string literal IR pools. The native backend rereads the full extent, computes SHA-256 in Sounio using the canonical stdlib implementation and rejects any mismatch before emitting an executable. Other targets must reject the unknown opcode. Native source/asset SHA256 manifests and an asset-aware external cache catalog remain mandatory; a previously cached executable cannot be validated merely by its old IR hash.

The single-asset limit is31MiB. The native readonly pool is32MiB and the assembled ELF ceiling is192MiB; declarations and accessors agree, and existing attempted-byte overflow/refusal guards remain active. These byte budgets do not change instruction, register, relocation, function or optimizer capacities. Multiple blobs share the readonly pool and may cause a fail-closed aggregate overflow. The existing legacy16MiB writer remains separately bounded.

Required acceptance: C++23-generated binary and zero-size fixtures; SHA256 concordance with independent file hashes; defining-module path resolution under another working directory; actual readonly segment verification at the returned runtime address; missing/nonliteral/absolute/over-budget refusals; same-length mutation rejection through serialized IR; no source mutation across build and final manifests. The original narrow-width fixtures and source-qualified compiler rebuild remain required. Full Unicode and other architectures are separate qualification claims.
