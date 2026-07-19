# Single-closure native compile witness

This fixture proves the canonical native-v2 route passes one caller-owned
`ModuleClosure` and one caller-owned `[Program; 256]` by the same references
through snapshot validation, visibility, and lowering; the resulting IR then
reaches codegen. The validator checks the collection generation,
physical/logical path compatibility, and import-edge topology; it does not
fingerprint `Program.items` or arena contents.

The paired snapshot self-test deliberately collects the fixture twice in one
compiler process. The second collection must supersede the first generation;
the canonical input validator must reject the stale generation before checker
or lowering work begins.

Acceptance gate:

```bash
SOUNIO_SINGLE_CLOSURE_RAW_BIN=/path/to/madaros-raw \
SOUNIO_SINGLE_CLOSURE_EXPECTED_RAW_SHA256=<sha256-from-build-receipt> \
  bash scripts/ci/madaros_single_closure_compile_gate.sh
```

The runtime proof requires an explicitly pinned RAW SHA-256, exactly one
collection on the normal `build` path, a nonempty ELF, exit status zero, and
stdout `42\n`. The gate proves behavior for that exact binary; the SHA's
source-fresh provenance comes from the external build receipt, not from this
gate alone. The legacy public compile entry remains as a one-collection
compatibility adapter; this witness does not claim that thin-link, LLVM, GPU,
or other legacy compiler routes have adopted the snapshot API.
