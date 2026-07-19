# Single-closure native compile witness

This fixture proves the canonical native-v2 route keeps one caller-owned
`ModuleClosure` and its pinned `[Program; 256]` from visibility through
lowering and codegen.

The paired snapshot self-test deliberately collects the fixture twice in one
compiler process. The second collection must supersede the first generation;
the canonical input validator must reject the stale generation before checker
or lowering work begins.

Acceptance gate:

```bash
SOUNIO_SINGLE_CLOSURE_RAW_BIN=/path/to/source-fresh-madaros \
  bash scripts/ci/madaros_single_closure_compile_gate.sh
```

The runtime proof requires exactly one collection on the normal `build` path,
a nonempty fresh ELF, exit status zero, and stdout `42\n`. The legacy public
compile entry remains as a one-collection compatibility adapter; this witness
does not claim that thin-link, LLVM, GPU, or other legacy compiler routes have
adopted the snapshot API.
