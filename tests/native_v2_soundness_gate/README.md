# native-v2 soundness gate

Guards the typecheck-skip hole in `--native-v2-compile`.

Before the fix, `run_native_v2_compile_mode` (self-hosted/compiler/main.sio) called
`bridge_lower_single_module_box` directly and only checked parse/lower errors — it
NEVER ran the typechecker. Ill-typed source was therefore silently miscompiled into
a runnable ELF, while `--check` correctly rejected the same source.

The fix runs `preflight_multimodule_frontend` (the same typecheck entry point
`--check` uses) before emitting any ELF, and refuses to emit on type/parse errors.

## Run

    bash tests/native_v2_soundness_gate/run.sh <mc.elf>

Prints `N/N` and exits non-zero on any failure.

## Cases

- `illtyped/` — clean type errors (E001/E016/E010/E008). For each: `--native-v2-compile`
  must emit NO ELF and exit non-zero, AND `--check` must reject it.
- `undefined-var-return.sio` — exercises a SEPARATE pre-existing checker gap
  (single-module name resolution is skipped, so `--check` prints "check: OK"). The
  gate only asserts that `--native-v2-compile` emits no ELF and exits non-zero here
  (it currently SIGSEGVs downstream rather than cleanly rejecting). This is not the
  hole this gate fixes; it is documented so a future resolve fix can tighten it.
- `control/` — well-typed programs that MUST keep compiling: `--check` passes and
  `--native-v2-compile` emits a correct-running ELF (exit == EXPECTED.txt value).
