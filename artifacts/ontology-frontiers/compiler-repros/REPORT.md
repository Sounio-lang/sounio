# Compiler reproductions — ontology-frontiers lane

Minimal, verified reproductions of compiler limitations discovered while
writing the ontology-frontiers prototypes. All outputs below were captured by
actually running the commands shown (branch
`research/zd-fiber-antisymmetry-lemma-20260731`, wrapper `bin/souc`, engine
Madaros v0.80.0). Limitations 1–2 were captured 2026-08-02 in round 6;
limitations P1–P5 were captured 2026-08-02 in round 7 (lane
`miscompile-hunt-20260802`, agent kimi-swarm). No compiler source, shared
scripts, or existing docs were modified; this directory only adds new files.

The ontology-frontiers prototypes
(`artifacts/ontology-frontiers/*/alignment_repair.sio`, `claim_status.sio`,
`version_chain.sio`) were deliberately written to avoid both pitfalls: they
enforce contracts as runtime assertions instead of `where` refinements, and
they model record collections as parallel primitive arrays with splat
initialization (`var a: [f64; N] = [0.0; N]`).

---

## LIMITATION 1 — `where` refinement clauses do not parse

**Symptom.** A postcondition refinement of the form
`fn f<T>(...) -> Knowledge<T> where result.confidence >= max(...)` (as used in
`examples/epistemic_dempster_shafer.sio`) is rejected by the Madaros parser
with `parse error: expected token` at the exact `where` position. The errors
appear only in stdout — there is no dedicated error exit contract: observed
exit code was 1 in this run (the wrapper does not print a distinct
"check failed" marker; `run_check_mode: AST closure incomplete` is the only
summary line).

**Minimal repro.** `where_refinement_parse.sio` — one generic function with a
`where` clause returning `Knowledge<bool>`, plus an empty `main`.

**Command.**

```
./bin/souc check artifacts/ontology-frontiers/compiler-repros/where_refinement_parse.sio
```

**Observed output (verbatim, key lines).**

```
parse error: expected token at line 12
:22
 expected=184
 actual=31

parse error: expected token at line 17
:12
 expected=131
 actual=124

parse error: expected token at line 19
:1
 expected=185
 actual=0

run_check_mode: AST closure incomplete nodes=0
 unresolved=0
 saturated=false
EXIT=1
```

Line 12, column 22 is exactly the `where` keyword in
`) -> Knowledge<bool> where result.confidence >= max(...)` — the parser has no
production for a refinement clause after the return type, and the two
follow-on errors at lines 17/19 are cascade from the same failure.

Cross-check on the original example (`./bin/souc check
examples/epistemic_dempster_shafer.sio`) fails the same way, with the first
error at `line 15 :19` — again exactly the `where` token of
`combine_evidence`:

```
parse error: expected token at line 15
:19
 expected=184
 actual=-8965959058170792059
```

**Expected behavior.** Either the parser accepts `where <predicate>` after the
return type and hands the predicate to the refinement/SMT layer (falling back
to a runtime assertion per the documented refinement policy), or it rejects it
with a single targeted diagnostic naming the unsupported construct.

**Suspected area.** Parser (Madaros, self-hosted): the function-signature
grammar lacks a `where`-clause production after the return type.

---

## LIMITATION 2a — arrays of structs segfault at runtime

**Symptom.** `var ms: [M; 2]` followed by struct-literal element stores
compiles and links cleanly (`Compilation successful!`), but the produced ELF
segfaults immediately when executed; `souc run` reports the shell's
`Segmentation fault` line and exits 139.

**Minimal repro.** `struct_array_segfault.sio` — a two-field struct
`M { id: i64, conf: f64 }`, a `[M; 2]` local, two struct-literal stores, one
`println`.

**Command.**

```
./bin/souc run artifacts/ontology-frontiers/compiler-repros/struct_array_segfault.sio
```

**Observed output (verbatim, key lines).**

```
imported_compile: typecheck ok
...
Compilation successful!
   Output: /workspace/.tmp/madaros-run.UgGJaG/main.elf
/workspace/sounio/bin/madaros: line 634: 2022661 Segmentation fault      "$out" "$@"
EXIT=139
```

Note that `println("OK")` never fires — the fault happens at (or before) the
first element store.

**Expected behavior.** Either the program runs and prints `OK`, or the
compiler rejects fixed-size arrays of structs at check time with a diagnostic.

**Suspected area.** Codegen / stack allocation for fixed-size arrays whose
element type is a struct (layout or store lowering); structs are otherwise
handle-based in this runtime, so an inline aggregate slot may be mis-sized or
mis-addressed.

---

## LIMITATION 2b — arrays without splat initialization segfault at runtime

**Symptom.** `var a: [f64; 3]` with no initializer compiles and links cleanly,
but the ELF segfaults at the first element store; `souc run` exits 139. The
identical program with splat initialization `= [0.0; 3]` runs and prints `OK`.

**Minimal repros.** `uninit_array_segfault.sio` (faulting) and
`splat_array_ok.sio` (working control, prints `OK`).

**Commands.**

```
./bin/souc run artifacts/ontology-frontiers/compiler-repros/uninit_array_segfault.sio
./bin/souc run artifacts/ontology-frontiers/compiler-repros/splat_array_ok.sio
```

**Observed output (verbatim, key lines) — uninit.**

```
Compilation successful!
   Output: /workspace/.tmp/madaros-run.5RMEEk/main.elf
/workspace/sounio/bin/madaros: line 634: 2025972 Segmentation fault      "$out" "$@"
EXIT=139
```

**Observed output (verbatim, key lines) — splat control.**

```
Compilation successful!
   Output: /workspace/.tmp/madaros-run.lmz4r9/main.elf
OK
EXIT=0
```

**Expected behavior.** An uninitialized fixed-size array declaration should
either be rejected at check time (definite-initialization rule) or lowered to
a zero-initialized allocation; it must not silently produce an ELF that faults
on first use.

**Suspected area.** Codegen / stack allocation: the no-initializer path
appears to skip the actual array allocation (or leaves a null/garbage base
pointer) that the splat-initializer path sets up.

---

## Related entries in the existing known-limitations registry

A registry already exists at `docs/compiler/KNOWN_LIMITATIONS.md`
(reconciled against `docs/serious-language/public-claim-registry.v1.tsv`).
The two closest entries:

1. **Refinement types (general)** — `docs/compiler/KNOWN_LIMITATIONS.md:80`:
   "`refinement.types = prototype` — Beta/prototype; runtime fallback
   dominates non-trivial predicates." Limitation 1 is the parser-level front
   edge of this same gap: the `where` surface syntax never even reaches the
   refinement layer.
2. **Multi-module / `lower_array` segfault residuals** —
   `docs/compiler/KNOWN_LIMITATIONS.md:93` (D3, issues #901, #921) and the
   extended note at `docs/compiler/KNOWN_LIMITATIONS.md:117` (SIGSEGV at
   `lower_array: dep_begin 2`). Those cover *compile-time* segfaults in array
   lowering; Limitations 2a/2b are distinct *runtime* segfaults of the emitted
   ELF for single-module programs, so they are not duplicates.

These repros are intentionally not merged into that registry by this task
(documentation ownership stays with the registry maintainers); they are
self-contained evidence the registry can cite.

---

# Round 7 — propagating the round-6 pitfall list (P1–P5)

Round 6 (see `artifacts/ontology-frontiers/README.md`) reported five new
compiler pitfalls. This section promotes each to a verified minimal repro. All
measurements below were taken on 2026-08-02 with the same wrapper/engine as
above; several thresholds turned out to be **statement-shape-dependent**, so
the exact measured boundary and shape is recorded for each.

## LIMITATION P1 — statements past a silent per-function cap are dropped

**Symptom.** Adding one more statement to a function makes tail statements
(including the final `return`) silently vanish. No diagnostic from
`souc check` or `souc build`; the emitted ELF just behaves as if the
statements were never written.

**Measured boundary (this build, this statement shape).** With
`x = x + 1` (scalar read-modify-write) the boundary is exactly **316
statements kept / 317 dropped**:

- `p1_stmt_cap_control.sio` — 316 statements (1 `var` + 314 increments + 1
  `return`). Expected and observed: exit code 58 (314 mod 256).
- `p1_stmt_cap_dropped.sio` — 317 statements (315 increments). Expected exit
  59; **observed exit 0** — the tail never executes.

**The cap is not a raw statement count.** Measured on identical harnesses:

| statement shape              | observed fate                          |
|------------------------------|----------------------------------------|
| `x = 7` (scalar store)       | >702 statements survive                |
| `x = x + 1` (scalar RMW)     | cap 316 kept / 317 dropped             |
| `a[i] = 1.0` (f64 elem store)| >259 survive                           |
| `a[0] = a[0] + 1` (i64 RMW)  | >302 survive                           |
| `a[0] = a[0] + 1.0` (f64 RMW)| cap 256 kept / 257 dropped (see P3)    |

Round 6 measured ">682 statements" on its own mix; the consistent
interpretation is a per-function budget on lowered instructions/nodes, not on
surface statements, and every shape crosses it at a different statement count.

**Commands.**

```
./bin/souc build artifacts/ontology-frontiers/compiler-repros/p1_stmt_cap_control.sio -o /tmp/p1c.elf && /tmp/p1c.elf; echo $?
./bin/souc build artifacts/ontology-frontiers/compiler-repros/p1_stmt_cap_dropped.sio -o /tmp/p1d.elf && /tmp/p1d.elf; echo $?
```

**Observed.** control `58`; dropped `0` (expected 59). `souc check` on the
dropped file emits no related warning.

**Suspected area.** IR lowering / function-body buffering: a fixed-size
per-function instruction or node accumulator silently truncates instead of
erroring when full.

---

## LIMITATION P2 — module-level splat-initialized arrays have garbage leading elements

**Symptom.** A module-level `var g: [i64; 4] = [0; 4]` reads back with a
garbage leading element; a module-level `var gb: [bool; 3] = [false; 3]`
reads back as non-false garbage. Identical arrays as `main` locals are
correct.

**Minimal repros.** `p2_module_splat_garbage.sio` (module-level) and
`p2_module_splat_control.sio` (main-local control).

**Commands.**

```
./bin/souc run artifacts/ontology-frontiers/compiler-repros/p2_module_splat_garbage.sio
./bin/souc run artifacts/ontology-frontiers/compiler-repros/p2_module_splat_control.sio
```

**Observed (verbatim program output).**

Module-level (garbage):

```
4202496     <- g_i64[0], expected 0
0
0
0
1           <- g_bool[0], expected 0 (false)
32          <- g_bool[1], garbage byte read as bool
64          <- g_bool[2]
```

Main-local control: `0 0 0 0 0 0 0` (all expected values).

**Expected behavior.** Module-level splat initialization must produce the
same all-splat contents as the main-local form (or be rejected at check
time).

**Suspected area.** Global/BSS initialization lowering: the splat appears to
size the allocation but skip (or partially skip) the initializer stores for
leading elements; note `4202496 = 0x401400`, pointer-shaped garbage.

---

## LIMITATION P3 — f64 array-element updates in a non-main function silently stop at a hard cap of 256

**Symptom.** Round 6 reported "f64 array-element assignment outside `main`
is a silent no-op". Probing shows single f64 stores through `&!` in non-main
functions work fine (single-module, named-import multimodule, nested calls,
loops, computed values, [f64; 1024] fills — all correct). The verified
failure is sharper: **read-modify-write f64 element updates
(`a[0] = a[0] + 1.0`) inside a non-main function silently stop taking effect
after exactly 256 updates** — the same silent-truncation family as P1, with
the lowest threshold of all shapes measured.

**Minimal repros.** `p3_f64_store_control.sio` (255 updates in helper
`bump`) and `p3_f64_store_noop.sio` (257 updates).

**Commands.**

```
./bin/souc run artifacts/ontology-frontiers/compiler-repros/p3_f64_store_control.sio
./bin/souc run artifacts/ontology-frontiers/compiler-repros/p3_f64_store_noop.sio
```

**Observed.** control prints `255.000000` (expected 255.0); the 257-update
file prints `256.000000` (expected 257.0) — update #257 silently never
applies. Bisected: 256.0 first appears at exactly 257 updates.

**Why round 6 saw it as "f64 stores never work outside main".** Round 6's
generated drivers put long runs of f64 element RMWs into non-main functions;
that shape hits the silent truncation first (256), far earlier than scalar
code (316+). Chunking updates into per-call batches of ≤255 or keeping them
in `main` avoids it — matching round 6's i64-per-10000 workaround.

**Suspected area.** Same per-function lowering budget as P1; f64 element RMW
lowers to the most instructions per statement (address + load + fp-op +
store), so it exhausts the budget first.

---

## LIMITATION P4 — multimodule thin-link fails past ~10.2k assignment statements in the imported module

**Symptom.** With a named import of a generated leaf module, lowering and IR
merge succeed but the final native-binary write fails:

```
Error: Failed to write native binary to /workspace/.tmp/madaros-run.XXXX/main.elf rc=19
Compilation failed!
Errors:
  error: multimodule native thin-link compilation failed
error: madaros build: compiler produced no ELF at ...
```

**Measured boundary (this build, this shape).** The leaf carries
N × 200 `x = x + 1` assignments (200 per function, so every function is
individually under the P1 cap):

- N = 51 (10,200 assignments): **builds and prints 400** — control.
- N = 52 (10,400 assignments): **fails as above**.

Round 6 reported the threshold as "~24k" on its differently-shaped module;
as with P1 the boundary is statement-shape-dependent.

**Minimal repros.** `p4_gen_fixture.py` (generator, stdlib-only),
`p4_thinlink_control_leaf.sio` (51 fns), `p4_thinlink_control_main.sio`,
`p4_thinlink_leaf.sio` (52 fns), `p4_thinlink_main.sio`.

**Commands.**

```
./bin/souc run artifacts/ontology-frontiers/compiler-repros/p4_thinlink_control_main.sio   # prints 400
./bin/souc run artifacts/ontology-frontiers/compiler-repros/p4_thinlink_main.sio           # rc=19 thin-link failure
```

**Related single-module failure.** The same 52 functions pasted into ONE
module (import removed, `pub` kept) crash the compiler itself:
`Segmentation fault` from the raw Madaros ELF during build (no ELF
produced). So the multimodule path fails later and more gracefully (rc=19 at
binary write) than the single-module path (SIGSEGV during compile), but both
die at the same module size — the thin-link failure is the multimodule face
of a shared code-emission size limit.

**Suspected area.** Native emitter / thin-link writer: a fixed-size buffer
or relocation table overflows past ~10k lowered assignment instructions in a
module (rc=19 from the binary writer; SIGSEGV in the single-module emitter).

---

## LIMITATION P5 — qualified-import form `use m; m::f(...)` miscompiles

**Symptom.** The qualified-import form compiles cleanly but the emitted ELF
is wrong: `&!` mutations through imported functions are silently lost,
imported helpers' array stores never happen, and a scalar qualified call
whose result is used segfaults (exit 139). The named-import form
`use m::{f}` compiles the identical program correctly.

**Minimal repros.** `p5_qualified_leaf.sio`, `p5_qualified_main.sio`
(faulting), `p5_named_control_main.sio` (control).

**Commands.**

```
./bin/souc run artifacts/ontology-frontiers/compiler-repros/p5_named_control_main.sio
./bin/souc run artifacts/ontology-frontiers/compiler-repros/p5_qualified_main.sio
```

**Observed.** Control prints `1 42 7 7 7 7`. Qualified prints `0` (mutation
lost) then `Segmentation fault` on the scalar call.

**Root cause (bisected, read-only).** `m::f(...)` callees are lowered to the
Type-method mangled name `m_f`
(`self-hosted/ir/lower.sio:15698-15717`, `expr_to_callee_name_ref`);
imported functions are registered under the plain name `f`, so
`lowerer_find_or_add_fn_id_mut` (`self-hosted/ir/lower.sio:16928`) fabricates
a body-less stub `m_f` and the call silently targets it. Confirmed
empirically on the unmodified compiler: defining `fn m_f` locally captures
the qualified call (printed its body result), defining plain `fn f` does
not. Full analysis, probes, and an UNAPPLIED candidate fix:
`docs/audit/QUALIFIED_IMPORT_MISCOMPILE_2026-08-02.md` and
`qualified_import_fix_candidate.diff` (dry-run-applies at
`self-hosted/ir/lower.sio:15698`).
