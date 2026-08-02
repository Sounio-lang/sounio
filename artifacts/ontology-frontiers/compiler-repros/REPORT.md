# Compiler reproductions — ontology-frontiers lane

Minimal, verified reproductions of two compiler limitations discovered while
writing the ontology-frontiers prototypes. All outputs below were captured by
actually running the commands shown (branch
`research/zd-fiber-antisymmetry-lemma-20260731`, wrapper `bin/souc`, engine
Madaros v0.80.0, 2026-08-02). No compiler source, shared scripts, or existing
docs were modified; this directory only adds new files.

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
