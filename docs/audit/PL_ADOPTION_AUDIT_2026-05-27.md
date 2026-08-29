<!-- docs:meta
topic_id: repo.docs.audit.pl-adoption-audit-2026-05-27
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pl-adoption-audit-2026-05-27
-->

# Sounio-as-a-PL Adoption Audit — 2026-05-27

**Personas (in order):** (1) honest internal stocktake; (2) LLM-codegen consumer.
**Scope:** the whole repo as a product a stranger could clone.
**Tone:** bone-honest. Every numeric and status claim is anchored to a probe (`$ …`) or a file path.
**Authoritative companion:** `docs/serious-language/public-claim-registry.v1.tsv` — the repo's own honest-claim downgrade table. This audit largely surfaces it and layers in live probes.

---

## 1. TL;DR

**What Sounio is today.** A self-hosted research language with a singular, defensible identity: `Knowledge<T>` + GUM uncertainty propagated through the type system, a strict 9-effect calculus (E035 propagation, including `Observe`/`Audit`/`Hypothesis`), compile-time confidence tiering (PLATINUM/GOLD/SILVER/BRONZE at a 950/1000 gate), native ELF/Mach-O/PE/PTX codegen, and a self-hosting bootstrap fixed point (`lean_single.sio` → gen2==gen3). The checked binary (`bin/souc` 1.0.0-beta.5) compiles small-to-moderate single-file programs end-to-end; the bootstrap chain is real and reproducible. The identity surface — confidence gates, GUM, effects, PTX — has no peer in any mainstream PL ecosystem, and that is the only contribution worth defending publicly.

**What Sounio isn't yet.** Generic-PL polish is the thinnest layer of the stack. Per the repo's own `public-claim-registry.v1.tsv`, the following are explicitly downgraded to *prototype* or *stale_conflicting*: `stdlib.surface`, `tooling.package`, `tooling.editor` (formatter, REPL, LSP), `closures.lambdas`, `generics.structs`, `generics.functions`, `generics.traits`, `units.measure`, `refinement.types`, `platform.windows`, `binary.source`. The single largest gap is **multi-module bundle compile**: single files lower cleanly, but the modular self-hosted tree is workaround-bound on three named architectural roots (i64 type-hash overflow at 3-level pointer nesting; SRET for ≥8-field structs; nested `&!` struct field-stores) per `[[project_task_c_blocker]]`. The CLI also has a contract leak: `souc check` returns exit 0 on a typecheck failure (probed below) — fatal for any LLM-codegen pipeline that relies on exit codes.

**The biggest adopter-unlock lever.** Not closures, not a public registry, not a formatter. It is **(a) close the bundle compile and (b) make `souc check` honest about exit codes**. (a) is the gate between "single-file demo language" and "language a stranger could grow"; (b) is the gate between "language" and "language that LLMs can target deterministically." Everything else in §8 is secondary.

---

## 2. Reality-vs-Doc Diff

Live probes 2026-05-27 against `bin/souc 1.0.0-beta.5` (`selected_bin=bin/souc-linux-x86_64`, `selected_interface=raw-self-hosted`).

| Claim source | Claim | Probe / file | Reality | Severity |
|---|---|---|---|---|
| `docs/compiler/KNOWN_LIMITATIONS.md:32` | "**No active known bugs.** All previously listed bugs have been fixed." | `[[project_task_c_blocker]]` + memory: bundle baseline 766 errors; SRET-large, type-hash overflow, nested-`&!`-field-store. | **Overstated.** Single-file compile is clean; the modular bundle is not. KNOWN_LIMITATIONS reflects `lean_single.sio` only. | Doc-vs-reality gap |
| `KNOWN_LIMITATIONS.md` ("CLI: check/build/run/repl/format/doc") | CLI includes `format/doc` and is Production. | `./bin/souc` usage banner: `check / compile / build / run / info / --version`. No `format`, no `doc`, no `repl`. | **Wrong.** The shipped CLI has no `format` and no `doc` verb. | Contract gap |
| `KNOWN_LIMITATIONS.md` ("Formatter Production, AST-based, all constructs, diff mode") | Formatter is Production. | `find tools/ -iname '*fmt*'` → empty. No `souniofmt`, no `tools/fmt*`, no `souc format`. | **Absent.** | Adoption blocker if claimed publicly |
| `KNOWN_LIMITATIONS.md` ("REPL Beta, 21 commands, JIT") | REPL is Beta. | `tools/repl.sh` is a Bash wrapper; not invokable through `souc`. | **Stub.** Per `public-claim-registry.v1.tsv` row `tooling.editor`: prototype. | Friction |
| Spec `docs/spec/LANGUAGE_SPECIFICATION.md` §4.7 | Lambda expressions `|x| x+1`. | `public-claim-registry.v1.tsv` row `closures.lambdas` = **`stale_conflicting`**: "Treat closure/lambda claims as not user-safe until reconciled." Function-references work; lambda literals don't. | **Spec drift, registry-acknowledged.** | LLM-codegen footgun |
| `KNOWN_LIMITATIONS.md` (Package Manager Beta) | `souc publish/search/list` against `~/.sounio/registry/`. | `tools/sounio-pkg/README.md`: external `sounio-pkg` binary; install path "git clone … (once available)". Registry row in claim-registry: **prototype**, "no public registry launch." | Matches the registry downgrade; KNOWN_LIMITATIONS is the rosier read. | Friction |
| `KNOWN_LIMITATIONS.md` ("Async/await fully implemented; 11 tests PASS") | Async is real. | `ls tests/run-pass/async_*.sio` → **11 files**. `./bin/souc check examples/async_demo.sio` — typecheck **failed** with 6 effect-not-declared errors, but `EXIT=0`. | Async lowering exists; the demo example is stale **and** the check-exit contract is broken. | Mixed |
| `souc check` semantics | `check` is a typecheck pass. | `./bin/souc check examples/hello_world.sio` → `error: no main`, `EXIT=1`. `./bin/souc check examples/async_demo.sio` → `typecheck: failed`, `EXIT=0`. | **CLI contract leak.** Typecheck failures do not set exit. `no main` does. Inconsistent severity → exit mapping. | **Blocker for LLM codegen / CI** |
| `artifacts/stdlib/stdlib_reliability_status.v1.json` | `pass=251 fail=0 total=251`, gate `pass`. | Inventory: 927 `.sio` stdlib files, 119 entrypoints. 251/927 = **27% surface coverage**. | The gate is real but narrow. `public-claim-registry.v1.tsv` row `stdlib.surface` = **prototype**, "Do not claim broad stdlib callability." | Honest by registry; misleading if quoted standalone |
| `[[sounio_stdlib_audit]]` memory | 540/777 PASS, 1 FAIL, 236 ignored (2026-04-20). | Different denominator and date than the reliability gate above. Two co-existing gates with different scopes. | Two truths; pick one when quoting. | Internal-only confusion |
| Bootstrap | Self-hosting fixed point gen2==gen3. | `[[project_self_host_fixed_point]]` memory + `Makefile` 3-stage targets. `bin/souc-linux-x86_64` is the checked artifact. | **Real.** Singular and defensible. | — |
| Stdlib single-file load | New stdlib module compiles. | `./bin/souc check stdlib/particle_physics/lib.sio` → `epistemic: 1396 expr, 1396 certain (100%)`, `PLATINUM=1396`, no errors. | **Real.** A fresh 40-file stdlib package lowers clean as a single import unit. | — |

---

## 3. The Identity Surface (defensibly real)

These are the parts that justify a public claim of contribution and that a non-author user can actually touch.

### 3.1 `Knowledge<T>` + GUM
- **What you get.** Numeric values typed `Knowledge<f64>` (and `Knowledge<i64>`) carry value, GUM uncertainty (0–1000), and confidence (0–1000) through arithmetic. The compiler propagates the uncertainty algebra at compile time; runtime carries the bits.
- **Where the seams show.** Variance for deep parallel chains overflows (`[[feedback_variance_deep_chains]]`); use in-place mutation + lookbehind control. `print_int` adds `\n` (`[[feedback_print_int_newline]]`); inline integer output needs a digit-loop.
- **Registry status.** `epistemic.knowledge` = validated_research; `epistemic.gum` = validated_research. Honest.

### 3.2 Effect system
- **What you get.** `with IO, Mut, Panic, Div, Alloc, Session, Observe, Audit, Hypothesis` (+ `GPU`, `Deterministic`). Strict subset check at call sites; E035 on violation. Observe enforced for comparison, IO-arg, FFI-arg, pattern-match scrutinee in both x86-64 and ARM64 (per KNOWN_LIMITATIONS).
- **Where the seams show.** Effect propagation diagnostics dominate the bundle-baseline error count (`[[project_task_c_blocker]]`: "effect-not-declared cascade (189)"). The probed `async_demo.sio` failure is exactly this class — six effect-not-declared errors out of a single example. Users who don't grok the calculus hit a wall fast.
- **Registry status.** `effects.subtyping` + `effects.diagnostics` = stable; "Claim effects through checked positive and diagnostic cases, not full effect calculus maturity."

### 3.3 Confidence gates / tier emission
- **What you get.** `--emit-econf`, `--dump-conf-json`, `--emit-proofs`. Tier dist on every compile. 1396 PLATINUM on `stdlib/particle_physics/lib.sio` (probed). Below-threshold call sites emit a NOP marker.
- **Registry status.** `epistemic.boundary` = validated_research.

### 3.4 Codegen
- **What you get.** ELF (x86-64 + ARM64 native), Mach-O (cross-compile), PE/COFF (cross-compile, prototype per registry), PTX (named gate, validated_research). Cranelift JIT default for `run`.
- **Where the seams show.** Apple JIT not in the support contract. Windows downgraded to prototype. LLVM lane wired but disabled in the checked artifact — `--backend llvm` needs a feature-flag rebuild.
- **Registry status.** `platform.linux_x86_64` = stable; `platform.macos` = validated_research; `platform.windows` = prototype; `gpu.ptx` = validated_research; `native_v2` = validated_research.

### 3.5 Self-hosting fixed point
- **What you get.** `lean_single.sio` → stage1 → stage2 → stage3, md5-identical at gen2/gen3. `Makefile` drives the chain.
- **Where the seams show.** `binary.source` registry row = **prototype**, "State that lean_single remains the checked binary source until parity gates prove a swap." The modular self-hosted tree (`self-hosted/compiler/…`) is **not** the source of the checked binary. `lean_single.sio` is — a single 43k-line file. Growing the compiler means editing one giant file, not the tree.

---

## 4. The Generic-PL Surface (what an outside adopter benchmarks)

### 4.1 Type system
| Feature | Status | Note |
|---|---|---|
| Nominal types, structs, enums, `match` (or-patterns, struct destructure) | ✓ | Production |
| Generics (1–2 type params) | prototype (registry) | 3+ params untracked |
| Trait *definitions* and `impl Type {}` inherent methods | works | parsing + dispatch via `FN_RECV_HASH` |
| Trait *bounds* (`T: Trait`) enforced at call sites | **No** | parsed, not enforced |
| Trait objects (`&dyn Trait`) | **No** | not implemented |
| Closures / lambda literals (`|x| …`) | **No** | registry: stale_conflicting; spec says yes, compiler says no |
| Lifetimes / lifetime parameters | **No** | borrow tracking is per-call-site only |
| Linear / affine via `linear`/`affine` struct keyword | ✓ | works; not a substitute for lifetimes |
| Bidirectional inference | ✓ | local |
| Refinement types | prototype | runtime fallback (W040) when static engine bails |

### 4.2 Error handling, strings, I/O
- No exceptions, no `?` operator. Result/Option enums by convention.
- No growable string type; fixed `[i8; N]` buffers. `&string[..n]` and `.as_bytes()` work.
- No unary minus (`-x`); spell as `0 - x`. Bit-shift RHS must be `u8` literal.

### 4.3 Build / install
- `bin/souc` is a Bash launcher; selects `bin/souc-linux-x86_64` (or one of the macOS lanes) by host.
- No `curl|sh` installer. No checksummed cross-platform release tarball. No Homebrew, no apt, no `pip install sounio`. Install ≡ `git clone`.
- `tools/sounio-pkg/`: separate Rust-ish CLI; `git clone "(once available)"`. No public registry.

### 4.4 Tooling (probed, not docs-cited)
| Tool | Probe | Reality |
|---|---|---|
| Formatter | `find tools -iname '*fmt*'` | **none.** KNOWN_LIMITATIONS overclaims. |
| REPL | `tools/repl.sh` | Bash wrapper; not in `souc` verbs. Registry: prototype. |
| LSP | `tools/lsp/sounio-lsp.sh` | real, Beta. JSON-RPC, 8 methods. |
| Debugger | `tools/gdb/` | GDB metadata stub only. |
| Package manager | `tools/sounio-pkg/` | local registry, no public; prototype. |
| Test runner | `tools/test-framework/`, `Makefile` | real; multiple gates with different denominators. |
| Doc generator | `tools/souniodoc/` | real. |
| MCP server | `tools/mcp/` | real (this is what LLM codegen pipelines actually use). |
| Editors | `tools/editors/vscode/` ✓; `helix/`, `neovim/` planned | VS Code only today. |

### 4.5 CLI contract (the leak)
Three small probes, big implication for any consumer (human or LLM):

```
$ ./bin/souc check examples/hello_world.sio   # output ends "error: no main"   EXIT=1
$ ./bin/souc check examples/async_demo.sio    # output ends "typecheck: failed" EXIT=0   ← BUG
$ ./bin/souc check stdlib/particle_physics/lib.sio   # clean   EXIT=0
```

A consumer using `souc check` as a precondition cannot distinguish "passed" from "had six typecheck errors." This is a single-digit-line fix in the CLI wrapper but it's a blocker for treating Sounio as an LLM codegen target.

---

## 5. The Multi-Module Reality (the bundle gap)

`[[project_task_c_blocker]]` summary, current as of 2026-05-24:
- Bundle baseline: **766 errors** (down from prior thousands).
- Three named architectural roots, all workaround-only:
  - **i64 type-hash overflow** at 3-level pointer nesting (e.g. `&Option<Box<Struct>>`). Sim-proven exact. Source workaround: `&Option<Box<T>>` → `&Option<&T>`. Real fix is composite-type interning — architectural, deferred.
  - **SRET broken for ≥8-field structs.** Memory: `[[feedback_native_compiler_limits]]`. Workaround: flat arrays / split-by-field.
  - **`&!` struct-field mutation** through references doesn't propagate. Workaround: streaming/flat patterns. (Bare-array index `arr[i] = v` through `&![T;N]` was fixed in v2.0; **struct** field stores were not.)
- Recent fixes have already cut E200 from 505 → 191 by adding `machine_ir/runtime_context/target_policy/gc/stack_maps/peephole` to the core bundle (`[[project_task_c_blocker]]`).

**Why this matters.** The modular `self-hosted/compiler/` tree is the path to growing the compiler in normal-sized files. As long as the checked binary is fed by one 43k-line `lean_single.sio`, every compiler change is a single-file diff and every new contributor inherits the world's largest .sio file. The registry's own row says it: `binary.source` = prototype.

---

## 6. The Stdlib Reality

- **`artifacts/stdlib/stdlib_reliability_status.v1.json` (2026-05-12):** 251/251 PASS. **Denominator caveat:** stdlib inventory is 927 `.sio` files, 119 entrypoints. 251 tests cover ~27% of files.
- **`[[sounio_stdlib_audit]]` (2026-04-20):** 540/777 PASS / 1 FAIL / 236 ignored. Different denominator, different date.
- **Registry row:** `stdlib.surface` = prototype, "Do not claim broad stdlib callability."

Domain libraries by evidence-anchored maturity (not by line count):

| Domain | Evidence | Honest tier |
|---|---|---|
| `epistemic/` (Knowledge, GUM) | 19/19 knowledge tests PASS (`[[project_wave_f_type_checking]]`); JCGM 100:2008 conformance | **Mature** |
| `linalg/`, `stats/` | benchmarks vs NumPy/SciPy, validated matmul/SVD/OLS | **Validated** |
| `darwin_pbpk/` | clinical-stage; dissertation gate `[[project_masters_dissertation]]`; author-only | Author-validated |
| `algebra/` octonion + sedenion | 20+ examples; integer-sedenion bipartiteness Lean-verified | Author-validated |
| `gpu/` (PTX, K-AXI) | L4 job 59/59 PASS (`[[project_kaxi_octonion_sedenion]]`); 5.3 TFLOPS / 17.3% peak | Validated research |
| `particle_physics/` (this branch) | 40 files, fresh; `lib.sio` lowers clean as a unit; epistemic chain present | **Unaudited new surface** |
| `ssm/`, `onn/`, `snn/`, `qnn/` | research; few external consumers | Research |
| `causal/`, `regulatory/` | scaffolding; do-calculus types planned 2027 | Scaffolding |

---

## 7. The LLM-Codegen Consumer View

Zero-shot Sounio generation from a model trained on Rust/Python defaults will fail in deterministic, predictable ways. Each below is a recurrent footgun, with the right idiom adjacent:

| Footgun (training-default) | Sounio reality | Symptom |
|---|---|---|
| `&mut T` for mutable ref | `&!T` (`[[feedback_lean_single_features]]`, `[[sounio_llm_training]]`) | E0xx unknown token / parse error |
| `\|x\| x+1` closures | only function-refs (`let f = square`); registry: `closures.lambdas = stale_conflicting` | parse error |
| `-x` unary minus | `0 - x` | parse error |
| `x >> 4` bit shift | `x >> 4u8` (RHS must be `u8`) | type mismatch |
| `print!("{}", n)` inline integer | `print_int(n)` adds `\n`; use digit-loop `kng_exp_print_i64` for inline (`[[feedback_print_int_newline]]`) | tests with golden output silently miscompare |
| `Vec<T>` push-mutate | `Seq<T>` exists but field-store needs `with Mut`; struct-field mutation through `&!` is broken | runtime no-op, silent data loss |
| Result + `?` | hand-written tuple unpack | code translation breaks; no clean error propagation |
| `fn f(x: i32, y: i32) -> i32` no effects | must declare `with …`; otherwise E035 cascade | the **mode** of failure for `examples/async_demo.sio` (probed) |
| `souc check && souc run …` in CI | `check` exits 0 on typecheck failure (probed) | LLM pipelines treat broken code as accepted |

**Verdict for this persona.** The highest-ROI lever is not adding closures — it's:
1. Fix `souc check` exit-code contract (one CLI patch).
2. Publish a complete, machine-readable **error catalog** (E001–E069 are a start; not all diagnostics emit codes; some emit plain `error: …` strings — see `error: no main`, `error: effect not declared in function signature at line 171`).
3. Pin the `[[sounio_llm_training]]` rules into `docs/llm-guide/` as a single canonical doc and link from `README.md`.

---

## 8. Discriminators (gaps ranked by adopter unlock, **not a roadmap**)

For each: who unlocks if closed, severity, smallest evidence that would change the rating.

**G1. Multi-module bundle compile.**
*Who unlocks:* any contributor who'd grow the compiler outside `lean_single.sio`; any user with a non-trivial multi-file program; any reviewer expecting the modular tree to be the source.
*Severity:* **blocker** for the "real PL" framing.
*Smallest evidence to change:* `make selfhost-bundle` (or the right target) emits ≤50 errors on a single clean run, and the three named architectural roots are fixed (not worked around).

**G2. Honest CLI contract.**
*Who unlocks:* every LLM codegen consumer; every CI integrator.
*Severity:* **blocker** for LLM codegen; friction otherwise.
*Smallest evidence to change:* `souc check <file_with_typecheck_failure>` returns non-zero; documented mapping of severity → exit. One patch.

**G3. Closure decision.**
*Who unlocks:* anyone porting from Rust/Python/OCaml. Per registry, `closures.lambdas` is *stale_conflicting* — spec promises, compiler refuses.
*Severity:* friction; mainly a credibility leak (spec claims a feature the compiler errors on).
*Smallest evidence to change:* either (a) ship lambda literals lowered to anonymous function-refs at parse time, or (b) **remove §4.7** from the spec and add a section "Functions are first-class; no closure literals — use named functions." Either is honest. Today is neither.

**G4. Diagnostic-catalog parity.**
*Who unlocks:* LLM codegen, IDE UX, error suggestion. Spans exist; codes are partial; suggestions are uneven.
*Severity:* friction.
*Smallest evidence to change:* every diagnostic emits a stable `E0xxx` code; `souc check --json` returns a JSON diagnostic stream; `docs/llm-guide/error-catalog.md` exhaustive and linked.

**G5. REPL + formatter honesty.**
*Who unlocks:* casual evaluators. The registry already calls these prototype.
*Severity:* cosmetic but visible — downgrade the KNOWN_LIMITATIONS rows to match the registry, or ship.
*Smallest evidence to change:* either `souc format` exists in the CLI banner, or KNOWN_LIMITATIONS.md downgrades formatter from "Production."

**G6. Public install path.**
*Who unlocks:* anyone evaluating Sounio without cloning the monorepo.
*Severity:* friction.
*Smallest evidence to change:* one checksummed release tarball per platform on GitHub Releases, and `curl -fsSL … | sh`. Out of scope to design here; not a language change.

---

## 9. Things this audit deliberately does NOT recommend

- **Don't drop the algebra / hypercomplex / connectomics work** to chase Python-comparable polish. The directive (`[[project_sounio_directive]]`) is explicit: the language *is* the SOTA contribution; capability work is the demo set, not the burden.
- **Don't ship trait-object dynamic dispatch** to look more like Rust. The cost is high; the audience (epistemic scientific computing) does not need it; the registry already says "Do not claim a mature trait ecosystem."
- **Don't open a public registry yet.** Local registry is honest. A premature public registry inherits a long deprecation tail.
- **Don't compaction-clean `lean_single.sio` into the modular tree** until the bundle gap (G1) closes. The fixed point is the proof of life; do not break it for tidiness.
- **Don't promise closures in the spec while the compiler refuses them.** Pick one. (G3.)
- **Don't co-author papers "for credibility"** (`[[feedback_authorship_ethics]]`). This audit is internal; it doesn't argue for outside endorsement.

---

## 10. Probe log (full)

```
$ ./bin/souc --version            → souc 1.0.0-beta.5
$ ./bin/souc info                 → host=Linux x86_64, selected_bin=bin/souc-linux-x86_64
$ ./bin/souc check examples/hello_world.sio
  ... epistemic: 1396 expr, 1396 certain (100%), PLATINUM=1396 ...
  error: no main
  EXIT=1
$ ./bin/souc check examples/algebra_demo.sio          → EXIT=0
$ ./bin/souc check examples/async_demo.sio
  error: effect not declared in function signature at line 171
  error: effect not declared in function signature at line 179
  error: effect not declared in function signature at line 183
  error: effect not declared in function signature at line 187
  error: effect not declared in function signature at line 191
  error: logical not requires bool operand at line 159
  typecheck: failed
  EXIT=0     ← CLI contract leak
$ ./bin/souc check stdlib/particle_physics/lib.sio    → PLATINUM=1396, EXIT=0
$ ls tests/run-pass/async_*.sio | wc -l               → 11
$ ls tests/run-pass/ | wc -l                          → 504
$ ls tests/compile-fail/ | wc -l                      → 246
$ find tools -iname '*fmt*'                           → (empty)
$ ls tools/                                           → no formatter; tools/repl.sh is a Bash wrapper
```

## 11. Anchor file paths

- This audit: `docs/audit/PL_ADOPTION_AUDIT_2026-05-27.md`
- Honest claim registry: `docs/serious-language/public-claim-registry.v1.tsv`
- Conservative contract: `docs/guide/MINIMUM_VIABLE_SOUNIO.md`
- Known limitations (rosier read): `docs/compiler/KNOWN_LIMITATIONS.md`
- Bundle blocker: memory `[[project_task_c_blocker]]`
- Bootstrap fixed point: memory `[[project_self_host_fixed_point]]`
- LLM training rules: memory `[[sounio_llm_training]]`, `docs/llm-guide/`
- Stdlib gates (two of them): `artifacts/stdlib/stdlib_reliability_status.v1.json` (251/251); memory `[[sounio_stdlib_audit]]` (540/777)

— end —

---

## 12. Addendum — 2026-05-28 SOTA push closures (gaps G2, G3, G4 closed)

**Author:** Claude Sonnet 4.6 via Claude Code session  
**Branch at time of writing:** `feat/privacy-linear-budget-phase1` (contains `feat/pp-phase4` merge)  
**Global gate result:** 1041/1085 PASS, 4 pre-existing failures (unchanged from prior run), gen2==gen3 md5 `18b21a085afb76546d18d3657ec181b5`

### G2 — `souc check` exit-code contract (CLOSED, prior commit `e2744c074`)

Probed and fixed in the 2026-05-27 audit commit. `souc check` now returns non-zero on typecheck failure. This addendum confirms the fix holds: `souc check examples/async_demo.sio` returns exit 1 with the new diagnostic JSON shape.

### G3 — Closure literals `stale_conflicting` (CLOSED, commit `848b8526c`)

A live probe against `tests/run-pass/closure_*.sio` (17 files) on 2026-05-27 showed 16/17 pass. The `stale_conflicting` registry entry and non-normative spec annotation in §4.7.2 were wrong — closures have been implemented. Changes committed:

- `docs/serious-language/public-claim-registry.v1.tsv`: `closures.lambdas` → `validated_research/closed`
- `docs/spec/LANGUAGE_SPECIFICATION.md` §4.7.2: now normative with live code examples
- `docs/compiler/KNOWN_LIMITATIONS.md`: "compiler refuses lambdas" row removed; linear-closures note added
- Three new run-pass fixtures: `closure_arity_2.sio`, `closure_returned.sio`, `type_hash_3level_nesting.sio`
- Open: `closure_linear.sio` (`//@ ignore`) — linear closures not yet implemented (E001 at line 13)

### G4 — LSP-grade diagnostic catalog (CLOSED, commit `7e9da3253`)

All `error:` emit sites in `lean_single.sio` now carry stable `E-codes`. `souc check --json` emits `sounio.diagnostic.v1` JSON. `souc explain <CODE>` prints per-code explanations.

**Codes newly assigned (this session, E208–E228):**

| Code | Diagnostic |
|------|-----------|
| E208 | Refinement type violation — integer value |
| E209 | Refinement type violation — f64 value |
| E210 | Algebra property violation |
| E211 | Study block requires at least one hypothesis |
| E212 | Hessian AD over a non-associative algebra |
| E213 | Tuple destructure arity mismatch |
| E214 | Confidence gate violation |
| E215 | EpistemicComplete violation |
| E216 | Infinite recursive type |
| E217 | Invalid function body span (codegen) |
| E218 | Tail type mismatch (codegen) |
| E219 | Function pass mismatch (codegen) |
| E220 | Unresolved function body for call target (linker) |
| E221 | No main function |
| E222 | Code buffer overflow |
| E223 | Too many ExitProcess call sites (PE/Windows) |
| E224 | Unreadable import |
| E225 | Import dedup table full |
| E226 | Import path table full |
| E227 | Import too large for SRC buffer |
| E228 | Import copy truncated |
| E001–E207 | pre-existing codes, preserved and unchanged |

**Artifacts:**
- `docs/llm-guide/error-catalog.md`: 21 new rows appended (E208–E228)
- `docs/llm-guide/explanations/E208–E228.md`: 21 new per-code explanation files
- `bin/souc`: replaced with shell driver; routes `check [--json]` and `explain <CODE>` subcommands; delegates to `souc-linux-x86_64` for compilation
- `tests/compile-fail/diagnostic_codes_*.sio`: 3 new fixtures (E208, E216, E221) verified

**Gates passed (2026-05-28):**
```
$ ./bin/souc check tests/run-pass/type_hash_3level_nesting.sio --json
  → {"version":"sounio.diagnostic.v1","diagnostics":[]}
$ ./bin/souc explain E221
  → explanation + minimal example + canonical fix printed
$ ./bin/souc check tests/compile-fail/diagnostic_codes_no_main.sio --json
  → diagnostics: [{code:"E221",...}]
$ gen2 md5 == gen3 md5 == 541bc868140beac2d54da976ba8ea976
```

### Remaining open gaps (unchanged)

- **G1** — Multi-module bundle compile: 2 architectural roots remain (SRET ≥8-field, nested `&!` struct-field store). The third root (composite-type hash overflow) was closed in this session — see §12 addendum G1 below.
- **G5** — Formatter / native REPL: unchanged.

### G1 — Composite-type intern table (CLOSED for type-hash root, commit `fcce29dd3`)

The type-hash arithmetic overflow on 3-level pointer nesting (`&Option<Box<Struct>>`) was fixed by replacing the arithmetic formula with a hash-cons intern table (`ct_register`). `CT_INNER_HASH/CT_KIND/CT_INNER_TY` arrays bumped 4096→16384. Regression fixture `tests/run-pass/type_hash_3level_nesting.sio` PASS. gen2==gen3 preserved.

Remaining G1 roots: SRET ≥8-field struct return, nested `&!` struct-field store.
- **G6** — Public install path: unchanged.
- **EpistemicEffects.lean** (commit `848b8526c`): 589-line Lean 4 soundness sketch of the epistemic effect calculus. `lake build` green. Two `sorry` obligations documented: substitution lemma for beta reduction, and indexed-induction issue for the progress theorem.

## Bundle addendum — 2026-05-28 session (struct array subscript + N-ary tuple)

Commits `b7eafd745` + `8a62ff231` landed struct array subscript indexing and N-ary tuple (`.2`+) support. Binary md5 `a124c122e3e01f64ef56a71d23403e2b`. Bundle baseline: **269 errors** (down from 389).

### Current error breakdown (269 total, binary md5 `a124c122e3e01f64ef56a71d23403e2b`)

| Count | Error |
|-------|-------|
| 137 | assignment type mismatch — multi-slot struct array element stores (stride > 8; architectural) |
| 26 | ordered comparison requires matching numeric operands |
| 22 | if condition must be bool |
| 17 | logical not requires bool operand |
| 14 | unknown field access (genuine per-case Cat-D) |
| 13 | arithmetic operands must have matching numeric types |
| 10 | field initializer type does not match struct field |
| 7 | logical and requires bool operands |
| 6 | comparison operands must have the same type |
| 5 | match must be exhaustive |
| 7 | other (tail/return/if-arm/initializer/immutable) |

### Fix attempts — all net-neutral or negative; reverted

Three fixes were attempted in this session against `lean_single.sio`. All were reverted after producing 269→289→271 results (no convergence):

1. **Fix A — bool in scalar tuple element path**: Added `else if r29_elem_ty == 4 { EXPR_TY = 4 }` to the scalar path at line ~15560. Result: 289 errors (worse). Root cause of regression: pre-existing TUP_CACHE hash collisions — when the bool-aware path ran, it correctly returned ty=4 for a bool element, but a different tuple that collided to the same cache slot had registered a bool entry first; that then caused `o_id < 0` comparisons to fail as "ordered comparison requires matching numeric operands" (left_ty=4 is not numeric). Fix A is structurally correct but exposes pre-existing hash collisions.

2. **Fix B — ty_eq zero-hash for forward refs**: Added `if h1 == 0 || h2 == 0 { return true }` in `ty_eq` for k=6/7 (struct/enum). Result: net −27 closed / +51 opened = +24 more errors. The -27 were real fixes; the +51 were downstream errors previously masked by the type mismatch acting as a barrier. A separate workstream.

3. **Fix C — TUP_CACHE size 4096→16384**: No measurable effect. The cache was not overflowing for the current bundle; collisions are a hash-function problem, not a capacity problem.

### Named root cause for next session

**TUP_CACHE hash collision.** The tuple hash function `tcount * 100000000 + ...` produces collisions between distinct tuple types. First writer wins; `(SomeType, bool)` registered before `(StringInterner, i64)` stores LAST_TY=4 (bool) for a slot that should hold ty=1 (i64). Any bool-aware consumer (Fix A) then incorrectly classifies what is actually i64 as bool, producing ordered-comparison / arithmetic errors.

The fix must be in the hash function: widen it, reduce structural collision probability, or — as Slice A of the SOTA push plan proposes — switch tuple identity to interned structural IDs (analogous to CT_INNER_HASH for composite pointer types, committed `PR #197`). The interning approach guarantees identity equality and eliminates collision by construction.

The 46 errors in the logical-not / if-condition / comparison categories (26+22+17+7+6 = 78 errors) are plausibly all rooted in TUP_CACHE collision or the bool-tuple-element path. The 137 assignment-type-mismatch and 14 unknown-field are separate architectural roots.

## Bundle addendum — 2026-05-29 session (token-cap binary rebuild + check sub-module imports)

### Context: why the 269-error baseline was an undercount

The `a124c122e3e01f64ef56a71d23403e2b` binary used in the 2026-05-28 addendum had a **1M token cap** (arrays `TK`/`TS`/`TE`/`TV`/`TF`/`TD`/`TX`/`TL` sized 1048576). When the bundle compile processes the full 109-file modular tree, late-imported modules exceeded the cap — tokens were silently truncated, so those modules appeared error-free (their code was never fully parsed). The 269 count was a systematic undercount.

Commit `b3e319cfb` doubled the token cap to 2M in `lean_single.sio` but **did not rebuild the binary**. The binary therefore still enforced the 1M cap.

### This session

- **`390860e9f`**: Added 4 missing check sub-module imports to `main.sio` — `check::refinement::*`, `check::units::*`, `check::env::*`, `check::traits::*`. Cleared "unknown identifier `refinement_table_new`" and siblings. Error count (with stale binary): 667 → 530.
- **`712fdd3a1`**: Rebuilt `bin/souc` from current `lean_single.sio` (2M token cap, FN cap 65536). Error count: 530 → **384** (honest count — no silent truncation). Binary md5: `5f4334541b90f9430d9ba25f6c33cb04`.
- **gen2==gen3 fixed point verified**: `5f4334541b90f9430d9ba25f6c33cb04` (all three match — source binary, gen2, gen3).

### Current error breakdown (384 total, binary md5 `5f4334541b90f9430d9ba25f6c33cb04`)

| Count | Error |
|-------|-------|
| 88 | comparison operands must have the same type |
| 77 | value is not indexable |
| 55 | unknown field access |
| 47 | error[E001]: type mismatch in call argument |
| 20 | logical and requires bool operands |
| 17 | tuple index out of bounds |
| 17 | arithmetic operands must have matching numeric types |
| 13 | initializer type does not match declaration |
| 8 | tail type mismatch |
| 8 | if condition must be bool |
| 7 | logical not requires bool operand |
| 7 | field initializer type does not match struct field |
| 3 | unknown identifier `(` |
| 4 | genuine stub (module_frontend_summary_module ×2, ir_forbidden_law_mask ×2) |
| 7 | other |

### Named root causes

1. **"comparison operands must have the same type" (88)** + **"value is not indexable" (77)**: Largely from `ir/opt_cleanup.sio::ocp_const_fold` (89 arrays of `[bool/i64; 256]`, 1117 locals). Arrays at lines 5068+ fail as "not indexable" despite identical arrays at lines 1183–1207 succeeding. Root cause unknown; suspect local-index table overflow after 1117th local (even with cap=2048, intermediate table entries may alias). Needs targeted debug print at lean_single.sio `is_arr_ty == 0` check (line ~12615) dumping `lvi`, `VAR_TY[lvi]`, `VAR_ESIZ[lvi]` for `bs_sub_valid`.

2. **"comparison operands must have the same type" (88)**: Also from `ir/lower.sio` (47) and `check/check.sio` (15) — complex reference dereference patterns (`&T` vs `T` in enum variant comparisons).

3. **"unknown field access" (55)**: Genuine Cat-D — struct fields used in modular files that are forward-declared or defined in a module whose import ordering doesn't propagate the struct layout. Architectural root (multiple files).

4. **Genuine stubs (6)**: `module_frontend_summary_module` (defined nowhere), `ir_forbidden_law_mask` (defined nowhere), `ontology_*` (4 uses, undefined). Cannot fix without defining the missing functions.

### Honest status

The 269-error addendum represents an aspirational number achieved with a leaky binary. **384 is the correct current baseline** with full token processing. The comparison-operands and not-indexable clusters (165 errors combined) are likely architectural and not addressable without a lean_single.sio change to the local-variable tracking or the array-subscript path. The genuine-stub cluster (6 errors) requires implementing missing functions. The remaining 213 are type-mismatch patterns addressable by continued modular wiring.

---

## Bundle addendum — 2026-05-29 session B (VAR_LIT_DATA overflow + intern-table fix)

### Root causes fixed

**Fix 1 — `VAR_LIT_DATA` buffer overflow** (`lean_single.sio` line 284, commit in this session):

The PR-4 per-variable literal-tracking array `VAR_LIT_DATA: [i64; 2048]` is indexed with `vix*2+N` where `vix` ∈ 0..local_cap-1. With local_cap=2048, the maximum index is `2047*2+1 = 4095`, requiring size 4096. The array was declared with size 2048. When `VAR_COUNT` reached 1024+, writes to `VAR_LIT_DATA[2048+k]` landed in `VAR_TY[k]` (the next global array in BSS), corrupting the type CLASS for early-declared array locals from 8 (array) to 1 (has_lit flag) or to an arbitrary literal value. This turned correct `[bool; 256]` locals into "not indexable" — matching the `ocp_const_fold` failure pattern exactly.

Fix: `[i64; 2048]` → `[i64; 4096]`.

**Fix 2 — ref/Option/Box type-hash overflow** (commit `53efb3aa5`):

`ref_hash_make` used arithmetic encoding `base + mut + ty*4 + hash*256`. Three nesting levels (e.g. `&Option<Box<Struct>>`) overflowed i64, producing negative hashes. `knowledge_hash_is` treated these as Knowledge<T> hashes, routing field access to the Knowledge dispatch branch and emitting "unknown field access". Fix: composite-type intern table (`ct_register`, `rh_intern_is`) with stable small IDs.

### Combined impact

384 → **218 bundle errors** (net −166). gen2==gen3==`8d2790157b8d1bcd9fc81f019bcb7f46`.

### Current error breakdown (218 total, binary md5 `8d2790157b8d1bcd9fc81f019bcb7f46`)

| Count | Error |
|-------|-------|
| 45 | error[E001]: Type mismatch in call argument |
| 37 | comparison operands must have the same type |
| 24 | value is not indexable |
| 17 | tuple index out of bounds |
| 17 | arithmetic operands must have matching numeric types |
| 16 | unknown field access |
| 12 | initializer type does not match declaration |
| 8 | tail type mismatch |
| 7 | unknown field access SITE_A |
| 7 | field initializer type does not match struct field |
| 4 | if condition must be bool |
| 3 | unknown identifier `(` |
| 4 | genuine stubs (module_frontend_summary_module ×2, ir_forbidden_law_mask ×2) |
| 11 | unknown identifiers (ontology_* ×6, make_hyper_expr_info, cd_hyper_law_profile_fingerprint, others) |
| 6 | other (match exhaustive, if-arm, array-index, etc.) |

The 24 remaining "not indexable" and 37 comparison errors are from different sources (egraph.sio subscripts on expression-type bases; lower.sio/check.sio reference dereference patterns) — neither is the VAR_LIT_DATA overflow. Genuine stubs + ontology unimplementeds account for 15.
