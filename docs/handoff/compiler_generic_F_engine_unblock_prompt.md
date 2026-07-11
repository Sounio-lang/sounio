<!-- docs:meta
topic_id: repo.docs.handoff.compiler-generic-f-engine-unblock-prompt
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.compiler-generic-f-engine-unblock-prompt
-->

# Prompt — fable5: unblock the generic `<F>` exact Cayley-Dickson engine (end-to-end)

**For:** fable5 (compiler-internals agent)
**Authored by:** Claude (exact-algebra-core lane — now merged to main, PR #631 `f2bf998ce`), 2026-07-05
**Type:** self-hosted compiler (parser + type checker + monomorphizer), **both engines** (lean_single + Madaros)
**Serialized surfaces:** `self-hosted/compiler/lean_single.sio` + `bin/souc-linux-x86_64` (coordinate with Lane 4). Fixed-point + output-verified gates mandatory.

---

## 0. Goal (the one acceptance target)

Make the **generic** exact Cayley-Dickson engine compile **and run correctly**, so the exact-algebra lane can replace its hand-monomorphized concrete engine (`stdlib/algebra/cayley_dickson_exact_i64.sio`, on main) with a single generic one over a coefficient ring `F`:

```
struct CDElementExact<F: ExactRing> { c: [F; 2048], bits: i32 }
fn cd_mul_exact<F: ExactRing>(a: CDElementExact<F>, b: CDElementExact<F>, zero: F) -> CDElementExact<F> ...
fn cd_associator_exact<F: ExactRing>(...) -> CDElementExact<F>
fn zd_exact<F: ExactRing>(...) -> bool
```

The **exact consumer skeleton already exists on main**: `stdlib/algebra/cayley_dickson_exact.sio` (does not compile — that's what you're fixing) with a methods-only `trait ExactRing` and `impl ExactRing for i64 / Rational`. Read it first; it is written to MINIMIZE the compiler ask (methods-only trait — no associated/`Self`-returning fns; ±1 sign selects add/sub so no `from_int`; `cd_sigma` reused verbatim).

## 1. The prerequisite set (all spike-verified; do them in order)

The `<F>` engine needs FOUR compiler capabilities. Sub-prompts + minimal repros are on main in `docs/handoff/`:

1. **generic-struct-return** — a fn returning a generic struct parameterized by its own type param (`fn f<F>(..) -> Wrapper<F>`) is rejected by both engines. Symptoms: `turbofish.sio` known-failure; the spike yields `expected CDExact, found CDExact__T` (mangler leaks the type-param name); lean_single **"tail type mismatch"** @ `lean_single.sio:27281`. **Prompt:** `docs/handoff/compiler_generic_struct_return_fix_prompt.md` (merged via PR #636). Root-map: `mono_mangle`@5740, MONO_* registry + discovery @ ~25150–25325, Pass 0c+0d @ 26178.
2a. **bodyless trait-method signatures** — `trait ExactRing { fn er_add(self, o: Self) -> Self }` fails to PARSE (only empty `trait {}` parses today). Spike: `docs/handoff/repros/`.
2b. **`impl Trait for Type`** — `impl ExactRing for i64 { .. }` fails to parse at the `for` keyword; zero *compiling* occurrences in stdlib. **Prompt (covers 2a+2b):** `docs/handoff/compiler_impl_trait_for_type_fix_prompt.md`.
3. **trait-bounded generic dispatch** — `fn f<F: ExactRing>(a: F) -> F { a.er_add(b) }` needs (after 2a/2b) the checker to resolve `a.er_add(b)` to the concrete `impl` per monomorphized `F`, and emit per-instantiation code (reuse the generic-fn mono machinery). **Prompt:** `docs/handoff/compiler_trait_bounded_dispatch_fix_prompt.md`.

Full matrix + rationale: `docs/handoff/exact_engine_prereqs.md`.

## 2. Acceptance criteria (ALL required; output-verified; both engines)

1. `tests/run-pass/turbofish.sio` → **run-pass** (remove the `known-failure` annotation); all three asserts print `PASS`.
2. Each of the four spikes (generic-struct-return + the two trait spikes) compiles and runs correct output.
3. **THE REAL TARGET:** `stdlib/algebra/cayley_dickson_exact.sio` compiles, AND a new `tests/run-pass/cd_exact_generic_i64.sio` instantiates `CDElementExact<i64>`, runs `cd_mul_exact`/`zd_exact` on the canonical sedenion zero-divisor pair `a=e₃+e₁₀, b=e₆−e₁₅`, and:
   - proves it annihilates (all 16 comps 0), AND
   - **produces byte-identical results to the concrete `cayley_dickson_exact_i64.sio` engine** on the same inputs (the generic engine must reproduce the monomorphic one).
4. Self-host fixed point preserved (canonical compiler gate; gen2==gen3). No regressions: existing `generic_struct_*`, `closure_generic_hof`, and the whole exact-algebra suite (`sedenion_zd_census_168`, `octonion_*`, `bignat_selftest*`, `sedenion_cd_full16_q`, …) still pass. Full run-pass fail-count not worse than baseline.

## 3. HARD-WON GOTCHAS from the exact-algebra lane (read these — they cost me hours)

- **souc miscompiles SILENTLY.** rc is a **false-green** (exit 0 on failure); worse, several builds emit **wrong values with a clean compile** (observed: swapped add/sub, sign flips, aliased structs). NEVER trust a bare `PASS`. Verify every numeric result against an **independent non-souc oracle** — the exact-algebra lane ships Python oracles (`scripts/research/{verify_zd168,bignat,ratbig,cd16}_oracle.py`) and a 9-face gate (`scripts/ci/sedenion_zd168_crosscheck_gate.sh`, in CI). Add a cross-check for your `<F>` test the same way.
- **CI uses a FRESH stage2 souc built from source** (`SOUNIO_TEST_SOUC_BIN=/tmp/souc-stage2`), NOT the committed `bin/souc` (which is older). **Local pass ≠ CI pass.** To reproduce a CI-only failure, download the exact artifact: `gh run download <runid> -n native-compiler-linux-x86_64` → `souc-stage2` (interface: `mini_native <src> <out>`; `chmod +x` the ELF). Test your fix against a stage2 build, not just `bin/souc`.
- **Related defects to keep in view** (repros in `docs/handoff/souc_v0800_defects.md`, issues #637/#638/#639/#641/#643/#645): D6/#643 `var r = a; r.field = x` on a struct **aliases the caller** (use fresh literals); D7/#645 **data-carrying enum variants** rejected (E200 undefined-var @ ~`lean_single.sio:14486`) — not needed for the CD engine (structs only), but on your radar since the monomorphizer touches aggregate construction. `[F; 2048]` where `F` is a struct (e.g. `Rational`) is a large by-value aggregate — watch #637 (cross-module aggregate SIGSEGV) and the ~24-struct-fn whole-program capacity wall.
- Build on **current main** (has the recent a64 codegen fixes #630/#632/#633/#640/#642/#644).

## 4. Protocol

Fresh worktree off `origin/main`; CLAIM in `artifacts/omega/agent_handoff.log.md`; serialize `lean_single.sio` + the `bin/souc` token with Lane 4 (nv2-hardening). Rebuild to fixed point; **output-verify** every witness (exit-code gates false-green). On RELEASE: ping the exact-algebra consumer via the log so the lane swaps its concrete-i64 engine for `CDElementExact<F>` (and adds `F=Rational` over `stdlib/math/rational.sio`, then `F=BigInt` over `stdlib/math/bignat.sio` for unbounded ℚ — the piece the lane could NOT compose without generics).

## 5. Out of scope

Don't touch the exact-algebra consumer files beyond removing the skeleton's "does-not-compile" caveats once it compiles. No trait system beyond what criteria 1–3 need (no HKT, no generic-method-on-generic-struct unless the engine requires it). Smallest change that makes generic-struct **return + param** positions and trait-bounded method dispatch monomorphize correctly.
