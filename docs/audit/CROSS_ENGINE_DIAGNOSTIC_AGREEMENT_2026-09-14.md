<!-- docs:meta
topic_id: repo.docs.audit.cross-engine-diagnostic-agreement-2026-09-14
authority: repo_only
audience: users
last_validated: 2026-09-14
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.cross-engine-diagnostic-agreement-2026-09-14
-->

# Cross-engine diagnostic agreement, classified by hand (2026-09-14)

## What this measures

`scripts/research/cross_engine_diagnostic_agreement.sh` asks both Sounio compilers whether they
ACCEPT or REJECT every versioned `.sio` file outside `archive/` and `bootstrap/`:

- **AGREE_ACCEPT** or **AGREE_REJECT:** both engines give the same answer.
- **LEAN_ONLY:** lean_single rejects, Madaros accepts.
- **MADAROS_ONLY:** Madaros rejects, lean_single accepts.

This report takes every disagreement from one full sweep and assigns it a cause. The cause comes
from reading the file, and where reading was not enough, from a minimal program run on both engines.

It is **not a defect rate**. Neither engine is ground truth, a large share of the disagreements are
intentional or artifacts of the instrument, and a count is only meaningful together with its class.

## Headline

- 8,232 files: 3,113 AGREE_ACCEPT, 1,921 AGREE_REJECT, 2,440 LEAN_ONLY, 726 MADAROS_ONLY, 32 TIMEOUT.
- All 3,166 disagreements were classified; none is left unexamined.
- 919 (29%) are an instrument artifact: lean_single's `check` is a full compile, so it rejects every
  file without `main`.
- 785 (25%) are intentional. 705 of these are tests that declare `//@ requires: madaros`.
- 604 are lean_single defects, 439 are Madaros defects, and 60 contain defects of both engines.
- 250 are rules Madaros enforces and lean_single never implemented. Most of them expose a real bug in
  the file (private functions imported across modules, missing effect declarations).
- 104 remain unresolved: the evidence does not decide which engine is right.

## Provenance

| Item | Value |
|---|---|
| Sweep start / end (UTC) | 2026-09-14T02:16:24Z / 2026-09-14T07:18:11Z, script exit 0 |
| Branch, HEAD at start / end | `feat/w1-qd128-transcend`, `4c0e2fcbdd` / `c6e7cbbce6` |
| Madaros | built by `scripts/ci/build_modular_madaros.sh` from a `git archive` snapshot of `59e82e3d50`; sha256 prefix `a26d5284510dcff8` |
| lean_single (sweep) | `SOUNIO_SOUC_ENGINE=lean_single bin/souc check`, which runs the worktree's `bin/souc-lean-single-x86_64` (an uncommitted rebuild; sha256 prefix `3c5e4315e47529b6` at start and end) |
| lean_single (experiments) | `bin/souc-linux-x86_64 self-hosted/compiler/lean_single.sio`, built at `59e82e3d50` |
| Environment | `SOUNIO_STDLIB_PATH` pinned to this worktree's `stdlib/`; `SOUC_BIN`, `SOUNIO_SOUC_BIN`, `MADAROS_RAW_BIN`, `SOUNIO_MADAROS_BIN` unset |
| Positive controls | `tests/run-pass/_diag_sobol.sio` → AGREE_ACCEPT, `tests/compile-fail/f256_v0b_arithmetic_rejected.sio` → MADAROS_ONLY, `stdlib/verify/interval.sio` → AGREE_REJECT; all OK before the sweep |
| Files swept | 8,232 (`git ls-files '*.sio'` minus `archive/`, `bootstrap/`); per-file timeout 25 s |
| Source newer than the Madaros build | `self-hosted/ir/lower.sio` (not called by `--check`), and `self-hosted/check/epistemic.sio` and `self-hosted/check/units.sio` from commits landed during the sweep (`a3b2964594`, `c6e7cbbce6`). `stdlib/` did not change during the run. Only 3 disagreement rows fall on files changed after the build, all LEAN_ONLY. |

Data for this run is under `artifacts/research/cross_engine_diagnostic_agreement/20260914T021624Z/`:
- the raw verdicts (`agreement.tsv`)
- the per-file classification (`classified.tsv`)
- the full group table (`tables.md`)
- the classification scripts and the manual assignments (`tools/`)
- the minimal programs used as evidence (`experiments/`)

## Results

### Verdicts

| Verdict | Files |
|---|---:|
| AGREE_ACCEPT | 3,113 |
| LEAN_ONLY | 2,440 |
| AGREE_REJECT | 1,921 |
| MADAROS_ONLY | 726 |
| TIMEOUT | 32 |

A timeout is not counted as a disagreement:
- 21 files timed out on both engines, almost all of them compiler drivers under `self-hosted/compiler/`.
- 5 timed out on lean_single and were rejected by Madaros.
- 5 timed out on lean_single and were accepted by Madaros.
- 1 timed out on Madaros and was accepted by lean_single (`artifacts/ontology-frontiers/compiler-repros/p4_thinlink_control_main.sio`).

### Disagreements by class

| Class | Meaning | LEAN_ONLY | MADAROS_ONLY | Total |
|---|---|---:|---:|---:|
| HARNESS | artifact of the instrument | 919 | 0 | 919 |
| INTENTIONAL | the divergence is declared or documented | 749 | 36 | 785 |
| DEFECT-LEAN | lean_single rejects a program the language documents or a tracked test requires, or accepts one a test requires it to reject | 566 | 38 | 604 |
| DEFECT-MADAROS | the same, for Madaros | 109 | 330 | 439 |
| MADAROS-GUARANTEE | Madaros enforces a rule lean_single does not have | 0 | 250 | 250 |
| UNRESOLVED | the evidence does not decide | 37 | 67 | 104 |
| BOTH | the file carries a defect of each engine | 60 | 0 | 60 |
| SOURCE-BUG | the file is wrong and the stricter engine is right | 0 | 5 | 5 |

## Method

1. **Group by first diagnostic.** For each disagreement, the rejecting engine's first diagnostic
   was normalised: line numbers, byte spans, qualified names and backticked identifiers removed,
   error codes kept. Files were grouped by that signature.
2. **Read each group.** That meant representative logs, the source at the reported line or byte
   span, the file's own header, and the docs or tracked tests that define the construct.
3. **Test a minimal program when reading was not enough.** It was checked on both engines. 111 such
   programs are kept under `experiments/`; the ones that decide a finding are quoted below.
4. **Check each group against its mechanism.** Where a group mixed causes, it was split:
   - Every E221 file was checked for having no `fn main`.
   - Every lean E035 site was classified as a local-variable write, a global write, a field store
     through a parameter, or an IO or effectful call. The file was then assigned by site kind.
   - A signature group that did not share one cause was split into mechanism rules.
5. **Tracked test files were also judged by their own contract.** This covers:
   - `//@ run-pass`, `//@ current-source-run-pass` and `//@ check-only` tests rejected by one engine
   - `//@ compile-fail` and `//@ typecheck-fail` tests accepted by one engine
   - `//@ requires: madaros` and `//@ known-failure` declarations

   For compile-fail rows the declared `//@ error-pattern` was checked against the rejecting
   engine's output. It was found in 70 of 80 rows, absent in 6, and not declared in 4; each row
   in the group table says which.

How each of the 3,166 disagreements was decided:

| Basis | Files |
|---|---:|
| A mechanism rule established by reading and experiments | 2,192 |
| The test declares `requires: madaros` | 724 |
| The test's run-pass / compile-fail contract, cause not isolated further | 187 |
| Single file read by hand | 46 |
| The test's contract, with the cause also identified | 17 |

## Findings

Counts are files. The complete list of 216 groups, with an example file for each, is in `tables.md`.

### Instrument artifact (919)

lean_single has no check-only mode, and the script's lean leg is a full compile. Every library, data
or fixture file without `fn main` is rejected with `error[E221]: no main`. All 919 files were
confirmed to contain no `fn main`.

### lean_single defects (604, plus the lean half of 60 BOTH)

| Mechanism | Files | Evidence |
|---|---:|---|
| `Mut` demanded for mutation of a function-local `var` | 337 (+58 mixed) | `docs/guide/LLM_PROGRAMMING_GUIDE.md` lines 239-252 and spec §7.2.1: `Mut` is for caller-observable mutation, nothing for a local `var`. Minimal `fn bump(n: i64) -> i64 { var y = 1 y = y + n y }`: lean E035, Madaros accepts. |
| Code inside `/* */` is analysed | 70 | Every "Aspirational example preserved below" stub keeps its original program in a block comment. A commented `#[cfg]`, `email.contains("@")` or `f64@h` struct literal is rejected by lean and ignored by Madaros. |
| Compile-fail tests lean accepts | 28 | The test contract. For 22 of them the declared error-pattern is present in Madaros's output; for 4 it is absent, and 2 declare none. |
| A comma inside generic type arguments is counted as a parameter separator | 24 | `fn g(pa: &Hyper<Octonion, f64>, pH: &[Hyper<Octonion, f64>; 16], pout: &![f64; 256]) with GPU {}` called with 3 arguments: lean "expected 5 got 3". All of `tests/gpu/*_tile.sio`. |
| `contest [M] on x` not supported | 18 | lean E200 on the model name. |
| A reference to a call result, `&f()` | 16 | `get(&mk())`: lean "unknown identifier `mk`". |
| Madaros builtins absent (`second_order_mean`, `correlate`) | 15 | Both are builtins in `self-hosted/check/check.sio`; neither name appears in `lean_single.sio`. |
| A struct name resolved by bare name across modules | 9 | Two imported modules each defining `struct Name`: lean picks the wrong one ("unknown field access"). Madaros has a sibling defect in the other import order (E046). |
| A reference to an indexed element as an argument, `&a[i]` | 8 | `total(&a[1])` with `a: [S; 2]`: lean E001. |
| A `sqrt` method on an imported type shadows the free `sqrt(f64)` | 7 | A library with `impl P { pub fn sqrt(self: &P) }` plus a caller of `sqrt(4.0)`: lean E001. |
| Negative probes accepted | 7 | `tests/frontend/parser_stability/invalid/`, `tests/selfhost/native_typecheck/`, `*_must_reject.sio` |
| Smaller mechanisms | 65 | See the bullet list below. |

The 65 smaller-mechanism files:
- type alias in a tuple return (5)
- `import X::*` (5)
- data-carrying enum variant `V::M { .. }` (5)
- qualified call `a::b::f()` (5)
- `loop { }` (4)
- `pub use` re-exports not followed (4)
- `[0; N]` as a function tail (4)
- tuple index > 1 (3)
- array `var` without initializer (3)
- i128/i256 (2)
- `[ints]` into `[i8; N]` (3)
- `[0; N]` as an argument (2)
- empty `[]` (2)
- u8/u16/u32 (2)
- unresolved `pub use` accepted (2)
- run-pass tests rejected, cause not isolated (8)
- single files (empty `match`, `handle<E> { }`, nested array type on a local, implicit borrow into `&!`, generic struct, `let` bound to a nested call)

The same lean mechanisms also appear in 705 tests that declare `requires: madaros`, counted as
INTENTIONAL: local `var` `Mut` (338), `pub use` re-exports not followed (318), and others.

### Madaros defects (439, plus the Madaros half of 60 BOTH)

| Mechanism | Files | Evidence |
|---|---:|---|
| Run-pass tests rejected, cause not isolated beyond the diagnostic | 120 | The test contract. By first diagnostic: E001 18, E009 16, E137 15, E008 12, E004 10, unresolved module closure 9, E037 9, E012 7 (async `.await`), E036 4, E015 3, E011 3, other 14. |
| Caller-observable mutation or IO without the effect | 64 (+58 mixed) | Same guide passage: "Madaros currently requires `Mut` for neither case". A global array write or `println` in a function without `Mut`/`IO`: lean E035, Madaros accepts. A write through `&!` is rejected by both. |
| Method calls on non-struct receivers (E019) | 56 | `x.sqrt()` on `f64`. Two tracked run-pass tests, `tests/run-pass/associator_variance_mc.sio` and `octonion_basic_demo.sio`, fail. |
| Builtins used by tracked code missing | 41 | `append_file` 23, `ln` 9, `print_i64` 5, `read_line` 3, `atan2` 1. `ln(` appears in 8 run-pass tests and `print_i64(` in 3; `read_line(` is in the guide. lean_single implements all five. |
| Compile-fail tests Madaros accepts | 32 | The test contract, with lean's declared diagnostic present in 30. They cover refinement predicate violations (E208/E209), confidence gates (E214), linear and affine use, match exhaustiveness, and others. |
| Parse failure with no diagnostic printed | 27 | `run_check_mode: module failed to parse` with no `parse error` line. |
| Knowledge-on-Knowledge arithmetic declared unsupported (E245) | 20 | The diagnostic says there is no lowering. 4 of these are documented known failures. |
| `[0u8; N]` typed literal in an array repeat | 11 | Minimal struct literal with `data: [0u8; 4]`: Madaros parse error, lean accepts. |
| `read_file(..)` result untyped | 11 | `read_file` is bound as an unknown import (`check.sio` near line 4483), so `.as_bytes()` fails (E011). |
| Compile-time division-by-zero false positive | 10 | A copy of `stdlib/data/csv.sio` `csv_field_fixed` (`var pow = 1` grown in a loop, then `x / place` inside `while place >= 1`) gets E056. The check ignores both the reassignment and the guard. |
| Float-literal array into `[f32; N]` | 8 | `let a: [f32; 2] = [1.0, 0.0]` is rejected; the scalar `let a: f32 = 1.0` is accepted. |
| A named import of a function the module does not define is accepted | 8 | `use helpers::{no_such_fn}` followed by a call: lean rejects, Madaros accepts. `examples/particle_physics/exp7_gum_xi_tension_transfer.sio` and `exp8` import `m_w_tension_tree` from `particle_physics::ew_precision`, which has no such function. |
| An imported user-declared effect is not resolved (E246) | 7 | `pub effect HostTrust {}` in a library, used through `use lib::{HostTrust, ..}`: Madaros "unknown effect". The declaration exists at `stdlib/coordination/fleet_transaction.sio:13`. |
| A value returned from a function with no return type is accepted | 5 | `fn f() with IO { return 0 }`, `return x` and `return 1.5` are all accepted, while `fn f() -> i64 { return true }` gets E008. |
| Smaller mechanisms | 20 | See the bullet list below. |

The 20 smaller-mechanism files:
- if-branch joins with a literal or in statement position (4)
- a refinement returned as its base type (3), including tracked run-pass `tests/frontend/refinement_interval_entailment.sio`
- a char literal as an array-repeat element (3)
- `500_mg` underscore suffix, documented in `docs/guide/programming.md` (2)
- `str` alias missing (2)
- `f64<m/s>` compound unit, tracked run-pass `tests/frontend/unit_f64_unit_expr_velocity_current_source.sio` (2)
- `-> T where result...` clause (2)
- `[str; N]` (1)
- field store through `*mut` by auto-deref (1)

### Rules Madaros enforces and lean_single does not (250)

| Mechanism | Files |
|---|---:|
| A private function used across modules (E175) | 100 |
| A caller missing an effect its callee declares (E035) | 98 |
| Compile-fail tests declaring `requires: madaros` | 19 |
| An effectful function passed where an effect-free function type is expected (`docs/compiler/CLOSURE_TYPE_THEORY.md`: `fn(T) -> U with E`) | 8 |
| An error in an imported module that lean_single does not report | 7 |
| Negative controls, block scope, correlated-uncertainty independence (E230), f64/f32 mixing, `KCoreKnowledge` misuse, i64 compared to an enum, a CI reject fixture | 18 |

The first two groups expose real bugs in the stdlib, examples and repros. For example:
- the stdlib `pcg64_new`, `smp_pcg64_new` and `dst_pcg64_new` are private
- `particle_physics::rge::active_flavors` calls `mass_charm` without its effects
- `self-hosted/compiler/knowledge_runtime_guard_lowering_plan.sio:347` reads a field
  `KnowledgeRuntimeObligationDescriptor` does not have

### Intentional (785)

- 705 tests declare `//@ requires: madaros`.
- 52 files are Knowledge-annotation probes. `docs/audit/KNOWLEDGE_ANNOTATION_SURFACE_MADAROS_ONLY_2026-08-19.md` records that lean_single has no provenance parsing.
- 28 files are documented reservations and fixtures: f128/f256 (E249) and `extern "C"` (E250).

### Source bugs (5)

- `loop` used as a function name, but it is a reserved keyword (4).
- A Rust format string passed to `print` (1).

### Unresolved (104)

- Module search roots differ: lean_single also resolves imports under `self-hosted/` or a parent
  directory (46).
- Field privacy: lean enforces it, Madaros has no check, and the docs are silent (13).
- An imported stdlib module does not parse in Madaros, and the construct was not isolated (10).
- `print_char` argument type (5).
- Borrows of disjoint fields (2), `extern "C" system(string)` (2), and `Knowledge<T where {..}>`
  checked statically vs by runtime guard (2).
- 24 single files, each recorded in `tools/manual.tsv` with the reason it stays open.

## Limitations

- **One run.** One sweep, one run per file, no repeat measurement.
- **First diagnostic only.** A file rejected for two independent reasons is grouped by the first,
  except E035, where every site was examined.
- **Contract rows are not fully explained.** For 187 rows the verdict comes from the test's
  contract; the rejecting diagnostic is recorded but its cause was not isolated further.
- **The lean side is an uncommitted binary.** It is the worktree's rebuilt seed, not a committed
  binary; its hash is recorded above.
- **The Madaros binary is older than some source.** It predates the checker commits that landed
  during the sweep, listed under provenance.
- **A first run was discarded.** It started 2026-09-13T23:43:46Z and was stopped after 3,175 files.
  - It ran in a tmux window that inherited `SOUNIO_STDLIB_PATH=/workspace/sounio/stdlib` and
    `SOUC_BIN=/workspace/sounio/bin/souc` from the tmux server's global environment.
  - `/workspace/sounio` is another branch whose `stdlib/` differs from this worktree's in 39
    entries, so every stdlib-importing file was checked against the wrong library.
  - `bin/souc` does not read `SOUC_BIN` on the lean_single path, so only the stdlib was wrong.
  - The hazard was already known: the provenance note in `bin/souc` (around line 390) records agents
    who "carefully unset the poisoned SOUC_BIN and SOUNIO_STDLIB_PATH first".
  - Any caller of this script should pin `SOUNIO_STDLIB_PATH`.
- **Two open changes are not reflected.** #2497 (Madaros lowers compound assignment and adds E050
  for `f64 %=`) and #2498 (lean_single refuses compound assignment with E260) postdate these
  engines. Once #2498 lands, every compound assignment becomes a new LEAN_ONLY disagreement that
  this report does not count.

## Reproduce

```bash
bash scripts/ci/build_modular_madaros.sh /tmp/madaros-src
env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u MADAROS_RAW_BIN -u SOUNIO_MADAROS_BIN \
  SOUNIO_STDLIB_PATH="$PWD/stdlib" \
  CROSS_ENGINE_MADAROS_BIN=/tmp/madaros-src CROSS_ENGINE_OUT_DIR=/tmp/xeng-out \
  bash scripts/research/cross_engine_diagnostic_agreement.sh
```

To re-derive the classification from that output, run the scripts in `tools/` from the repository
root, in this order:
1. `sig.sh`
2. `e035_sites.sh`
3. `classify.sh`, using `tools/manual.tsv`
4. `classify_post.sh`
5. `report_tables.sh`
