# M1.2 — N-v2 driver segfault clusters (2026-05-09 inventory)

Run-pass parity inventory on 2026-05-09 main HEAD reported 230
`nv2_compile` failures. Drilling into the per-test logs:

  - **157 of 230 (68%) are silent segfaults** (rc=139, 0-byte log file).
    The driver crashes before emitting any diagnostic. These look like
    "punch-list misses" in the parity report but are actually a parser
    robustness gap — N-v2 null-derefs on unhandled constructs instead
    of raising `unsupported_frontend reason=...`.
  - 73 of 230 emit clean `kind=NNN text=XXX` diagnostics. Top markers:
    `gpu_thread_id_x` (10), `variance_of` (9), `hessian_of` (9),
    `sensitivity_of` (6), `imported_add` (2), `Some` (2), `str_concat`
    (2), `getpid` (2), `syscall6` (2), `sed` (2).

## Bisect of 4 segfaulting test files

Each of the 4 smallest segfaulting tests reduces to a narrow 1-feature
trigger. All four crash at rc=139 with no output; the remaining 153
segfaults likely cluster under similar narrow shapes.

### Cluster S-A — type-inferred array-literal RHS

  ```
  // CRASH (rc=139)
  fn main() -> i32 {
      let arr = [0.0; 16]
      return 0
  }
  ```

  ```
  // OK
  fn main() -> i32 {
      let arr: [f64; 16] = [0.0; 16]
      return 0
  }
  ```

  Trigger: `let NAME = [V; N]` with NO explicit type annotation.
  Crashes regardless of element type (`[0; N]` int also crashes),
  size (4, 64, 1024, 4096, 65536 all crash), or function effects.
  PUNCH_LIST.md note (2026-04-30) said array-init was "PARTIALLY
  CLOSED" — the partial closure landed type-annotated form; the
  type-inferred form is the remaining gap and now segfaults instead
  of unsupported_frontend.

### Cluster S-B — turbofish call syntax `::<T>`

  ```
  // CRASH (rc=139)
  fn identity<T>(x: T) -> T { return x }
  fn main() -> i32 {
      let a = identity::<i32>(42)
      ...
  }
  ```

  Defining `identity<T>` and calling it without turbofish (`identity(42)`)
  works. The crash is on the `::<...>` token-sequence in CALL position.
  Likely `parse_fn_call_ir` (line 2941) doesn't recognize the `::<` shape
  and falls through to a null deref.

### Cluster S-C — `Box::new(...)` constructor

  ```
  // CRASH (rc=139)
  fn main() -> i32 with IO, Mut, Panic {
      let boxed = Box::new(42)
      0
  }
  ```

  Same `::` token shape as turbofish — but here `Box::new` is a
  type-method-call expression. Different parse path from S-B but same
  surface symptom.

### Cluster S-D — refinement type in param position

  ```
  // CRASH (rc=139)
  fn divide(num: i32, denom: { d: i32 | d != 0 }) -> i32 with Panic {
      num / denom
  }
  ```

  Refinement type `{ d: T | predicate }` in parameter position. Works
  fine when the parameter is a plain type. Type parser probably has
  no handler for `{` opening a refinement expression in type context
  and null-derefs.

## Recommended attack path (next session)

Two complementary directions:

1. **Defensive backstop**: in `self-hosted/compiler/native_compile_driver.sio`,
   audit every `*ptr` / array-index that follows a parser dispatch where
   the dispatched function might have failed. Replace silent null-deref
   crashes with explicit unsupported_frontend emit + early return.
   Convert all 157 segfaults into 157 clean diagnostics. Higher leverage
   than per-feature fixes because each deref is a one-line guard.

2. **Pick highest-leverage cluster + fix**: S-A (array-literal type-inference)
   is the cleanest shape — the explicit-type form already works, just
   need to teach the let-without-type path to peek the RHS, infer the
   array type, and dispatch to the existing array-init handler.

Either way, the fixed-point gate (`scripts/ci/lean_single_fixed_point_gate.sh`)
and the wall-clock timeout from `b845c522` must stay green at every
intermediate commit.

## Reproducers

Saved under `/tmp/bisect/` (regenerate from this file if needed):
  - `v3.sio`   — turbofish (S-B)
  - `b1.sio`   — `let arr = [0.0; 16]` (S-A)
  - `b2.sio`   — `Box::new(42)` (S-C)
  - `r1.sio`   — refinement param (S-D)

Repro:
  ```
  STAGE1=/tmp/sounio-parity-inventory.<HASH>/nv2_driver.stage1
  bash -c "$STAGE1 /tmp/bisect/b1.sio /tmp/_out.elf >/dev/null 2>&1; echo rc=$?"
  # Expected: rc=139
  ```

To rebuild the stage1 binary:
  ```
  bash scripts/ci/track_a_nv2_parity_inventory.sh examples/native
  ```
(it builds the stage1 driver from `self-hosted/compiler/native_compile_driver.sio`
in $OUT_DIR/nv2_driver.stage1 as a side effect).

## Inventory delta this session

| Date | corpus | ok | nv2_compile fail | of which silent segv |
|--|--:|--:|--:|--:|
| 2026-04-30 baseline | 392 | 50 (12.7%) | 324 | — |
| 2026-05-03 delta    | 392 | 76 (19.4%) | 291 | — |
| 2026-05-09          | 407 | 129 (31.7%) | 230 | 157 |
| 2026-05-10 (S-A)    | 407 | 129 (31.7%) | 230 | 156 |
| **2026-05-10b**     | **407** | **145 (35.6%)** | **194** | **0** |

## 2026-05-10b update — S-B/S-C/S-D ALL CLOSED

After the V2_UFN buffers were doubled (`213c5c27`, parallel agent),
S-B/S-C/S-D landed cleanly:

- **S-B/S-C** (`d025215d`): `parse_atom_ir` extends the bare-fn-ref
  guard to exclude TK_COLONCOLON and adds a small refusal block
  setting reason=2 (`pathsep_in_expr`).
- **S-D** (`34075a4e`): param-type scanner detects `IDENT : {` (where
  `find_fn_lbrace` returned the refinement's brace as the
  incorrectly-detected fn body) and refuses with reason=3
  (`refinement_type`).

Result: **0 silent segfaults remain** in tests/run-pass. All 156 silent
crashes converted to clean `unsupported_frontend` diagnostics. As a
side-effect of dropping invalid-but-now-cleanly-refused functions, 16
tests graduated to ok (129→145, 31.7%→35.6%) and 20 newly compile-but-diverge
(44→64 nv2_run). The remaining 194 nv2_compile failures are the actual
M1.2 punch-list — all clean diagnostics with identifiable kind/text/reason.

## 2026-05-10 update — S-A landed (superseded)

(see 2026-05-10b row above; previous status retained for history)

S-A defensive backstop landed (commit `dfe0894a`): `parse_stmt_ir` refuses
type-inferred `let X = [...]` cleanly via `refuse_let_array_lit/2` helper.
Inline guard tipped `parse_stmt_ir` over the V2_UFN_OPS[2048] budget; the
helper-extraction approach kept self-compile fixed-point intact.

Regressions vs 2026-05-03 (only 2):
  - `tests/run-pass/import_basic.sio` — clean `unresolved_call text=imported_add`
    (something in import resolution broke between commits)
  - `tests/run-pass/mc_struct_field_write.sio` — segfault (likely cluster S-?)
