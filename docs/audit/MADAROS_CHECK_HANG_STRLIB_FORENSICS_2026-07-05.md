# Madaros checker hang on `str::lib` — forensic narrowing — 2026-07-05

Companion to `docs/audit/MADAROS_CHECK_HANG_STRLIB_2026-07-05.md` (the blocker
declaration). This document narrows the hang to a 4-line reproducer, isolates
the compiler phase, identifies the regression window at the binary level, and
proposes the fix dispatch. Read-only forensic session; no repo files modified
other than this document. All probes ran under `timeout ≤ 60 s`, one at a time,
scratch files in `/tmp/strbisect/` only.

Blocker-ID: BLK-MADAROS-CHECK-HANG-STRLIB (a.k.a. BLK-20260705-madaros-check-hang-strlib)

## 1. Reproduction (measured)

Worktree `/workspace/sounio`, branch `gpu/epistemic-tensor-core-next`,
HEAD `908f4ab06`, 2026-07-05 ~17:05 UTC:

| Command | Exit | Wall time |
|---|---:|---:|
| `timeout 60 ./bin/souc check stdlib/str/lib.sio` | 124 | 60.0 s (killed) |
| `timeout 60 env SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check stdlib/str/lib.sio` | 1 | 0.22 s (`error: no main` — expected for a library on the lean lane; typecheck itself completes) |

The hang reproduces identically through the raw ELF
(`./bin/madaros-linux-x86_64 --check <file>`), so it is not wrapper routing.

## 2. Minimal reproducer (4 lines, verified)

The bisection of `stdlib/str/lib.sio` (halving by top-level decl, then
construct-stripping — 20+ probes, all logged in the session) converged on a
construct that has nothing to do with `Str`, fixed-size arrays, or `&!` params:

```sio
fn main() with Mut {
    var x: i64 = 0
    x = { 1 }
}
```

`timeout 25 ./bin/souc check /tmp/strbisect/p2.sio` → **exit 124 (hang)**.
`SOUNIO_SOUC_ENGINE=lean_single` on the same file → **exit 0** in 0.2 s.

**Trigger law (established by the probe matrix): an assignment *statement*
(`StmtAssign`) whose right-hand side contains a block `{ … }` hangs the default
checker.** Every variant below was measured:

| RHS of `x = …` (inside a fn body) | Result |
|---|---|
| `{ 1 }` (bare block) | HANG |
| `if true { 1 } else { 2 }` | HANG |
| `if true { 1 }` (no else) | HANG |
| `(if true { 1 } else { 2 })` (parenthesised) | HANG |
| `1 + if true { 1 } else { 2 }` (nested in binary) | HANG |
| `x += if true { 1 } else { 2 }` (compound op) | HANG |
| `match true { _ => { 1 } }` (arm with block body) | HANG |
| `match true { _ => 1 }` (no block anywhere) | **OK** |
| `1`, `g(1)` (literal / call) | **OK** |
| `let y = if true { 1 } else { 2 }` (let-init, same expr) | **OK** |
| `var y = if true { 1 } else { 2 }` (var-init) | **OK** |
| `if true { x = 1 }` (assignment *inside* a block) | **OK** |

So the two earlier shape hypotheses in the blocker doc (struct arrays,
`Str`/`StrSplit` signatures) are dead; the trigger in `stdlib/str/lib.sio`
is its `result.data[i as usize] = if ch >= … { … } else { … }` statements
(`str_to_upper` L281, `str_to_lower` L297, `str_replace_char` L358 — the file
bisection isolated exactly the L275–L305 window first). Any module importing
`str::lib` re-parses it and inherits the hang, matching the observed blast
radius (`stdlib/eisa/*`, `tests/stdlib/str/*`).

## 3. Phase evidence: the hang is in the PARSER (module load), not the checker

Three independent measurements:

1. **Multimodule banner ordering (decisive).** The multimodule `--check` lane
   prints `run_check_mode: about to check N modules` *after* all
   `load_module_file` calls and *before* `check_modules_verdict_boot4`
   (`self-hosted/compiler/main.sio:1936-1938`). A control file
   (`use math::dd64::*` + `x = 1`) prints that line and `check: OK` in 0.3 s.
   The same file with `x = if true { 1 } else { 2 }` hangs **without ever
   printing the `about to check` line** → the process never finishes
   *loading* (lexing+parsing) the entry module.
2. **Parse-error ordering.** `run_check_mode` panics on
   `parser_last_error_count() > 0` *before* invoking the typechecker
   (`main.sio:1886-1888`). A file with only garbage (`@@@`) exits 1 in 0.2 s;
   a file with garbage at line 1 *plus* the reproducer hangs → the parser
   reaches the construct and never returns.
3. **Process state.** During the hang: state `R`, 100 % of one CPU, `wchan 0`
   (not blocked), VmRSS **flat** at ~184,400 kB across repeated samples (no
   allocation growth), VmSize ~3.5 GB (pre-mapped arena). This is a
   non-allocating CPU spin, not runaway recursion and not memory exhaustion.
   `strace`/`ltrace` are not installed on the pod; `gdb` is.

GDB sampling (binary is stripped — no symbols) puts 6/6 interrupt samples in
one tight loop, back-edge `0x65f7048 → jmp 0x65f6a1f` (~1.6 KB of code). The
loop body: load a cursor slot (`rbp-0xba0`), dereference an Option tag
(`cmp $1` = Some), bulk-copy a large (>0x88-byte) struct field-by-field from
the payload (`(*node).head` of a linked list — Stmt-sized), then test the tag
against 0 (None → exit at `0x65f704d`) and loop. The cursor value never
changes between iterations: **a list-traversal loop whose
`cursor = (*node).tail` advance is lost**, so the same `Some` node is
re-visited forever. No `call`/alloc inside the loop — consistent with the
flat RSS.

## 4. Suspect code (evidence for the dispatch — NOT fixed here)

The construct-conditioned path is unambiguous in the Madaros parser sources:

- `self-hosted/parser/stmts.sio:335-377` — `parse_expr_or_assign_stmt`: on
  seeing `=` it parks the target in the **global** `LAST_ASSIGN_TARGET`
  (`stmts.sio:359` → `stmt_store_assign_target`, `stmts.sio:13-30`) and then
  parses the RHS via `parse_expr_entry`. When (and only when) the RHS contains
  a `{`, the nested `parse_block` runs **while the assignment is in flight**.
- `self-hosted/parser/stmts.sio:132-215` — `parse_block`: per-statement loop,
  including the tuple-let drain loop `while true { match TPLET_PENDING {
  Some(node) => { …copy (*node).head…; TPLET_PENDING = (*node).tail; … }
  None => break } }` (`stmts.sio:194-203`) — a `while true`+`match` over an
  `Option<Box<StmtList>>` global, exactly the compiled loop shape observed in
  gdb (Some-tag test, big head copy, tail advance, None exit).
- Both `TPLET_PENDING` and the drain loop were introduced by commit
  **`b8eb99ea9`** (2026-06-25 22:11, "fix(parser): desugar tuple-destructuring
  let `(a,b)=e`") — *inside the regression window established below*.

Precise suspect: the `Option<Box<…>>` **tail-advance store being lost** in one
of the two `while`+`match` list loops on the `StmtAssign`-with-block path —
`stmts.sio:194-203` (TPLET drain; global store `TPLET_PENDING = (*node).tail`)
being the best structural match for the observed non-allocating spin. Whether
the defect is (a) in these *sources* or (b) **wrong code generated by the
lean_single seed that built this Madaros binary** cannot be settled without a
rebuild — see §5. Given that the identical sources check fine under
lean_single's own (separate) parser, and given today's seed campaign
(`cacd3c358`, `9203671c0`, `17c0aeb6d`, `c647203a6`… — an all-day series of
**lvalue/place-store wrong-code fixes in the seed**), hypothesis (b) is the
stronger prior.

## 5. Binary provenance (the stale-binary trap, per docs/MADAROS_STATUS.md)

- Active raw ELF: `bin/madaros-linux-x86_64`, mtime 2026-07-05 10:32,
  sha256 `1c03193…a15781` — **matches** the gate receipt
  `artifacts/self-hosted/madaros.gate-receipt`
  (`created_utc=2026-07-05T10:31:55Z`, `smt_tests=6/6`, built in scratchpad
  worktree `wt-rel`; the receipt records no source SHA). The receipt-gated
  `artifacts/self-hosted/madaros` ELF itself is absent, so resolution falls
  through to this prebuilt.
- The prebuilt was committed as **`bc8d381a2`** (2026-07-05 10:33), commit
  message: "Rebuilt from `7236bf055`" (2026-07-05 10:27).
- **Regression window at the binary level (measured):** the *previous*
  prebuilt, extracted from git (`81cbecf62`, 2026-06-24 20:00), checks both
  reproducers (`x = { 1 }` and `x = if …`) **OK in < 1 s**. The current
  prebuilt hangs on both. So the regression entered between 2026-06-24 20:00
  and 2026-07-05 10:33 — a window that contains BOTH the tuple-let parser
  commit `b8eb99ea9` (06-25) AND every seed state prior to today's
  lvalue-store fix campaign (12:17–17:24, i.e. **after** the 10:31 build).
- **Consequence:** the binary was seeded by a lean_single that *predates*
  today's place-resolver/aggregate-store fixes. Rebuilding Madaros with the
  current seed (`make build-madaros`, serialized via
  `scripts/dev/souc-build-lock.sh`) is plausibly sufficient to change the
  result and MUST be the dispatch's first step before any source-level blame
  of `self-hosted/parser/stmts.sio`. Not run in this session (heavy build;
  build-lock discipline; out of read-only scope).

## 6. History answers

- `stdlib/str/lib.sio`: last touched `2e67d5a9d` — long before the window;
  the file is innocent.
- `self-hosted/parser/{stmts,exprs}.sio`: last touched `b8eb99ea9`
  (2026-06-25) — inside the window, introduces the exact loop shape observed.
- `self-hosted/check/`, `self-hosted/compiler/module_frontend.sio`: multiple
  July commits, but §3 rules the checker out as the hanging phase.

## 7. Blocker record (per `.claude/PARALLEL_BLOCKER_CONTRACT.md`)

```text
Blocker-ID: BLK-20260705-madaros-check-hang-strlib
Status: classified
Severity: B1 (default-lane check blocked for any module whose source, or transitive import, contains `<place> = <expr containing a block>`; lean_single lane unaffected)
Class: bootstrap-runtime (primary hypothesis: seed-miscompiled Madaros parser binary; secondary: compiler-semantics in self-hosted/parser/stmts.sio b8eb99ea9)
Owner: unassigned (fix dispatch needed; seed campaign owner is the natural owner given today's lvalue-store fix series)
Lane: default-compiler-lane (bin/souc → Madaros)
Worktree: /workspace/sounio
Branch: gpu/epistemic-tensor-core-next @ 908f4ab06
Files-Owned: docs/audit/MADAROS_CHECK_HANG_STRLIB_FORENSICS_2026-07-05.md (this doc only)
Files-Read-Only: stdlib/str/lib.sio, self-hosted/parser/*, self-hosted/check/*, self-hosted/compiler/main.sio, bin/madaros-linux-x86_64
Do-Not-Touch: bin/souc, scripts/lib/resolve_souc.sh, scripts/ci/build_modular_madaros.sh (serialized surfaces)
Repro: printf 'fn main() with Mut {\n    var x: i64 = 0\n    x = { 1 }\n}\n' > /tmp/r.sio && timeout 30 ./bin/souc check /tmp/r.sio
Observed: exit 124; banner only; 100% CPU spin in a stripped-binary loop 0x65f6a1f–0x65f704d (Option-tagged list traversal whose tail-advance never lands); RSS flat
Expected: exit 0 (`check: OK`), as lean_single and the 2026-06-24 prebuilt (81cbecf62) both produce in <1 s
Acceptance-Gate: timeout 60 ./bin/souc check stdlib/str/lib.sio → exit 0, plus the 4-line reproducer above → exit 0, plus make madaros-full-gate green
Evidence-Level: E2 (classified: construct isolated, phase isolated, binary regression window bounded; unrelated shape hypotheses separated and falsified)
Evidence: this document; probe corpus /tmp/strbisect/ (session-local); gate receipt artifacts/self-hosted/madaros.gate-receipt; git prebuilt 81cbecf62 vs bc8d381a2 differential
Fallback-Path: SOUNIO_SOUC_ENGINE=lean_single (named, in use by the EISA track as validated_lane: lean_single)
Legacy-Kept: yes (lean_single engine untouched and green)
LLM-Offload: not-required (no math claim, no clinical code, no external artifact; forensic audit doc only)
Next-Action: rebuild Madaros with the current post-campaign seed under the build lock (scripts/dev/souc-build-lock.sh + make build-madaros) and re-run the 4-line reproducer; if it still hangs, bisect b8eb99ea9's TPLET drain loop (self-hosted/parser/stmts.sio:194-203) and the LAST_ASSIGN_TARGET in-flight interplay (stmts.sio:13-30,359) as a source defect; if it passes, refresh the prebuilt + receipt and close as seed-wrong-code fixed upstream
```

## 7a. Addendum (2026-07-05 18:25 UTC): rebuild executed — seed hypothesis REFUTED

The Next-Action rebuild was run (`scripts/ci/build_modular_madaros.sh`, which
takes the build lock internally; an initial attempt deadlocked by wrapping it
in an outer `souc-build-lock.sh` — do not double-wrap). Result:

- New binary: `artifacts/self-hosted/madaros`, md5 `d80375330e61d358591e606edb45c9d8`,
  built 18:25 UTC from the **post-campaign** seed (`bin/souc-lean-single-x86_64`
  17:24 UTC, md5 `40ffdbecdb7cde6baddeba95cddbb691`, includes today's full
  lvalue/place-store fix series through `2dc581185`).
- 4-line reproducer: still **exit 124** (banner, then spin).
- `--check stdlib/str/lib.sio`: still **exit 124**.
- Control: `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check` on the reproducer
  → exit 0 in <1 s.
- Shape probe: the `TPLET_PENDING`-style drain loop (global
  `Option<Box<List>>`, `while true { match ... Some => advance, None => break }`)
  compiled by the current seed terminates correctly with the right sum
  (`/tmp/probe_tplet_shape.sio`, PROBE OK). The loop *shape* is not what the
  seed miscompiles — if the defect is in codegen at all.

Consequence: the primary hypothesis (pre-campaign seed wrong-code, fixed
upstream) is **refuted** — a post-campaign-seed rebuild reproduces the hang.
The secondary hypothesis is now primary: a source defect introduced by
`b8eb99ea9` (2026-06-25), which is fully consistent with the 2026-06-24
prebuilt (`81cbecf62`) passing. Given the drain-loop shape probe passes, the
sharpest remaining suspects are the `LAST_ASSIGN_TARGET` in-flight interplay
with the *nested* `parse_block` reached through
`parse_expr_or_assign_stmt` → `parse_expr_entry` → `parse_block_expr`
(stmts.sio:359 stores the target, then the RHS parse re-enters block parsing
before `stmt_take_assign_target` at :363), and non-advancing statement
consumption inside the nested block loop. Next dispatch: source bisect of
`b8eb99ea9` under lean_single-built Madaros probes; owner remains the parser
campaign owner. Blocker stays open, class updated to
**compiler-semantics (parser source)**, evidence level E2.

## 8. Session hygiene

Every hang probe ran under `timeout` (25–60 s); after each timed-out check the
`timeout` parent reaped the child. Final `pgrep` sweep confirms no process
started by this session remains. Two long-running `madaros --check` processes
owned by a different agent (`/tmp/neurodyn_fn_cuts/*`, under `timeout 120`)
were observed and deliberately left untouched.
