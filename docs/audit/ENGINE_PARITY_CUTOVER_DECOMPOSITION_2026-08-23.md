<!-- docs:meta
topic_id: repo.docs.audit.engine-parity-cutover-decomposition-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.engine-parity-cutover-decomposition-2026-08-23
-->

# The lean_single-to-Madaros cutover gap, decomposed

**Date:** 2026-08-23
**Measured by:** the `qwen` lane, rounds 3–5, read-only on every worktree.
**Why this document exists:** the lane's tmux window did not survive the
`fleet` session restart at 13:52:05. Its findings existed only as two handoff
files in `/tmp`, which is shared and reaped. Five rounds of measurement were one
cleanup away from being lost. The raw material is preserved verbatim under
`artifacts/audit/engine_parity_20260823/`; this file is the readable account.

Its central claims were re-verified here before being written down. Where a
number of mine disagrees with the lane's, both are given.

## 1. The parity numbers, and why the headline moved for a reason that is not a regression

    agree = 731   diverge = 88   madaros-only = 108   lean-only = 242   neither = 602

`JOBS=4` and `JOBS=8` produce identical verdicts, so the count is not a
parallelism artefact.

**The 291 new NEITHERs are the E035 `Mod` migration, not new breakage.** The
baseline was cut 2026-08-10, eleven days before `Mod` became effect id 29
(#2059, 08-21), so every program that needs `Mod` changed category on its own.
Eleven of twelve sampled fail with "effect not declared (missing: Mod)".

**Consequence, and it is the actionable one:** do NOT wire the parity workflow
until the refinereturn migration lands and the baseline is refreshed with
`--update-baseline`. Wired today it fails every PR on migration churn rather
than on regressions, which is the fastest way to teach everyone to ignore it.
The draft sits in `artifacts/audit/engine_parity_20260823/` deliberately, not in
`.github/workflows/`.

## 2. LEAN-ONLY = 39: what the cutover actually owes

Measured gate-faithfully — compiled under `ulimit -v 8000000`, the same bound
the gate's worker uses, then run.

    compile_fail = 28      run_crash = 11      run_ok = 0

The 28 that do not compile:

| count | cause |
|---:|---|
| 9 | `lorenz_i256_*_imported` — imported-module array lowering dies at `lower_array: seed_begin` |
| 17 | clean refusals: the #1876 print-dispatch family, the lorenz trajectory2 family, two viz receipts |
| 2 | parse failure on the `on` keyword — see §3 |

The 11 that compile and then crash include **5 closure codegen SIGSEGVs** —
`closure_fn_ref`, `closure_higher_order`, `closure_sort_by`,
`closure_lambda_lift`, `closure_effect_checked`. All compile `rc=0` and die at
run time; lean_single runs all five `rc=0`. That is the single largest
run-crash family and it points at Sprint-228 function-reference lowering.

## 3. `on` is a reserved word in one engine and an identifier in the other

Re-verified here, not taken on report. `self-hosted/parser/parser.sio:649`
lexes the byte pair `('o','n')` into `TokenKind::On`, for the Contest syntax
`[m1, m2] on subject`. `self-hosted/compiler/lean_single.sio` contains no such
class at all. So

    fn f(on: i32) -> i32 { on }

parse-fails under Madaros and compiles clean under lean_single. Confirmed on
both the committed August ELF and a fresh from-source build.

Four stdlib files use `on` as an identifier — `graphics/surface.sio`,
`graphics/text.sio`, `viz/ir.sio`, `theorem/cdcl.sio`. (The lane counted about
16 sites; a line-based count here gives 3/4/2/23, which counts lines rather than
occurrences and does not exclude comments. The mechanism is what matters; the
site count should be taken from a real edit, not from either grep.)

**One parse failure blocks a chain, which is why this is worth more than two
programs.** `surface.sio` is imported by `canvas_ext`, which is imported by
`renderer3d`, which is imported by `viz/ir` — so every `viz_*` test is blocked
behind a single file that will not parse.

Renaming `on` to `is_on` in `surface.sio` alone makes it parse and compile.

**But the rename is not obviously the right fix, and this is a founder call, not
a lane call.** Making `on` contextual — reserved only in Contest position, after
a `]` — keeps the identifier legal and puts the two engines back in agreement.
Renaming four stdlib files makes the symptom go away while leaving the engines
disagreeing about what a valid identifier is, which is a divergence this
repository has repeatedly paid for elsewhere.

## 4. The order the lane suggested

1. Settle `on`: contextual keyword, or the stdlib rename. This unblocks the
   graphics/viz/render chain.
2. Wait for the refinereturn `Mod` migration, refresh the baseline, then wire
   the parity workflow.
3. The 5 closure codegen SIGSEGVs — biggest run-crash family.
4. The 9 lorenz imported-module lowering SIGSEGVs.
5. The #1876 print-dispatch refusals, whose three missing branches in
   `lower.sio` were already located by an earlier bisect.

## Preserved material

`artifacts/audit/engine_parity_20260823/` holds the lane's own words and data:
the round 4 and round 5 handoffs, the draft workflow, the LEAN-ONLY path lists
(baseline, current, and the 39 new), and the run script. Nothing there was
edited; where this document and those files disagree, they are the primary
source and this is the reading.
