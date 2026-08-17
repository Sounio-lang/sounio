<!-- docs:meta
topic_id: repo.docs.audit.pr1702-ref-deref-verification.reference.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pr1702-ref-deref-verification.reference.readme
-->

# Issue #1702 — defect verification (do NOT port)

This directory is the witness corpus for the verification of [issue #1702](https://github.com/Sounio-lang/sounio/issues/1702), which records work stranded in the July SOIR/Arena-v2 stack: PR #889's `explicit_ref_deref` fix and PR #887's heap-bridge + ambiguous-public fixtures.

**Decision: do not port either fix.** The verification surfaced two findings more valuable than the ports would have been.

## Critical caveat — instrument not built from current source

All witnesses in this directory were run against the prebuilt `bin/madaros-linux-x86_64` (sha256 `437bdd8f96a205906d53ca50a2a29ccf5f03a71c2e98e020b54d01351a0bff44`, identity `Madaros v0.80.0`). The blob was committed at `3d1f143e7a` (2026-08-17 07:09:56 UTC). Current `origin/main` is `db750980b4` — roughly 7 hours later, with ~14 commits past the prebuilt's source. Per the FLEET-WIDE WARNING, `./bin/souc` is prebuilt and editing compiler source does not change it; a stale binary caused one prior false investigation (#1689). The proper path is build-from-current-source, but Slurm is currently broken (`launch failed requeued held`) and a full self-compile trips the pod's CPU-saturation liveness probe.

**All findings below are conditional on the prebuilt being representative of current main.** A CI-bound rerun against a current-source Madaros is required before any port lands.

---

## Finding A — #889 defect still reproduces, but the discriminator is PRIOR STATE, not aliasing

The PR #889 fix in `self-hosted/ir/lower.sio`'s field-access branch in `lower_assign_stmt_ref` changes the first argument of `field_idx_for_base_ref`:

```
// auto path (current main, broken when prior state exists)
field_idx_for_base_ref(&(*target_expr).left, ...)
                                          ^--- deref-typed

// fix path (PR #889)
field_idx_for_base_ref(&(*base_expr).left, ...)
                                ^--- inner of the deref
```

Without the fix, the explicit-deref store `(*pair).field = value` writes to a wrong field offset. The witness `collision_direct.sio` reproduces this:

```
before: second=7.500000 guard=101.500000
after:  second=7.500000 guard=8.500000
expected after: second=8.5 guard=101.5
```

`(*collision).second = 8.5` lands in the `guard` slot. The original PR's `ref_field_autoderef_semantics_witness.sio` panics on `assert(pair.second == 44.5)` after `(*pair).second = 44.5` for the same reason — silent offset confusion, then a failed assertion.

### Prior state is the discriminator, not aliasing

The minimum failing witness contains **no alias** — it writes `(*pair).field = value` directly through a `&!T` function parameter. The relevant dimensions that were held constant across the failing witnesses:

- All write through `&!T` (mut ref). `&T` (shared) is rejected by type-check.
- All targets are field accesses, not index ops. The array-write path passes (`array_explicit.sio`).
- All writes use the explicit `(*x).f = v` form. The auto-deref form `x.f = v` is not affected.

What varied and what survived the variation:

| Witness | Alias? | Prior state? | Result |
|---|---|---|---|
| `just_read.sio`         | no  | no  | pass |
| `just_store.sio`        | no  | no  | pass |
| `array_explicit.sio`    | no  | no  | pass |
| `alias_path.sio`        | yes (untyped)   | no  | pass |
| `annotated_alias.sio`   | yes (typed)    | no  | pass |
| `local_borrow.sio`      | yes (auto-borrow `&!local`) | no | pass |
| `collision_clean.sio`   | no  | no  | pass |
| `collision_direct.sio`  | no  | yes | **fail** — offset confusion |
| `collision_no_annot.sio`| yes (untyped)  | yes | **fail** — same offset confusion |
| `collision_only.sio`    | via function explicit write | yes | **fail** — same offset confusion |

The `no` vs `yes` (prior state) column is the only one that flips. The alias column is irrelevant: removing the alias does not fix the bug, and adding the alias does not cause it. **The discriminator is the lowerer's accumulated struct-layout state, not the syntactic shape of the write.**

Why this matters: the PR's gate (`ref_field_autoderef_static_gate.sh`) is purely syntactic — it greps for `explicit_ref_kind == 2` and `> 0` in `lower.sio`. It cannot detect a regression where the field-offset lookup is wrong in a state-dependent way, because the regression would still contain the right string. The correct regression gate for this defect is **dynamic**: `collision_direct.sio` plus its cleaned-up variant. Pass/fail on that witness — not on the static grep — is what tells you whether the fix is alive.

---

## Finding B — #887's gate passes vacuously; the original diagnosis was wrong

The `scripts/ci/madaros_visibility_context_gate.sh` IS in main. It tests:

- `duplicate_private_single_main.sio` and `duplicate_private_18_main.sio` (E175 privacy)
- `visibility_fn_private_main.sio`, `visibility_struct_private_main.sio`, `visibility_enum_private_main.sio` (E175/E176/E177)

It does **not** test ambiguous-public names. The three `ambiguous_public_*.sio` fixtures (a/b/main) are absent.

The original witness (`ambiguous_public_main.sio`) imports `load_ambiguous_a` and `load_ambiguous_b` only — **not** `same_public_name`. It then calls `same_public_name()` undeclared. The compiler raises `error[E137]: use of undeclared variable`.

E137 is raised for **any** undeclared variable, not specifically for ambiguity. Confirmed by the control `ambiguous/no_use.sio`:

```
// no imports at all
fn main() -> i64 {
    same_public_name()
}
```

```
error[E137] in ambiguous/no_use::main at 112..128: use of undeclared variable
```

The original ambiguity detection (if it exists) is not exercised by the witness. **The gate passes vacuously.** Porting a fix for a defect whose test coverage is broken would be wasted work.

Stronger witnesses I tried:

- `ambiguous/strong.sio` — `as` aliases not supported by the parser (token 185 vs actual 30).
- `ambiguous/strong2.sio` — bare `use module` (no braces) parses but does not bind names; same E137 downstream.
- `ambiguous/strong3.sio` — `use a::{same_public_name}; use b::{same_public_name};` parses; **type-check passes**; `run_check_mode` reports `nodes=1 unresolved=2` (one call node, two ambiguous candidates) but **no ambiguity-specific error** is emitted. The closure walk detects the situation but does not surface it as a distinct diagnostic.

Three possible states for the defect:

1. The defect was fixed by some other route since the PR was closed.
2. The defect is still present but the test infrastructure does not surface it.
3. The compiler never had an ambiguity check; the PR's gate was always vacuous.

**Without a non-vacuous positive control, these three are indistinguishable.** This is why the fix is not ported.

---

## Why the port is rejected

Porting #889's fix now would discard Finding A. Finding A is more valuable than the fix because:

- It pins down the **discriminator** (prior state), not the symptom (aliasing). The fix changes a single arg to a single function call. If the fix lands and a future change re-breaks the same path with a different surface area, Finding A's discriminator points at the right regression test; the static gate does not.
- It exposes that the PR's static gate is **insufficient**: it greps for string equality but the bug is dynamic. A green gate against a broken state-dependent witness would be a false negative. The fix has to be validated against `collision_direct.sio` (or a successor), not against the grep.

Porting #887's fix now would discard Finding B. Finding B is more valuable than the fix because:

- It is a **refutation of the closed issue's premise** — the issue asserts the fix is needed because of an ambiguity-detection regression, and the refutation shows the gate was vacuous. That refutation applies whether or not the compiler actually has an ambiguity check.
- A port of the fix without a non-vacuous positive control is unverifiable: there is no test that distinguishes "fix worked" from "no test exists that the fix could have failed."

Both findings are recorded here, with sources, so a future investigation can re-run the witnesses against a current-source Madaros and either confirm them or refute them.

---

## Witness inventory

Each witness has a `.sio` (source) and a `.elf` (compiled binary output, kept for forensic reproducibility).

Reads:

- `just_read.sio` — minimal read, shared ref, both auto and explicit deref.

Stores (single-struct):

- `just_store.sio` — direct explicit write on a clean module, via `&!T` parameter.
- `array_explicit.sio` — explicit write to a fixed-size array field.
- `alias_path.sio` — untyped `let alias = pair; (*alias).second = value`.
- `annotated_alias.sio` — typed `let alias: &!T = pair; (*alias).second = value`.
- `local_borrow.sio` — `var local; let alias = &!local; (*alias).second = value` inside a parameterless function.

Stores (multi-struct, prior-state dependent):

- `collision_clean.sio` — control: direct explicit write on `RefFieldCollision` with no prior struct activity. PASSES.
- `collision_direct.sio` — same as above but with prior writes to `RefFieldPair`. **FAILS.**
- `collision_no_annot.sio` — untyped alias variant, with prior state. **FAILS.**
- `collision_only.sio` — variant using a function-based explicit write on `RefFieldPair`, with prior state. **FAILS.**

Combined:

- `all_paths.sio` — runs every path from the original witness, using `print` instead of `assert`. Identifies which step panics. Used to isolate the prior-state trigger.

Original witness (panics):

- `ref_field_autoderef.sio` — verbatim copy of the PR #889 witness from `tests/native-v2/`. Panics silently at `assert(pair.second == 44.5)` after `(*pair).second = 44.5` for the same reason as `collision_direct.sio`, but expressed through the original test surface.

Ambiguous-public (vacuous-pass investigation):

- `ambiguous/a.sio`, `ambiguous/b.sio`, `ambiguous/main.sio` — verbatim copies of the PR #887 fixtures from `tests/compiler/madaros_visibility_context/`.
- `ambiguous/no_use.sio` — control showing E137 is undeclared-variable, not ambiguity.
- `ambiguous/strong.sio` — `as`-alias attempt (parser rejects).
- `ambiguous/strong2.sio` — bare-`use` attempt (does not bind).
- `ambiguous/strong3.sio` — both-imports attempt; type-check passes but no ambiguity diagnostic.

---

## Run instructions

From the worktree root, with the prebuilt Madaros:

```bash
SOUNIO_STDLIB_PATH="$(pwd)/stdlib" bin/souc compile <witness>.sio -o <witness>.elf
chmod +x <witness>.elf
SOUNIO_STDLIB_PATH="$(pwd)/stdlib" <witness>.elf
```

For the ambiguous subdirectory, change directory first so the `use ambiguous_public_a::{...}` imports resolve:

```bash
cd docs/audit/pr1702_ref_deref_verification/reference/ambiguous
SOUNIO_STDLIB_PATH="<repo>/stdlib" <repo>/bin/souc compile main.sio -o main.elf
```

A current-source Madaros build would replace the prebuilt with one whose source matches `origin/main`'s tree, after which every witness in this corpus should be re-run before any decision about whether the defects still survive.
