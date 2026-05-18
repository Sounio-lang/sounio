<!-- docs:meta
topic_id: repo.docs.audit.r3-0-transitive-alias-chain.dispatch
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.r3-0-transitive-alias-chain.dispatch
-->

# DISPATCH R.3.0 — rng.sio Phase D Wiring Blocker (discovery)

**Opened.** 2026-05-18 (R.2.9 §4 follow-up, with R.2.9 §4's "transitive
alias" hypothesis explicitly *superseded* by this dispatch's §2).
**Class.** Discovery-only — characterise and locate the bug.
**Priority.** P4 — purely cosmetic; rng.sio inlining costs ~100 LOC
duplication. Not blocking PBPK, dissertation, oracles, or any
operational consumer.
**Branch.** `sounio-pure/r2-1-park-miller`.
**Time budget.** 2h Phase A (diagnose only). Phase B/C/D scoped at the
end of Phase A, not now. **No fix proposed in this dispatch.**

---

## §0 — Sounio-Pure constraint

Read-only on `self-hosted/compiler/lean_single.sio` for Phase A
diagnosis. Probes written under
`docs/audit/r3_0_transitive_alias_chain/reference/`. No bootstrap
chain edits, no umbrella regression tolerance.

---

## §1 — What fails

Applying R.2.7+R.2.8's Path A wiring to `stdlib/random/rng.sio` —
specifically `type Pcg64 = PcgState` plus collapsing the
`pcg64_next_*` family to one-line forwards into `pcg_step` /
`pcg_next_f64` / `pcg_next_f64_nonzero` — produces this failure when
compiling any consumer (e.g. `rng_oracle_gen.sio`):

```
error: assignment type mismatch at line 14   (consumer: `rng = r.0`)
error: assignment type mismatch at line 94   (rng.sio pcg64_bounded: `current = r.0`)
error: error[E001]: Type mismatch in call argument at line 303
                                                (rng.sio rng_new: `rng_splitmix_step(s)`)
error: assignment type mismatch at line 114  (rng.sio xoshiro256_new fn signature)
```

The first three reproduce on every consumer that imports the wired
rng.sio. R.2.8 deferred this; R.2.9's compiler patch closed the
1-field-struct scalar branch (line 13597+) but did **not** close this
failure mode.

---

## §2 — Ruled out (what is NOT the cause)

R.2.9 SYNTHESIS §4 claimed "transitive alias `pcg64_core::PcgState →
rng::Pcg64 → consumer` doesn't propagate." That claim is **wrong**
and is hereby superseded.

Evidence (`reference/transitive_*.sio`):

| Probe | Shape | Result |
|---|---|---|
| `transitive_pass.sio` | direct alias of imported struct in single file | **PASS** |
| `transitive_main.sio` + `transitive_lib.sio` | 2-hop alias chain through imported module | **PASS** |
| `transitive_lib2.sio` + `transitive_main2.sio` | 2-hop + extra state machine (SplitMix64) | **PASS** |
| `transitive_lib3.sio` + `transitive_main3.sio` | 2-hop + RngPcgWrapper struct field of aliased type + xoshiro256 + pcg64_next_f64_nonzero (var/while pattern) | **PASS** |

The minimal failing case isn't yet bisected. Bisection covered every
structural feature of rng.sio's PCG64 + xoshiro256 + RngPcgWrapper
sections and reproducing the failure required *more* of rng.sio's
content than fits in those four probes. Phase A's first task is to
finish the bisection.

Things definitely ruled out:
- Single-hop alias (handled by R.2.7 ce9810ee9).
- Two-hop alias through one re-aliasing module.
- 1-field-struct scalar-branch propagation (closed by R.2.9 3fb8986bd).
- Struct field whose type is an aliased struct.
- Co-presence of an unrelated state machine (splitmix64) using
  R.2.9-affected `var sm = fn(seed); sm = r.0` pattern.

---

## §3 — Attack plan (Phase A only)

### Phase A — Bisect rng.sio to minimum failing case (2h)

1. Start from passing `transitive_main3.sio` + `transitive_lib3.sio`.
2. Copy rng.sio into a probe file, then **remove halves** until the
   failure disappears. Each removal that still fails narrows the cause.
   Each removal that flips to passing identifies a necessary
   constituent of the bug.
3. The failure surfaces at:
   - line 14 (consumer `rng = r.0`)
   - line 94 (`current = r.0` inside `pcg64_bounded`)
   - line 303 (`rng_splitmix_step(s)` — passing `step1.0` of `(i64, i64)` tuple)
   - line 114 (xoshiro256_new signature line, which suggests an error
     position drift; the actual offence is probably in xoshiro256_new's
     body or in xoshiro256_next_i64).
4. Goal: a minimum reproducer that fails with one of these errors
   under souc `c7ea6a4d…` (the post-R.2.9 binary) but passes when one
   specific element is removed.
5. Add the minimum reproducer to `reference/`. Add a probe matrix
   commentary to SYNTHESIS §2.

### Phase B/C/D — Scoped after Phase A converges

Once the minimum failing case is in hand, scope Phase B (compiler fix
direction) based on what the failure actually shows. **Do not**
pre-commit to a code-path or a fix shape now — three prior dispatches
(R.2.8, R.2.9, R.2.9 §4) had pre-committed diagnoses that turned out
wrong; this one explicitly avoids that pattern.

---

## §4 — Operational status quo

Acceptable steady state:

- R.2.7 / R.2.8 / R.2.9 remain RESOLVED-PARTIAL. The Phase D rng.sio
  refactor is the only outstanding aspirational change.
- `stdlib/random/rng.sio` has ~100 LOC of PCG64 inlining redundancy
  with `pcg64_core`. Not blocking anything. Not a correctness issue.
- All oracles (R.2.4 distributions 1024/1024, R.2.5 rng self-oracle
  1024/1024, dissertation PBPK suite 7/7) are bit-exact.
- Umbrella 12/12 PASS at HEAD (`84278435c..1b8e43d82`).

R.3.0 can sit OPEN without blocking any operational work. The
priority is *understanding*, not *shipping*.

---

## §5 — Out of scope

- Any rng.sio source edit (R.3.0 is read-only on the stdlib).
- Compiler patch (deferred to R.3.1 if Phase A converges).
- Bootstrap chain.
- Algorithmic changes to PCG64 / SplitMix64 / Xoshiro.

---

## §6 — Halt conditions

- **Phase A bisection produces no minimum failing case in 90 min.**
  Document partial state; close R.3.0 as DEFERRED with notes for
  whoever picks it up next.
- **Bisection reveals a pre-existing failure mode unrelated to alias
  resolution.** Pivot: re-scope R.3.0 to that finding.
- **Phase A discovers the failure depends on simultaneous edits to
  pcg64_core or sampling.sio.** Pivot: re-scope to the multi-file
  interaction.

---

## §7 — Deliverables (Phase A close)

1. `docs/audit/r3_0_transitive_alias_chain/SYNTHESIS.md` — Phase A
   diagnosis writeup with minimum failing case + which lean_single.sio
   code path is implicated.
2. `docs/audit/r3_0_transitive_alias_chain/reference/transitive_*.sio` —
   already present (the *passing* controls that rule out hypotheses).
3. `docs/audit/r3_0_transitive_alias_chain/reference/<minimum_repro>.sio` —
   to be added during Phase A.
4. **No code changes.**

---

## §8 — Acceptance (Phase A)

R.3.0 Phase A is **VALIDATED** iff:

1. ✓ Minimum failing reproducer committed to `reference/` and verified
   FAIL under souc `c7ea6a4d…`.
2. ✓ Same reproducer passes when *one* specific element is removed —
   that element is the cause locus.
3. ✓ SYNTHESIS §2 maps the locus to a specific code path in
   `lean_single.sio` (or identifies "not in lean_single.sio" if the
   failure is in a different layer).
4. ✓ R.3.1 scope drafted with explicit Phase B fix direction grounded
   in the Phase A locus — *not* in any pre-existing hypothesis.

If 1–3 succeed but the fix direction is unclear: ship R.3.0 SYNTHESIS
documenting just the locus; defer R.3.1 dispatch authoring to a
follow-up session.

If 1 fails after 90 min: close R.3.0 as DEFERRED with bisection notes.

---

## §9 — Notes

- **The three preceding dispatches each anchored their §3 plan to a
  wrong cause hypothesis.** R.3.0 explicitly avoids that pattern by
  refusing to propose a fix direction before bisection converges.
  This is method discipline, not pessimism.
- The cost of holding open is zero — the steady state is operationally
  fine. The cost of shipping a fourth wrong hypothesis is another
  cycle of partial-resolution status notes.

**END OF DISPATCH.**
