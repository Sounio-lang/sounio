<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r20-2026-07-28
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r20-2026-07-28
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R20 — provenance binding: the check passes, the derivation is not in the tree

**Date:** 2026-07-28
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `PROVENANCE_BINDING_IMPLEMENTED__CITED_DERIVATION_MUST_EXIST`
**Parents:** `self_falsifying_compilation_line_r17_2026-07-28.md` (witness binding — the mechanism this completes), `self_falsifying_compilation_line_r15_2026-07-28.md` (what a token can and cannot see)
**Harness:** `scripts/research/self_falsifying_compilation_line_r20_contract.py` (+ `scripts/research/r20/`)
**Gate:** `scripts/ci/self_falsifying_compilation_line_r20_gate.sh`

---

## 1. Result

> **A contract in this repository can be green, emit exactly the verdict token
> its claim declares, and match a witness fingerprint, while the derivation it
> says it rests on is not in the tree. Two such artifacts underpin
> `ZD_FIBER_SPECTRUM_COMPLETE_INVARIANT_N_LE_8` — the very claim this arc has
> been studying — and at discovery both lived on a branch that was never merged
> here. They have since been restored and re-verified (§1.3); a third, on which
> the orbit theorem's own verification depends, was never committed anywhere at
> all and remains absent.**

Verdict: `SELF_FALSIFYING_R20_VERDICT PROVENANCE_BINDING_IMPLEMENTED__CITED_DERIVATION_MUST_EXIST`.

Every mechanism this line has built reads what a gate **computes and emits**:
the exit status (R0), the proposition (R2), the evidence fingerprint (R17).
**None reads what a claim cites.** That is the gap, and it is not hypothetical.

### 1.1 The audit

916 files scanned across `scripts/research/`, `docs/research/` and
`scripts/ci/`; **2 155 distinct repository artifacts cited; 93 absent from the
tree.** Of those, **8 were committed to another branch**.

**The distinction that matters is not where a citation went, but whether it is
prose or load-bearing.** A docstring mention is a citation; an
`exec_module`/`open`/`bash` on a path is a **dependency**. Separating them:
**6 of the missing are hard dependencies** — loaded or executed with no
fallback. See §1.3; the first version of this spec dismissed the
never-committed bucket as "mostly planned names", which was too quick.

The two that matter:

| artifact | role | cited by | lives on |
|---|---|---|---|
| `cd_tower_collapse_isomorphism.py` | the explicit parity-collapse map Φ — the **upper bound of the completeness pincer** | `..._spectral_classifier_contract.py`, `..._signed_localization_contract.py` | `lean/cd-seamflip-forall-n` |
| `cd_tower_fiber_geometry_collision.py` | the construction the classifier states it replicates **verbatim** | the classifier's contract and spec | `lean/cd-seamflip-forall-n` |

So on this branch the completeness claim's upper half is not reproducible, and
the construction's stated provenance dangles. Both were committed — `aa60dd45a`
and `13466f3e0` — and simply never arrived here.

**This is also the claim R18 bound a witness to.** The witness matches. It
cannot see any of this.

### 1.2 The mechanism

A claim may declare `provenance = "<path>"`. The compiler refuses codegen when
the path is not in the tree (`CLAIM_PROVENANCE_MISSING`, code 8), checked after
the witness so a claim declaring all three must satisfy the proposition, its
grounds, and the availability of what it cites.

**Existence only.** Content drift is the witness's job — a gate that
fingerprints its inputs already covers it — and claiming more here would
overstate what one `stat` establishes.

| probe | gate | rc | ELF | outcome |
|---|---|---:|---|---|
| present | exits 0, token ✓, cited file in tree | 0 | yes | `CLAIM_PASS` |
| **missing** | **exits 0, token ✓, cited file absent** | **1** | **no** | **`CLAIM_PROVENANCE_MISSING`** |
| compat | no `provenance` field | 0 | yes | `CLAIM_PASS` |

The missing-probe cites a **synthetic** path, and that is itself a lesson: it
first cited the real absence, and when that absence was repaired the probe
silently stopped demonstrating anything. **A fixture built on a real defect
self-destroys when the defect is fixed.** Real absences belong in the audit
data; fixtures need a path that stays absent.

### 1.3 The cascade, and the worst one

Restoring the two artifacts above did not end it. Each repair revealed the next
gap:

1. `cd_tower_collapse_isomorphism.py` restored — it runs, and **all its collapse
   isomorphisms verify at n = 6, 7, 8 with 0 mismatches**.
2. Its docstring cites `cd_tower_auto_action_on_zd_fibers.py`, the orbit
   theorem's verification script — **also absent**. Restored from the same
   branch.
3. That script loads `cd_tower_automorphism_oracle.py` via
   `spec_from_file_location` + `exec_module`, with no fallback — and that file
   was **never committed anywhere, on any branch, in the repository's entire
   history** (`git log --all` returns nothing).

So the orbit theorem's verification script — described in Φ's own docstring as
*"PROVEN forall n and untouched"* — **cannot run in any checkout of this
repository**. That is not a merge oversight like the first two; it is an
artifact that never entered version control at all, and no gate can detect it
because the script that needs it is not run by CI either.

---

## 2. The instrument failed first, and that is the better half

The first version of the audit reported 65 missing artifacts, confidently, and
**did not include the file whose absence motivated the rung.**

Its rule for a bare filename was: count it as a citation only if that basename
is `git ls-files`-tracked. That was meant as prose-filtering. What it actually
does is make existence a **precondition for being checked** — so a file cited by
bare name and absent from the tree can never be reported.

> **The filter's precondition was the negation of the thing being detected.**
> A missing-artifact detector that only considers artifacts that are present.

The fix does not hardcode a whitelist, because a whitelist is a second thing to
keep in sync and would silently stop covering new families. It derives the
corpus's own naming families from the tracked artifacts — the token before the
first underscore of every tracked research script — and accepts a bare basename
in one of those families **whether or not it exists**. 65 → 93, including both
load-bearing ones.

Seventh self-catch of this line, and the first where the instrument's blind spot
was structurally identical to the corpus defect it was built to find.

---

## 3. What this is NOT

- **Not a claim that the completeness result is wrong.** It is a claim that its
  upper bound is **not reproducible from this branch**. The artifact exists; it
  is elsewhere. Merging it would close the finding without touching the
  mathematics.
- **Not content verification.** §1.2. A file can exist and be the wrong file;
  `provenance` does not look inside it.
- **Not a defect count of 93.** Most citations are prose. The load-bearing
  number is **6 hard dependencies**, and within those the one that was never
  committed at all (§1.3). The first version of this spec called the
  never-committed bucket "mostly planned names" and was corrected by looking.
- **Not automatic.** A claim must declare its provenance. Nothing infers
  dependencies from a contract's imports or citations.
- **Not a solution to shared misinterpretation.** R0 §3 is untouched, as in R17.

---

## 4. Reproduce

```bash
python3 scripts/research/r20_provenance_audit.py        # the corpus audit
python3 scripts/research/self_falsifying_compilation_line_r20_contract.py
bash scripts/ci/self_falsifying_compilation_line_r20_gate.sh
```

The compile arm (`SFCL_R20_RUN_COMPILE=1`) needs a provenance-binding Madaros at
`artifacts/self-hosted/madaros-provenance`; the build is CPU-heavy and
serialises on a lock shared with other agents. As in R17, the behaviour receipt
is bound to the executor's sha256, so editing the executor invalidates it rather
than certifying stale behaviour.

---

## 5. AI disclosure

Audit, executor change, fixtures, contract, gate and spec drafted under human
direction (2026-07-28). The audit's own filter defect in §2 was hit, not
anticipated, and is reported with the number it wrongly produced. The behaviour
rows are transcribed from an actual run of a compiler built from this executor
source. No clinical content. GAIDeT-ICMJE 2025.
