<!-- docs:meta
topic_id: repo.docs.research.scientific-compilation-obligation-2026-08-01
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.scientific-compilation-obligation-2026-08-01
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Scientific Compilation Obligation (SCO)

**Status:** research definition (Step I) — not an implementation claim  
**Date:** 2026-08-01  
**Branch / worktree:** `research/novel-compiler-20260801` @ `/tmp/sounio-novel-compiler-research`  
**Authority:** definitions only. No claim that any compiler currently satisfies SCO.

---

## 1. Motivation (one paragraph)

Scientific programs do not only compute values. They compute *reports*: magnitudes together with uncertainty, algebraic legality, and observation status. A compilation path that preserves the value while silently destroying the report is a successful build and a failed instrument. The object of this note is a compilation judgment that treats that failure as primary — not as a library bug, not as a test flake, and not as “numerical noise.”

This note defines the judgment. It does not survey prior compilers, and it does not inventory any existing tree.

---

## 2. Primitives

### 2.1 Scientific type

A **scientific type** \(\tau\) is a type whose meaning includes at least one *reportable scientific component* beyond a bare carrier value (e.g. a float payload).  
Examples of kinds of scientific component (not an API list):

- uncertainty / coverage / degrees of freedom attached to a quantity;
- algebraic structure that restricts which term rewrites are identities;
- observation or conditioning status (“this quantity was fixed by an observation”);
- effectful scientific labels that change which rewrites or evaluations are legal.

A type with only a machine number and no reportable scientific component is **ordinary**, not scientific, for the purposes of SCO.

### 2.2 Observable package \(\mathsf{Obs}(\tau)\)

For a scientific type \(\tau\), fix a finite **observable package**

\[
\mathsf{Obs}(\tau) = (o_1,\ldots,o_n)
\]

where each \(o_i\) is a **reportable** quantity or flag that a correct scientific reading of a value of type \(\tau\) must be able to recover after compilation and execution.

**Design rule:** \(n\) is small. Bloated \(\mathsf{Obs}\) cannot be gated.  
A minimal package sufficient to start the programme:

| Symbol | Role | Report form |
|--------|------|-------------|
| \(v\) | carrier value (when scientifically meaningful) | number / structure |
| \(u\) | combined standard uncertainty (or explicit “absent”) | non-negative scalar or \(\bot\) |
| \(k\) | coverage factor used for an expanded uncertainty (or \(\bot\)) | positive scalar or \(\bot\) |
| \(\nu\) | effective degrees of freedom (or \(\bot\)) | positive scalar or \(\bot\) |
| \(\alpha\) | algebraic legality class (e.g. free reassoc / restricted / forbidden) | finite enum |
| \(\omega\) | observation barrier status (open / fixed / barred) | finite enum |

Not every \(\tau\) uses every component. Unused components are \(\bot\) and are ignored by \(\preceq\).

**Reading:** \(\mathsf{Obs}\) is *what the scientist is entitled to read back*. It is not the full type theory of the language.

### 2.3 Observable preorder \(\preceq\)

Let \(O, O'\) be observable packages of the same shape (same components present).  
Define

\[
O \preceq O'
\]

to mean: **the report did not scientifically worsen** from \(O\) (source-level entitlement) to \(O'\) (after compile + run).

Concretely, for present components:

| Component | \(O \preceq O'\) requires |
|-----------|---------------------------|
| \(v\) | agreement under the problem’s exactness convention (bit-identical, ulp bound, or algebraic identity — fixed per corpus item, not hand-waved) |
| \(u\) | \(u' \ge u\) whenever both defined — **non-contraction of uncertainty** under optimisations that claim to preserve meaning; *or* an explicit, typed permission to refine \(u\) that is part of \(\tau\), never silent |
| \(k\) | \(k' = k\) when both defined and the coverage *procedure* is part of the report; silent collapse of Student-\(t\) coverage to a normal default is a violation |
| \(\nu\) | \(\nu' = \nu\) when both defined and material to \(k\); silent loss of finite-dof identity is a violation |
| \(\alpha\) | \(\alpha'\) is at most as permissive as \(\alpha\) (no illegal widening of rewrite freedom) |
| \(\omega\) | observation barriers are not erased (barred stays barred; fixed stays fixed) |

**Silent improvement of confidence** (smaller \(u\), larger effective trust, more rewrite freedom) without a typed justification is a **violation**, not a win.

### 2.4 Theory of equivalence \(E(\tau)\)

Each scientific type carries a **theory of equivalence** \(E(\tau)\): the set of term rewrites and algebraic identities the compiler may use when optimising code at type \(\tau\).

\[
E(\tau) \subseteq \{ \text{rewrites on the compiler IR / term language} \}
\]

Rules:

1. If a rewrite \(r \notin E(\tau)\), applying \(r\) to a term of type \(\tau\) is a **legality fault**.  
2. \(E(\tau)\) may depend on effects and strategies in the context (e.g. observation barriers shrink \(E\)).  
3. \(E\) is part of the *meaning* of compilation for scientific types, not a cost-model plug-in.

SCO will require that every optimisation step used on a scientific subterm is justified by the ambient \(E\).

---

## 3. The judgment: Scientific Compilation Obligation

### 3.1 Statement

**Scientific Compilation Obligation (SCO).**  
A compilation of a closed program \(P\) (possibly multi-module) **satisfies SCO** for a designated set of scientific sinks \(s_1,\ldots,s_m\) in \(P\) when:

1. **Emission:** \(P\) compiles and links to executable code \(C\) on the **default multi-module native path** under test (the path a user actually runs — not a secondary engine, not check-only).  
2. **Observable recovery:** executing \(C\) yields observable packages \(O'_1,\ldots,O'_m\) at those sinks.  
3. **Source entitlement:** the source (or a source-level reference semantics) determines entitlements \(O_1,\ldots,O_m\).  
4. **Preservation:** \(O_i \preceq O'_i\) for all \(i\).  
5. **Theory respect:** every IR rewrite applied to a subterm of scientific type \(\tau\) is in \(E(\tau)\) under the ambient context (effects, strategies, barriers).  
6. **Non-vacuity:** there exists a controlled ablation (disable \(E\), or disable scientific lowering checks) under which some \(O_i \preceq O'_i\) **fails** or a legality fault appears — so SCO is not satisfied by “never optimising” alone without a demonstrated obligation.

Clause 6 is part of the *research* obligation: a compiler that never touches scientific terms can pass (1)–(5) vacuously; novelty requires that \(E(\tau)\) is *doing work*.

### 3.2 What SCO is not

- Not “the program typechecks.”  
- Not “the float value looks close.”  
- Not “a library routine implements GUM on the host before codegen.”  
- Not “a separate verification toolchain could in principle prove something.”  
- Not bit-identical self-hosting of the compiler binary (that is a trust-anchor problem, orthogonal unless used as a delivery vehicle for SCO).

### 3.3 Adversary

The **SCO adversary** may:

- split the program across modules and force imported callees;  
- request optimisations on the default path;  
- choose scientific types and effects that *invite* illegal rewrites (non-associative structure, finite-dof coverage, observation barriers).

The adversary wins if \(C\) runs and prints a **plausible** report that violates \(\preceq\) or if a rewrite outside \(E(\tau)\) was applied.

The adversary does not need a crash. Crashes are ordinary compiler defects. **Plausible lies** are SCO defects.

---

## 4. Three conceptual violations

These are *shapes* of failure. They define what the corpus must eventually force. They are not bug tickets against any named file.

### Violation V1 — Silent coverage collapse

**Setup.**  
A quantity is reported with finite effective degrees of freedom, inducing a coverage factor \(k\) materially larger than the large-sample normal default. Source entitlement: \((v, u, k, \nu)\) with \(\nu\) small and \(k = k(\nu)\).

**Fault.**  
After multi-module native compile and run, the printed expanded uncertainty uses \(k' \approx 1.96\) (or any \(k' \neq k(\nu)\)) while \(v'\) remains plausible and the process exits zero.

**Why it is SCO-class.**  
The instrument still “works.” The scientist is misled about risk.  
\(O \npreceq O'\) via \(k\) (and typically \(\nu\)).

**Corpus sketch.**  
Module `A` builds a finite-dof uncertainty object; module `B` imports and prints \(k\) or \(U_{95}\) as an integer sentinel. Pass only if sentinel matches source entitlement exactly.

### Violation V2 — Illegal algebraic reassociation

**Setup.**  
A term inhabits a scientific type whose \(E(\tau)\) **forbids** a reassociation (or other rewrite) that would be legal for ordinary floats. Source entitlement includes \(\alpha = \mathsf{restricted}\) or \(\mathsf{forbidden}\) for that rewrite class.

**Fault.**  
The default optimisation path applies the forbidden rewrite. The carrier \(v'\) may even look “simpler.” Algebraic identity required by the domain fails, or a domain-specific sentinel (e.g. associator magnitude class) changes while the build stays green.

**Why it is SCO-class.**  
Ordinary FP optimisation theory is the wrong theory. Applying it is not a missed optimisation — it is a **category error** about \(E(\tau)\).  
\(O \npreceq O'\) via \(\alpha\) and usually \(v\).

**Corpus sketch.**  
A multi-module program whose source-level algebraic sentinel is exact; optimisation either preserves the sentinel or must refuse the rewrite under \(E(\tau)\). Ablation of \(E\) must change the outcome (clause 6).

### Violation V3 — Erased observation barrier

**Setup.**  
A quantity is under an observation/conditioning barrier: \(\omega = \mathsf{barred}\) or \(\mathsf{fixed}\). \(E(\tau)\) must not equate terms across the barrier.

**Fault.**  
Optimisation or lowering merges, forwards, or rewrites across the barrier so that post-run \(\omega' = \mathsf{open}\) or the observable behaves as if the observation never fixed the quantity — while the value stream remains smooth and the exit code is zero.

**Why it is SCO-class.**  
Observation is part of scientific meaning, not a comment. Erasing it is report fraud.  
\(O \npreceq O'\) via \(\omega\).

**Corpus sketch.**  
Sink prints a barrier tag or a downstream quantity that must differ when the barrier is respected; illegal merge collapses the distinction.

---

## 5. Minimal interface to later steps (no implementation)

| Step | Consumes this note | Produces |
|------|--------------------|----------|
| **II** Corpus | V1–V3 shapes | programs + exact sentinels |
| **III** Install \(E(\tau)\) | SCO + \(E\) | default-path behaviour that enforces theory respect |
| **IV** Gate + small lemma | \(\preceq\), non-contraction / legality | CI obligation + one formal lemma |
| **V** Separate compilation | SCO multi-module | preservation across imported callees |

**Acceptance for Step I (this document):**  
A reader can implement a gate *in principle* from §§2–4 without asking what “worse report” means.

---

## 6. Claim discipline

Until a default multi-module native path passes a corpus built from V1–V3 **and** meets clause 6 (non-vacuity):

- Do **not** claim that SCO holds for any production compiler.  
- Do **not** treat the existence of unrelated infrastructure as progress on SCO.  
- Do claim only: *the obligation and adversary are defined.*

That is the entire status of this note.

---

## 7. Open definitional choices (resolve in Step II, not by rhetoric)

1. **Exactness convention per sink:** bit-identical vs fixed-integer encoding of floats vs algebraic equality — must be per corpus item.  
2. **Permission to refine \(u\):** is any decrease of uncertainty ever allowed under a typed “analysis strengthening” effect, or is non-contraction absolute for v1?  
3. **Multi-sink programs:** SCO is universal over designated sinks; which sinks are designated (annotations vs whole-program scientific types)?  
4. **Separate compilation:** is module-level SCO stated with a relational \(\preceq\) on linked observables, or only whole-program for v1?

Recommended default for the first corpus: **absolute non-contraction of \(u\)**, **exact integer sentinels**, **annotated sinks**, **whole-program multi-module** (imports allowed, single final executable).

---

## 8. One-sentence research target

**Invent and discharge a compilation obligation under which multi-module native code cannot silently worsen scientific reports, because every rewrite on scientific types is justified by a type-carried theory of equivalence, and observable packages are gated end-to-end.**

## 9. Forward links

- Step II corpus (V1–V3 drivers + gate):  
  [`sco_corpus_2026-08-01.md`](./sco_corpus_2026-08-01.md) · `examples/sco_corpus/` · `scripts/ci/sco_corpus_gate.sh`
- Step III \(E(\tau)\) install:  
  [`sco_etau_2026-08-02.md`](./sco_etau_2026-08-02.md) · `scripts/ci/sco_etau_install_gate.sh` · `self-hosted/ir/lower.sio` markers `SCO_ETAU_*`

End of Step I.
