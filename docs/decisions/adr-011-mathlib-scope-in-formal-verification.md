<!-- docs:meta
topic_id: repo.docs.decisions.adr-011-mathlib-scope-in-formal-verification
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-011-mathlib-scope-in-formal-verification
-->

# ADR 011: Boundary and Scope of Mathlib in Sounio Formal Verification

- **Status:** accepted
- **Date:** 2026-09-24
- **Context:** Sounio maintains a lightweight formal verification suite in `formal/` and `formal/lean4/` intended to run with standard Lean 4 toolchains without external dependency friction. However, higher-dimensional non-associative algebras (such as octonions $\mathbb{O}$ and sedenions $\mathbb{S}$) involve multivariate polynomial identities in 16 to 32 variables (e.g. Degen eight-square identity, alternativity, Moufang identities). Lean 4 core linear arithmetic (`omega`) cannot discharge degree-2 and degree-4 multivariate identities ($x_i \cdot y_j$).
- **Decision:**
  1. The core Lean 4 formal tree (`formal/`) shall remain self-contained for linear and basis-checked theorems. Distributivity (`oct_mul_add_left`, `oct_mul_add_right`) are formally proved via `simp only + omega` and `decide` without Mathlib.
  2. The package `formal/omega_mathlib/` (already present in the repository) is designated as the official locus for `Mathlib.Tactic.Ring`.
  3. Universal polynomial theorems that cannot be discharged by linear arithmetic alone are asserted with explicit epistemic tracking in `formal/AXIOM_INVENTORY.md` marked as `NÃO ESTABELECIDO` until proven via `formal/omega_mathlib/`.
- **Consequences:** Eliminates formal inconsistency while keeping the standard CI build fast and deterministic.
