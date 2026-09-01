# Conjecture 6.8 IS A THEOREM — Zhilina, arXiv 2608.26890 (2026-08-27)

**Svetlana Zhilina, "Diameter of the commutativity graph of the real sedenions",
arXiv:2608.26890 (27 Aug 2026); J. Math. Sci. lineage.**

Abstract (verbatim): "The commutativity graph of the real sedenion algebra is
considered. It is shown that those elements whose imaginary part is not a zero
divisor correspond to isolated vertices of this graph. All other elements form a
connected component whose diameter equals 3."

## Theorem 4.13 (= Guterman–Zhilina Conjecture 6.8, RESOLVED)
The diameter of Γ_C^Z(𝕊) equals 3.

Chronology: arXiv 2608.26903 (Guterman–Zhilina, "Relation graphs of the sedenion
algebra") states this as **Conjecture 6.8**; the companion paper 2608.26890
(Zhilina, same 27 Aug 2026 arXiv batch) **proves it** as Theorem 4.13. The
conjecture was resolved simultaneously with being posed. **2608.26890 was already
in our manuscript's reference list; we never fetched it.**

## The proof (the construction we independently rediscovered and VERIFIED, 8/8 configs)
Given zero divisors x=(a,b), x'=(c,d):
1. dim Im C_S((c,d)) = 5 (Lemma 4.2(2), the centralizer).
2. The four conditions (system 4.5)
     ⟨a,a'⟩−⟨b,b'⟩=0, ⟨a,b'⟩+⟨b,a'⟩=0, ⟨a',ab⟩=0, ⟨b',ab⟩=0
   are exactly (a',b') ∈ span{(a,−b),(b,a),(ab,0),(0,ab)}^⊥ (codim ≤ 4).
3. **dim(Im C_S((c,d)) ∩ that⊥) ≥ 5 − 4 = 1** ⟹ a nonzero (a',b') exists.
4. System 4.5 ⟹ the Lemma 4.12 criterion (its determinant column vanishes) ⟹
   d(x,(a',b')) ≤ 2, i.e. ∃ u ∈ Im C(x) with [u,(a',b')]=0.
5. (a',b') ∈ Im C(x') ⟹ adjacent to x'. Path x — u — (a',b') — x', length ≤ 3.
6. With Prop 4.7 (diam ≥ 3, from O_S(a,b)∩O_S(b,a)=0), diameter = 3. ∎

**Why our year-long attack missed it:** we attacked the witness as [u,w]=0 directly
(a determinantal locus D of degree 6/7/9 with an even 4-dim residual — genuinely
resistant to parity/degree/topology). Zhilina's proof finds w FIRST in a codim-4
space (trivial dimension count 5−4≥1), THEN u via Lemma 4.12 — sidestepping the
even residual entirely. The right decomposition, not a new deep invariant.

## Our residual contribution (still valid, still worth publishing)
The exact rank laws §§2–5 of conj68_manuscript_draft.md (im T ⊥ {x,x̃,x',x̃'};
the ghost/sector rank-9 law; the associator rank-7 relations), all mechanically
verified (Sounio i64 + Lean), are independent of the diameter theorem and stand as
a study of the commutator map T. The obstruction map (proof_strategy §§4–15) is a
catalogue of why the *hard* decomposition resists — of interest but not needed now
that Zhilina's easy decomposition is known.

Found via codex-1 (GPT-5.6 SOL, neighbouring tmux lane) web-searching the companion
series; construction verified computationally here (8/8 configs, real length-3
witness in every case).
