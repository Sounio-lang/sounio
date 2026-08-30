# xai (Grok 4.5) adversarial review — `EpistemicEffectsNS.lean` (2026-08-30)

**Dispatch:** `bin/llm-offload -t math-review -p xai` (Grok grok-4.5, `OFFLOAD_MAX_TOKENS=16384`),
packet = the full `.lean` file (62.5 KB) + nine named attack surfaces (modelling, definitions,
claim↔statement gap; compilation explicitly out of scope). Artifact under review: PR #2308 /
lane commit `98aee98965`.

**Verdict:** 6 × [OK], 3 × [TIGHTENABLE], 0 × [FAIL]. All three tightenables acted on below.

---

## Referee report (verbatim)

**1. Aff / inner / trueVar** — **[OK]**
`inner a b = Σ_{(s,c)∈a} c·coeff b s` equals `Σ_s coeff_a(s)·coeff_b(s)` with dup-summing (`coeff_append`, dup expansion); `trueVar a = Σ_s coeff(a,s)²` = Var under i.i.d. unit-variance `ε_s`.

**2. Covers membership-based** — **[TIGHTENABLE]**
Zero-coeff monomials still force `nsMem`, so `Covers` is stricter than nonzero support; strengthens rejection (completeness↓), no unsoundness direction.

**3. Label = noise-symbol axiom** — **[TIGHTENABLE]**
Distinct labels on physically correlated measures type as disjoint and slip through; not circular/`Covers` is an invariant from `measure`/`certain`, but soundness is relative to honest labelling only.

**4. Exact ⇒ reported=true at reached Knowledge** — **[OK]**
`exact_preservation` + `preservation` close under red/subst/shift; `kunc`/`kconf` erase to bare `lit_real` (claim is only about Knowledge values, not extracted scalars).

**5. AGFree only on already-kraw operands** — **[OK]**
`soundness_star` re-derives `AGFree` from preserved `HasTy` at every `⇒*` residual, so when congruence finally makes operands `kraw` values the cov check is live.

**6. kmul_red / delta-method form** — **[OK]**
`scale y a ++ scale x b` is the correct first-order linearisation at `(x,y)`; `gMulMeta` matches `trueVar_mul` up to exactly the `2xy⟨a,b⟩` gap.

**7. nsDisjoint / E230 / untypability strength** — **[OK]**
`⊤` never disjoint matches (Add-Indep); `∀ Γ T E, ¬HasTy Γ (kadd xk xk) T E` quantifies all derivations, hence all operand annotations `N` (via `support_over_approx` + `nsDisjoint_of_shared`).

**8. Progress / preservation** — **[OK]**
No extra stuck states beyond untypable E230; `t_sub` only widens effects (not `N`); `weakening`/`substClosed` preserve `Covers`/`kvalid` on `kraw`.

**9. Claim ↔ file gaps** — **[TIGHTENABLE]**
Not proved: conf semantics, higher-order (non-linearised) product variance, physical correlation beyond labels. Mild overclaim: "true variance" after `kmul` is first-order/delta variance only; "no anti-garbling" is `AGFree` (cov=0 at kraw-valued operands), not a semantic cost.

---

## Disposition

| # | Finding | Action | Where |
|---|---|---|---|
| 2 | `Covers` is membership-based; paper's Lemma 2 is worded on *actual* (nonzero-coefficient) support | added `covers_coeff : Covers N a → ∀ s, coeff a s ≠ 0 → nsMem s N` — the coefficient-based statement is derived from the membership invariant, so the theorem is *at least* as strong as the wording; the direction is conservative (rejects more), as the referee notes | `EpistemicEffectsNS.lean` §B; gate C4 |
| 3 | soundness is relative to honest labelling (distinct `measure` labels = distinct physical sources) | promoted from a header remark to an explicit boundary — residual (iv) in the closure note and in §6.4's closing paragraph; it is the modelling axiom of every affine-arithmetic / noise-symbol system (Comba–Stolfi, Fluctuat) and is *not* discharged by the type system — the type system tracks sources, it does not discover them | note §4, paper §6.4 |
| 9 | "true variance" after `kmul` is first-order only; `AGFree` is cov=0 at value-operands | wording: "true **first-order** variance" wherever the exactness row/claim appears; `AGFree`'s formal content stated as such. "conf semantics" is out of scope by design (the paper's §10 already calls confidence decay heuristic) | note, paper §6.4 table, PR body |

Nothing in the report identifies an unsound direction; the two substantive tightenables (2, 3)
both concern *completeness* / *modelling scope*, which is where the paper already places its
stated boundaries (§6.5).
