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

---

## Second-provider status (2026-08-31) — BLOCKED, not skipped

The repo's math-review policy is a two-provider fan-out (xai + zai). The zai leg was
dispatched on the post-`covers_coeff` file with the identical packet and **could not run**
for account reasons, none of which are about the proof:

| Route | Result |
|---|---|
| Z.AI GLM-5.2 direct (`api.z.ai`, coding plan) | `1313` Fair-Usage lockout — "request frequency has been limited… submit a request" (2 attempts, ~10 min apart) |
| GLM-4.6 via OpenRouter (driver fallback, forced) | `402` no credits on the OpenRouter account |
| DeepSeek V4 Pro (independent substitute) | `401` API key invalid |
| Groq Llama 3.3 70B (last independent key) | `401` API key invalid |

Only xAI is operational. `xai-fast` (Grok 4.1) is the same vendor and does not count as
an independent second opinion, so it was not used as a stand-in. The zai review remains
**open**: restore Z.AI access (or fund OpenRouter / rotate the DeepSeek–Groq keys) and re-run
`OFFLOAD_MAX_TOKENS=16384 bin/llm-offload -t math-review -p zai -i <packet>`; the packet is
the `.lean` file preceded by the nine attack surfaces listed at the top of this record.

Until then the mechanization's external-review status is: **one adversarial review (Grok
4.5), 0 FAIL, all tightenables acted on; second opinion pending on operator action.**

---

## Round 2 — xai Grok **4.6** (2026-08-31), same packet (post-`covers_coeff` file)

Requested by the operator as the second reviewer (Z.AI blocked, see above). Same vendor as
round 1, so it is a *stronger model*, not an *independent* opinion; recorded as such.
**Verdict:** 6 × [OK], 2 × [TIGHTENABLE], 1 × [FAIL]/[TIGHTENABLE] (item 9: missing
witnesses + wording). Both substantive findings were correct and are now fixed in the file.

### Referee report (verbatim)

**1. Aff / inner / trueVar — [OK]**  
`inner` folds `c * coeff b s` over `a`; with `coeff` summing duplicates this is exactly `Σ_s coeff_a(s)·coeff_b(s)` (and `inner_comm`/`trueVar_append` hold). `trueVar a = ⟨a,a⟩ = Σ_s c_s²` is Var of `Σ c_s ε_s` for independent unit-variance `ε_s` (constant lives in the payload).

**2. `Covers` membership vs coefficients — [TIGHTENABLE]**  
Membership `Covers` is strictly stronger than “nonzero coeff ⇒ s∈N” (`covers_coeff`); zero-monomials only extra-reject, never license a bad `nsDisjoint`. No unsoundness direction; Lemma 2’s informal “actually carries uncertainty” is the weaker coeff-based statement.

**3. Label = noise symbol; circularity — [OK]**  
Distinct labels can still name physically correlated sites (theorem is relative to that axiom, never an under-approx of correlation). Not circular/vacuous: `Covers` is generated by `meas_red`/`covers_single` and preserved, not merely assumed; `x_plus_y_typable` is a real inhabitant.

**4. `Exact` ⇒ reported=true at reached Knowledge — [OK]**  
`Exact` is structural (incl. λ-bodies, kraw payloads); `exact_shift`/`exact_subst` close beta/let. `kvalue`/`kunc`/`kconf` drop to payloads/literals (`measure`’s `c` is a syntactic `Int`, not a computed variance). `t_kraw` does *not* require `gumVar = trueVar`; `soundness_star` correctly adds initial `Exact`.

**5. `AGFree` delayed to kraw operands — [OK]**  
Vacuous until both sides are `kraw lit_real`, but `soundness_star` re-obtains `AGFree` from `preservation`+`typed_agfree` on every `⇒*` term, so when congruence finally exposes a redex the inner-zero check is live. No extra AGFree-preservation lemma needed.

**6. Delta-method product / `gMulMeta` — [OK]**  
`scale y a ++ scale x b` is the first-order form of `uv` at `(x,y)`; `trueVar_mul` vs `gMulMeta` differs by exactly `2xy⟨a,b⟩`.

**7. `nsDisjoint` / E230 / annotation quantifiers — [TIGHTENABLE]**  
`⊤` never disjoint matches (Add-Indep)/E230-as-untypable; `x_plus_x_untypable : ∀ Γ T E, ¬ HasTy …` really is “no annotation works” (inversion, not a term-level `N`). `uk` is *not* ⊤-only (form still has source 0; same proof as `x+x`), so `x_plus_top_untypable` does not isolate the ⊤ clause.

**8. Progress / preservation — [OK]**  
Closed typed terms are not stuck where the paper wouldn’t be (`measure` emptyE+`meas_arg`; `kadd`/`kmul` forced through `canon_know`/`canon_real`). `t_sub`/`weakening`/`substClosed` only re-effect values; they cannot shrink `N` or drop `Covers`. (No NS-subtyping at all.)

**9. Claim gaps — [FAIL] missing / [TIGHTENABLE] overclaim**  
Missing vs §6.4: source-level `measure s + measure s` untypability (only the runtime `xk` term); a genuine ⊤-never-disjoint witness; any probabilistic/GUM adequacy (algebraic `⟨a,a⟩` only). Overclaim: comment that `uk` is “typed only at ⊤”; Lemma 2 wording as coeff-support; `AGFree` advertised as “no operator *is* an anti-garbling” while it only constrains already-valued operands. File correctly does *not* claim typing ⇒ `Exact`.


### Disposition

| # | Finding | Action |
|---|---|---|
| 7 / 9 | **Correct catch.** `uk`'s form `[(0,1)]` shares source 0 with `x`, so the old `x_plus_top_untypable` was `x+x` in disguise — it did not isolate the ⊤ clause; the docstring "typed only at ⊤" was false (any `kraw` can be annotated with any superset of its support) | Added the constructor `opaque e` (the paper's `opaque_knowledge()` fixture): `t_opaque : e : tknow T N → opaque e : tknow T ⊤`, `opaque_red` a value no-op (only the *type* forgets), threaded through shift/subst/`Exact`/`AGFree`/progress/preservation/exactness. New `x_plus_top_untypable : ∀ Γ T E, ¬ HasTy Γ (kadd xk (opaque yk)) T E` — `y` is on source 1, disjoint from `x`; `x + y` is admitted (`x_plus_y_typable`) and `opaque y` alone is well-typed (`opaque_y_typable`); the sum is rejected **purely** by ⊤-never-disjoint (`invOpaque` + `nsDisjoint_top_right`). `uk` removed. |
| 9 (missing) | source-level `measure s + measure s` untypability was only shown on the runtime term `xk` | Added `invMeasure` (a `measure` term is typed at exactly `{s}`) and `measure_plus_measure_untypable : ∀ Γ T E, ¬ HasTy Γ (kadd mx mx) T E` for the unreduced program text `mx = measure 10 1 1000 0`. |
| 9 (missing) | "any probabilistic/GUM adequacy (algebraic ⟨a,a⟩ only)" | Scope, stated: Lemma 1 is the algebraic identity for first-order affine forms under independent unit-variance symbols; the calculus does not model distributions. Same boundary as §6.5 first-order. |
| 2 | Lemma 2 wording is the coefficient-based statement | `covers_coeff` (round 1) is exactly that derived statement; the note's theorem map says so. |
| 9 (overclaim) | `AGFree` constrains already-valued operands; docstring "no operator *is* an anti-garbling" | Docstring kept precise: the check fires at the redex (both operands values), which is the only point at which an operator *computes*; `soundness_star` re-derives it at every ⇒* term (referee's own item 5). Wording in the paper table already says "no *reached* … operator". |
| 3, 4, 5, 6, 8 | OK — notably 3: "not circular/vacuous: `Covers` is generated by `meas_red`/`covers_single` and preserved, not merely assumed; `x_plus_y_typable` is a real inhabitant" | none |

Gate after round 2: `NS_METATHEORY_LEAN_GATE_PASS`, 12 theorems in the axiom footprint,
still ⊆ {propext, Quot.sound, Classical.choice}, sorry-free.

**External-review status now:** two adversarial rounds from xAI (Grok 4.5, Grok 4.6): 0
unsound findings; 5 tightenables + 1 missing-witness FAIL, all acted on. Independent
second-vendor opinion (Z.AI) still pending on operator account action.
