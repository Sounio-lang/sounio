<!-- docs:meta
topic_id: repo.docs.research.paper-a-ns-metatheory-xai-review-2026-08-30
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-ns-metatheory-xai-review-2026-08-30
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

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

---

## Round 3 — xai Grok **4.6** (2026-08-31), file with `opaque` (attack the round-2 fixes)

Packet = round-2 packet + item 10 (break the two fixes) + item 11 (round-2 leftovers, report
[CLOSED]/[OPEN]). **Verdict:** 7 × [OK], 2 × [TIGHTENABLE], 2 × [FAIL] (3: labelling
wording; 9: probabilistic adequacy + missing let-bound control). Round-2 fixes: **both
[OK]**; leftovers: source-level witness [CLOSED], genuine ⊤ witness [CLOSED],
probabilistic adequacy [OPEN → scoped; then **measured** 2026-08-31: Var_MC/Var_T = 0.999, decision agreement 99.4 %, `paper_A_rq4_mc_adequacy_2026-08-31.md` — a measurement, not a theorem], `AGFree` wording [OPEN → fixed].

### Referee report (verbatim)

**1. Aff / inner / trueVar.** [OK] `inner a b = Σ_{(s,c)∈a} c·coeff b s` unfolds to `Σ_s coeff_a(s)·coeff_b(s)` even with duplicate keys (`coeff` sums); `trueVar a = ⟨a,a⟩ = Σ_s c_s²` is Var of `Σ c_s ε_s` for i.i.d. unit `ε_s` (constant lives in the payload).

**2. `Covers` membership vs coeff.** [TIGHTENABLE] Membership is strictly stronger than “nonzero coeff ⇒ s∈N” (`covers_coeff`); dummy/`scale 0` monomials inflate `N`. Conservative (more E230), not unsound — no under-approx of true support.

**3. Labels = noise symbols.** [FAIL] Programmer-chosen `s` can tag physically correlated sites with distinct labels; calculus then treats them as `nsDisjoint` and `inner=0` in-model. File’s “sharing is an over-approx, never an under-approx” is false. Not circular: `Covers` is established by `meas_red`/`cert_red` and preserved, not assumed for source terms; it *is* tautological for handwritten `kraw`.

**4. Exact ⇒ reported=true at reached Knowledge.** [OK] `Exact` walks every `kraw` (incl. under `lam`); `exact_shift`/`exact_subst`/`exact_preservation` close beta/let/opaque. `kunc`/`kconf`/`kvalue` peel metadata to bare `treal`/`T` (untracked, but not Knowledge). Well-typed ⇏ `Exact` (`t_kraw` only `kvalid`); that extra hyp is as paper §6.1.

**5. `AGFree` value-only.** [OK] Vacuous until both sides are `kraw`; `soundness_star` re-checks every `⇒*` intermediate, so once congruence has valued the operands the inner-product clause fires. Wording of `typed_agfree` is slightly stronger than the def.

**6. Delta-method product.** [OK] `y·a ++ x·b` is the linearisation of `uv` at `(x,y)`; `trueVar_mul` vs `gMulMeta` differs by exactly `2xy⟨a,b⟩`. (Silent: second-order `a·b` noise dropped — first-order only.)

**7. E230 / untypability vs “every N”.** [OK] `t_kadd`/`t_kmul` = (Add/Mul-Indep) with `⊤` never disjoint. `∀ Γ T E, ¬HasTy (kadd xk xk)` quantifies result `T` (hence operand `N`s by inversion); any covering `N` contains source 0, so `nsDisjoint` dies. Annotations are those in the derivation, not a separate `∀ N`.

**8. Progress / preservation.** [OK] No extra stuck states for closed well-typed terms (`opaque`/`measure` of values step; `kadd` of non-`treal` Knowledge is untypable). `t_sub` mutates only `E`, never `N`; weakening/subst leave `a`/`N` intact.

**9. Claim gap / overclaim.** [FAIL] File does **not** prove probabilistic/sampling adequacy of `trueVar`, nor let-bound `let x=measure s in x+x` as a named control. Overclaim: comment “never an under-approximation”; `typed_agfree` text vs value-restricted `AGFree`. Algebraic Thm 6.4 (`soundness_star`) matches the packet if “true variance” = `⟨a,a⟩`.

**10. Round-3 fixes.**
- (a) `opaque`. [OK] `opaque_red` retypes at `⊤` via `covers_top` (widen, not shrink); no new stuck state; `Exact`/`AGFree` unchanged (ghost form `a` not erased — type-only). `x_plus_top_untypable` + `x_plus_y_typable` + `opaque_y_typable` is a genuine ⊤-clause witness. Residual: after `opaque_red` the *term* `kraw … a` still *admits* a precise `t_kraw` (principal-type instability); harmless inside subject reduction.
- (b) `invMeasure`. [OK] Inversion complete: only `t_measure`/`t_sub`, `T` fixed to `{s}`. [TIGHTENABLE] `measure s + measure s` ≠ let-bound `x+x`; weaker than a §8.2 shared-variable control (though that program is untypable by the same `nsDisjoint` on `t_var`).

**11. Round-2 leftovers.** source-level witness [CLOSED]; genuine ⊤ witness [CLOSED]; probabilistic adequacy [OPEN]; `AGFree` informal wording [OPEN].


### Disposition

| # | Finding | Action |
|---|---|---|
| 3 [FAIL] | the header sentence "sharing a label is an over-approximation, never an under-approximation" reads as a claim that the calculus never under-approximates; with distinct labels on physically correlated sites it *does* | **Wording was wrong, fixed.** The header now states the honest-labelling axiom as an *assumption, not a theorem*: under it, shared labels only over-approximate; without it the calculus under-approximates the true covariance and every theorem is relative to the axiom (= residual (iv)). "The type system tracks sources; it does not discover them." The referee's "tautological for handwritten `kraw`" is accurate and intended: `Covers` is a premise for runtime values and *derived* for source programs (`meas_red`/`cert_red`). |
| 9 [FAIL] / 10b | no let-bound `let x = measure s in x + x` control (the §8.2 *shared-variable* case; `measure s + measure s` is only duplicated text) | Added `invVar`, `invLet`, `var_plus_var_untypable'` and **`let_x_plus_x_untypable : ∀ Γ T E, ¬ HasTy Γ (letE mx (kadd (var 0) (var 0))) T E`** — (Measure) fixes `x : Knowledge⟨ℝ,{0}⟩`, both `var 0` look up the same `{0}`, `nsDisjoint {0} {0} = false`. This is the paper's `x + x` control verbatim (de Bruijn). |
| 9 [FAIL] / 11 | probabilistic / sampling adequacy of `trueVar` not proved | **Scoped, not claimed.** Header now says: `trueVar a = ⟨a,a⟩` is the variance of `Σ c_s ε_s` under independent unit-variance symbols *by definition*; no distributional semantics is modelled; Lemma 1 and Theorem 6.4 are algebraic. Matches the paper (Lemma 1 is an algebraic identity; §6.5 first-order). |
| 9 / 11 | `AGFree` informal wording stronger than the value-restricted definition | Docstrings of `AGFree` and `typed_agfree` and the header now say "whose operands are runtime values", with the reason (an operator computes only at a redex; `soundness_star` re-derives it on every reduct — the referee's own item 5). |
| 10a | `opaque` fixes | [OK] — "genuine ⊤-clause witness"; residual noted: after `opaque_red` the term `kraw … a` again admits a precise `t_kraw` (principal-type instability), "harmless inside subject reduction" — recorded, no action. |
| 2 | membership vs coefficient `Covers` | as round 1/2: `covers_coeff`; conservative direction only. |
| 6 | second-order product noise dropped | scope, §6.5. |

Gate after round 3: `NS_METATHEORY_LEAN_GATE_PASS`, **13** theorems in the footprint,
⊆ {propext, Quot.sound, Classical.choice}, sorry-free.

**External-review status:** three adversarial rounds from xAI (4.5, 4.6, 4.6-on-fixes).
Cumulative: **0 unsound findings**; every [FAIL]/[TIGHTENABLE] either closed by a new theorem
(`covers_coeff`, `opaque`+`x_plus_top_untypable`, `measure_plus_measure_untypable`,
`let_x_plus_x_untypable`) or converted into an explicitly stated boundary (honest labelling,
algebraic scope, first-order). Independent second-vendor opinion (Z.AI) still pending on
operator account action.

---

## Independent second vendor — **Kimi K3** (Moonshot), 2026-08-31

**Route:** the Kimi Code K3 lane (`fleet:10`, kimi-cli2 auth) via tmux dispatch — `kimi-code
0.38.0 -p` crashes on any tool-using prompt, and the `api.moonshot.ai` key is not provisioned;
Z.AI remained locked (1313). Same packet as round 3 (file as merged to `main` `cd325f66fc`,
12 items; item 12 = "you are the independent vendor, no xAI text provided"). The referee
confirms in item 12 that no prior review content was visible.

**Verdict:** 10 × [OK], 2 × [TIGHTENABLE], **0 × [FAIL]**. Round-2/3 fixes: [OK]. Leftovers:
source-level witness [CLOSED], genuine ⊤ witness [CLOSED], `AGFree` wording [CLOSED],
probabilistic adequacy [OPEN — "a scoped gap rather than a defect"].

### Referee report (verbatim)


1. [OK] — `inner a b` sums `c · coeff b s` per entry; grouping duplicate entries of `a` by `s` gives `Σ_s coeff a s · coeff b s` since `coeff b s` is fixed within each group; the bilinearity/commutativity the file actually consumes (`inner_append_left/right`, `inner_comm`) are proved, and `trueVar a = ⟨a,a⟩` is exactly the declared unit-variance-independent model.

2. [OK] — membership-based `Covers` is strictly *stronger* than coefficient-based containment (zero-coefficient entries still force membership), which is the safe direction: it can only over-reject, never under-reject; `covers_coeff` discharges the bridge to Lemma 2's "actually carries uncertainty" wording. No unsoundness direction. Residual completeness cost (not a flaw): `scale 0 b` from a zero payload yields phantom `(s,0)` entries that stay in the tracked set and can block later disjoint sums whose true covariance is genuinely 0.

3. [OK] — two physically correlated sites are treated as disjoint only by violating the declared honest-labelling axiom (distinct labels); the file states this and the under-approximation direction explicitly (§6.4 residual iv). Not circular: `Covers` as a `t_kraw` premise is the invariant, `support_over_approx` just extracts it; ⊤-coverage is admissible but self-defeating since ⊤ is never disjoint. Nice corroboration that laundering is blocked: `t_certain`/`t_measure` require `emptyE` subterms, so `certain (measure …)` is untypable — the effect discipline closes the obvious evasion. (Edge: source-level `kraw` literals can fabricate metadata — covered by Theorem 6.4's stated Exact hypothesis.)

4. [OK] — `Exact` is an explicit hypothesis of `soundness_star` (= the paper's "initially report their true variance"), not smuggled into `t_kraw` (which requires only `Covers` + `kvalid`); `exact_subst`/`exact_shift` close substitution/shifting; `kvalue_red` unwraps with `Exact v` preserved; `kunc_red`/`kconf_red` exit to bare `lit_real` — no metadata channel survives, and what they read out equals the true variance precisely because `Exact` held pre-step.

5. [OK] — `kadd_red`/`kmul_red` fire only on `kraw (lit_real _)` operands, and well-typedness forces `nsDisjoint` on the annotations, which via `Covers` + `inner_disjoint` yields `inner = 0` exactly at the redex; the vacuous window (reached operator, operands not yet values) is benign because computation requires values and `soundness_star` re-establishes `AGFree` on every reduct. Additionally: the AGFree obligation for non-`lit_real` payloads is vacuous but unreachable, since `t_kadd`/`t_kmul` force `tknow treal` and `canon_real` collapses payloads to `lit_real`.

6. [OK] — `scale y a ++ scale x b` is the correct first-order delta form of the product at the observed point `(x,y)` (gradient `(y,x)`), and `trueVar_mul` gives `y²Var a + x²Var b + 2xy⟨a,b⟩` with `gMulMeta` dropping exactly the `2xy` term; linearization at the pre-step payloads is the right point.

7. [OK] — `x_plus_x_untypable : ∀ Γ T E, ¬ HasTy Γ (kadd xk xk) T E` quantifies over *all derivations*, and every admissible annotation must cover source 0 (`support_over_approx`), so both operands always share 0 — including ⊤, handled by the `none` cases of `nsDisjoint_of_shared`; "for every annotation N" is fully captured. The Bool `nsDisjoint` premise is the decidable form of (Add-Indep); Lemma 2 is what makes the checked form imply the paper's rule.

8. [TIGHTENABLE] — progress and preservation both hold (`t_sub` is effects-only, no N-subsumption anywhere; `weakening`/`substClosed` are the standard development; `canon_know`+`canon_real` close the kadd/kmul value cases; `kadd` of non-real Knowledge is ill-typed rather than stuck). But `meas_red`/`t_measure` accept payloads of *any* type, including arrows: `measure (lam …)` steps to a `kraw`-wrapped lambda whose "variance" `c²` is formally exact and semantically meaningless. Harmless to every proved theorem; a one-line payload restriction (`treal`/`tnat`) would align the calculus with §6's intent.

9. [TIGHTENABLE] — Not proved (named, all honestly scoped in-file): (i) `treal` is modelled by `Int` — the mechanization is ring algebra over integer "reals"; the paper's prose must not claim ℝ-valued measure theory; (ii) "true variance" is the algebraic `⟨a,a⟩` under declared independent unit-variance symbols — no sampling/distributional semantics (declared in the design notes; keep the qualifier in §6.4 prose); (iii) Theorem 6.4's Exact hypothesis is load-bearing against fabricated source-level `kraw` literals — do not drop the hypothesis when citing. No overclaim found in the file: the `AGFree` value-restriction and the labelling axiom are both disclosed.

10. (a) [OK] — `opaque_red` preservation re-types at ⊤ via `covers_top`, sound since `Exact`/`AGFree` are N-independent and both recurse through `opaque` unchanged; no new stuck state (`canon_know` + progress); no annotation shrink is possible — `invOpaque` forces `nsTop` and `t_sub` cannot touch `T`. `x_plus_top_untypable` is a genuine ⊤-clause witness: both controls are present (`x_plus_y_typable` admits the un-opaqued sum, `opaque_y_typable` types `opaque yk` alone), so the rejection is attributable to the ⊤ clause of (Add-Indep) alone (`nsDisjoint_top_right`). (b) [OK] — `invMeasure` is complete: only `t_measure`/`t_sub` derive `measure` terms and `t_sub` preserves `T`, and the system has no N-widening rule, so a `measure` term can carry no annotation other than `{s}`; with `measure_plus_measure_untypable` and `let_x_plus_x_untypable` (the shared-de-Bruijn-variable control), the source-level witnesses match §8.2's "x + x flagged" at program-text level.

11. source-level witness: [CLOSED] (`measure_plus_measure_untypable`, `let_x_plus_x_untypable`, both `∀ Γ T E`). genuine ⊤ witness: [CLOSED] (`x_plus_top_untypable` with both controls). probabilistic adequacy: [OPEN] — still no distributional semantics; variance is the declared algebraic proxy; correctly labelled in-file, so a scoped gap rather than a defect. *(Measured the same day on the RQ4 cohort: Var_MC/Var_T = 0.999, 99.4 % decision agreement — see `paper_A_rq4_mc_adequacy_2026-08-31.md`.)* `AGFree` wording: [CLOSED] — header discloses the value restriction and `soundness_star` re-establishes it per step.

12. [OK] — verdicts above were derived from the packet alone; no xAI review text was provided or consulted (the file's own comments attribute earlier fixes to prior rounds, but no review content was visible).

### Disposition

| # | Finding | Action |
|---|---|---|
| 8 | `measure`/`meas_red` accept payloads of any type, including arrows; a `kraw`-wrapped lambda has a formally exact but meaningless variance | **Kept generic, documented.** `Knowledge⟨T,N⟩` is generic by design (parity with V2's `Knowledge<T>` and the paper's §5.1); `kadd`/`kmul` are typed at `treal` only, so non-numeric metadata is inert — it can never enter an arithmetic redex. Stated in the header. |
| 9 (i) | `treal` is modelled by `Int`: ring algebra over integer "reals" | **Boundary (vi), stated** in the header and §6.4: no ℝ-valued measure theory is claimed; first-order propagation is a polynomial identity, for which `Int` is exact. |
| 9 (ii) | variance is the algebraic `⟨a,a⟩` under declared independent unit-variance symbols | already boundary (v); kept. |
| 9 (iii) | Theorem 6.4's `Exact` hypothesis is load-bearing against fabricated source-level `kraw` literals | **Stated** in the header: `Exact` is a hypothesis of `soundness_star`, not a theorem — do not drop it when citing. |
| 2 (residual) | `scale 0 b` leaves phantom `(s, 0)` monomials that stay in the tracked set and can block a later genuinely-disjoint sum | completeness cost, not soundness; noted. `covers_coeff` is the direction that matters. |
| 3 (corroboration) | `t_certain`/`t_measure` require `emptyE` subterms, so `certain (measure …)` is untypable — the effect discipline closes the laundering evasion | recorded as an observation we had not made ourselves. |

Gate after the header edit: `NS_METATHEORY_LEAN_GATE_PASS`, 13 theorems.

**External-review status, final for this cycle:** four adversarial reads — xAI Grok 4.5, Grok
4.6, Grok 4.6-on-fixes, and **Kimi K3 as the independent second vendor**. Cumulative: **0 unsound
findings**; every [FAIL]/[TIGHTENABLE] closed by a theorem (`covers_coeff`, `opaque` +
`x_plus_top_untypable`, `measure_plus_measure_untypable`, `let_x_plus_x_untypable`) or stated as an
explicit boundary (honest labelling; algebraic scope; first order; `Int` reals; generic payloads;
`Exact` as hypothesis). The two-vendor policy of `.claude/AGENT_OFFLOAD_POLICY.md` is satisfied.
