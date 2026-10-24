<!-- docs:meta
topic_id: repo.docs.decisions.adr-016-epistemic-merge-and-provenance-trust
authority: repo_only
audience: users
last_validated: 2026-10-24
validated_by: formal/TrustSemiring.lean; docs/audit/EPISTEMIC_MERGE_2026-10-24/peelle.cpp
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-016-epistemic-merge-and-provenance-trust
-->

# ADR-016: Epistemic merge, contradiction, typed zeros and provenance trust

**Status**: accepted (implementation pending)
**Date**: 2026-10-24

## Context

ADR-015 made `⊥` and `⊤` tokens of their own and left their meaning open. Fixing that meaning exposed a hazard: the same symbol names opposite things in the two orders Sounio uses.

- In the **subsumption** order of the ontology runtime, `⊥` is the empty concept (contradiction) and `⊤` is everything.
- In the **information** order that `Knowledge<T>` lives in, `⊥` is "nothing known" and `⊤` is contradiction.

Under the theory–model Galois connection these two orders are antitone, so the two `⊥` sit at opposite ends. A grammar that let context pick the meaning silently would compute a lie of exactly the kind the language exists to refuse. The founder's intent for `0_⊥` in the ontology-algebra record was a zero that **moves** from one reading to the other as evidence accumulates. This ADR gives that trajectory a type, and settles everything the trajectory touches: how two sources merge, what a contradiction is, which zeros exist, and how provenance carries trust.

## Decision

The founder took these decisions on 2026-10-24.

### 1. `⊥` and `⊤`

1. `⊥` and `⊤` are nullary tokens, like `true` and `false`.
2. They are valid only in a type that **declares its order**; the declared order fixes their meaning. In a type with no declared order, using them is an error.
3. `Knowledge<T>` declares the information order: `⊥` is **Absent** (decision 10) and `⊤` is contradiction, which is realisable only as `Contest<T>` (decision 2).
4. Concept types of the ontology runtime declare the subsumption order: `⊥` is the empty concept.
5. `0_⊥` is not a constant. It names the trajectory of decision 11.
6. A statistical-independence operator, if one is ever added, is spelled `⫫` (U+2AEB), never `⊥`.

### 2. Contradiction lives in `Contest<T>`, never in `Knowledge<T>`

1. A function that receives `Knowledge<T>` can never receive a contradiction.
2. `meet(a, b) -> Knowledge<T>` fuses two sources. On the certified layer (`e`, the certified interval of D-34) an empty intersection is a contradiction, and `meet` **refuses** with a location and both provenances.
3. `contest(a, b) -> Contest<T>` always succeeds and carries both sources with their provenance. Inside `Contest<T>`, `⊤` is an ordinary value.

### 3. Consistency on the statistical layer (`u`)

1. Overlapping distributions are never formally contradictory, so `meet` measures **degree** of incompatibility: the normalised error `E_n = |x₁ − x₂| / √(U₁² + U₂²)` (ISO 13528) for two sources, the Birge ratio for more.
2. Above the threshold, `meet` **refuses** and reports `E_n` and both sources.
3. The threshold is an **explicit policy** on the type or the call (`consistency: En(k)`); the default `En(1.0)` is stated here and is never hidden in the runtime.
4. Birge expansion of the combined uncertainty exists only as an explicit policy (`consistency: Birge`) and is recorded in provenance. It is a modelling choice about the cause of the disagreement, so the programmer makes it, not the language.
5. Accepting inconsistent sources and returning a narrower interval than either input is never done.
6. Because `E_n` depends on measured values, the refusal of `meet` is in general a **run-time contract**, not a compile-time check. It becomes a static guarantee only through the typestate of decision 11, where a certificate proves the consistency predicate over a certified enclosure.

### 4. Resolving a `Contest<T>`

1. A contest is resolved only by **new evidence**, never by recomputing over the same data:
   - **Third source**: leave-one-out consistency identifies the outlier; requires the lower bound of `n_eff` (decision 5) to be at least 3;
   - **Traceability**: a source with a certified, higher metrological chain prevails;
   - **Prior**: a declared reliability prior, recorded in provenance.
2. With two sources and nothing else, the conflict is not identifiable (either source's bias explains the data equally), and the type refuses to resolve it.
3. `retract(c, source, because: evidence)` is the only exit from contradiction. It is explicit and recorded; the retracted source stays in provenance and never disappears.

### 5. Independence of sources

1. Provenance is a graph: each source points to its calibrant, reference material, method, instrument model and software version.
2. Sharing an **instance** (the same reference-material lot) shares a Type B component and correlates the sources: `r = u_shared² / (u₁ u₂)`.
3. Sharing only a **class** (the same method, different lots) gives a partial correlation **declared by the method** (for example from its interlaboratory validation).
4. The effective number of independent sources is `n_eff = 1ᵀ R⁻¹ 1`.
5. Where provenance is unknown, `n_eff` is an **interval**, computed with `r = 1` and with `r = 0` for the unknown pairs. Resolution by a third source uses the lower bound.

### 6. Uncertainty budget representation

Type A components stay diagonal. Type B components carry a **provenance key**, so correlation between sources is exact where they share a key. This refines the 2026-09-23 Madaros design (diagonal by default, `Correlated` inferred by type): shared provenance is an operation that correlates.

### 7. Negative GLS weights

When the generalised-least-squares combination gives a negative weight, `meet` **refuses** with a Peelle diagnostic: the weights and the shared provenance keys. The exits are `contest`, or an explicit model of the correlated error (decision 8).

### 8. Multiplicative error

1. The **dimension** of a component decides its error model. A component in the units of the quantity is additive; a dimensionless relative component (calibrant purity, volumetric tolerance, normalisation) is multiplicative.
2. A shared multiplicative component is combined in the **log domain** when the quantity is proven positive (decision 9); otherwise as an explicit nuisance parameter with iterated GLS.
3. Covariance of a relative component is evaluated at the **estimate**, never at the measured values.
4. Witness (Peelle 1987: `1.5` and `1.0`, 10 % independent, 20 % shared normalisation), `docs/audit/EPISTEMIC_MERGE_2026-10-24/`:

| model | result | weights |
|---|---|---|
| additive GLS, covariance at measured values | 0.8824 ± 0.2183 (below both inputs) | −0.2353, +1.2353 |
| log domain | 1.2247 ± 0.2598 (21.21 %) | +0.5000, +0.5000 |

### 9. Positivity

1. Positivity is a **support refinement** of `Knowledge<T>` (`supp ⊆ (0, ∞)`), consistent with the 2026-09-01 support-refined `Knowledge` decision.
2. It comes from the **quantity kind** (concentration, mass, dose), is preserved by closure (`×`, `÷`, `+`, `exp`), and is lost by subtraction unless the certified lower bound of the difference is proven greater than 0.
3. A positive support requires a **positive-support family** (log-normal by default). A Gaussian on a positive quantity is a type error when `u / x̂` exceeds a stated policy threshold (default 10 %); below it the Gaussian approximation is allowed and recorded in provenance.

### 10. Typed zeros

Four zeros become first-class types:

| zero | meaning | rule |
|---|---|---|
| **Structural** | exactly 0 by definition (the drug was not given) | a certified point; enters any computation |
| **Censored** (`< LOQ`) | somewhere in `[0, LOQ)` | an interval; `log` gives `(−∞, log LOQ)`; substitution (for example `LOQ/2`, Beal's M5/M6) is refused; the likelihood path (Beal's M3) is the supported one |
| **Absent** | not measured | `⊥` of the information order; never converted to 0 without an explicit, recorded decision |
| **Numerical** (`0_ε`) | underflow below the format | a rounding bound, carried in the rounding component of D-34 |

The ontological `⊥` (empty concept) belongs to concept types (decision 1.4), not to this table.

### 11. The trajectory as a typestate

1. A value's stage moves along the information order: **Absent → Knowledge → Contest**. This is the trajectory `0_⊥` names.
2. Stages are indexed by **evidence predicates** (dependent types). For example, resolving a contest by a third source requires a proof that the lower bound of `n_eff` is at least 3.
3. **Search outside, certificates inside.** The checker emits each proof obligation with the hash of the code that generated it. An external, swappable prover searches for the proof and writes a **certificate**; the compiler only **checks** certificates, with a small deterministic kernel. A missing or stale certificate is a compile error that names the obligation.
4. The compiler never calls a solver, so the build stays deterministic and bit-identical, and no third-party solver is a compiler dependency. A third-party solver may be used as a developer tool; a prover written in Sounio may replace it later without changing what is trusted.
5. Predicates over floating-point values are proven over the **certified enclosure** of D-34 (Marco 0), not over the real number the code approximates. Certificate forms: Farkas multipliers (linear), sum-of-squares identities (polynomial), re-run interval evaluation with a fixed rounding mode (float predicates).
6. The certificate checker's soundness is to be proven in Lean.

### 12. Provenance trust

1. Trust levels: **signed** > **inferred** > **assumed**. A signed certificate beats an ontology inference, which beats an `assume`; an `assume` is always recorded and lowers `Q`.
2. The ontology is the medium of representation. Each fact carries its trust level, and each conclusion carries the trust of its premises through a provenance semiring (Green, Karvounarakis and Tannen 2007).
3. The annotation is, for each level, the **fewest inference steps** of any derivation reaching at least that level: the product of three tropical semirings `(min, +)` on `Option Nat`. `⊕` is pointwise min, `⊗` is pointwise sum.
4. Proven in `formal/TrustSemiring.lean`, without Mathlib: commutative semiring laws, distributivity, idempotence of `⊕` (no double counting of correlated derivations), absorption `1 ⊕ x = 1` (the 0-stability under which Datalog fixpoints converge), and preservation of well-formedness. `#print axioms`: `propext`, `Quot.sound` only.
5. The lexicographic "pick the best derivation" design is rejected: it is not distributive. The counterexample is `Trust.lex_not_distrib` in the same file.
6. The ontology runs under the open-world assumption, so independence is never derivable from it alone. A signed certificate that states its chain is **complete** closes the world for that chain; only then can independence be proven. Without it, decision 5.5 applies.

## Prior art and claimed contribution

An adversarial prior-art search on 2026-10-24 tried to kill each part of this ADR. Every mechanism below already exists and is cited, not claimed.

| mechanism | prior art |
|---|---|
| a value carried with its GUM budget as a type; correlation arising from shared inputs tracked by persistent identifiers | GUM Tree Calculator (Hall 2022); METAS UncLib; `Uncertain<T>` (Bornholt, Mytkowicz, McKinley, ASPLOS 2014) |
| `n_eff = 1ᵀR⁻¹1`; `E_n` (ISO 13528) and the Birge ratio as consistency tests | standard statistics and metrology; analyst-side calculators |
| the statistical remedies for Peelle's puzzle: normalisation as an explicit parameter, covariance at the estimate, iterated reweighting, log-normal encoding of relative uncertainty, log-normal above a relative-uncertainty threshold | D'Agostini 1994; Chiba & Smith 1991; Lachaume 2021; Walton et al., *Sci. Rep.* 2025; AMC Technical Brief 88 (2019) |
| contradiction as the top of an information order; retraction of a premise from a contradiction | Fitting, bilattice logic programming; LVars (Kuper et al., POPL 2014); propagators with a truth-maintenance system (Radul & Sussman; Radul 2009) |
| dependently typed typestate with evidence-carrying transitions | Idris `ST` |
| untrusted search, small certified checker | SMTCoq (Ekici et al. 2016) |
| bounds on float expressions proven by interval computation inside a proof checker | Melquiond 2008; Martin-Dorel & Melquiond 2015 |
| tropical provenance; trust conditions over provenance | Green, Karvounarakis, Tannen 2007; Green et al., VLDB 2007 |
| the product of tropical semirings | standard algebra; the Lean file is a machine check, not a discovery |

**Claimed contribution.** Each of these is, in current practice, a choice the analyst makes or a check that runs and reports at execution time. This ADR makes them **obligations the language imposes**:

1. fusion split into a `meet` that refuses (with location and both provenances) and a `contest` that always succeeds and carries the conflict, with the consistency threshold as an explicit type- or call-level policy;
2. the physical dimension of an uncertainty component, already tracked by the unit system, selecting the additive or multiplicative error model, with positivity as a support refinement selecting the log-domain or nuisance-parameter route, so that Peelle's puzzle cannot arise silently;
3. contradiction lifted from a run-time state to a static typestate, so that `Knowledge<T>` cannot hold a contradiction by construction, and the exit from `Contest<T>` untypable without a certificate of identifiability (in particular, refused with two sources and no further evidence);
4. partial correlation declared by a shared method class, and `n_eff` as an interval under unknown provenance;
5. ranked trust levels as a provenance semiring, which the VLDB 2007 work left as future work.

No search found a precedent for these as language-level obligations; absence after a bounded search is not proof of absence. Not yet checked: GUM Tree Calculator and METAS UncLib documentation on relative uncertainty in full, Neudecker et al. 2014 in full, Radul's thesis in full, Plaid and the original Strom & Yemini typestate papers, IEEE 1788, and knowledge-fusion / truth-discovery literature.

## Consequences

- Decision 11 depends on D-34 (Marco 0): dependent proofs over floating-point values are sound only over a certified enclosure. Marco 0 is now on the critical path of this ADR.
- PBPK work that substitutes `LOQ/2` for below-quantification data stops compiling. The supported path is Beal's M3 likelihood.
- Merging sources becomes stricter by default: inconsistent sources refuse instead of producing a falsely precise number. Every relaxation is an explicit, recorded policy.
- `0_⊥` in the ontology-algebra research record now means the trajectory of decision 11; the record should cite this ADR.
- The kill test of the ontology-algebra line ("provenance-semiring fixpoint tractable at `elplus.sio` scale") now has a fixed semiring to run against.

## Implementation

Not started. Proposed order: the trust semiring in the ontology runtime (decision 12); the typed zeros (decision 10); `meet`/`contest` with the `E_n` policy (decisions 2–3); budget keys and `n_eff` (decisions 5–6); then the typestate and certificate kernel (decision 11), after Marco 0.

## Grounded in

- `formal/TrustSemiring.lean` (built by the `formal/` Lake package).
- `formal/EpistemicSafety.lean`: a minimal certified-layer calculus (`Knowledge`, `Contest`, `meet`, `contest`, `resolve`) with explicit refusals, and its theorems: a well-typed `Knowledge` term never evaluates to a contradiction (`safety`); a successful `meet` implies its inputs intersect (`meet_no_silent`); disjoint inputs refuse (`meet_refuses`); resolution succeeds only when the evidence is compatible with exactly one side (`resolve_identifiable`, `resolve_refuses_both`). `#print axioms`: `propext` only. - `formal/EpistemicStat.lean`: the statistical layer. The `E_n ≤ num/den` policy is decided exactly on squares, `(x₁ − x₂)² · den² ≤ num² · (U₁² + U₂²)`, with no real or float in the decision. Theorems: a successful `meet` satisfies the declared policy (`meet_sound`); above it `meet` refuses (`meet_refuses`); the decision is symmetric (`consistent_comm`); loosening the policy never turns acceptance into refusal (`consistent_mono`). Witness checked by `decide`: 100 ± 4 vs 112 ± 5 (E_n ≈ 1.87) is refused under the ISO 13528 default and accepted under `E_n ≤ 2`.
- `formal/EpistemicJoint.lean`: both layers in one `Knowledge` value. A joint `meet` succeeds only when the certified and the statistical layer both accept (`meet_sound`), its interval is never a contradiction (`meet_ok_interval`), and a refusal names its layer, certified first (`meet_refuses_certified`, `meet_refuses_statistical`). The statistical contest resolves by a third source only when it is consistent with exactly one side (`resolve_identifiable`, `resolve_refuses_both`). Witness by `decide`: A = 100 ± 4 and B = 112 ± 5 conflict; C = 111 ± 4 is consistent with B only.
- `docs/audit/EPISTEMIC_MERGE_2026-10-24/peelle.cpp` and its README.
- Belnap N. D. (1977), "A useful four-valued logic". Ginsberg M. L. (1988), "Multivalued logics: a uniform approach to reasoning in AI", *Computational Intelligence* 4.
- Green T. J., Karvounarakis G., Tannen V. (2007), "Provenance semirings", PODS.
- Abo Khamis M., Ngo H. Q., Pichler R., Suciu D., Wang Y. R. (2022), "Convergence of Datalog over (pre-)semirings", PODS.
- Peelle R. W. (1987), "Peelle's Pertinent Puzzle", ORNL memorandum. Chiba S., Smith D. L. (1991), ANL/NDM-121.
- ISO 13528 (normalised error `E_n`). Birge R. T. (1932), *Physical Review* 40.
- Beal S. L. (2001), "Ways to fit a PK model with some data below the quantification limit", *J. Pharmacokinet. Pharmacodyn.* 28.
- Hall B. D. (2022), "The uncertain number: a data model for measurement", IMEKO TC6 M4Dconf, Berlin. https://www.m4dconf2022.ptb.de/fileadmin/documents/m4dconf2022/Material/Paper/IMEKOTC6-M4Dconf-2022-P17-HALL.pdf
- Bornholt J., Mytkowicz T., McKinley K. S. (2014), "Uncertain<T>: a first-order type for uncertain data", ASPLOS.
- D'Agostini G. (1994), "On the use of the covariance matrix to fit correlated data", *Nucl. Instrum. Methods A* 346, 306–311. https://doi.org/10.1016/0168-9002(94)90719-6
- Chiba S., Smith D. L. (1991), OSTI report. https://doi.org/10.2172/5881035
- Lachaume R. (2021), "Bias-free model fitting of correlated data in interferometry", arXiv:2104.07082.
- Walton N. A. W. et al. (2025), "Understanding Peelle's Pertinent Puzzle bias in generalized least squares regression through eigenspectrum analysis", *Sci. Rep.* 15, 40885. https://doi.org/10.1038/s41598-025-24706-0
- Analytical Methods Committee (2019), AMCTB No. 88, "Why do we need the uncertainty factor?", *Anal. Methods*.
- Kuper L., Turon A., Krishnaswami N. R., Newton R. R. (2014), "Freeze after writing: quasi-deterministic parallel programming with LVars", POPL.
- Radul A., Sussman G. J., "Revised report on the propagator model", MIT; Radul A. (2009), *Propagation Networks*, PhD thesis, MIT. https://dspace.mit.edu/handle/1721.1/49525
- Ekici B. et al. (2016), "Extending SMTCoq, a certified checker for SMT", arXiv:1606.05947.
- Melquiond G. (2008), "Proving bounds on real-valued functions with computations", IJCAR. Martin-Dorel É., Melquiond G. (2015), *J. Autom. Reasoning* 57(3), 187–217. https://doi.org/10.1007/s10817-015-9350-4
- Green T. J., Karvounarakis G., Ives Z. G., Tannen V. (2007), "Update exchange with mappings and provenance", VLDB.
