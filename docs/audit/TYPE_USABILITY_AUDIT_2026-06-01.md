# Sounio Type Usability Audit (2026-06-01)

**No: of the 94 audited Sounio types, only 31 (33%) are usable end-to-end; 28 parse but cannot produce a running typed value, and 35 are dead enum slots unreachable from source — so most of the epistemic/causal/decision type surface is not usable today.**

(Note: the audit brief referenced "all 98 Sounio types"; that 98 over-counted by ~4 comment-word false positives in the enum scan. There are **94** distinct `TypeKind` variants, all audited.)

## Critical clarifications (do not misread the verdicts)

- **`TyEpistemic` DEAD ≠ `with Epistemic` broken.** The `with Epistemic(N)` *effect*
  gate is a separate, fully-working mechanism (56 files use it). The verdict here is
  about the *type* `Epistemic<T>` (`ty_epistemic()` has **0 callers** → the TypeKind
  slot is unreachable). The working epistemic *type* is `TyKnowledge` (`Knowledge<T>`),
  which is USABLE.
- **DEAD TypeKind ≠ name unusable.** `Sample`, `Distribution`, etc. exist as ordinary
  **userland structs** (resolve to `TyNamed`, USABLE) — but the *special* `TySample` /
  `TyDistribution` TypeKinds those names were meant to denote are dead. Likewise
  `Gradient`/`Jacobian` appear only in one frontend (parse-only) test, not as the
  special TypeKinds.
- **PARSE_ONLY = real syntax, no runtime.** The dedicated keyword/parse path is
  genuine; what's missing is a value constructor or a producing chain that
  type-checks (e.g. the whole policy/plan family decls compile but their producer
  builtins — `contest`, `plan_acquisition`, `admit_action` — resolve as unknown
  identifiers under the canonical souc).

## Summary

| Verdict | Count | % of 94 | Meaning |
|---|---|---|---|
| USABLE | 31 | 33% | Surface name resolves to the real type AND a typed value compiles + runs end-to-end |
| PARSE_ONLY | 28 | 30% | Dedicated keyword/parse path is genuine, but no value reaches runtime (no constructor, or producing chain fails to typecheck) |
| DEAD | 35 | 37% | Enum slot unreachable from source; bare annotation is silently accepted as an unknown generic (byte-identical to a bogus control) |
| UNCERTAIN | 0 | 0% | — |
| **Total** | **94** | **100%** | |

## USABLE (31)

- TyKnowledge (Knowledge<T>) — keyword + parse path + `Knowledge(...)` constructor; witness `k.value` compiled and ran rc=0.
- TyAleatoric (Aleatoric<T>) — keyword + lowering; mismatched Knowledge→Aleatoric coercion raised type-specific P0003 a bogus name never would.
- TyUnobserved (Unobserved<T>) — no token but real nominal resolution; `u == 7` raises type-specific E36 observation-boundary error.
- TyValidation (validation Name for T {}) — `validation` keyword + item parse; manifest chain ran rc=42 (expected).
- TyDiffPrivate (DiffPrivate) — keyword + `lower_diffprivate_type`; witness ran rc=7 with byte-distinct ELF (type-specific lowering, not silent accept).
- TyDPBudget (DPBudget) — keyword + `lower_dp_budget_type`; witness compiled and ran rc=7 via distinct path.
- TyHyper (Hyper<Algebra,T>) — dedicated `Hyper` name match + algebra resolution + IR serialization; Quaternion and Clifford(1,3) witnesses compiled/ran rc=0 (field access `.w` is a gap, but inner value usable).
- TyProof (Proof) — `Proof` keyword → `lower_proof_annotation`; `f(x: Proof<i64>)` compiled and ran rc=0 (carrier is inner type).
- TyLemma (Lemma) — `Lemma` keyword → `lower_lemma_annotation`; witness ran rc=0.
- TyAxiom (Axiom) — `Axiom` keyword → `lower_axiom_annotation`; witness ran rc=0.
- TyI64 (i64) — nominal resolution; arithmetic witness rc=0.
- TyF64 (f64) — nominal resolution; constructed/used, rc=0.
- TyBool (bool) — nominal resolution; `if c` branch printed and ran rc=0.
- TyUnit (()) — dedicated parse path; `let u:()=nothing()` ran rc=0.
- TyString (string) — `string` ident + StringLit; `"hi"` printed, rc=0.
- TyChar (char) — built via `65 as char` round-trip (NB: literal `'A'` rejected E200; construction is via cast).
- TyArray ([T; N]) — dedicated sized-array parse; `[i64;3]` indexed sum=60, rc=0.
- TyRef (&T) — `&` → `parse_ref_type` → ty_ref; `*x` through `&i64` = 77, rc=0.
- TyRefMut (&!T) — `&!` → ty_ref_mut; sized `&![i64;3]` mutated sum=66 (NB: scalar `&!i64` write-back broken).
- TySlice (&[T]) — unsized `&` → ty_slice; `sum(s:&[i64])` = 15, rc=0.
- TySliceMut (&![T]) — unsized `&!` → ty_slice_mut; in-place fill sum=66, rc=0.
- TyTuple ((T, U)) — comma → TypeTuple; `(99,true).0` = 99, rc=0.
- TyFn (fn(T,...)->U) — `fn` → parse_fn_type → TyFn; higher-order apply = 9, rc=0.
- TyNever (!) — `!` dedicated parse; divergent `fn boom()->!` compiles/runs rc=0 (no value by definition).
- TyI8 (i8) — nominal; participated in sum=87, rc=0.
- TyI32 (i32) — nominal; used in arithmetic, rc=0.
- TyU8 (u8) — nominal; sum=87, rc=0.
- TyU32 (u32) — nominal; used, rc=0.
- TyU64 (u64) — nominal; sum=87, rc=0.
- TyF32 (f32) — nominal; constructed/used, rc=0.
- TyNamed (Foo struct/enum) — fallback parse; `Point{x:3,y:4}` field sum=7, rc=0 (realness proven by struct construction, not annotation).

## PARSE_ONLY (28)

- TyIntervention (Intervention) — keyword + `parse_epistemic_wrapper_type` → ty_intervention; annotation parses but no value constructor (compiles like bogus).
- TyCounterfactual (Counterfactual) — keyword + parse path → ty_counterfactual; no expression-level constructor.
- TyCausalEffect (CausalEffect) — keyword + `lower_causal_effect_annotation`; no value constructor anywhere.
- TyPotentialOutcome (PotentialOutcome) — keyword + `lower_potential_outcome_type`; only caller is annotation lowering.
- TyPolicy (Policy<T>) — token + parse path; `policy ... {}` item compiles but consuming `prove_robust` chain fails (`unknown identifier contest`).
- TyDecisionPolicy (DecisionPolicy<T>) — token + parse; item compiles but `admit_action(...)` is `unknown identifier`.
- TyDeferralPolicy (DeferralPolicy<T>) — token + parse; item compiles but `defer_action(...)` chain fails typecheck.
- TyValidated (Validated<T>) — token + parse; `validate_manifest` chain fails at preceding `contest`; bare annotation non-load-bearing.
- TyAdmissible (Admissible<T>) — token + parse; `admit_action(...)` unresolved, typecheck failed.
- TyDeferred (Deferred<T>) — token + parse; `defer_action(...)`/`deferral_reason` unknown identifiers.
- TyContest (Contest<...>, `contest [..] on subj`) — tokens + dedicated `parse_contest_expr`; canonical souc still resolves `contest` as unknown identifier, typecheck fails.
- TyRobust (Robust<...>) — token + parse; builtin `prove_robust` never reached because required `contest` operand is unknown.
- TyAcquisitionPolicy (AcquisitionPolicy / acquisition_policy) — keywords + checker rules; decl runs rc=0 but is not a value; `plan_acquisition`/`contest` unknown.
- TyAcquisitionPlan (AcquisitionPlan via plan_acquisition()) — token + checker; witness fails `unknown identifier plan_acquisition`.
- TyRecoursePolicy (RecoursePolicy / recourse_policy) — keywords + checker; decl runs but producing chain (`plan_recourse`) unknown.
- TyRecoursePlan (RecoursePlan via plan_recourse()) — token + checker; fails `unknown identifier recourse_reason`/`defer_action`.
- TyAlternativePolicy (AlternativePolicy / alternative_policy) — keywords + checker; decl runs but `propose_alternatives` chain unknown.
- TyAlternativeSet (AlternativeSet via propose_alternatives()) — token + checker; `contest`/`defer_action`/`propose_alternatives` unknown.
- TyAlternativeOption (AlternativeOption via alternative_option()) — token + checker; depends on unresolvable `propose_alternatives`/`alternative_option`.
- TyTransitionPolicy (TransitionPolicy / transition_policy) — keywords + checker; decl runs but `commit_alternative` pipeline unreachable.
- TyTransitionPlan (TransitionPlan via commit_alternative()) — token + checker; whole producing chain unknown.
- TyMonitoringPolicy (MonitoringPolicy / monitoring_policy) — keywords + checker; decl runs but `observe_transition`/`rollback_transition` unknown.
- TyObservedTransition (ObservedTransition via observe_transition()) — token + checker; fails `unknown identifier observe_transition`.
- TyRollbackCertificate (RollbackCertificate via rollback_transition()) — token + checker; fails `unknown identifier rollback_transition`.
- TyModel (Model<In,Out>) — token + `lower_model_type`; type-level decl runs rc=0 but only value path `contest` fails typecheck.
- TyModelFamily (models NAME = [...]) — `models` token + `collect_models_def`; type-level decl runs but it is a compile-time type-arg with no runtime constructor.
- TySession (session NAME {...}) — token + item parse, but ItemSession is an explicit no-op in collect AND typecheck; never binds a TySession entry.
- TyChan (chan PROTOCOL) — token + `lower_chan_type`; annotation compiles but no channel-creation expression exists and protocol resolves only as unknown (session is no-op).

## DEAD (35)

- TyEpistemic (Epistemic, no constructible surface name) — no token, ty_epistemic() zero callers; annotation byte-identical to FooBogus control; only produced internally from Aleatoric ops.
- TyDistribution (Distribution<T,kind>) — no parse branch, ty_distribution zero external callers; annotation ignored (= bogus).
- TySample (Sample<T,D>) — lowercase `sample` is vestigial; no TySample parser consumer; behaves like SampleBogus.
- TyConditionalDist (Conditional<T,E>) — no token/branch; ty_conditional_dist zero external callers; = ConditionalBogus.
- TyEntropic (Entropic<T,H>) — no token/branch; only internal entropy_combine callers; = EntropicBogus.
- TyMutualInfo (MutualInfo<X,Y>) — no token/branch; only mutual_info_from_entropics; = MutualInfoBogus.
- TyKLBounded (KLBounded<T,delta>) — no token/branch; only kl_to_tv_pinsker; = KLBoundedBogus.
- TyELBO (ELBO<Q,P,delta>) — no token/branch; only internal VI bridges; = ELBOBogus.
- TyVariationalFamily (VarFamily<Q,Params>) — no token/branch; ty_variational_family zero callers anywhere; = VarFamilyBogus.
- TyCondIndep (IndepKnowledge) — `IndepKnowledge<` is only a print formatter; ty_cond_indep zero callers; falls to parse_named_type like bogus.
- TyATE (ATE) — print formatter only; ty_ate reached only by callerless dead bridges; = bogus control.
- TyTransportable (Transport) — print formatter only; ty_transportable zero callers; = FooBogus.
- TySelectionDiagram (SelectionDiagram) — print formatter only; ty_selection_diagram zero callers; = bogus.
- TyFairPrediction (FairPrediction) — no token/branch; ELF byte-identical to FooBogus; test only mentions it in comments.
- TyFairnessGap (FairnessGap) — no token/branch; only print/compat/check internals; no surface name resolves (no witness possible).
- TyEffectVar (no surface name) — internal effect-row tail; ty_effect_var zero callers; live effect checking uses separate Effect type.
- TyEffectRow (no surface name) — ty_effect_row zero callers; consumers only self-recursive; unreachable.
- TyGradedEffect (GradedEffect) — ty_graded_effect zero callers; annotation byte-identical to bogus; name only in test comments.
- TyScopedEffect (ScopedEffect) — ty_scoped_effect zero callers; runs identically to FooBogusXyz; tests use only `with IO,Div`.
- TyVecShaped (Vec<T,n>) — lower_named_type_with_args matches only Hyper/Unobserved; ty_vec_shaped zero external callers; = ZqxBogus.
- TyMatrixShaped (Matrix<T,m,n>) — no name match; ty_matrix_shaped only an internal Jacobian bridge; not source-reachable.
- TyBroadcastable (Broadcastable<T,m,n>) — no name match; ty_broadcastable zero callers; = ZqxBogus.
- TySingleton (Singleton<T,name>) — no name match; ty_singleton zero callers; = ZqxBogus.
- TyDifferentiable (Differentiable) — no token/branch; ty_differentiable only via callerless diff_compose; = FooBogusXYZ.
- TyGradient (Gradient) — no token/branch; ty_gradient only via callerless gradient_bridge_complete.
- TyJacobian (Jacobian) — no token/branch; reached only via callerless jacobian_from_matrix_shape bridge.
- TyMarkovChain (Markov) — no token/branch; ty_markov_chain no callers; test mentions Markov only in a comment.
- TySDE (SDE) — no token/branch; ty_sde only via self-referential dead bridge (langevin_from_gradient).
- TyMartingale (Martingale) — no token/branch; ty_martingale zero callers; = FooBogusXYZ.
- TyStationaryDist (StationaryDist) — no token/branch; ty_stationary_dist zero callers; unreachable.
- TyBigO (BigO) — no keyword/TypeBigO; ty_bigo zero callers; compiles like FooBogus.
- TyAmortized (Amortized) — no keyword/TypeAmortized; ty_amortized zero callers; indistinguishable from bogus.
- TySessionEnd (internal terminal) — no token; produced only on parser-unreachable branches (TypeSession never emitted; `chan` always has inner); witness resolves as unknown name, not the terminal.
- TyError (internal) — checker error sentinel; no surface spelling/token/constructor.
- TyUnknown (internal) — inference placeholder; surface `_` maps to TypeInfer, not TyUnknown; internal only.

## Method & limits

- **Compiler under test:** canonical souc fixed point `5a144ce6` (the self-reproducing `bin/souc`).
- **Discriminating-witness approach:** each type was probed with a minimal `.sio` program intended to construct and use a real typed value, then compile + run; a verdict of USABLE required the value to reach runtime (rc as expected), not merely to type-check.
- **Bogus-control guard:** Sounio silently accepts unknown generic names (e.g. `FooBogus<T>`) in type-annotation position, so a *bare annotation that compiles* proves nothing. Auditors compared each candidate's emitted ELF / error behavior against a bogus control of the same shape; byte-identical output ⇒ no nominal resolution ⇒ DEAD. USABLE/PARSE_ONLY required a discriminating signal (distinct ELF, type-specific error, dedicated parse path, or a running value).
- **PARSE_ONLY vs DEAD:** PARSE_ONLY types have a genuine dedicated lexer keyword + parser/checker lowering (the surface *type* is real) but no reachable value constructor — typically because the entire decision/causal/recourse value pipeline hinges on `contest`/`defer_action`/`plan_*`/`commit_alternative`/`propose_alternatives` builtins that canonical souc reports as `unknown identifier`. DEAD types have no parse path at all and resolve as unknown generics.
- **Known caveats inside USABLE:** TyChar literals (`'A'`) are rejected (construction is via `as char`); scalar `&!i64` deref write-back does not propagate (array/slice mut forms work); TyHyper field access (`.w`/`.real`) fails to lower. These are recorded as gaps, not promotions.
- **No UNCERTAIN verdicts** were returned; every type had a decisive witness or a decisive structural reason (zero callers + no parse path).
