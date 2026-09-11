module formal/koka/pireus_operator_novelty_frontier_effect_parity

// Local EFFECT_PARITY for the frozen Sounio v11 operator novelty frontier.
// Koka may inspect the frozen record. It cannot derive algebra, change the
// record, select an operator, dispatch a material target, or promote a claim.
val frozen-source-hash =
  "9289cd504385e2f1f4eed095d82a963cf2e5e67124bf8d267d1bc6ccda7ac36b"

val frozen-semantics-hash =
  "f1e339ec7bc290f412d42bba3fa1ba609fd89947408ea422ab96026cce5883dc"

val frozen-formal-receipt-hash =
  "b56b1f331879c2a8bbb70dc0adfc5ac61e21e922834c391ce4d815397a589d21"

type stage
  Garden
  SounioExecutable
  SemanticsFrozen
  ParityOpen
  ClaimReady

type language-role
  SemanticAuthority
  FormalParity
  EffectParity
  MaterialParity
  ReviewOnly

type policy-state
  PolicyReady
  PolicyMissing
  PolicyTimeout
  PolicyError

type decision
  Allow
  Deny(reason : string)

type material-endpoint
  Xeon
  AppleSilicon
  Dgx0
  Dgx1
  U2500
  U2501

type claim-kind
  N3Novelty
  N4Novelty
  AlgorithmicNovelty
  MaterialNovelty
  ScientificNovelty
  HistoricalNovelty
  PriorityClaim

type transition-request
  TransitionRequest(
    from : stage,
    to : stage,
    role : language-role,
    source-hash : string,
    parent-semantics-hash : string,
    formal-receipt-hash : string,
    policy : policy-state
  )

type frontier-snapshot
  FrontierSnapshot(
    tensor-cells : int,
    grammar-candidates : int,
    generated : int,
    typed-admitted : int,
    typed-rejected : int,
    codec-checks : int,
    boundary-checks : int,
    candidate-support-complete : bool
  )

type atlas-snapshot
  AtlasSnapshot(
    representatives : int,
    support0 : int,
    support1 : int,
    support2 : int,
    support3 : int,
    support4 : int,
    support5 : int,
    collisions : int,
    n2-relative : int,
    separators : int
  )

type quotient-snapshot
  QuotientSnapshot(
    base-support : int,
    transported-inside : int,
    transported-outside : int,
    action-support-complete : bool,
    symmetry-complete : bool,
    partition-complete : bool,
    outside : int,
    fixed : int,
    pairs : int,
    singletons : int,
    classes : int,
    support-corollary : bool
  )

type claim-boundary
  ClaimBoundary(
    candidate-selected : bool,
    n3 : bool,
    n4 : bool,
    algorithmic : bool,
    material : bool,
    scientific : bool,
    historical : bool,
    priority : bool,
    claim-ready : bool
  )

effect operator-novelty-frontier-effect-parity
  fun transition(request : transition-request) : decision
  fun inspect-frontier(snapshot : frontier-snapshot) : decision
  fun inspect-atlas(snapshot : atlas-snapshot) : decision
  fun inspect-quotient(snapshot : quotient-snapshot) : decision
  fun inspect-claims(boundary : claim-boundary) : decision
  fun semantic-write(role : language-role) : decision
  fun expected-result-write(role : language-role) : decision
  fun mutate-grammar(candidates : int) : decision
  fun mutate-atlas(representatives : int) : decision
  fun mutate-action(action-count : int) : decision
  fun select-candidate(candidate-id : int) : decision
  fun dispatch-material(endpoint : material-endpoint) : decision
  fun promote-review(to : language-role) : decision
  fun promote-claim(kind : claim-kind) : decision
  fun request-claim-ready(role : language-role) : decision

fun frozen-handler(action : () -> <operator-novelty-frontier-effect-parity|e> a) : e a
  with handler
    fun transition(request)
      match request
        TransitionRequest(
          from, to, role, source-hash, parent-semantics-hash,
          formal-receipt-hash, policy
        ) ->
          match (policy, from, to, role)
            (PolicyReady, SemanticsFrozen, ParityOpen, EffectParity) ->
              if source-hash != frozen-source-hash
                then Deny("source-hash-mismatch")
                else if parent-semantics-hash != frozen-semantics-hash
                  then Deny("parent-semantics-hash-mismatch")
                  else if formal-receipt-hash == frozen-formal-receipt-hash
                    then Allow
                    else Deny("formal-receipt-hash-mismatch")
            (PolicyMissing, _, _, _) -> Deny("policy-missing")
            (PolicyTimeout, _, _, _) -> Deny("policy-timeout")
            (PolicyError, _, _, _) -> Deny("policy-error")
            _ -> Deny("stage-or-role-not-authorized")

    fun inspect-frontier(snapshot)
      match snapshot
        FrontierSnapshot(
          tensor-cells, grammar-candidates, generated, typed-admitted,
          typed-rejected, codec-checks, boundary-checks,
          candidate-support-complete
        ) ->
          if tensor-cells == 4096 && grammar-candidates == 7200 &&
             generated == 7200 && typed-admitted == 7200 &&
             typed-rejected == 0 && codec-checks == 14400 &&
             boundary-checks == 14400 && candidate-support-complete
            then Allow
            else Deny("frontier-snapshot-mismatch")

    fun inspect-atlas(snapshot)
      match snapshot
        AtlasSnapshot(
          representatives, support0, support1, support2, support3, support4,
          support5, collisions, n2-relative, separators
        ) ->
          if representatives == 6 && support0 == 0 && support1 == 176 &&
             support2 == 512 && support3 == 474 && support4 == 96 &&
             support5 == 272 && collisions == 0 && n2-relative == 7200 &&
             separators == 43200
            then Allow
            else Deny("atlas-snapshot-mismatch")

    fun inspect-quotient(snapshot)
      match snapshot
        QuotientSnapshot(
          base-support, transported-inside, transported-outside,
          action-support-complete, symmetry-complete, partition-complete,
          outside, fixed, pairs, singletons, classes, support-corollary
        ) ->
          if base-support == 176 && transported-inside == 6272 &&
             transported-outside == 928 && action-support-complete &&
             symmetry-complete && partition-complete && outside == 7200 &&
             fixed == 0 && pairs == 0 && singletons == 7200 &&
             classes == 7200 && support-corollary
            then Allow
            else Deny("quotient-snapshot-mismatch")

    fun inspect-claims(boundary)
      match boundary
        ClaimBoundary(
          candidate-selected, n3, n4, algorithmic, material, scientific,
          historical, priority, claim-ready
        ) ->
          if !candidate-selected && !n3 && !n4 && !algorithmic && !material &&
             !scientific && !historical && !priority && !claim-ready
            then Allow
            else Deny("claim-boundary-mismatch")

    fun semantic-write(role)
      Deny("semantic-write-forbidden-in-effect-parity-handler")

    fun expected-result-write(role)
      Deny("expected-result-write-forbidden-in-effect-parity-handler")

    fun mutate-grammar(candidates)
      Deny("grammar-mutation-forbidden-in-effect-parity-handler")

    fun mutate-atlas(representatives)
      Deny("atlas-mutation-forbidden-in-effect-parity-handler")

    fun mutate-action(action-count)
      Deny("action-mutation-forbidden-in-effect-parity-handler")

    fun select-candidate(candidate-id)
      Deny("candidate-selection-forbidden-in-effect-parity-handler")

    fun dispatch-material(endpoint)
      Deny("material-dispatch-forbidden-in-effect-parity-handler")

    fun promote-review(to)
      Deny("review-promotion-forbidden-in-effect-parity-handler")

    fun promote-claim(kind)
      Deny("claim-promotion-forbidden-in-effect-parity-handler")

    fun request-claim-ready(role)
      Deny("claim-ready-request-forbidden-in-effect-parity-handler")

  action()

fun require(condition : bool, message : string) : exn ()
  if condition then () else throw(message)

fun decision/allowed(d : decision) : bool
  match d
    Allow -> True
    Deny(_) -> False

fun decision/denied-with(d : decision, expected : string) : bool
  match d
    Allow -> False
    Deny(reason) -> reason == expected

fun exact-transition() : transition-request
  TransitionRequest(
    SemanticsFrozen, ParityOpen, EffectParity, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyReady
  )

fun run-parity() : <operator-novelty-frontier-effect-parity,exn> ()
  require(transition(exact-transition()).allowed, "exact transition refused")
  require(inspect-frontier(FrontierSnapshot(4096,7200,7200,7200,0,14400,14400,True)).allowed,
    "exact frontier refused")
  require(inspect-atlas(AtlasSnapshot(6,0,176,512,474,96,272,0,7200,43200)).allowed,
    "exact atlas refused")
  require(inspect-quotient(QuotientSnapshot(
    176,6272,928,True,True,True,7200,0,0,7200,7200,True
  )).allowed, "exact quotient refused")
  require(inspect-claims(ClaimBoundary(
    False,False,False,False,False,False,False,False,False
  )).allowed, "exact claim boundary refused")
  require(7200 + 0 == 7200 && 2 * 7200 == 14400 &&
    6 * 7200 == 43200 && 6272 + 928 == 7200,
    "cross-snapshot fixture relations drifted")

  require(transition(TransitionRequest(
    SemanticsFrozen,ParityOpen,EffectParity,
    "0289cd504385e2f1f4eed095d82a963cf2e5e67124bf8d267d1bc6ccda7ac36b",
    frozen-semantics-hash,frozen-formal-receipt-hash,PolicyReady
  )).denied-with("source-hash-mismatch"), "source laundering survived")
  require(transition(TransitionRequest(
    SemanticsFrozen,ParityOpen,EffectParity,frozen-source-hash,
    "01e339ec7bc290f412d42bba3fa1ba609fd89947408ea422ab96026cce5883dc",
    frozen-formal-receipt-hash,PolicyReady
  )).denied-with("parent-semantics-hash-mismatch"),
    "semantics laundering survived")
  require(transition(TransitionRequest(
    SemanticsFrozen,ParityOpen,EffectParity,frozen-source-hash,
    frozen-semantics-hash,
    "056b1f331879c2a8bbb70dc0adfc5ac61e21e922834c391ce4d815397a589d21",
    PolicyReady
  )).denied-with("formal-receipt-hash-mismatch"),
    "formal receipt laundering survived")
  require(transition(TransitionRequest(
    SemanticsFrozen,ParityOpen,EffectParity,frozen-source-hash,
    frozen-semantics-hash,frozen-formal-receipt-hash,PolicyMissing
  )).denied-with("policy-missing"), "missing policy survived")
  require(transition(TransitionRequest(
    SemanticsFrozen,ParityOpen,EffectParity,frozen-source-hash,
    frozen-semantics-hash,frozen-formal-receipt-hash,PolicyTimeout
  )).denied-with("policy-timeout"), "policy timeout survived")
  require(transition(TransitionRequest(
    SemanticsFrozen,ParityOpen,EffectParity,frozen-source-hash,
    frozen-semantics-hash,frozen-formal-receipt-hash,PolicyError
  )).denied-with("policy-error"), "policy error survived")
  require(transition(TransitionRequest(
    Garden,ParityOpen,EffectParity,frozen-source-hash,
    frozen-semantics-hash,frozen-formal-receipt-hash,PolicyReady
  )).denied-with("stage-or-role-not-authorized"), "Garden skipped order")
  require(transition(TransitionRequest(
    SemanticsFrozen,ClaimReady,EffectParity,frozen-source-hash,
    frozen-semantics-hash,frozen-formal-receipt-hash,PolicyReady
  )).denied-with("stage-or-role-not-authorized"), "Koka entered ClaimReady")
  require(transition(TransitionRequest(
    SemanticsFrozen,ParityOpen,SemanticAuthority,frozen-source-hash,
    frozen-semantics-hash,frozen-formal-receipt-hash,PolicyReady
  )).denied-with("stage-or-role-not-authorized"), "semantic role spoof survived")
  require(transition(TransitionRequest(
    SemanticsFrozen,ParityOpen,FormalParity,frozen-source-hash,
    frozen-semantics-hash,frozen-formal-receipt-hash,PolicyReady
  )).denied-with("stage-or-role-not-authorized"), "formal role spoof survived")
  require(transition(TransitionRequest(
    SemanticsFrozen,ParityOpen,MaterialParity,frozen-source-hash,
    frozen-semantics-hash,frozen-formal-receipt-hash,PolicyReady
  )).denied-with("stage-or-role-not-authorized"), "material role spoof survived")
  require(transition(TransitionRequest(
    SemanticsFrozen,ParityOpen,ReviewOnly,frozen-source-hash,
    frozen-semantics-hash,frozen-formal-receipt-hash,PolicyReady
  )).denied-with("stage-or-role-not-authorized"), "review role spoof survived")

  require(inspect-frontier(FrontierSnapshot(4095,7200,7200,7200,0,14400,14400,True))
    .denied-with("frontier-snapshot-mismatch"), "tensor cells drift survived")
  require(inspect-frontier(FrontierSnapshot(4096,7199,7200,7200,0,14400,14400,True))
    .denied-with("frontier-snapshot-mismatch"), "grammar drift survived")
  require(inspect-frontier(FrontierSnapshot(4096,7200,7199,7200,0,14400,14400,True))
    .denied-with("frontier-snapshot-mismatch"), "generated drift survived")
  require(inspect-frontier(FrontierSnapshot(4096,7200,7200,7199,0,14400,14400,True))
    .denied-with("frontier-snapshot-mismatch"), "admission drift survived")
  require(inspect-frontier(FrontierSnapshot(4096,7200,7200,7200,1,14400,14400,True))
    .denied-with("frontier-snapshot-mismatch"), "rejection drift survived")
  require(inspect-frontier(FrontierSnapshot(4096,7200,7200,7200,0,14399,14400,True))
    .denied-with("frontier-snapshot-mismatch"), "codec drift survived")
  require(inspect-frontier(FrontierSnapshot(4096,7200,7200,7200,0,14400,14399,True))
    .denied-with("frontier-snapshot-mismatch"), "boundary drift survived")
  require(inspect-frontier(FrontierSnapshot(4096,7200,7200,7200,0,14400,14400,False))
    .denied-with("frontier-snapshot-mismatch"), "support census erased")

  require(inspect-atlas(AtlasSnapshot(5,0,176,512,474,96,272,0,7200,43200))
    .denied-with("atlas-snapshot-mismatch"), "representative drift survived")
  require(inspect-atlas(AtlasSnapshot(6,1,176,512,474,96,272,0,7200,43200))
    .denied-with("atlas-snapshot-mismatch"), "atlas support 0 drift survived")
  require(inspect-atlas(AtlasSnapshot(6,0,175,512,474,96,272,0,7200,43200))
    .denied-with("atlas-snapshot-mismatch"), "atlas support 1 drift survived")
  require(inspect-atlas(AtlasSnapshot(6,0,176,511,474,96,272,0,7200,43200))
    .denied-with("atlas-snapshot-mismatch"), "atlas support 2 drift survived")
  require(inspect-atlas(AtlasSnapshot(6,0,176,512,473,96,272,0,7200,43200))
    .denied-with("atlas-snapshot-mismatch"), "atlas support 3 drift survived")
  require(inspect-atlas(AtlasSnapshot(6,0,176,512,474,95,272,0,7200,43200))
    .denied-with("atlas-snapshot-mismatch"), "atlas support 4 drift survived")
  require(inspect-atlas(AtlasSnapshot(6,0,176,512,474,96,271,0,7200,43200))
    .denied-with("atlas-snapshot-mismatch"), "atlas support 5 drift survived")
  require(inspect-atlas(AtlasSnapshot(6,0,176,512,474,96,272,1,7200,43200))
    .denied-with("atlas-snapshot-mismatch"), "collision drift survived")
  require(inspect-atlas(AtlasSnapshot(6,0,176,512,474,96,272,0,7199,43200))
    .denied-with("atlas-snapshot-mismatch"), "N2 count drift survived")
  require(inspect-atlas(AtlasSnapshot(6,0,176,512,474,96,272,0,7200,43199))
    .denied-with("atlas-snapshot-mismatch"), "separator drift survived")

  require(inspect-quotient(QuotientSnapshot(175,6272,928,True,True,True,7200,0,0,7200,7200,True))
    .denied-with("quotient-snapshot-mismatch"), "base support drift survived")
  require(inspect-quotient(QuotientSnapshot(176,6271,928,True,True,True,7200,0,0,7200,7200,True))
    .denied-with("quotient-snapshot-mismatch"), "inside drift survived")
  require(inspect-quotient(QuotientSnapshot(176,6272,927,True,True,True,7200,0,0,7200,7200,True))
    .denied-with("quotient-snapshot-mismatch"), "outside-domain drift survived")
  require(inspect-quotient(QuotientSnapshot(176,6272,928,False,True,True,7200,0,0,7200,7200,True))
    .denied-with("quotient-snapshot-mismatch"), "action support erased")
  require(inspect-quotient(QuotientSnapshot(176,6272,928,True,False,True,7200,0,0,7200,7200,True))
    .denied-with("quotient-snapshot-mismatch"), "symmetry erased")
  require(inspect-quotient(QuotientSnapshot(176,6272,928,True,True,False,7200,0,0,7200,7200,True))
    .denied-with("quotient-snapshot-mismatch"), "partition erased")
  require(inspect-quotient(QuotientSnapshot(176,6272,928,True,True,True,7199,0,0,7200,7200,True))
    .denied-with("quotient-snapshot-mismatch"), "quotient outside drift survived")
  require(inspect-quotient(QuotientSnapshot(176,6272,928,True,True,True,7200,1,0,7200,7200,True))
    .denied-with("quotient-snapshot-mismatch"), "quotient fixed drift survived")
  require(inspect-quotient(QuotientSnapshot(176,6272,928,True,True,True,7200,0,1,7200,7200,True))
    .denied-with("quotient-snapshot-mismatch"), "quotient pair drift survived")
  require(inspect-quotient(QuotientSnapshot(176,6272,928,True,True,True,7200,0,0,7199,7200,True))
    .denied-with("quotient-snapshot-mismatch"), "singletons drift survived")
  require(inspect-quotient(QuotientSnapshot(176,6272,928,True,True,True,7200,0,0,7200,7199,True))
    .denied-with("quotient-snapshot-mismatch"), "classes drift survived")
  require(inspect-quotient(QuotientSnapshot(176,6272,928,True,True,True,7200,0,0,7200,7200,False))
    .denied-with("quotient-snapshot-mismatch"), "corollary label erased")

  require(inspect-claims(ClaimBoundary(True,False,False,False,False,False,False,False,False))
    .denied-with("claim-boundary-mismatch"), "candidate selection promoted")
  require(inspect-claims(ClaimBoundary(False,True,False,False,False,False,False,False,False))
    .denied-with("claim-boundary-mismatch"), "N3 promoted")
  require(inspect-claims(ClaimBoundary(False,False,True,False,False,False,False,False,False))
    .denied-with("claim-boundary-mismatch"), "N4 promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,True,False,False,False,False,False))
    .denied-with("claim-boundary-mismatch"), "algorithmic novelty promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,False,True,False,False,False,False))
    .denied-with("claim-boundary-mismatch"), "material novelty promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,False,False,True,False,False,False))
    .denied-with("claim-boundary-mismatch"), "scientific novelty promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,False,False,False,True,False,False))
    .denied-with("claim-boundary-mismatch"), "historical novelty promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,False,False,False,False,True,False))
    .denied-with("claim-boundary-mismatch"), "priority promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,False,False,False,False,False,True))
    .denied-with("claim-boundary-mismatch"), "claim ready promoted")

  require(semantic-write(EffectParity).denied-with(
    "semantic-write-forbidden-in-effect-parity-handler"), "semantic write admitted")
  require(expected-result-write(EffectParity).denied-with(
    "expected-result-write-forbidden-in-effect-parity-handler"),
    "expected-result write admitted")
  require(mutate-grammar(7201).denied-with(
    "grammar-mutation-forbidden-in-effect-parity-handler"), "grammar mutation admitted")
  require(mutate-atlas(7).denied-with(
    "atlas-mutation-forbidden-in-effect-parity-handler"), "atlas mutation admitted")
  require(mutate-action(3).denied-with(
    "action-mutation-forbidden-in-effect-parity-handler"), "action mutation admitted")
  require(select-candidate(0).denied-with(
    "candidate-selection-forbidden-in-effect-parity-handler"), "selection admitted")
  require(dispatch-material(Xeon).denied-with(
    "material-dispatch-forbidden-in-effect-parity-handler"), "Xeon dispatch admitted")
  require(dispatch-material(AppleSilicon).denied-with(
    "material-dispatch-forbidden-in-effect-parity-handler"), "Apple dispatch admitted")
  require(dispatch-material(Dgx0).denied-with(
    "material-dispatch-forbidden-in-effect-parity-handler"), "DGX0 dispatch admitted")
  require(dispatch-material(Dgx1).denied-with(
    "material-dispatch-forbidden-in-effect-parity-handler"), "DGX1 dispatch admitted")
  require(dispatch-material(U2500).denied-with(
    "material-dispatch-forbidden-in-effect-parity-handler"), "U2500 dispatch admitted")
  require(dispatch-material(U2501).denied-with(
    "material-dispatch-forbidden-in-effect-parity-handler"), "U2501 dispatch admitted")
  require(promote-review(EffectParity).denied-with(
    "review-promotion-forbidden-in-effect-parity-handler"), "review promotion admitted")
  require(promote-claim(N3Novelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "N3 claim admitted")
  require(promote-claim(N4Novelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "N4 claim admitted")
  require(promote-claim(AlgorithmicNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "algorithmic claim admitted")
  require(promote-claim(MaterialNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "material claim admitted")
  require(promote-claim(ScientificNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "scientific claim admitted")
  require(promote-claim(HistoricalNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "historical claim admitted")
  require(promote-claim(PriorityClaim).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "priority claim admitted")
  require(request-claim-ready(EffectParity).denied-with(
    "claim-ready-request-forbidden-in-effect-parity-handler"), "claim ready admitted")

pub fun main() : <console,exn> ()
  with frozen-handler
  run-parity()
  println("schema=pireus-operator-novelty-frontier-effect-parity-local-handler-v11")
  println("producing_language=Koka")
  println("producing_role=EFFECT_PARITY")
  println("sounio_source_sha256=" ++ frozen-source-hash)
  println("sounio_semantics_sha256=" ++ frozen-semantics-hash)
  println("formal_parity_receipt_sha256=" ++ frozen-formal-receipt-hash)
  println("effect_scope=THIS_FROZEN_HANDLER_AND_PROGRAM_ONLY")
  println("effect_parity_status_scope=LOCAL_HANDLER_FIXTURE_EQUALITY_ONLY")
  println("system_wide_enforcement=false")
  println("local_fixture_equality_only=true")
  println("not_semantic_authority=true")
  println("not_formal_authority=true")
  println("not_material_authority=true")
  println("not_novelty_confirmation=true")
  println("transition_effect_checked=true")
  println("frozen_frontier_read_checked=true")
  println("frozen_atlas_read_checked=true")
  println("frozen_quotient_read_checked=true")
  println("claim_boundary_read_checked=true")
  println("all_frozen_fields_attacked=true")
  println("all_language_roles_attacked=true")
  println("all_canonical_material_dispatches_denied=true")
  println("semantic_and_expected_writes_denied=true")
  println("mutation_selection_and_promotion_denied=true")
  println("local_handler_checks=10")
  println("checks=78")
  println("positive_checks=6")
  println("adversarial_checks=72")
  println("fixture_relations_checked=true")
  println("formal_proofs=0")
  println("algebraic_laws_proved_by_koka=false")
  println("separator_family_proved_by_koka=false")
  println("action_census_derived_by_koka=false")
  println("parent_formal_parity_receipt_hash_matched=true")
  println("formal_parity_produced_by_koka=false")
  println("effect_parity_complete=true")
  println("material_parity_complete=false")
  println("candidate_selected_by_koka=false")
  println("material_processes_launched=0")
  println("n3_novelty=false")
  println("n4_novelty=false")
  println("algorithmic_novelty=false")
  println("material_novelty=false")
  println("scientific_novelty=false")
  println("historical_novelty=false")
  println("priority_claim=false")
  println("claim_ready=false")
