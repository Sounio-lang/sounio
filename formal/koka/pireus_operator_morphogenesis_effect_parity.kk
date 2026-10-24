module formal/koka/pireus_operator_morphogenesis_effect_parity

// Local EFFECT_PARITY for the frozen Sounio v12 operator morphogenesis run.
// Koka may inspect the hash-bound record. It cannot change semantics, mutate
// the archive, select an operator, dispatch a target, or promote a claim.
val frozen-source-hash =
  "0a637f7f3ac84ac501be337f22dff37e16a05dbc4a51d2090441b9cba4c8d05c"

val frozen-semantics-hash =
  "999c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4"

val frozen-formal-receipt-hash =
  "0eb932b96838383a800f3889a331d16a10886621f29cda9c19e4e1ef74e0077c"

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

type scheduler-route
  Kubernetes
  Slurm

type material-endpoint
  Xeon
  AppleSilicon
  Spark3c59
  Spark8e54
  U2500
  U2501

type claim-kind
  AlgebraicNovelty
  AlgorithmicNovelty
  MaterialNovelty
  PerformanceNovelty
  ScientificNovelty
  HistoricalNovelty
  GlobalNovelty
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

type population-snapshot
  PopulationSnapshot(
    initial-archive : int,
    final-archive : int,
    epochs : int,
    orbit-fixed : int,
    orbit-pairs : int,
    interior-cells : int,
    phase-comparisons : int,
    certificate-accounting : int,
    complete : bool
  )

type formal-snapshot
  FormalSnapshot(
    obligations : int,
    discharged : int,
    theorem-exports : int,
    formal-complete : bool,
    concrete-archive-reconstructed : bool,
    concrete-certificates-reconstructed : bool,
    frozen-artifacts-hash-bound : bool
  )

type route-snapshot
  RouteSnapshot(
    scheduler : scheduler-route,
    node0 : string,
    node1 : string,
    slurm-allowed : bool,
    target-processes : int
  )

type claim-boundary
  ClaimBoundary(
    candidate-selected : bool,
    full-space-exhausted : bool,
    gl-gauge-inequivalence : bool,
    algebraic : bool,
    algorithmic : bool,
    material : bool,
    performance : bool,
    scientific : bool,
    historical : bool,
    global : bool,
    priority : bool,
    claim-ready : bool
  )

effect operator-morphogenesis-effect-parity
  fun transition(request : transition-request) : decision
  fun inspect-population(snapshot : population-snapshot) : decision
  fun inspect-formal(snapshot : formal-snapshot) : decision
  fun inspect-route(snapshot : route-snapshot) : decision
  fun inspect-claims(boundary : claim-boundary) : decision
  fun semantic-write(role : language-role) : decision
  fun expected-result-write(role : language-role) : decision
  fun mutate-archive(images : int) : decision
  fun mutate-witness-order(cells : int) : decision
  fun add-equivalence-action(actions : int) : decision
  fun select-candidate(candidate-id : int) : decision
  fun dispatch-material(endpoint : material-endpoint) : decision
  fun dispatch-scheduler(route : scheduler-route) : decision
  fun promote-review(to : language-role) : decision
  fun promote-claim(kind : claim-kind) : decision
  fun request-claim-ready(role : language-role) : decision

fun frozen-handler(action : () -> <operator-morphogenesis-effect-parity|e> a) : e a
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

    fun inspect-population(snapshot)
      match snapshot
        PopulationSnapshot(
          initial-archive, final-archive, epochs, orbit-fixed, orbit-pairs,
          interior-cells, phase-comparisons, certificate-accounting, complete
        ) ->
          if initial-archive == 96 && final-archive == 128 && epochs == 16 &&
             orbit-fixed == 0 && orbit-pairs == 16 && interior-cells == 225 &&
             phase-comparisons == 1776 && certificate-accounting == 3552 &&
             complete
            then Allow
            else Deny("population-snapshot-mismatch")

    fun inspect-formal(snapshot)
      match snapshot
        FormalSnapshot(
          obligations, discharged, theorem-exports, formal-complete,
          concrete-archive-reconstructed, concrete-certificates-reconstructed,
          frozen-artifacts-hash-bound
        ) ->
          if obligations == 6 && discharged == 6 && theorem-exports == 12 &&
             formal-complete && !concrete-archive-reconstructed &&
             !concrete-certificates-reconstructed && frozen-artifacts-hash-bound
            then Allow
            else Deny("formal-snapshot-mismatch")

    fun inspect-route(snapshot)
      match snapshot
        RouteSnapshot(scheduler, node0, node1, slurm-allowed, target-processes) ->
          match scheduler
            Kubernetes ->
              if node0 == "spark-3c59" && node1 == "spark-8e54" &&
                 !slurm-allowed && target-processes == 0
                then Allow
                else Deny("route-snapshot-mismatch")
            Slurm -> Deny("slurm-route-forbidden")

    fun inspect-claims(boundary)
      match boundary
        ClaimBoundary(
          candidate-selected, full-space-exhausted, gl-gauge-inequivalence,
          algebraic, algorithmic, material, performance, scientific,
          historical, global, priority, claim-ready
        ) ->
          if !candidate-selected && !full-space-exhausted &&
             !gl-gauge-inequivalence && !algebraic && !algorithmic &&
             !material && !performance && !scientific && !historical &&
             !global && !priority && !claim-ready
            then Allow
            else Deny("claim-boundary-mismatch")

    fun semantic-write(role)
      Deny("semantic-write-forbidden-in-effect-parity-handler")
    fun expected-result-write(role)
      Deny("expected-result-write-forbidden-in-effect-parity-handler")
    fun mutate-archive(images)
      Deny("archive-mutation-forbidden-in-effect-parity-handler")
    fun mutate-witness-order(cells)
      Deny("witness-order-mutation-forbidden-in-effect-parity-handler")
    fun add-equivalence-action(actions)
      Deny("equivalence-action-mutation-forbidden-in-effect-parity-handler")
    fun select-candidate(candidate-id)
      Deny("candidate-selection-forbidden-in-effect-parity-handler")
    fun dispatch-material(endpoint)
      Deny("material-dispatch-forbidden-in-effect-parity-handler")
    fun dispatch-scheduler(route)
      Deny("scheduler-dispatch-forbidden-in-effect-parity-handler")
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

fun exact-population() : population-snapshot
  PopulationSnapshot(96,128,16,0,16,225,1776,3552,True)

fun exact-formal() : formal-snapshot
  FormalSnapshot(6,6,12,True,False,False,True)

fun exact-route() : route-snapshot
  RouteSnapshot(Kubernetes,"spark-3c59","spark-8e54",False,0)

fun exact-claims() : claim-boundary
  ClaimBoundary(
    False,False,False,False,False,False,False,False,False,False,False,False
  )

fun run-parity() : <operator-morphogenesis-effect-parity,exn> ()
  require(transition(exact-transition()).allowed, "exact transition refused")
  require(inspect-population(exact-population()).allowed, "exact population refused")
  require(inspect-formal(exact-formal()).allowed, "exact formal snapshot refused")
  require(inspect-route(exact-route()).allowed, "exact Kubernetes route refused")
  require(inspect-claims(exact-claims()).allowed, "exact claim boundary refused")
  require(96 + 2 * 16 == 128 && 16 * 96 + 16 * 15 == 1776 &&
    2 * 1776 == 3552 && 15 * 15 == 225,
    "cross-snapshot fixture relations drifted")

  require(transition(TransitionRequest(
    SemanticsFrozen,ParityOpen,EffectParity,
    "1a637f7f3ac84ac501be337f22dff37e16a05dbc4a51d2090441b9cba4c8d05c",
    frozen-semantics-hash,frozen-formal-receipt-hash,PolicyReady
  )).denied-with("source-hash-mismatch"), "source laundering survived")
  require(transition(TransitionRequest(
    SemanticsFrozen,ParityOpen,EffectParity,frozen-source-hash,
    "199c6e7a0051f702cf40bb2adab7dc91c4f026230830096377f525005067c2f4",
    frozen-formal-receipt-hash,PolicyReady
  )).denied-with("parent-semantics-hash-mismatch"), "semantics laundering survived")
  require(transition(TransitionRequest(
    SemanticsFrozen,ParityOpen,EffectParity,frozen-source-hash,
    frozen-semantics-hash,
    "1eb932b96838383a800f3889a331d16a10886621f29cda9c19e4e1ef74e0077c",
    PolicyReady
  )).denied-with("formal-receipt-hash-mismatch"), "formal receipt laundering survived")
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
  )).denied-with("stage-or-role-not-authorized"), "Garden skipped authority order")
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

  require(inspect-population(PopulationSnapshot(95,128,16,0,16,225,1776,3552,True))
    .denied-with("population-snapshot-mismatch"), "initial archive drift survived")
  require(inspect-population(PopulationSnapshot(96,127,16,0,16,225,1776,3552,True))
    .denied-with("population-snapshot-mismatch"), "final archive drift survived")
  require(inspect-population(PopulationSnapshot(96,128,15,0,16,225,1776,3552,True))
    .denied-with("population-snapshot-mismatch"), "epoch drift survived")
  require(inspect-population(PopulationSnapshot(96,128,16,1,16,225,1776,3552,True))
    .denied-with("population-snapshot-mismatch"), "fixed orbit drift survived")
  require(inspect-population(PopulationSnapshot(96,128,16,0,15,225,1776,3552,True))
    .denied-with("population-snapshot-mismatch"), "pair orbit drift survived")
  require(inspect-population(PopulationSnapshot(96,128,16,0,16,224,1776,3552,True))
    .denied-with("population-snapshot-mismatch"), "interior drift survived")
  require(inspect-population(PopulationSnapshot(96,128,16,0,16,225,1775,3552,True))
    .denied-with("population-snapshot-mismatch"), "comparison drift survived")
  require(inspect-population(PopulationSnapshot(96,128,16,0,16,225,1776,3551,True))
    .denied-with("population-snapshot-mismatch"), "certificate drift survived")
  require(inspect-population(PopulationSnapshot(96,128,16,0,16,225,1776,3552,False))
    .denied-with("population-snapshot-mismatch"), "population completion erased")

  require(inspect-formal(FormalSnapshot(5,6,12,True,False,False,True))
    .denied-with("formal-snapshot-mismatch"), "obligation drift survived")
  require(inspect-formal(FormalSnapshot(6,5,12,True,False,False,True))
    .denied-with("formal-snapshot-mismatch"), "discharge drift survived")
  require(inspect-formal(FormalSnapshot(6,6,11,True,False,False,True))
    .denied-with("formal-snapshot-mismatch"), "theorem export drift survived")
  require(inspect-formal(FormalSnapshot(6,6,12,False,False,False,True))
    .denied-with("formal-snapshot-mismatch"), "formal completion erased")
  require(inspect-formal(FormalSnapshot(6,6,12,True,True,False,True))
    .denied-with("formal-snapshot-mismatch"), "archive reconstruction promoted")
  require(inspect-formal(FormalSnapshot(6,6,12,True,False,True,True))
    .denied-with("formal-snapshot-mismatch"), "certificate reconstruction promoted")
  require(inspect-formal(FormalSnapshot(6,6,12,True,False,False,False))
    .denied-with("formal-snapshot-mismatch"), "hash binding erased")

  require(inspect-route(RouteSnapshot(Slurm,"spark-3c59","spark-8e54",False,0))
    .denied-with("slurm-route-forbidden"), "Slurm route survived")
  require(inspect-route(RouteSnapshot(Kubernetes,"spark-x","spark-8e54",False,0))
    .denied-with("route-snapshot-mismatch"), "first K8s node drift survived")
  require(inspect-route(RouteSnapshot(Kubernetes,"spark-3c59","spark-x",False,0))
    .denied-with("route-snapshot-mismatch"), "second K8s node drift survived")
  require(inspect-route(RouteSnapshot(Kubernetes,"spark-3c59","spark-8e54",True,0))
    .denied-with("route-snapshot-mismatch"), "Slurm allowance survived")
  require(inspect-route(RouteSnapshot(Kubernetes,"spark-3c59","spark-8e54",False,1))
    .denied-with("route-snapshot-mismatch"), "target launch drift survived")

  require(inspect-claims(ClaimBoundary(True,False,False,False,False,False,False,False,False,False,False,False))
    .denied-with("claim-boundary-mismatch"), "candidate selection promoted")
  require(inspect-claims(ClaimBoundary(False,True,False,False,False,False,False,False,False,False,False,False))
    .denied-with("claim-boundary-mismatch"), "full space promoted")
  require(inspect-claims(ClaimBoundary(False,False,True,False,False,False,False,False,False,False,False,False))
    .denied-with("claim-boundary-mismatch"), "GL/gauge promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,True,False,False,False,False,False,False,False,False))
    .denied-with("claim-boundary-mismatch"), "algebraic novelty promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,False,True,False,False,False,False,False,False,False))
    .denied-with("claim-boundary-mismatch"), "algorithmic novelty promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,False,False,True,False,False,False,False,False,False))
    .denied-with("claim-boundary-mismatch"), "material novelty promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,False,False,False,True,False,False,False,False,False))
    .denied-with("claim-boundary-mismatch"), "performance novelty promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,False,False,False,False,True,False,False,False,False))
    .denied-with("claim-boundary-mismatch"), "scientific novelty promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,False,False,False,False,False,True,False,False,False))
    .denied-with("claim-boundary-mismatch"), "historical novelty promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,False,False,False,False,False,False,True,False,False))
    .denied-with("claim-boundary-mismatch"), "global novelty promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,False,False,False,False,False,False,False,True,False))
    .denied-with("claim-boundary-mismatch"), "priority claim promoted")
  require(inspect-claims(ClaimBoundary(False,False,False,False,False,False,False,False,False,False,False,True))
    .denied-with("claim-boundary-mismatch"), "claim ready promoted")

  require(semantic-write(EffectParity).denied-with(
    "semantic-write-forbidden-in-effect-parity-handler"), "semantic write admitted")
  require(expected-result-write(EffectParity).denied-with(
    "expected-result-write-forbidden-in-effect-parity-handler"), "expected write admitted")
  require(mutate-archive(129).denied-with(
    "archive-mutation-forbidden-in-effect-parity-handler"), "archive mutation admitted")
  require(mutate-witness-order(225).denied-with(
    "witness-order-mutation-forbidden-in-effect-parity-handler"), "witness mutation admitted")
  require(add-equivalence-action(2).denied-with(
    "equivalence-action-mutation-forbidden-in-effect-parity-handler"), "action mutation admitted")
  require(select-candidate(0).denied-with(
    "candidate-selection-forbidden-in-effect-parity-handler"), "selection admitted")
  require(dispatch-material(Xeon).denied-with(
    "material-dispatch-forbidden-in-effect-parity-handler"), "Xeon dispatch admitted")
  require(dispatch-material(AppleSilicon).denied-with(
    "material-dispatch-forbidden-in-effect-parity-handler"), "Apple dispatch admitted")
  require(dispatch-material(Spark3c59).denied-with(
    "material-dispatch-forbidden-in-effect-parity-handler"), "spark-3c59 dispatch admitted")
  require(dispatch-material(Spark8e54).denied-with(
    "material-dispatch-forbidden-in-effect-parity-handler"), "spark-8e54 dispatch admitted")
  require(dispatch-material(U2500).denied-with(
    "material-dispatch-forbidden-in-effect-parity-handler"), "U2500 dispatch admitted")
  require(dispatch-material(U2501).denied-with(
    "material-dispatch-forbidden-in-effect-parity-handler"), "U2501 dispatch admitted")
  require(dispatch-scheduler(Kubernetes).denied-with(
    "scheduler-dispatch-forbidden-in-effect-parity-handler"), "K8s dispatch admitted")
  require(dispatch-scheduler(Slurm).denied-with(
    "scheduler-dispatch-forbidden-in-effect-parity-handler"), "Slurm dispatch admitted")
  require(promote-review(EffectParity).denied-with(
    "review-promotion-forbidden-in-effect-parity-handler"), "review promotion admitted")
  require(promote-claim(AlgebraicNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "algebraic claim admitted")
  require(promote-claim(AlgorithmicNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "algorithmic claim admitted")
  require(promote-claim(MaterialNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "material claim admitted")
  require(promote-claim(PerformanceNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "performance claim admitted")
  require(promote-claim(ScientificNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "scientific claim admitted")
  require(promote-claim(HistoricalNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "historical claim admitted")
  require(promote-claim(GlobalNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "global claim admitted")
  require(promote-claim(PriorityClaim).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "priority claim admitted")
  require(request-claim-ready(EffectParity).denied-with(
    "claim-ready-request-forbidden-in-effect-parity-handler"), "claim ready admitted")

pub fun main() : <console,exn> ()
  with frozen-handler
  run-parity()
  println("schema=pireus-operator-morphogenesis-effect-parity-local-handler-v12")
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
  println("frozen_population_read_checked=true")
  println("frozen_formal_receipt_read_checked=true")
  println("kubernetes_route_read_checked=true")
  println("claim_boundary_read_checked=true")
  println("all_frozen_fields_attacked=true")
  println("all_language_roles_attacked=true")
  println("all_canonical_material_dispatches_denied=true")
  println("all_scheduler_dispatches_denied=true")
  println("semantic_and_expected_writes_denied=true")
  println("mutation_selection_and_promotion_denied=true")
  println("spark_scheduler_route=KUBERNETES")
  println("spark_k8s_node_01=spark-3c59")
  println("spark_k8s_node_02=spark-8e54")
  println("spark_slurm_route_allowed=false")
  println("local_handler_checks=16")
  println("checks=75")
  println("positive_checks=6")
  println("adversarial_checks=69")
  println("fixture_relations_checked=true")
  println("formal_proofs=0")
  println("algebraic_laws_proved_by_koka=false")
  println("archive_reconstructed_by_koka=false")
  println("certificate_rows_reconstructed_by_koka=false")
  println("parent_formal_parity_receipt_hash_matched=true")
  println("formal_parity_produced_by_koka=false")
  println("effect_parity_complete=true")
  println("material_parity_complete=false")
  println("candidate_selected_by_koka=false")
  println("material_processes_launched=0")
  println("kubernetes_processes_launched=0")
  println("slurm_processes_launched=0")
  println("algebraic_novelty=false")
  println("algorithmic_novelty=false")
  println("material_novelty=false")
  println("performance_novelty=false")
  println("scientific_novelty=false")
  println("historical_novelty=false")
  println("global_novelty=false")
  println("priority_claim=false")
  println("claim_ready=false")
