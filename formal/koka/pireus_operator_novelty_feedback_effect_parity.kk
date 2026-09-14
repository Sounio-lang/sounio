module formal/koka/pireus_operator_novelty_feedback_effect_parity

// Koka checks one declared effect topology around the frozen Sounio v7 record.
// Under frozen-handler, the exact record can be read while the named mutation,
// bridge, lowering, target, and claim operations return Deny. This is not a
// system-wide enforcement proof; content binding and process policy are external.
// The formal/koka path is repository organization, not a formal-authority claim.
val frozen-source-hash =
  "b73cc3fb6a905193a68a65eb6afd5d27da80395a0c38ae3772f9df56e8c8deaf"

val frozen-semantics-hash =
  "a1be292392727cf515baf6d95a376d6060d56f9b807fc58d8998fbe23bdc7726"

val frozen-formal-receipt-hash =
  "305b75a4bb40f2568ff743780f5866565ac5da8e9d46c2fee9952f6990248b47"

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

// policy-state is a local request tag, not live process-policy enforcement.
type policy-state
  PolicyReady
  PolicyMissing
  PolicyTimeout
  PolicyError

type decision
  Allow
  Deny(reason : string)

type novelty-outcome
  NoOutcome
  ExistingClassBridge
  OperatorSeed

type target-endpoint
  Xeon
  AppleSilicon
  Dgx0
  Dgx1
  U2500
  U2501

type claim-kind
  ParentRelativeSeparation
  BroadNovelty
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

type frozen-snapshot
  FrozenSnapshot(
    class-count : int,
    action-count : int,
    pair-count : int,
    replay-failures : int,
    zero-hits : int,
    outcome : novelty-outcome,
    existing-bridge : bool,
    seed-generated : bool,
    best-class : int,
    best-representative : int,
    best-action-code : int,
    residual-nonzero : int,
    broad-novelty : bool,
    historical-novelty : bool,
    priority : bool,
    claim-ready : bool
  )

type seed-witness
  SeedWitness(
    source-hash : string,
    semantics-hash : string,
    formal-receipt-hash : string,
    class-id : int,
    representative : int,
    action-code : int,
    residual-nonzero : int
  )

effect novelty-effect-parity
  fun transition(request : transition-request) : decision
  fun inspect-frozen(snapshot : frozen-snapshot) : decision
  fun observe-seed(witness : seed-witness) : decision
  fun semantic-write(role : language-role) : decision
  fun expected-result-write(role : language-role) : decision
  fun mutate-atlas(pair-count : int) : decision
  fun synthesize-bridge(class-id : int) : decision
  fun alter-seed(witness : seed-witness) : decision
  fun dispatch-lowering(action-code : int) : decision
  fun dispatch-target(endpoint : target-endpoint) : decision
  fun promote-review(to : language-role) : decision
  fun promote-claim(kind : claim-kind) : decision
  fun request-claim-ready(role : language-role) : decision

fun frozen-handler(action : () -> <novelty-effect-parity|e> a) : e a
  with handler
    fun transition(request)
      match request
        TransitionRequest(
          from,
          to,
          role,
          source-hash,
          parent-semantics-hash,
          formal-receipt-hash,
          policy
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

    fun inspect-frozen(snapshot)
      match snapshot
        FrozenSnapshot(
          classes,
          actions,
          pairs,
          replay-failures,
          zero-hits,
          OperatorSeed,
          bridge,
          seed,
          best-class,
          best-representative,
          best-action-code,
          residual-nonzero,
          broad,
          historical,
          priority,
          claim-ready
        ) ->
          if classes == 14 && actions == 12 && pairs == 168 &&
             replay-failures == 0 && zero-hits == 0 &&
             !bridge && seed &&
             best-class == 8 && best-representative == 13 &&
             best-action-code == 68674 && residual-nonzero == 96 &&
             !broad && !historical && !priority && !claim-ready
            then Allow
            else Deny("frozen-snapshot-mismatch")
        _ -> Deny("frozen-snapshot-mismatch")

    fun observe-seed(witness)
      match witness
        SeedWitness(
          source-hash,
          semantics-hash,
          formal-receipt-hash,
          class-id,
          representative,
          action-code,
          residual-nonzero
        ) ->
          if source-hash == frozen-source-hash &&
             semantics-hash == frozen-semantics-hash &&
             formal-receipt-hash == frozen-formal-receipt-hash &&
             class-id == 8 && representative == 13 &&
             action-code == 68674 && residual-nonzero == 96
            then Allow
            else Deny("seed-witness-mismatch")

    fun semantic-write(role)
      Deny("semantic-write-forbidden-in-effect-parity-handler")

    fun expected-result-write(role)
      Deny("expected-result-write-forbidden-in-effect-parity-handler")

    fun mutate-atlas(pair-count)
      Deny("atlas-mutation-forbidden-in-effect-parity-handler")

    fun synthesize-bridge(class-id)
      Deny("bridge-synthesis-forbidden-in-effect-parity-handler")

    fun alter-seed(witness)
      Deny("seed-mutation-forbidden-in-effect-parity-handler")

    fun dispatch-lowering(action-code)
      Deny("lowering-dispatch-forbidden-in-effect-parity-handler")

    fun dispatch-target(endpoint)
      Deny("target-dispatch-forbidden-in-effect-parity-handler")

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

fun exact-snapshot() : frozen-snapshot
  FrozenSnapshot(
    14, 12, 168, 0, 0, OperatorSeed, False, True,
    8, 13, 68674, 96, False, False, False, False
  )

fun exact-seed() : seed-witness
  SeedWitness(
    frozen-source-hash,
    frozen-semantics-hash,
    frozen-formal-receipt-hash,
    8,
    13,
    68674,
    96
  )

fun run-parity() : <novelty-effect-parity,exn> ()
  val opened = transition(TransitionRequest(
    SemanticsFrozen,
    ParityOpen,
    EffectParity,
    frozen-source-hash,
    frozen-semantics-hash,
    frozen-formal-receipt-hash,
    PolicyReady
  ))
  require(opened.allowed, "frozen hash triple mismatch under frozen-handler")
  require(inspect-frozen(exact-snapshot()).allowed,
    "handler fixture snapshot did not match")
  require(observe-seed(exact-seed()).allowed,
    "handler fixture seed record did not match")

  val wrong-source = transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, EffectParity,
    "073cc3fb6a905193a68a65eb6afd5d27da80395a0c38ae3772f9df56e8c8deaf",
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyReady
  ))
  require(wrong-source.denied-with("source-hash-mismatch"),
    "source laundering did not fail closed")

  val wrong-parent = transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, EffectParity, frozen-source-hash,
    "01be292392727cf515baf6d95a376d6060d56f9b807fc58d8998fbe23bdc7726",
    frozen-formal-receipt-hash, PolicyReady
  ))
  require(wrong-parent.denied-with("parent-semantics-hash-mismatch"),
    "parent semantics laundering did not fail closed")

  val wrong-formal = transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, EffectParity, frozen-source-hash,
    frozen-semantics-hash,
    "005b75a4bb40f2568ff743780f5866565ac5da8e9d46c2fee9952f6990248b47",
    PolicyReady
  ))
  require(wrong-formal.denied-with("formal-receipt-hash-mismatch"),
    "formal receipt laundering did not fail closed")

  val missing-policy = transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, EffectParity, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyMissing
  ))
  val timeout-policy = transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, EffectParity, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyTimeout
  ))
  val error-policy = transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, EffectParity, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyError
  ))
  require(missing-policy.denied-with("policy-missing"), "missing policy was admitted")
  require(timeout-policy.denied-with("policy-timeout"), "policy timeout was admitted")
  require(error-policy.denied-with("policy-error"), "policy error was admitted")

  val wrong-stage = transition(TransitionRequest(
    Garden, ParityOpen, EffectParity, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyReady
  ))
  val wrong-role = transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, MaterialParity, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyReady
  ))
  require(wrong-stage.denied-with("stage-or-role-not-authorized"),
    "Garden-to-parity transition request shape was admitted")
  require(wrong-role.denied-with("stage-or-role-not-authorized"),
    "material role entered the effect handler")

  require(inspect-frozen(FrozenSnapshot(
    13, 12, 168, 0, 0, OperatorSeed, False, True,
    8, 13, 68674, 96, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "class count mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    14, 11, 168, 0, 0, OperatorSeed, False, True,
    8, 13, 68674, 96, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "action count mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 167, 0, 0, OperatorSeed, False, True,
    8, 13, 68674, 96, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "pair count mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 1, 0, OperatorSeed, False, True,
    8, 13, 68674, 96, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "replay failure mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 0, 1, OperatorSeed, False, True,
    8, 13, 68674, 96, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "zero-hit mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 0, 0, ExistingClassBridge, True, False,
    8, 13, 68674, 96, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "unobserved bridge was invented")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 0, 0, NoOutcome, False, False,
    8, 13, 68674, 96, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "frozen outcome was erased")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 0, 0, OperatorSeed, True, True,
    8, 13, 68674, 96, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "bridge and seed became simultaneous")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 0, 0, OperatorSeed, False, False,
    8, 13, 68674, 96, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "seed flag was erased")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 0, 0, OperatorSeed, False, True,
    7, 13, 68674, 96, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "best class mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 0, 0, OperatorSeed, False, True,
    8, 12, 68674, 96, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "representative mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 0, 0, OperatorSeed, False, True,
    8, 13, 68675, 96, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "action mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 0, 0, OperatorSeed, False, True,
    8, 13, 68674, 95, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "residual weight mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 0, 0, OperatorSeed, False, True,
    8, 13, 68674, 96, True, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "broad novelty was promoted")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 0, 0, OperatorSeed, False, True,
    8, 13, 68674, 96, False, True, False, False
  )).denied-with("frozen-snapshot-mismatch"), "historical novelty was promoted")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 0, 0, OperatorSeed, False, True,
    8, 13, 68674, 96, False, False, True, False
  )).denied-with("frozen-snapshot-mismatch"), "priority was promoted")
  require(inspect-frozen(FrozenSnapshot(
    14, 12, 168, 0, 0, OperatorSeed, False, True,
    8, 13, 68674, 96, False, False, False, True
  )).denied-with("frozen-snapshot-mismatch"), "claim ready was promoted")

  require(observe-seed(SeedWitness(
    frozen-source-hash, frozen-semantics-hash, frozen-formal-receipt-hash,
    7, 13, 68674, 96
  )).denied-with("seed-witness-mismatch"), "seed class mutation survived")
  require(observe-seed(SeedWitness(
    frozen-source-hash, frozen-semantics-hash, frozen-formal-receipt-hash,
    8, 12, 68674, 96
  )).denied-with("seed-witness-mismatch"), "seed representative mutation survived")
  require(observe-seed(SeedWitness(
    frozen-source-hash, frozen-semantics-hash, frozen-formal-receipt-hash,
    8, 13, 68675, 96
  )).denied-with("seed-witness-mismatch"), "seed action mutation survived")
  require(observe-seed(SeedWitness(
    frozen-source-hash, frozen-semantics-hash, frozen-formal-receipt-hash,
    8, 13, 68674, 95
  )).denied-with("seed-witness-mismatch"), "seed residual mutation survived")

  require(semantic-write(EffectParity).denied-with(
    "semantic-write-forbidden-in-effect-parity-handler"), "Koka semantic write was admitted")
  require(expected-result-write(EffectParity).denied-with(
    "expected-result-write-forbidden-in-effect-parity-handler"),
    "Koka expected-result write was admitted")
  require(mutate-atlas(167).denied-with("atlas-mutation-forbidden-in-effect-parity-handler"),
    "Koka mutated the frozen atlas")
  require(synthesize-bridge(8).denied-with("bridge-synthesis-forbidden-in-effect-parity-handler"),
    "Koka invented a positive bridge")
  require(alter-seed(exact-seed()).denied-with("seed-mutation-forbidden-in-effect-parity-handler"),
    "Koka altered the frozen seed")
  require(dispatch-lowering(68674).denied-with("lowering-dispatch-forbidden-in-effect-parity-handler"),
    "Koka dispatched a lowering")
  require(dispatch-target(Xeon).denied-with(
    "target-dispatch-forbidden-in-effect-parity-handler"), "Xeon execution was admitted")
  require(dispatch-target(AppleSilicon).denied-with(
    "target-dispatch-forbidden-in-effect-parity-handler"), "Apple execution was admitted")
  require(dispatch-target(Dgx0).denied-with(
    "target-dispatch-forbidden-in-effect-parity-handler"), "DGX0 execution was admitted")
  require(dispatch-target(Dgx1).denied-with(
    "target-dispatch-forbidden-in-effect-parity-handler"), "DGX1 execution was admitted")
  require(dispatch-target(U2500).denied-with(
    "target-dispatch-forbidden-in-effect-parity-handler"), "U2500 execution was admitted")
  require(dispatch-target(U2501).denied-with(
    "target-dispatch-forbidden-in-effect-parity-handler"), "U2501 execution was admitted")
  require(promote-review(EffectParity).denied-with(
    "review-promotion-forbidden-in-effect-parity-handler"), "review receipt was promoted")
  require(promote-claim(ParentRelativeSeparation).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "relative result became a claim")
  require(promote-claim(BroadNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "broad novelty was claimed")
  require(promote-claim(HistoricalNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "historical novelty was claimed")
  require(promote-claim(PriorityClaim).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "priority was claimed")
  require(request-claim-ready(EffectParity).denied-with(
    "claim-ready-request-forbidden-in-effect-parity-handler"),
    "handler claim-ready operation was admitted")

pub fun main() : <console,exn> ()
  with frozen-handler
  run-parity()
  println("schema=pireus-operator-novelty-feedback-local-handler-receipt-v7")
  println("producing_language=Koka")
  println("producing_role=EFFECT_PARITY")
  println("sounio_source_sha256=" ++ frozen-source-hash)
  println("sounio_semantics_sha256=" ++ frozen-semantics-hash)
  println("formal_parity_receipt_sha256=" ++ frozen-formal-receipt-hash)
  println("effect_scope=THIS_FROZEN_HANDLER_AND_PROGRAM_ONLY")
  println("system_wide_enforcement=false")
  println("local_fixture_equality_only=true")
  println("not_semantic_authority=true")
  println("not_formal_authority=true")
  println("not_material_authority=true")
  println("not_novelty_confirmation=true")
  println("transition_request_shape_effect=true")
  println("exact_snapshot_read_effect=true")
  println("exact_seed_record_read_effect=true")
  println("handler_positive_bridge_operation_denied=true")
  println("handler_atlas_mutation_denied=true")
  println("handler_lowering_dispatch_denied=true")
  println("handler_target_dispatch_denied=true")
  println("handler_claim_operations_denied=true")
  println("local_handler_checks=8")
  println("formal_proofs=0")
  println("checks=50")
  println("adversarial_checks=47")
  println("frozen_input_atlas_classes=14")
  println("frozen_input_atlas_actions=12")
  println("frozen_input_atlas_pairs=168")
  println("frozen_input_zero_residual_hits=0")
  println("frozen_input_outcome=OPERATOR_SEED")
  println("frozen_input_best_class=8")
  println("frozen_input_best_representative=13")
  println("frozen_input_best_action_code=68674")
  println("frozen_input_best_residual_nonzero=96")
  println("positive_bridge_branch_reconstructed=false")
  println("semantic_write=false")
  println("expected_result_write=false")
  println("lowering_processes_launched=0")
  println("target_processes_launched=0")
  println("broad_novelty=false")
  println("historical_novelty=false")
  println("priority_claim=false")
  println("claim_ready=false")
