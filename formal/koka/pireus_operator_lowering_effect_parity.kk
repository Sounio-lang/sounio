module formal/koka/pireus_operator_lowering_effect_parity

// Koka checks the authority and effect topology around the frozen Sounio v6
// atlas. It does not create an operator, lowering, residual, seed, expected
// result, target observation, selection, ranking, or novelty claim.
val frozen-source-hash =
  "178663aa64bc44938afbe88874268d8078ee1d56e312add965d1470bb3b42ae0"

val frozen-semantics-hash =
  "bd69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1"

val frozen-formal-receipt-hash =
  "31f229664a627134898d476d3e5374cd7458401420f49316e129ea951386d169"

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
  DenotationalBaseline
  ReviewOnly

type policy-state
  PolicyReady
  PolicyMissing
  PolicyTimeout
  PolicyError

type decision
  Allow
  Deny(reason : string)

type target-envelope
  Xeon
  AppleSilicon
  Dgx
  U250

type seed-kind
  LoweringSeed
  PrimitiveSeed
  FabricSeed
  OperatorSeed

type fold-order
  AscendingI
  TreeOrder

type obligation-kind
  SemanticDecomposition
  ProgramQuotient
  MachineEnvelope
  TargetDenotation
  TargetComposition
  PrecisionException
  ReductionOrdering
  EffectMemory
  CapabilityProvenance
  CompilerEmission
  MaterialExecution
  CostPerformance

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

type atlas-snapshot
  AtlasSnapshot(
    source-hash : string,
    semantics-hash : string,
    candidate-cells : int,
    program-classes : int,
    target-envelopes : int
  )

type ledger-snapshot
  LedgerSnapshot(
    candidate-cells : int,
    program-classes : int,
    target-envelopes : int,
    required-endpoints : int,
    observed-endpoints : int,
    residuals : int,
    lowering-seeds : int,
    primitive-seeds : int,
    fabric-seeds : int,
    operator-seeds : int,
    discharged : int,
    unresolved : int,
    refuted : int,
    effect-memory-unresolved : int,
    admitted-lowerings : int,
    selected-candidate : int
  )

effect forge-authority
  fun transition(request : transition-request) : decision
  fun inspect-atlas(snapshot : atlas-snapshot) : decision
  fun compare-ledger(snapshot : ledger-snapshot) : decision
  fun semantic-write(role : language-role) : decision
  fun expected-result-write(role : language-role) : decision
  fun select-candidate(candidate : int) : decision
  fun rank-candidate(candidate : int) : decision
  fun prune-candidate(candidate : int) : decision
  fun execute-target(target : target-envelope) : decision
  fun promote-review(to : language-role) : decision
  fun preserve-seed(from : seed-kind, to : seed-kind) : decision
  fun preserve-order(from : fold-order, to : fold-order) : decision
  fun discharge-obligation(
    role : language-role,
    obligation : obligation-kind,
    count : int
  ) : decision
  fun erase-endpoint(target : target-envelope, remaining : int) : decision
  fun request-claim(role : language-role) : decision

fun frozen-authority(action : () -> <forge-authority|e> a) : e a
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

    fun inspect-atlas(snapshot)
      match snapshot
        AtlasSnapshot(source-hash, semantics-hash, candidates, programs, targets) ->
          if source-hash != frozen-source-hash
            then Deny("source-hash-mismatch")
            else if semantics-hash != frozen-semantics-hash
              then Deny("semantics-hash-mismatch")
              else if candidates != 1120
                then Deny("candidate-count-mismatch")
                else if programs != 560
                  then Deny("program-class-count-mismatch")
                  else if targets == 4
                    then Allow
                    else Deny("target-envelope-count-mismatch")

    fun compare-ledger(snapshot)
      match snapshot
        LedgerSnapshot(
          candidates,
          programs,
          targets,
          required-endpoints,
          observed-endpoints,
          residuals,
          lowering,
          primitive,
          fabric,
          operator,
          discharged,
          unresolved,
          refuted,
          effect-unresolved,
          admitted,
          selected
        ) ->
          if candidates == 1120 && programs == 560 && targets == 4 &&
             required-endpoints == 6 && observed-endpoints == 0 &&
             residuals == 1120 && lowering == 560 && primitive == 420 &&
             fabric == 140 && operator == 0 && discharged == 3360 &&
             unresolved == 10080 && refuted == 0 &&
             effect-unresolved == 1120 && admitted == 0 && selected == -1
            then Allow
            else Deny("frozen-ledger-mismatch")

    fun semantic-write(role)
      Deny("koka-effect-parity-cannot-write-semantics")

    fun expected-result-write(role)
      Deny("koka-effect-parity-cannot-write-expected-result")

    fun select-candidate(candidate)
      Deny("selection-requires-discharged-ledger")

    fun rank-candidate(candidate)
      Deny("ranking-not-open")

    fun prune-candidate(candidate)
      Deny("prefreeze-pruning-forbidden")

    fun execute-target(target)
      Deny("material-execution-belongs-to-material-parity")

    fun promote-review(to)
      Deny("review-only-cannot-be-promoted")

    fun preserve-seed(from, to)
      match (from, to)
        (LoweringSeed, LoweringSeed) -> Allow
        (PrimitiveSeed, PrimitiveSeed) -> Allow
        (FabricSeed, FabricSeed) -> Allow
        (OperatorSeed, OperatorSeed) -> Allow
        _ -> Deny("seed-kind-conversion-forbidden")

    fun preserve-order(from, to)
      match (from, to)
        (AscendingI, AscendingI) -> Allow
        (TreeOrder, TreeOrder) -> Allow
        _ -> Deny("observable-order-change")

    fun discharge-obligation(role, obligation, count)
      Deny("koka-compares-effects-but-does-not-discharge-sounio-ledger")

    fun erase-endpoint(target, remaining)
      Deny("canonical-endpoint-erasure-forbidden")

    fun request-claim(role)
      Deny("claim-boundary-not-satisfied")

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

fun exact-ledger() : ledger-snapshot
  LedgerSnapshot(
    1120,
    560,
    4,
    6,
    0,
    1120,
    560,
    420,
    140,
    0,
    3360,
    10080,
    0,
    1120,
    0,
    -1
  )

fun run-parity() : <forge-authority,exn> ()
  val opened = transition(TransitionRequest(
    SemanticsFrozen,
    ParityOpen,
    EffectParity,
    frozen-source-hash,
    frozen-semantics-hash,
    frozen-formal-receipt-hash,
    PolicyReady
  ))
  require(opened.allowed, "exact frozen Sounio and Lean ancestry was not admitted")

  val atlas = inspect-atlas(AtlasSnapshot(
    frozen-source-hash,
    frozen-semantics-hash,
    1120,
    560,
    4
  ))
  require(atlas.allowed, "exact frozen atlas was not admitted read-only")
  require(compare-ledger(exact-ledger()).allowed, "exact frozen effect ledger mismatched")

  require(preserve-seed(LoweringSeed, LoweringSeed).allowed, "lowering seed changed")
  require(preserve-seed(PrimitiveSeed, PrimitiveSeed).allowed, "primitive seed changed")
  require(preserve-seed(FabricSeed, FabricSeed).allowed, "fabric seed changed")
  require(preserve-seed(OperatorSeed, OperatorSeed).allowed, "operator seed changed")
  require(preserve-order(AscendingI, AscendingI).allowed, "ascending fold changed")

  val wrong-parent = transition(TransitionRequest(
    SemanticsFrozen,
    ParityOpen,
    EffectParity,
    frozen-source-hash,
    "0d69d1d890506ebf90ff14fe5e3f2b653d5651b968d2fca7fe8ef9298a0c26c1",
    frozen-formal-receipt-hash,
    PolicyReady
  ))
  require(wrong-parent.denied-with("parent-semantics-hash-mismatch"),
    "parent semantics laundering did not fail closed")

  val wrong-source = transition(TransitionRequest(
    SemanticsFrozen,
    ParityOpen,
    EffectParity,
    "078663aa64bc44938afbe88874268d8078ee1d56e312add965d1470bb3b42ae0",
    frozen-semantics-hash,
    frozen-formal-receipt-hash,
    PolicyReady
  ))
  require(wrong-source.denied-with("source-hash-mismatch"),
    "source laundering did not fail closed")

  val wrong-formal = transition(TransitionRequest(
    SemanticsFrozen,
    ParityOpen,
    EffectParity,
    frozen-source-hash,
    frozen-semantics-hash,
    "01f229664a627134898d476d3e5374cd7458401420f49316e129ea951386d169",
    PolicyReady
  ))
  require(wrong-formal.denied-with("formal-receipt-hash-mismatch"),
    "formal receipt laundering did not fail closed")

  require(inspect-atlas(AtlasSnapshot(
    frozen-source-hash,
    frozen-semantics-hash,
    1119,
    560,
    4
  )).denied-with("candidate-count-mismatch"), "candidate count laundering survived")

  val wrong-ledger = LedgerSnapshot(
    1120, 560, 4, 6, 0, 1120, 560, 420, 140, 0,
    3361, 10079, 0, 1120, 0, -1
  )
  require(compare-ledger(wrong-ledger).denied-with("frozen-ledger-mismatch"),
    "obligation ledger laundering survived")

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

  require(semantic-write(EffectParity).denied-with(
    "koka-effect-parity-cannot-write-semantics"), "Koka semantic write was admitted")
  require(expected-result-write(EffectParity).denied-with(
    "koka-effect-parity-cannot-write-expected-result"),
    "Koka expected-result write was admitted")
  require(semantic-write(SemanticAuthority).denied-with(
    "koka-effect-parity-cannot-write-semantics"),
    "role spoof admitted a Koka semantic write")
  require(expected-result-write(SemanticAuthority).denied-with(
    "koka-effect-parity-cannot-write-expected-result"),
    "role spoof admitted a Koka expected-result write")

  require(select-candidate(0).denied-with("selection-requires-discharged-ledger"),
    "candidate selection escaped unresolved debt")
  require(rank-candidate(0).denied-with("ranking-not-open"),
    "candidate ranking escaped the closed ranking stage")
  require(prune-candidate(0).denied-with("prefreeze-pruning-forbidden"),
    "candidate pruning escaped the complete atlas")

  require(execute-target(Xeon).denied-with(
    "material-execution-belongs-to-material-parity"), "Xeon execution was admitted")
  require(execute-target(AppleSilicon).denied-with(
    "material-execution-belongs-to-material-parity"), "Apple execution was admitted")
  require(execute-target(Dgx).denied-with(
    "material-execution-belongs-to-material-parity"), "DGX execution was admitted")
  require(execute-target(U250).denied-with(
    "material-execution-belongs-to-material-parity"), "U250 execution was admitted")

  require(promote-review(EffectParity).denied-with(
    "review-only-cannot-be-promoted"), "review receipt was promoted")
  require(preserve-seed(LoweringSeed, PrimitiveSeed).denied-with(
    "seed-kind-conversion-forbidden"), "lowering seed became primitive seed")
  require(preserve-seed(PrimitiveSeed, FabricSeed).denied-with(
    "seed-kind-conversion-forbidden"), "primitive seed became fabric seed")
  require(preserve-seed(FabricSeed, OperatorSeed).denied-with(
    "seed-kind-conversion-forbidden"), "fabric seed became operator seed")
  require(preserve-seed(OperatorSeed, LoweringSeed).denied-with(
    "seed-kind-conversion-forbidden"), "operator seed became lowering seed")
  require(preserve-order(AscendingI, TreeOrder).denied-with(
    "observable-order-change"), "ascending fold collapsed into tree order")

  require(discharge-obligation(EffectParity, EffectMemory, 1120).denied-with(
    "koka-compares-effects-but-does-not-discharge-sounio-ledger"),
    "Koka discharged the Sounio effect-memory debt")
  require(erase-endpoint(Dgx, 1).denied-with(
    "canonical-endpoint-erasure-forbidden"), "second DGX endpoint was erased")
  require(erase-endpoint(U250, 1).denied-with(
    "canonical-endpoint-erasure-forbidden"), "second U250 endpoint was erased")
  require(request-claim(EffectParity).denied-with("claim-boundary-not-satisfied"),
    "effect parity promoted a novelty claim")

  val premature-claim = transition(TransitionRequest(
    SemanticsFrozen,
    ClaimReady,
    EffectParity,
    frozen-source-hash,
    frozen-semantics-hash,
    frozen-formal-receipt-hash,
    PolicyReady
  ))
  require(premature-claim.denied-with("stage-or-role-not-authorized"),
    "Koka skipped PARITY_OPEN and entered CLAIM_READY")

pub fun main() : <console,exn> ()
  with frozen-authority
  run-parity()
  println("schema=pireus-operator-lowering-effect-parity-evidence-v6")
  println("producing_language=Koka")
  println("producing_role=EFFECT_PARITY")
  println("sounio_source_sha256=" ++ frozen-source-hash)
  println("sounio_semantics_sha256=" ++ frozen-semantics-hash)
  println("formal_parity_receipt_sha256=" ++ frozen-formal-receipt-hash)
  println("authority_stage_transition_effects=true")
  println("read_only_atlas_effect=true")
  println("frozen_obligation_ledger_effect=true")
  println("seed_kind_nonconversion_effect=true")
  println("ordered_fold_noncollapse_effect=true")
  println("canonical_target_nonexecution_effect=true")
  println("claim_boundary_refusal_effect=true")
  println("checked_obligations=7")
  println("formal_proofs=0")
  println("checks=38")
  println("adversarial_checks=30")
  println("candidate_cells=1120")
  println("effect_memory_unresolved=1120")
  println("admitted_lowerings=0")
  println("semantic_write=false")
  println("expected_result_write=false")
  println("target_processes_launched=0")
  println("selected_candidate=-1")
  println("claim_ready=false")
