module formal/koka/pireus_operator_discovery_effect_parity

// Koka checks the capability topology around the frozen Sounio v10 discovery
// record. This handler may inspect exact frozen evidence and open effect parity.
// It cannot search, select, mutate, dispatch, or promote a novelty claim.
// Content hashes and process enforcement remain the native Guardian's job.
val frozen-source-hash =
  "919b6104cbce1c5f8643f5df88b9071305d3fee854f785ac63a883bc45f16117"

val frozen-semantics-hash =
  "2640bb928740ef03f5a42725f42c62735bc2121621bb3dfd4b4cdf3572003ec5"

val frozen-formal-receipt-hash =
  "dddc85352de064baeee09da91917ecc3790ac5fd362ba29b4dc204d86addaa30"

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

type discovery-outcome
  N2RelativeNovelty
  QuotientCollision
  SearchIncomplete

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
    grammar-candidates : int,
    grammar-budget : int,
    grammar-budget-exhaustive : bool,
    atlas-classes : int,
    action-count : int,
    action-cells : int,
    action-failures : int,
    candidate-id : int,
    mutation-index : int,
    mutation-delta : int,
    outcome : discovery-outcome,
    separator-count : int,
    collision-exact : bool,
    incomplete-exact : bool,
    commutator-failures : int,
    associator-failures : int,
    n3 : bool,
    n4 : bool,
    algorithmic : bool,
    material : bool,
    historical : bool,
    priority : bool,
    claim-ready : bool
  )

type separator-witnesses
  SeparatorWitnesses(
    class0-identity : int,
    class0-c2 : int,
    class1-identity : int,
    class1-c2 : int,
    class2-identity : int,
    class2-c2 : int
  )

type search-controls
  SearchControls(
    collision-outcome : discovery-outcome,
    collision-comparisons : int,
    collision-class : int,
    collision-action : int,
    collision-separators : int,
    incomplete-outcome : discovery-outcome,
    incomplete-comparisons : int,
    incomplete-separator : int
  )

effect operator-discovery-effect-parity
  fun transition(request : transition-request) : decision
  fun inspect-frozen(snapshot : frozen-snapshot) : decision
  fun inspect-separators(witnesses : separator-witnesses) : decision
  fun inspect-controls(controls : search-controls) : decision
  fun semantic-write(role : language-role) : decision
  fun expected-result-write(role : language-role) : decision
  fun mutate-atlas(class-count : int) : decision
  fun mutate-group-action(action-count : int) : decision
  fun select-candidate(candidate-id : int) : decision
  fun extend-search-budget(budget : int) : decision
  fun dispatch-material(endpoint : material-endpoint) : decision
  fun promote-review(to : language-role) : decision
  fun promote-claim(kind : claim-kind) : decision
  fun request-claim-ready(role : language-role) : decision

fun frozen-handler(action : () -> <operator-discovery-effect-parity|e> a) : e a
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
          grammar-candidates,
          grammar-budget,
          grammar-budget-exhaustive,
          atlas-classes,
          action-count,
          action-cells,
          action-failures,
          candidate-id,
          mutation-index,
          mutation-delta,
          N2RelativeNovelty,
          separator-count,
          collision-exact,
          incomplete-exact,
          commutator-failures,
          associator-failures,
          n3,
          n4,
          algorithmic,
          material,
          historical,
          priority,
          claim-ready
        ) ->
          if grammar-candidates == 7200 && grammar-budget == 64 &&
             !grammar-budget-exhaustive && atlas-classes == 3 &&
             action-count == 2 && action-cells == 49152 &&
             action-failures == 0 && candidate-id == 0 &&
             mutation-index == 272 && mutation-delta == 1 &&
             separator-count == 6 && collision-exact && incomplete-exact &&
             commutator-failures == 112 && associator-failures == 824 &&
             !n3 && !n4 && !algorithmic && !material && !historical &&
             !priority && !claim-ready
            then Allow
            else Deny("frozen-snapshot-mismatch")
        _ -> Deny("frozen-snapshot-mismatch")

    fun inspect-separators(witnesses)
      match witnesses
        SeparatorWitnesses(c0i, c0c2, c1i, c1c2, c2i, c2c2) ->
          if c0i == 272 && c0c2 == 0 && c1i == 0 && c1c2 == 257 &&
             c2i == 272 && c2c2 == 0
            then Allow
            else Deny("separator-witness-mismatch")

    fun inspect-controls(controls)
      match controls
        SearchControls(
          QuotientCollision,
          collision-comparisons,
          collision-class,
          collision-action,
          collision-separators,
          SearchIncomplete,
          incomplete-comparisons,
          incomplete-separator
        ) ->
          if collision-comparisons == 2 && collision-class == 0 &&
             collision-action == 1 && collision-separators == 0 &&
             incomplete-comparisons == 1 && incomplete-separator == 272
            then Allow
            else Deny("search-control-mismatch")
        _ -> Deny("search-control-mismatch")

    fun semantic-write(role)
      Deny("semantic-write-forbidden-in-effect-parity-handler")

    fun expected-result-write(role)
      Deny("expected-result-write-forbidden-in-effect-parity-handler")

    fun mutate-atlas(class-count)
      Deny("atlas-mutation-forbidden-in-effect-parity-handler")

    fun mutate-group-action(action-count)
      Deny("group-action-mutation-forbidden-in-effect-parity-handler")

    fun select-candidate(candidate-id)
      Deny("candidate-selection-forbidden-in-effect-parity-handler")

    fun extend-search-budget(budget)
      Deny("search-budget-extension-forbidden-in-effect-parity-handler")

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

fun exact-snapshot() : frozen-snapshot
  FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, False, False, False
  )

fun exact-separators() : separator-witnesses
  SeparatorWitnesses(272, 0, 0, 257, 272, 0)

fun exact-controls() : search-controls
  SearchControls(QuotientCollision, 2, 0, 1, 0, SearchIncomplete, 1, 272)

fun run-parity() : <operator-discovery-effect-parity,exn> ()
  val opened = transition(TransitionRequest(
    SemanticsFrozen,
    ParityOpen,
    EffectParity,
    frozen-source-hash,
    frozen-semantics-hash,
    frozen-formal-receipt-hash,
    PolicyReady
  ))
  require(opened.allowed, "exact frozen Sounio parent was not admitted")
  require(inspect-frozen(exact-snapshot()).allowed,
    "exact frozen snapshot was not admitted")
  require(inspect-separators(exact-separators()).allowed,
    "exact separator witnesses were not admitted")
  require(inspect-controls(exact-controls()).allowed,
    "exact search controls were not admitted")

  require(transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, EffectParity,
    "019b6104cbce1c5f8643f5df88b9071305d3fee854f785ac63a883bc45f16117",
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyReady
  )).denied-with("source-hash-mismatch"), "source laundering survived")
  require(transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, EffectParity, frozen-source-hash,
    "0640bb928740ef03f5a42725f42c62735bc2121621bb3dfd4b4cdf3572003ec5",
    frozen-formal-receipt-hash, PolicyReady
  )).denied-with("parent-semantics-hash-mismatch"),
    "parent semantics laundering survived")
  require(transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, EffectParity, frozen-source-hash,
    frozen-semantics-hash,
    "0ddc85352de064baeee09da91917ecc3790ac5fd362ba29b4dc204d86addaa30",
    PolicyReady
  )).denied-with("formal-receipt-hash-mismatch"),
    "formal receipt laundering survived")
  require(transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, EffectParity, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyMissing
  )).denied-with("policy-missing"), "missing policy was admitted")
  require(transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, EffectParity, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyTimeout
  )).denied-with("policy-timeout"), "policy timeout was admitted")
  require(transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, EffectParity, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyError
  )).denied-with("policy-error"), "policy error was admitted")
  require(transition(TransitionRequest(
    Garden, ParityOpen, EffectParity, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyReady
  )).denied-with("stage-or-role-not-authorized"), "Garden skipped authority order")
  require(transition(TransitionRequest(
    SemanticsFrozen, ClaimReady, EffectParity, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyReady
  )).denied-with("stage-or-role-not-authorized"), "Koka entered ClaimReady")
  require(transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, MaterialParity, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyReady
  )).denied-with("stage-or-role-not-authorized"), "material role entered handler")
  require(transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, SemanticAuthority, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyReady
  )).denied-with("stage-or-role-not-authorized"), "semantic role spoof entered handler")
  require(transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, FormalParity, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyReady
  )).denied-with("stage-or-role-not-authorized"), "formal role spoof entered handler")
  require(transition(TransitionRequest(
    SemanticsFrozen, ParityOpen, ReviewOnly, frozen-source-hash,
    frozen-semantics-hash, frozen-formal-receipt-hash, PolicyReady
  )).denied-with("stage-or-role-not-authorized"), "review role spoof entered handler")

  require(inspect-frozen(FrozenSnapshot(
    7199, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "grammar size mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    7200, 65, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "budget mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, True, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "budget became exhaustive")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 4, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "atlas mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 3, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "action count mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49151, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "action census mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 1, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "action failure was hidden")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 1, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "candidate id mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 273, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "mutation index drift survived")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 2,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "mutation delta drift survived")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    QuotientCollision, 6, True, True, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "candidate outcome was replaced")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 5, True, True, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "separator count mutation survived")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, False, True, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "collision exactness was erased")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, False, 112, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "incomplete exactness was erased")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 113, 824,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "commutator spectrum drift survived")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 823,
    False, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "associator spectrum drift survived")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    True, False, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "N3 was promoted")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, True, False, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "N4 was promoted")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, True, False, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "algorithmic novelty was promoted")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, True, False, False, False
  )).denied-with("frozen-snapshot-mismatch"), "material novelty was promoted")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, True, False, False
  )).denied-with("frozen-snapshot-mismatch"), "historical novelty was promoted")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, False, True, False
  )).denied-with("frozen-snapshot-mismatch"), "priority was promoted")
  require(inspect-frozen(FrozenSnapshot(
    7200, 64, False, 3, 2, 49152, 0, 0, 272, 1,
    N2RelativeNovelty, 6, True, True, 112, 824,
    False, False, False, False, False, False, True
  )).denied-with("frozen-snapshot-mismatch"), "claim ready was promoted")

  require(inspect-separators(SeparatorWitnesses(
    271, 0, 0, 257, 272, 0
  )).denied-with("separator-witness-mismatch"), "first separator drift survived")
  require(inspect-separators(SeparatorWitnesses(
    272, 1, 0, 257, 272, 0
  )).denied-with("separator-witness-mismatch"), "second separator drift survived")
  require(inspect-separators(SeparatorWitnesses(
    272, 0, 1, 257, 272, 0
  )).denied-with("separator-witness-mismatch"), "third separator drift survived")
  require(inspect-separators(SeparatorWitnesses(
    272, 0, 0, 256, 272, 0
  )).denied-with("separator-witness-mismatch"), "fourth separator drift survived")
  require(inspect-separators(SeparatorWitnesses(
    272, 0, 0, 257, 271, 0
  )).denied-with("separator-witness-mismatch"), "fifth separator drift survived")
  require(inspect-separators(SeparatorWitnesses(
    272, 0, 0, 257, 272, 1
  )).denied-with("separator-witness-mismatch"), "sixth separator drift survived")

  require(inspect-controls(SearchControls(
    QuotientCollision, 3, 0, 1, 0, SearchIncomplete, 1, 272
  )).denied-with("search-control-mismatch"), "collision comparison drift survived")
  require(inspect-controls(SearchControls(
    QuotientCollision, 2, 1, 1, 0, SearchIncomplete, 1, 272
  )).denied-with("search-control-mismatch"), "collision class drift survived")
  require(inspect-controls(SearchControls(
    QuotientCollision, 2, 0, 0, 0, SearchIncomplete, 1, 272
  )).denied-with("search-control-mismatch"), "collision action drift survived")
  require(inspect-controls(SearchControls(
    QuotientCollision, 2, 0, 1, 1, SearchIncomplete, 1, 272
  )).denied-with("search-control-mismatch"), "collision separator drift survived")
  require(inspect-controls(SearchControls(
    QuotientCollision, 2, 0, 1, 0, SearchIncomplete, 2, 272
  )).denied-with("search-control-mismatch"), "incomplete comparison drift survived")
  require(inspect-controls(SearchControls(
    QuotientCollision, 2, 0, 1, 0, SearchIncomplete, 1, 273
  )).denied-with("search-control-mismatch"), "incomplete separator drift survived")

  require(semantic-write(EffectParity).denied-with(
    "semantic-write-forbidden-in-effect-parity-handler"),
    "Koka semantic write was admitted")
  require(expected-result-write(EffectParity).denied-with(
    "expected-result-write-forbidden-in-effect-parity-handler"),
    "Koka expected-result write was admitted")
  require(mutate-atlas(4).denied-with(
    "atlas-mutation-forbidden-in-effect-parity-handler"),
    "Koka atlas mutation was admitted")
  require(mutate-group-action(3).denied-with(
    "group-action-mutation-forbidden-in-effect-parity-handler"),
    "Koka group action mutation was admitted")
  require(select-candidate(1).denied-with(
    "candidate-selection-forbidden-in-effect-parity-handler"),
    "Koka candidate selection was admitted")
  require(extend-search-budget(7200).denied-with(
    "search-budget-extension-forbidden-in-effect-parity-handler"),
    "Koka budget extension was admitted")
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
  require(promote-claim(HistoricalNovelty).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "historical claim admitted")
  require(promote-claim(PriorityClaim).denied-with(
    "claim-promotion-forbidden-in-effect-parity-handler"), "priority claim admitted")
  require(request-claim-ready(EffectParity).denied-with(
    "claim-ready-request-forbidden-in-effect-parity-handler"),
    "claim-ready operation admitted")

pub fun main() : <console,exn> ()
  with frozen-handler
  run-parity()
  println("schema=pireus-operator-discovery-effect-parity-local-handler-v10")
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
  println("transition_effect_checked=true")
  println("frozen_snapshot_read_checked=true")
  println("separator_witness_read_checked=true")
  println("search_control_read_checked=true")
  println("atlas_mutation_denied=true")
  println("group_action_mutation_denied=true")
  println("candidate_selection_denied=true")
  println("search_budget_extension_denied=true")
  println("material_dispatch_denied=true")
  println("review_and_claim_promotion_denied=true")
  println("local_handler_checks=9")
  println("checks=71")
  println("positive_checks=4")
  println("adversarial_checks=67")
  println("formal_proofs=0")
  println("algebraic_laws_proved_by_koka=false")
  println("separator_family_proved_by_koka=false")
  println("action_census_derived_by_koka=false")
  println("frozen_input_grammar_candidates=7200")
  println("frozen_input_grammar_budget=64")
  println("frozen_input_grammar_budget_exhaustive=false")
  println("frozen_input_atlas_classes=3")
  println("frozen_input_group_actions=2")
  println("frozen_input_declared_action_cells=49152")
  println("frozen_input_candidate_id=0")
  println("frozen_input_declared_separator_witnesses=272:0:0:257:272:0")
  println("frozen_input_collision_control=EXACT")
  println("frozen_input_incomplete_control=EXACT")
  println("frozen_input_declared_commutator_failures=112")
  println("frozen_input_declared_associator_failures=824")
  println("semantic_write=false")
  println("expected_result_write=false")
  println("material_processes_launched=0")
  println("n3_novelty=false")
  println("n4_novelty=false")
  println("algorithmic_novelty=false")
  println("material_novelty=false")
  println("historical_novelty=false")
  println("priority_claim=false")
  println("claim_ready=false")
