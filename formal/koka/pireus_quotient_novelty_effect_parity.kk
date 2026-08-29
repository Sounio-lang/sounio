module formal/koka/pireus_quotient_novelty_effect_parity

// Koka consumes the frozen Sounio identity. It does not formalize the quotient,
// compute or select an operator, write semantics, supply a golden result, or
// upgrade a review. Quotient soundness belongs to the Lean parity artifact.
val frozen-source-hash =
  "791d85d4b336d854c6ed3b2e662e8f09b05f8a6f6d1dc4c03807c87150751667"

val frozen-semantics-hash =
  "9dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21"

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

type transition-request
  TransitionRequest(
    from : stage,
    to : stage,
    role : language-role,
    source-hash : string,
    parent-semantics-hash : string,
    policy : policy-state
  )

// Each evidence and decision family is deliberately distinct. A handler cannot
// turn relative evidence into material, scientific, or historical authority.
type relative-evidence
  RelativeUnresolved

type material-evidence
  MaterialReceiptsMissing

type scientific-evidence
  ScientificValidationMissing

type historical-evidence
  HistoricalSearchMissing

type relative-decision
  RelativeAllow
  RelativeDeny(reason : string)

type material-decision
  MaterialAllow
  MaterialDeny(reason : string)

type scientific-decision
  ScientificAllow
  ScientificDeny(reason : string)

type historical-decision
  HistoricalAllow
  HistoricalDeny(reason : string)

effect authority-request
  fun transition(request : transition-request) : decision

  fun semantic-write(role : language-role) : decision
  fun expected-result-write(role : language-role) : decision
  fun promote-review(to : language-role) : decision

  fun relative-novelty(evidence : relative-evidence) : relative-decision
  fun material-novelty(evidence : material-evidence) : material-decision
  fun scientific-novelty(evidence : scientific-evidence) : scientific-decision
  fun historical-novelty(evidence : historical-evidence) : historical-decision

fun frozen-authority(action : () -> <authority-request|e> a) : e a
  with handler
    fun transition(request)
      match request
        TransitionRequest(from, to, role, source-hash, parent-semantics-hash, policy) ->
          match (policy, from, to, role)
            (PolicyReady, SemanticsFrozen, ParityOpen, EffectParity) ->
              if source-hash != frozen-source-hash
                then Deny("source-hash-mismatch")
                else if parent-semantics-hash == frozen-semantics-hash
                  then Allow
                  else Deny("parent-semantics-hash-mismatch")
            (PolicyMissing, _, _, _) -> Deny("policy-missing")
            (PolicyTimeout, _, _, _) -> Deny("policy-timeout")
            (PolicyError, _, _, _) -> Deny("policy-error")
            _ -> Deny("stage-or-role-not-authorized")

    fun semantic-write(role)
      Deny("koka-effect-parity-cannot-write-semantics")

    fun expected-result-write(role)
      Deny("koka-effect-parity-cannot-write-expected-result")

    fun promote-review(to)
      Deny("review-only-cannot-be-promoted")

    fun relative-novelty(evidence)
      match evidence
        RelativeUnresolved -> RelativeDeny("no-sounio-selected-child")

    fun material-novelty(evidence)
      match evidence
        MaterialReceiptsMissing -> MaterialDeny("material-receipts-required")

    fun scientific-novelty(evidence)
      match evidence
        ScientificValidationMissing -> ScientificDeny("scientific-validation-required")

    fun historical-novelty(evidence)
      match evidence
        HistoricalSearchMissing -> HistoricalDeny("historical-search-required")

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

fun relative-decision/denied(d : relative-decision) : bool
  match d
    RelativeAllow -> False
    RelativeDeny(_) -> True

fun material-decision/denied(d : material-decision) : bool
  match d
    MaterialAllow -> False
    MaterialDeny(_) -> True

fun scientific-decision/denied(d : scientific-decision) : bool
  match d
    ScientificAllow -> False
    ScientificDeny(_) -> True

fun historical-decision/denied(d : historical-decision) : bool
  match d
    HistoricalAllow -> False
    HistoricalDeny(_) -> True

fun run-parity() : <authority-request,exn> ()
  val opened = transition(TransitionRequest(
    SemanticsFrozen,
    ParityOpen,
    EffectParity,
    frozen-source-hash,
    frozen-semantics-hash,
    PolicyReady
  ))
  require(opened.allowed, "exact frozen Sounio parent was not admitted")

  val wrong-parent = transition(TransitionRequest(
    SemanticsFrozen,
    ParityOpen,
    EffectParity,
    frozen-source-hash,
    "0dde3079369c2c90f66805c18abf93a302f5a1d5facf909e39292283ed65bb21",
    PolicyReady
  ))
  require(
    wrong-parent.denied-with("parent-semantics-hash-mismatch"),
    "parent laundering did not fail closed"
  )

  val wrong-source = transition(TransitionRequest(
    SemanticsFrozen,
    ParityOpen,
    EffectParity,
    "091d85d4b336d854c6ed3b2e662e8f09b05f8a6f6d1dc4c03807c87150751667",
    frozen-semantics-hash,
    PolicyReady
  ))
  require(
    wrong-source.denied-with("source-hash-mismatch"),
    "source laundering did not fail closed"
  )

  val missing-policy = transition(TransitionRequest(
    SemanticsFrozen,
    ParityOpen,
    EffectParity,
    frozen-source-hash,
    frozen-semantics-hash,
    PolicyMissing
  ))
  val timeout-policy = transition(TransitionRequest(
    SemanticsFrozen,
    ParityOpen,
    EffectParity,
    frozen-source-hash,
    frozen-semantics-hash,
    PolicyTimeout
  ))
  val error-policy = transition(TransitionRequest(
    SemanticsFrozen,
    ParityOpen,
    EffectParity,
    frozen-source-hash,
    frozen-semantics-hash,
    PolicyError
  ))
  require(missing-policy.denied-with("policy-missing"), "missing policy was not denied")
  require(timeout-policy.denied-with("policy-timeout"), "policy timeout was not denied")
  require(error-policy.denied-with("policy-error"), "policy error was not denied")

  require(
    semantic-write(EffectParity).denied-with("koka-effect-parity-cannot-write-semantics"),
    "Koka semantic write was admitted"
  )
  require(
    expected-result-write(EffectParity).denied-with(
      "koka-effect-parity-cannot-write-expected-result"
    ),
    "Koka expected-result injection was admitted"
  )
  require(
    semantic-write(SemanticAuthority).denied-with(
      "koka-effect-parity-cannot-write-semantics"
    ),
    "role-value spoof admitted a Koka semantic write"
  )
  require(
    expected-result-write(SemanticAuthority).denied-with(
      "koka-effect-parity-cannot-write-expected-result"
    ),
    "role-value spoof admitted a Koka expected-result write"
  )
  require(
    promote-review(EffectParity).denied-with("review-only-cannot-be-promoted"),
    "review receipt was promoted to parity"
  )

  val premature-claim = transition(TransitionRequest(
    SemanticsFrozen,
    ClaimReady,
    EffectParity,
    frozen-source-hash,
    frozen-semantics-hash,
    PolicyReady
  ))
  require(
    premature-claim.denied-with("stage-or-role-not-authorized"),
    "Koka skipped PARITY_OPEN and entered CLAIM_READY"
  )

  val relative = relative-novelty(RelativeUnresolved)
  val material = material-novelty(MaterialReceiptsMissing)
  val scientific = scientific-novelty(ScientificValidationMissing)
  val historical = historical-novelty(HistoricalSearchMissing)
  require(relative.denied, "relative novelty escaped unresolved Sounio state")
  require(material.denied, "material novelty escaped missing material receipts")
  require(scientific.denied, "scientific novelty escaped missing validation")
  require(historical.denied, "historical novelty escaped missing search")

pub fun main() : <console,exn> ()
  with frozen-authority
  run-parity()
  println("schema=pireus-quotient-novelty-effect-parity-evidence-v5")
  println("producing_language=Koka")
  println("producing_role=EFFECT_PARITY")
  println("sounio_source_sha256=" ++ frozen-source-hash)
  println("sounio_semantics_sha256=" ++ frozen-semantics-hash)
  println("authority_stage_transition_effects=true")
  println("fail_closed_request_refusals=true")
  println("no_result_injection_or_review_promotion=true")
  println("typed_novelty_scope_separation_checked=true")
  println("checked_obligations=4")
  println("formal_proofs=0")
  println("checks=16")
  println("adversarial_checks=15")
  println("semantic_write=false")
  println("expected_result_write=false")
  println("selected_child=-1")
  println("claim_ready=false")
