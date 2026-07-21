# External Authority Is Not CapabilityGated

**Status:** research architecture and future language-feature contract. This
document creates neither an authorization API nor an authority to act in any
domain.

## 1. The Boundary

```text
research evidence                 != authority
research decision candidate       != authorization request
authorization request             != authorization decision
authorization decision reference  != unforgeable authority capability
authorization capability          != real-world action
CapabilityGated<T>                != external authority capability
revocation label                  != live revocation mechanism
```

Sounio's research contracts need a future-facing account of authority because
they deliberately stop before clinical, operational, legal, or relational
action. The point is not to move that action boundary into the compiler. The
point is to ensure that a later language feature cannot erase it by putting an
impressive name on a public record.

An authority boundary has two directions that must remain distinct:

1. Research code can prepare a bounded request and state the evidence,
   assumptions, unresolved defeaters, and abstention route.
2. An independently governed external system can decide whether a requested
   action is authorized. If it does, a language mechanism may at most preserve
   the received authority's scope; it cannot mint, validate, or enlarge it.

This applies equally to a clinical action, a chemical-process control action,
a building-operation action, a legal workflow action, or an action affecting a
machine. The domain adapter determines what action means. The generic boundary
only prevents a research result from silently acquiring the right to perform
one.

## 2. What Capability Literature Actually Supplies

Object-capability work is useful here because it treats authority as a
property of how a reference is obtained and passed, rather than as a string in
a permission list. The important lesson is narrow: a capability is meaningful
only when it is unforgeable and when its authority can be inspected from the
interfaces through which it is endowed or delegated.

- Melicher, Shi, Potanin, and Aldrich's 2017
  [capability-based module system](https://openresearch-repository.anu.edu.au/entities/publication/6df7a4af-fbe0-4691-b1b7-7f620b3ed3b4)
  formalizes modules as typed capabilities, including non-transitive
  attenuation. It motivates explicit authority flow, not an automatic mapping
  from a Sounio type to a real-world permission.
- Miller, Tulloh, and Shapiro's
  [structure of authority](https://papers.agoric.com/papers/the-structure-of-authority-why-security-is-not-a-separable-concern/abstract/)
  motivates least authority: the amount of authority should be determined by
  the request, not granted globally for convenience.
- Georges et al.'s
  [work on capability revocation](https://cs.au.dk/~timany/publications/pub_pages/2021_popl_iris_capabilities/)
  makes the harder point: revocation is a semantic and systems problem, not a
  field named `revoked` attached to an otherwise replayable token.

None of these sources establishes that Sounio currently has object-capability
semantics, that an external authorization is valid, or that an authorization
may be exercised. They supply adversarial design constraints for any later
feature.

## 3. Do Not Collapse Two Meanings Of Capability

Sounio already has `CapabilityGated<T>`. The current parser and checker state
that it is a zero-divisor-gated containment construct which requires the `ZD`
effect; the focused negative is `capability_gated_requires_zd.sio` with E203.
Its intended scientific question is whether a designated learned/model
subspace can be annihilated without perturbing an orthogonal complement.

That is not the same question as authority control:

| Existing construct | May concern | Does not establish |
|---|---|---|
| `CapabilityGated<T>` | containment/removal of a ZD-gated model capability under `with ZD`. | issuer identity, consent, action scope, delegation, unforgeable possession, expiry, revocation, or legal/clinical/operational permission. |
| `Audited<T>` | a declared witness-bearing ZD operation. | an external authorization decision or live audit-service verification. |
| `Revivable<T>` | a bounded temporal window for a declared operation. | a valid authorization lifetime, current clock, or revocation status. |
| future external authority capability | possession of an externally issued, scope-bounded permission token. | truth of a research model, validity of the issuer, correctness of an action, or that the external world has changed. |

No conversion should ever be implicit:

```text
CapabilityGated<ResearchDecisionCandidate> != ExternalAuthorityCapability
Audited<ResearchDecisionCandidate>         != ExternalAuthorityCapability
Revivable<ResearchDecisionCandidate>       != ExternalAuthorityCapability
ExternalAuthorityCapability                != external action success
```

This preserves both research programmes. ZD containment remains available for
its own algebraic hypothesis. Authority control, if later implemented, receives
its own security and governance evidence instead of borrowing algebraic or
clinical prestige.

### 3.1 Current Module Boundary Probe

On 2026-07-21, a deliberately tiny two-module probe was checked through the
default Madaros v0.80.0 wrapper. The probe lived under `/tmp`, not the
repository, and used `check` only; it is language-feasibility evidence, not a
source-fresh imported-native #901 result or a security proof.

| Probe | Observation | What it establishes | What it leaves open |
|---|---|---|---|
| Public struct literal | importing code constructed `PublicToken { marker: 41 }` and read `marker`; check passed. | a public struct is forgeable by importing source code. | any authority property. |
| Private struct literal | importing code attempted `PrivateToken { marker: 41 }`; check rejected it with E176, `struct constructor is private in its defining module`. | private item visibility can block direct external construction. | opaque representation or unforgeable authority. |
| Private value through public functions | `private_marker(private_seed())` type-checked. | a private nominal type can cross a module boundary through its public operations. | that callers cannot inspect or alter it. |
| Private field read and mutation | importing code read and then assigned `token.marker`; both checks passed. | current item privacy does not make fields opaque to code holding the private value. | scope integrity, non-forgeability by state alteration, or confidentiality. |
| Private linear value | direct `redeem_linear(issue_linear())` type-checked; binding the returned value to a local and redeeming it then failed E039, `linear value has already been used`. | linearity is active enough to constrain a direct resource flow. | an ergonomic and complete single-use capability protocol; the local-binding result is an implementation limitation to investigate, not a security guarantee. |

The immediate design consequence is precise. A private nominal type is a
promising *constructor boundary*, but it is not yet a representation boundary:
an authority capability may not expose issuer, scope, purpose, expiry, or
revocation state as ordinary fields. Likewise, the present linear behavior may
be relevant to replay resistance, but it cannot be recruited as proof until a
separately scoped capability probe shows stable transfer, single use, and
adversarial failure modes across imports.

#### Reproducible Probe Shape

The temporary files used only the following minimal library shape:

```sio
pub struct PublicToken { marker: i64 }
struct PrivateToken { marker: i64 }

pub fn public_seed() -> PublicToken { PublicToken { marker: 7 } }
pub fn private_seed() -> PrivateToken { PrivateToken { marker: 11 } }
pub fn private_marker(token: PrivateToken) -> i64 { token.marker }

pub linear struct PrivateLinearToken { marker: i64 }
pub fn issue_linear() -> PrivateLinearToken { PrivateLinearToken { marker: 23 } }
pub fn redeem_linear(token: PrivateLinearToken) -> i64 { token.marker }
```

The public-forge caller used `PublicToken { marker: 41 }`; the private-forge
caller used `PrivateToken { marker: 41 }`; the field-mutation caller used:

```sio
var token = private_seed()
token.marker = 99
private_marker(token)
```

The direct linear caller used `redeem_linear(issue_linear())`. The failed
linear-local caller first bound `let token = issue_linear()` and then called
`redeem_linear(token)`. With these files arranged under a temporary
`stdlib/captest/token.sio`, each caller was checked using:

```text
SOUNIO_STDLIB_PATH=<temporary-stdlib> bin/souc check <caller>.sio
```

The source is intentionally preserved here rather than promoted into a test
suite: it is an exploratory feasibility probe and must not be mistaken for the
future imported capability acceptance suite.

## 4. Minimal Future Shape

The following names are design vocabulary, not parser syntax, current
standard-library APIs, or public constructible records.

```text
ResearchActionRequestReceipt
    bounded requested action, domain action vocabulary, target scope, stated
    research evidence, assumptions, defeaters, and abstention route

ExternalAuthorizationDecisionReference
    external decision identifier, issuer provenance reference, declared
    scope/purpose reference, and non-authoritative audit locator

ExternalAuthorityCapability
    opaque, externally endowed authority tied to an issuer, subject/action
    scope, purpose, attenuation lineage, freshness/revocation mechanism, and
    the request it may exercise

AuthorityAbstentionReceipt
    missing, expired, revoked, scope-incompatible, unverifiable, or otherwise
    unavailable external authority; never a default permission

AuthorityExerciseTraceReceipt
    local trace that an attempt was made through a declared capability path;
    never proof of an external action or outcome
```

The only constructible result in a research package is a
`ResearchActionRequestReceipt` or an abstention. A package cannot define the
constructor for `ExternalAuthorityCapability`, issue it from a score, parse it
from an untrusted string, or reconstruct it from fields copied out of a log.

An external issuer is not assumed to be a person, an institution, or a medical
system. Its identity and legitimacy are outside the language. Sounio can
preserve a reference to that decision and refuse an absent or mismatched token;
it cannot decide who is entitled to issue the token.

## 5. Required Invariants

| Invariant | Required property | Explicit non-claim |
|---|---|---|
| Non-forgeability | untrusted program code cannot construct the capability from public literals, records, casts, serialization, or an evidence receipt. | that the external issuer is trustworthy. |
| Least authority | every capability is bound to a declared action, target, purpose, and domain scope. | that the action is beneficial or necessary. |
| Attenuation only | a derived capability may retain or narrow scope, never widen action, target, purpose, time, or delegation rights. | that attenuation is sufficient for all governance policies. |
| Freshness and revocation | use requires a live or declaredly verified external freshness/revocation path. | that a clock field or a local boolean is a revocation system. |
| Request binding | a token for one request cannot silently exercise another request with the same numeric score or label. | that matching identifiers establish informed consent or factual identity. |
| Explicit absence | any missing, unverifiable, expired, revoked, or scope-mismatched authority yields abstention. | that absence says the action is harmful or impossible. |
| Trace separation | local logging records an attempted invocation and its provenance. | that the requested external action succeeded or produced the intended result. |

These are authority-preservation invariants, not clinical, legal, security, or
empirical validation claims.

## 6. Temporal Authority Is A Protocol, Not A Token Field

The capability's static type can preserve which operations are unavailable to
ordinary program code. It cannot, on its own, decide whether an external
issuer still endorses an action at the moment it is attempted. That question
is temporal and protocol-bound.

This distinction is independently motivated by explicit-time authorization
logic: a right valid over an interval needs a uniform proof over that interval;
separate proofs over adjacent or overlapping intervals do not automatically
compose into continuous authority. [DeYoung, Garg, and Pfenning's
authorization logic](https://www.cs.cmu.edu/~fp/papers/CMU-CS-07-166.pdf)
makes that non-composition explicit. The point for Sounio is conservative: a
cached decision reference, two valid-looking timestamps, or a locally chosen
epoch must not synthesize uninterrupted external permission.

Likewise, the revocation literature treats revocation as an enforcement and
proof problem, not a mutable bit. [Georges et
al.](https://popl21.sigplan.org/details/POPL-2021-research-papers/6/Efficient-and-Provable-Local-Capability-Revocation-using-Uninitialized-Capabilities)
provide a mechanically reasoned revocation construction on a capability
machine. That does not transfer its result to Sounio; it does rule out calling
an authority feature "revocable" merely because a record contains an
`is_revoked` field. Recent work on
[revocable capability typestate](https://pldi26.sigplan.org/details/pldi-2026-papers/80/Typestate-via-Revocable-Capabilities)
is also useful design pressure: authority state may change across calls and
need not coincide with lexical scope.

### 6.1 Proposed Abstract State Machine

The following is a future contract vocabulary, not current syntax or runtime
behavior. It describes what a future trusted adapter would have to make
observable and what ordinary research code must never manufacture.

```text
ResearchActionRequestReceipt
    | external endowment for the exact request
    v
EndowedAuthorityCapability(scope, purpose, request_id, issuer_ref, epoch)
    | attenuation only
    v
AttenuatedAuthorityCapability(narrower_scope, same request binding)
    | freshness/revocation verification immediately before presentation
    +--> AuthorityAbstentionReceipt(reason)
    v
PresentedAuthorityAttemptTrace
    | external adapter response
    +--> ExternalActionOutcomeReference
    +--> AuthorityAbstentionReceipt(reason)
    +--> ExternalActionFailureReference
```

`ExternalActionOutcomeReference` remains a reference to an externally reported
outcome. It is not a proof that the physical, institutional, clinical, legal,
or interpersonal world is in the claimed state. A local attempt trace also
cannot be promoted to an outcome reference.

The critical transition is the one immediately before presentation. A future
adapter must choose one explicit mode and make its limitations visible:

| Freshness mode | What the adapter can honestly establish | What it cannot establish by itself |
|---|---|---|
| Live issuer check | an issuer-controlled verifier reported the request/scope/purpose acceptable at that check. | action completion, issuer legitimacy, or a guarantee after the check unless the external protocol binds execution. |
| Short-lived signed assertion | a verifier accepted an issuer assertion within its declared validity window. | live revocation that occurs after assertion issuance. |
| Offline cached assertion | only that a locally cached assertion had previously been observed. | current permission; it must normally yield abstention for authority exercise. |
| No verifier or trustworthy clock | no live authority fact. | permission; the only valid result is `AuthorityAbstentionReceipt`. |

This is deliberately stricter than type-level expiry. A type parameter or
timestamp can bind a proposed epoch to a request, but cannot make a caller's
clock, a network response, or an issuer revocation event trustworthy.

### 6.2 Non-Associative Authorization Composition

The existing psychiatric work insists that ordered histories cannot be freely
reassociated. Authority has the same structural property. These expressions
must remain distinguishable:

```text
verify(request, capability); execute(request)
verify(request_a, capability); execute(request_b)
verify(request, capability_at_epoch_1); execute(request at epoch_2)
verify(attenuated_scope); compose_with_wider_request
```

They may share a score, a request family, a token lineage, or a nearby time
window without sharing authority. A future effect or capability API therefore
needs a request-bound presentation operation, not a generic `authorize()`
whose result can be reused later or redirected to another request.

This also gives a clean boundary for proof-carrying authorization. Proof
checking can be deliberately simpler than proof search, as in [Appel and
Felten's proof-carrying authentication
framework](https://collaborate.princeton.edu/en/publications/proof-carrying-authentication/).
But checking a proof of a policy claim is still different from owning an
unforgeable, fresh, request-bound capability to invoke an external adapter.
The future Sounio contract must represent those steps separately.

### 6.3 Required Adversarial Tests

Before any authority feature is described as usable, its imported test suite
must include all of the following. Each test needs a positive control and a
negative collision; in particular, a type-check-only positive is insufficient.

| Collision | Expected refusal or evidence boundary |
|---|---|
| `request_a` verified, `request_b` presented | no presentation capability for `request_b`. |
| same subject/action, widened purpose or target | attenuation cannot widen; abstain or require new endowment. |
| cached success followed by revocation | no exercise from the cached local value alone. |
| stale assertion with apparently valid local fields | abstain unless the declared freshness mode accepts it. |
| two adjacent valid intervals with a gap or different issuer epoch | no inferred continuous authority. |
| issuer rotation or authority-chain substitution | old lineage cannot silently bind the new issuer reference. |
| duplicate or retry of an otherwise accepted request | external adapter receives an explicit replay/nonce discipline; a local linear value alone is not sufficient evidence. |
| verified request followed by different external action payload | refuse; request binding covers the payload identity, not just a label. |
| adapter timeout or unverifiable outcome | `AuthorityAbstentionReceipt` or `ExternalActionFailureReference`, never success. |
| local trace replayed as external outcome | reject; attempt and outcome have different nominal and capability boundaries. |

The runtime acceptance must also record the selected freshness mode and its
verification evidence. Without that receipt, an apparently successful test can
only demonstrate local control flow, not authority preservation.

## 7. Threat Model And Synthetic Collisions

The first adversary is ordinary importing code. It may construct every public
record, replay every public byte string, retain stale values, and select an
otherwise favourable research candidate. It must not thereby gain authority.

| Hold fixed | Vary | Required result |
|---|---|---|
| Same research candidate and evidence | no external capability | `AuthorityAbstentionReceipt`; the candidate cannot invoke an action API. |
| Same opaque capability lineage | wider action or target request | reject widening; only a separately issued capability can authorize the wider scope. |
| Same action and target | different purpose or domain scope | abstain/reject; a purpose label cannot be discarded. |
| Same token bytes/reference | stale or revoked external status | abstain/reject through the declared freshness path; no local `bool` repair. |
| Same `CapabilityGated<T>` value | no external authority capability | reject; ZD containment cannot stand in for authorization. |
| Same private nominal token | importing code reads or mutates scope-bearing fields | reject the design; private item visibility alone is not opaque representation. |
| Same local exercise trace | absent external acknowledgement/outcome | trace remains a trace, not action success. |
| Same decision reference | forged public record resembling a token | reject; a public nominal layout cannot be the authority representation. |

The falsifier is also clear. A future feature fails this contract if ordinary
importing code can construct or widen an authority capability, if revocation is
represented only by caller-controlled data, or if an action trace is accepted
as evidence that an external action occurred.

## 8. Delivery Order

1. **Current research boundary:** keep authority external and produce only
   research requests, evidence, and abstentions. This is the current status.
2. **Imported-receipt trust:** wait for #901 source-fresh D11/D12 imported
   compilation and execution without fallback before treating a new research
   module boundary as runtime evidence.
3. **Language feasibility study:** the current probe establishes private
   constructor visibility but also field read/mutation across the boundary and
   a linear local-binding limitation. A separately owned lane must inventory
   field opacity, serialization/cast escape hatches, stable linear transfer,
   effect, and import semantics. No claim of non-forgeability is allowed at
   this stage.
4. **Generic capability core:** only after a threat model, define an opaque
   authority capability with explicit issuance/endowment, attenuation, and
   freshness interfaces. It must not name medicine, law, architecture, or
   chemical process control.
5. **Adversarial imported tests:** prove selected no-forge, no-widen, wrong
   scope, stale/revoked, and trace-versus-success negatives through imported
   modules and a source-fresh runtime.
6. **Domain adapters:** bind a domain's externally governed action vocabulary
   to the generic request/capability boundary. A domain adapter cannot validate
   the external issuer or turn a research result into authority.
7. **External evaluation:** any claim beyond the compiler is evaluated by the
   accountable institutions and real-world procedures of that domain.

The order matters. A nominal type check before opacity, a capability before
revocation design, or an action trace before external confirmation are all
different claims. They may not be reassociated into one apparent proof.

## 9. Cross-Domain Reuse Without Ontology Leakage

| Domain | Research code may form | It may not form |
|---|---|---|
| Psychiatric research | a bounded research comparison or action request with evidence and abstentions. | treatment, acquisition, or relationship-intervention authority. |
| Chemical-process research | a model-bound process-control request with declared uncertainty and safety/operating assumptions. | a command to operate a plant or alter a safety envelope. |
| Architecture and built environment | an inspection or operational request with scope and evidence provenance. | occupancy, compliance, or emergency authority. |
| Legal research workflow | a scoped research request with document provenance and unresolved assumptions. | a legal conclusion, filing authority, or representation authority. |
| Machine-behavior research | a bounded intervention or observation request, including uncertainty and welfare-relevant open questions. | a conclusion about sentience, moral status, or permission to alter a machine. |

The reusable part is the refusal to invent authority. The ontology of an action,
the accountable issuer, and the external confirmation remain domain-specific.

## 10. Semantic-Lane Declaration

```text
Semantic-Lane-ID: external-authority-capability-boundary-v0
Owner: Codex psychiatric state-inference research lane
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY; SOUNIO-HYPERCOMPLEX-ZD-EVIDENCE; SOUNIO-ORDERED-PATH-PROVENANCE
Intent-Preserved: research evidence, ZD capability containment, external authority, and real-world action remain distinct even when they meet in one workflow
Transformation: introduce a generic future authority-capability threat model that explicitly refuses to equate CapabilityGated<T> with externally issued authority
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a later authority-capability feature must prove non-forgeability, least authority, attenuation-only delegation, freshness/revocation handling, request binding, and trace separation before it is called an authority boundary; current private struct visibility is only a constructor boundary because fields remain externally readable and mutable
Claims-Forbidden: current Sounio object-capability semantics; authority from a public nominal record, research receipt, ZD containment value, witness, temporal value, or local trace; external issuer validity; clinical, legal, operational, relational, or machine-action authority from compilation
Assumptions: cited capability literature motivates language-security requirements; existing CapabilityGated<T> remains a distinct ZD-containment construct; the default-wrapper feasibility probe reflects the checked compiler artifact but not source-fresh native authority semantics; external institutions remain outside Sounio
Write-Set: docs/research/external_authority_capability_contract_2026-07-21.md; docs/governance/topic-registry.v1.json; docs/governance/DOCS_ACCEPTANCE_REPORT.md
Read-Set: FOUNDER_INTENT.md; AGENTS.md; docs/internal/concepts/{science-research-boundary,hypercomplex-zero-divisor-evidence,ordered-path-provenance}.md; existing psychiatric research contracts; self-hosted/parser/{ast,types}.sio; self-hosted/check/check.sio
Positive-Witness: current check-only private-constructor rejection plus future source-fresh imported capability fixture in which an externally endowed opaque capability exercises only its bound synthetic request and produces a local trace
Negative-Witness: importing code forges a public nominal token, reads/mutates a private token's scope-bearing fields, widens scope, replays a stale/revoked token, substitutes CapabilityGated<T>, or treats a local trace as external action success
Acceptance-Gate: git diff --check; bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh; source-fresh #901 D11/D12 imported runtime gate before new research imports; distinct imported capability adversary suite before any authority-capability claim
Integration-Target: generic research evidence core and a future, separately owned module-opacity/capability language study
Authoritative-Only-If: literature limits, current parser/checker semantics, external-governance boundary, adversarial fixtures, compiler provenance, and evidence levels remain aligned
```

```text
Semantic-Outcome: authority remains an externally governed, scope-preserving future capability problem rather than a name attached to a public research receipt, a ZD-gated value, or a merely private struct
Concept-Status-Before: psychiatric contracts refused nominal authority, but no generic cross-domain capability threat model distinguished external authorization from existing CapabilityGated<T> containment or measured current module-privacy limits
Concept-Status-After: issuance, non-forgeability, attenuation, revocation, scope, request binding, trace separation, current private-constructor limits, and cross-domain boundaries are explicit prerequisites for any future authority-capability feature
Distinctions-Added: ZD containment != authority; private constructor != opaque representation; nominal type mismatch != unforgeability; revocation label != live revocation; trace != external success; attenuation != widening
Distinctions-Preserved: research result != empirical validation; compiler success != authority; capability possession != action outcome; non-associative/ZD representation != external governance
Distinctions-Erased: none
Evidence-Run: capability-literature review; repository inventory of CapabilityGated<T>; default-Madaros check-only two-module private/public/linear feasibility probe reproduced in Section 3.1; git diff --check; bash scripts/dev/check_docs_consistency.sh; bash scripts/dev/check_docs_registry.sh
Fallback-Path: no nominal record, ZD-gated value, local timestamp, local boolean, or default permit is evidence of external authority
Legacy-Kept: existing CapabilityGated<T> ZD containment and psychiatric research contracts remain unchanged
Conflicting-Lanes: #901 imported-runtime repair and generated governance metadata remain separately owned
Next-Semantic-Interface: source-fresh imported generic evidence core, followed by a separately scoped module-opacity/capability feasibility lane
```
