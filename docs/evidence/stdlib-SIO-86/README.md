<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-86.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-86.readme
-->

# SIO-86: checked tropical trust in dense EL+ closure

Implements accepted ADR-016 decision12 from remote main be761447e1cb21fd0f9113dccf20c9e2aed8c7a4. The legacy Boolean ontology::elplus API and implementation remain untouched. New opt-in ontology::trust_semiring and ontology::elplus_trust carry a Pareto frontier of signed/inferred/assumed minimum derivation costs.

## Algebra and explicit limits

Each coordinate is a nonnegative i64 count or mathematical infinity encoded as-1. Finite MAX_I64 is distinct from infinity. Alternatives use pointwise min; joint premises use pointwise checked sum. The well-formed order is assumed<=inferred<=signed in the extended order. Status0 means available,1 invalid input,2 finite/counter overflow and3 incomplete round budget. Unavailable is never silently treated as semiring zero. Private annotation fields reject forged struct literals.

The existing formal/TrustSemiring.lean specifies the unbounded Option Nat algebra and its laws. It is unchanged and was not rebuilt in this slice. Native finite-domain checks do not prove this implementation correct for all inputs. Finite-count overflow is an explicit refusal, including conservatively refused dominated candidates; no saturation/wrap is presented as a derivation.

A caller labels its direct facts. A signed grade does not authenticate a certificate or prove source independence. Structural reflexivity/top/role identities have cost(0,0,0); every ontology completion, including role transitivity, has the stated inferred-rule cost(-1,1,1). Thus logical inference never manufactures a signed coordinate. This rule policy is separate from the semiring laws.

## Ontology integration

The annotated kernel implements all eight dense EL+ rules, weighted role-hierarchy closure and weighted chain premises. It continues when a fact already present gets a cheaper derivation. Capacities match the existing dense kernel:64 concepts,8 roles,8 chains. The caller owns110808 i64 state words. Packed concept/existential descriptors follow the existing EL+ table layout; the gate uses distinct interned atom identifiers.

Preflight validates dimensions, all triplet domains/order and active descriptor/existential/chain consistency before modifying the state. Invalid input leaves the entire state unchanged. Overflow or round-budget exhaustion can retain partial improvements; queries then return unavailable. max_rounds is explicit, and reaching its cap is not reported as convergence. Counters measure the in-place schedule and are not equated to the synchronous reference schedule.

## Executed acceptance

Final Slurm17077 completed0, using source-built compiler SHA d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6 on t560 Linux UID1000/stack524288KiB. This retained compiler is identified, not a new rebuild from current main. All fixture checks, compilations, native executions, C++ compilation/concordance and source/compiler hash verifications passed.

- C++23/Boost cpp_int generates the committed fixture deterministically and independently computes a synchronous least fixed point. Generated-source byte identity is checked on replay.
- 35 well-formed scalar annotations over{-1,0,1,2,3} exercise42875 triples for distributivity and associativity, plus identities, absorption, commutativity, idempotence, overflow and the frontier that rejects lexicographic selection.
- 16-concept/3-role and64-concept/8-role scenarios exercise all eight rules and compare every32466 active S/R/hierarchy/chain triplet, including unreachable facts. The C++ schedule stabilizes in7/9 rounds; native in-place schedules in3/3, with equal final costs.
- Boundary tests compare all110808 state words after invalid calls, test finite MAX, unavailable propagation, invalid descriptors/index/counts, zero/one-round budgets and overflow without committing wrapped counts.
- Private construction is rejected at check. Wrong fact and missing marker outputs are rejected by C++. A wrong-min scalar source mutant fails the laws with native13. A source mutant dropping the chain annotation still runs but is rejected by exact C++ concordance.

The final whole-program native observation, including scalar laws, both scenarios and output, is0.53s/47616KiB. The separate boundary program is0.05s/1628KiB. These are selected host observations, not isolated-kernel speedups or a universal tractability guarantee. No sparse4096-class annotated implementation or certificate/independence kernel is claimed.

Diagnostic jobs17070/17071/17073 checked early scalar/kernel/private controls.17075 preserved a gate failure because the wrong-min mutant failed law13 while the launcher expected12; only that expectation was corrected for passing17076.17077 repeats the complete gate after the generated tables were given distinct atom identifiers; the production kernel and algebra are unchanged. Prior attempts remain separately labelled, not rewritten as final acceptance.

## Replay and custody

Run on an authorized native Linux worker with explicit compiler selection:

    MADAROS_RAW_BIN=/absolute/madaros.elf TRUST_GATE_OUT=/absolute/fresh-output bash scripts/stdlib/ontology_trust_gate.sh

The central deep_science_gate.sh registers this focused gate. A future PR CI execution is separate from these Slurm receipts. native-receipts.tar.gz contains the final17077 receipts; diagnostic-receipts.tar.gz retains earlier attempts. Both contain text, per-phase exits, exact stdout, manifests and executable identities, without ELFs, compiler binaries, caches or full source snapshots. Extract into a fresh directory and verify its MANIFEST.sha256. Replaying excluded executables requires the identified source/compiler inputs; a hash is not a replay.

Human review only: no merge, queue entry or auto-merge is authorized. Notion SIO-86 remains subject to review. The typed-zero/D-34 interval decision is separate and was not resolved by this implementation.
