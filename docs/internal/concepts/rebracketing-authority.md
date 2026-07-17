<!-- docs:meta
topic_id: repo.docs.internal.concepts.rebracketing-authority
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.rebracketing-authority
-->

# Rebracketing Authority

Concept-ID: `SOUNIO-REBRACKETING-AUTHORITY`

Status: hypothesis. The scalar protocol, production cleanup pipeline, focused
same-block default native-v2 paths, and compiler-internal forward-CFG diamond
are executable under strict hash-bound gates. The source-to-IR extension adds
focused single-module and imported/merged forward-diamond witnesses, a
non-dominating join-copy control, a loop/backedge refusal, and an audit-only
link from checked ontology-typed source parameters to lowered functions. It
remains a bounded source-shape claim, not compiler-wide semantic preservation.

## Founder Intent

Parenthesization is semantic state. A compiler may change it only for one
identified occurrence after discharging the law, representation, ordering,
alias/use, and current-state obligations that apply there. A global operator
label such as "associative" is not sufficient authority.

This concept refines `SOUNIO-NONASSOCIATIVE-ORDER`; it does not weaken it.
Nonassociative, floating-point, uncertain, clinical, and history-sensitive
expressions remain ordered unless a later authority contract covers their exact
semantics.

## First Mathematical Slice

The admitted carrier is the 64-bit `i64` bit pattern, and the admitted operators
are only bitwise AND, OR, and XOR:

```text
(x & a) & b = x & (a & b)
(x | a) | b = x | (a | b)
(x ^ a) ^ b = x ^ (a ^ b)
```

These identities hold pointwise for every bit. The first slice does not admit
integer Add or Mul through the new authority, even though the legacy optimizer
still contains its older Add/Mul path. It admits no floating-point operation.

## Compiler Transaction

The production mutation point is Block L of
`self-hosted/ir/opt_cleanup.sio`. For a candidate
`(base op c1) op c2`, the transaction:

1. derives the inner and outer instructions from the live `IrFunction`;
2. requires plain `IrBinOp` nodes with the same admitted bitwise operator;
3. requires the ordered edge `outer.src1 == inner.dst`;
4. resolves both constants from live `IrLoadImm` definitions;
5. refuses float-marked registers and intervening conflicting writes;
6. admits canonical scalar data operands and canonical explicit control operands
   (`label`, `jump`, conditional branch, return, and sret return), while refusing
   calls, phi lists, packed/wide registers, explicit float IR, and
   metadata-carried register operands;
7. derives a private two-word flow/use certificate: same-block candidates keep
   the uninterrupted-region proof, while cross-block candidates require local
   constant definitions, a canonical forward acyclic CFG, and an inner block
   that dominates the outer block;
8. requires the `c2` register to have exactly one modeled consumer across the
   whole function;
9. seals the six-word occurrence and two-word certificate as one private
   eight-word authority;
10. recaptures the same indexed window and rederives the certificate through the same uninterrupted `&!`
   function reference;
11. recomputes both constants and the combined value from live IR; and
12. mutates only the `c2` definition and outer instruction.

Historically, pass-A1 Blocks AE, BG, and BM performed the same direct AND, OR,
and XOR constant-chain rewrites before Block L. That ordering made the guarded
transaction unreachable from lowered source even though its isolated probe
passed. Those three direct chain rewrites are now deferred to authority-owned
Block L. Their tracking tables and non-rebracketing identities remain intact.

The capability is private, is never returned or serialized, and is consumed in
the local transaction. Function scope is carried by the uninterrupted mutable
reference, not by a function-name hash. Audit fingerprints are computed only
after the structural check; they are diagnostic and cannot authorize mutation.

The six-word occurrence, two-word flow certificate, eight-word sealed authority,
nine-word transaction audit, five-word probe receipt, and fourteen-word module
receipt are deliberately at or below the aggregate boundary
exercised by the current compiler artifact. The module receipt is public audit
evidence only; it contains no private occurrence capability and cannot authorize
a mutation. The local investigation observed silent corruption when large model
records and nested instruction aggregates crossed that artifact's by-value
path. This observation motivates compact transport; it is not a general theorem
about all Sounio backends. Their exact counts are gate tripwires: record growth
must force an explicit review rather than silently changing the transport shape.
The machine-readable native-v2 trace additionally declares `schema=1`; changing
the text protocol requires an explicit schema and gate update.

Certificate refusal is encoded inside the private two-word value: an invalid
certificate has `inner_const_def_index = -1` and a negative path-binding word
whose magnitude is the refusal reason. A valid path binding exactly packs the
inner and outer block-start indices as `inner * IR_MAX_INSTRS + outer`, without
a collision-prone hash. This avoids transporting a second
heterogeneous three-tuple through the current bootstrap. The private issuer
asserts nonnegative indices, so a refusal-shaped value cannot become authority.

Occurrence-capture refusal follows the same closed-value discipline. An invalid
six-word occurrence has negative indices and stores the negated refusal reason
in `operator_key`; capture consumers require nonnegative indices and one of the
valid operator keys `3..5` before certification. This removes the remaining
heterogeneous capture tuple from the bootstrap path without broadening the
authority surface.

The 256-register admission limit is named in the implementation and mirrors the
existing fixed register-indexed tables in cleanup passes A2/B. It is a boundary
of this pass, not an asserted global compiler limit. The forward reachability
witness likewise names a 2048-instruction local capacity matching the current
`IrFunction` array. If the global IR capacity grows, this authority remains
fail-closed until its fixed proof storage and gate are deliberately revised.

## Evidence Layers

`scripts/ci/exact_bitwise_rebracket_authority_gate.sh` and
`scripts/ci/exact_bitwise_rebracket_source_ir_gate.sh` separate seven layers:

- The scalar Sounio kernel executes 11 protocol cases: valid AND/OR/XOR, replay, wrong
  occurrence, swapped tree, arithmetic, float marker, shared constant, stale
  epoch, and a diagnostic-hash collision across distinct scopes.
- Adjacent two-module fixtures require exact E175 and E176 diagnostics for the
  private issuer and private authority constructor.
- The production probe adds malformed, non-dominating, backward-edge, base-kill,
  and phi-bearing CFG refusals plus one positive dominated forward diamond. With
  the same-region and operand-model cases, it executes 21 focused cases, five
  applications, and sixteen unchanged refusals. A dedicated compiler mode calls
  that probe from inside the built Madaros binary.
- A cleanup-pipeline probe passes a cross-block AND diamond through the real
  pass-A1/Block-L ordering and requires exactly one recorded transaction,
  authorization, and application. This prevents an earlier optimizer block from
  silently consuming the shape and proves that the transaction is reachable
  through the production pass, not only through its direct helper.
- The default native-v2 executable layer requires an exact disabled receipt and
  correct runtime result without `-O`; with `-O`, it requires exactly one
  application and the same runtime result for both a single module and an
  imported/merged module.
- The source-to-IR layer distinguishes same-region applications, cross-block
  applications, refusal reasons, and source ontology-parameter links in public
  audit evidence. It requires one cross-block application for both focused
  source routes, zero transactions for a non-dominating if-expression lowered
  through a join register, and one reason-19 refusal for a source function
  containing a backedge. The applied diamond and refused loop carry the same
  ontology class identity, so the ontology parameter can request evidence but
  cannot manufacture CFG authority. The receipt establishes only same-function
  adjacency; it does not claim that the erased parameter is a dataflow input to
  the rewritten expression.
- Static source anchors bind that protocol to Block L, the one-use and
  operand-encoding checks, float refusal, private declarations, compact records,
  the pass-A1 handoff, optimization-intent routing, and the exact operator set.

The use scan remains whole-function and fail-closed. One unsupported operand
encoding still refuses bitwise rebracketing for that function, even outside the
local window. Canonical control instructions are explicit and counted globally.
Same-block candidates retain the original uninterrupted-region certificate.
For a cross-block candidate, each constant definition must be local to its
binop, all labels and targets must be unique and present, every CFG edge must go
strictly forward, the outer block must be entry-reachable, and removing the
inner block must make the outer block unreachable. Because block-start indices
are then a topological order, the path-exclusion scan is exact for this admitted
DAG. The capture scan also refuses every lexical write to the base, constants,
or intermediate result between the two binops; conservatively checking all such
writes, including unreachable or mutually exclusive ones, supplies the no-kill
fact needed by this narrow cross-block rewrite. A pass-level precheck avoids
repeating a known unsupported-use refusal for every candidate, while the
transaction repeats the live use scan and certificate derivation before
mutation. Legacy Add/Mul never enters the new transaction.

The E175/E176 fixtures are minimal language-level privacy controls, not imports
of the production optimizer module. Static source anchors inspect the real
declarations. The standalone modular runner remains a diagnostic characterization:
the historical self-hosted modules contain real cross-module private calls and
an unrelated `IrFunction` initializer mismatch, so accepting that runner would
require weakening boundaries that this lane must preserve. Strict acceptance
instead executes `--rebracket-authority-smoke` inside a fresh compiler, where
`compiler/main.sio` already imports the production cleanup module. Likewise, the
scalar hash-collision case demonstrates only that a diagnostic hash cannot
replace structural identity. The production transaction does not use that hash
as function scope.

Local classification is intentionally not merge authority:

```bash
bash scripts/ci/exact_bitwise_rebracket_authority_gate.sh
bash scripts/ci/exact_bitwise_rebracket_source_ir_gate.sh
```

The strict acceptance path must execute the internal production smoke and the
focused default native-v2 paths with a current-source compiler:

```bash
SOUNIO_REBRACKET_COMPILER_BIN=/path/to/current-source-madaros \
SOUNIO_REBRACKET_EXPECTED_COMPILER_SHA256=<sha256-of-that-elf> \
SOUNIO_REBRACKET_REQUIRE_COMPILER=1 \
bash scripts/ci/exact_bitwise_rebracket_authority_gate.sh

SOUNIO_REBRACKET_COMPILER_BIN=/path/to/current-source-madaros \
SOUNIO_REBRACKET_EXPECTED_COMPILER_SHA256=<sha256-of-that-elf> \
SOUNIO_REBRACKET_REQUIRE_COMPILER=1 \
bash scripts/ci/exact_bitwise_rebracket_source_ir_gate.sh
```

The strict form requires an explicit raw ELF, its expected SHA-256, and a clean
tracked worktree. Its receipt records both compiler and source hashes. The
current-source compiler must be built in the Sounio Compiler Foundry or the
approved Slurm path; the focused gate may then execute against the downloaded
artifact without running a full stress build in `/workspace/sounio`.

PR #1001 CI run `29436451446` built source SHA
`7e256c64177f7a66e92b2d065e81e707344ec0db`. Artifact `8351758740`
contained a 98,646,693-byte Madaros ELF with SHA-256
`9c46090f624363fee3fbc28c7c8d751018ace3c5ed87ce30185754821c9ecbfc`.
The strict gate returned:

```text
compiler_state=executable compiler_path=internal-smoke
source_sha=7e256c64177f7a66e92b2d065e81e707344ec0db merge_ready=1
```

That run closed the internal-smoke evidence gap only. The first default-path
attempts then exposed two independent problems rather than being accepted as
partial success: the audit trace was not initially machine-readable, and pass-A1
consumed the source chain before Block L. After repairing the trace and making
Block L the sole owner of exact bitwise chain reassociation, PR #1001 CI run
`29439952545`, job `87436301034`, built source SHA
`d7880b25d79071a1c2811895907afb49dfb79175`. Artifact `8353130310`
contained a 98,642,537-byte Madaros ELF with SHA-256
`0137f0be1c595a0d890076d2630af34e8b11508d35b92aaad48f574b343c6d45`.

The branchless runtime witnesses at source SHA
`fef880b255d2329806e5b1407f526d4a574e9bc3` were compiled by that ELF. The
strict gate returned:

```text
kernel=11/11 privacy=E175,E176 compiler_state=executable
compiler_path=internal-smoke+default-o native_v2_reachability=single-and-merged
compiler_sha256=0137f0be1c595a0d890076d2630af34e8b11508d35b92aaad48f574b343c6d45
source_sha=fef880b255d2329806e5b1407f526d4a574e9bc3 merge_ready=1
```

The witness binds `observed` to the return of `authority_apply(171)`. Both the
source and rewritten function return a value masked by `& 15`, so `observed` is
in `0..15`; consequently, `observed ^ 11` is zero exactly when the returned
value equals 11, and every returned-value mismatch produces one of `1..15`.
This avoids depending on an unrelated historical optimized
conditional-control-flow path while still executing the transformed function.

PR #1046 CI run `29552524219`, job `87798137029`, built source SHA
`394934a4e7c2a7d84b2c222743e66608cc1c5aac`. Artifact `8396343656`
contained a 98,756,167-byte Madaros ELF with SHA-256
`6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88`.
The downloaded ELF passed the inherited strict authority gate:

```text
compiler_path=internal-smoke+default-o native_v2_reachability=single-and-merged
compiler_sha256=6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88
source_sha=394934a4e7c2a7d84b2c222743e66608cc1c5aac merge_ready=1
```

The source-to-IR gate then compiled and executed all five runtime controls with
the same ELF and no fallback. Its exact audit receipts distinguished the
positive single-module and imported-leaf applications from the non-dominating
control and loop refusal:

```text
single:   transactions=1 applications=1 cross_block_applications=1 transaction_ontology_parameter_links=1 application_ontology_parameter_links=1 last_ontology_class_hash=7199034902620903764 combined=15
imported: transactions=1 applications=1 cross_block_applications=1 transaction_ontology_parameter_links=1 application_ontology_parameter_links=1 last_ontology_class_hash=7199034902620903764 combined=15
nondom:   transactions=0 applications=0 transaction_ontology_parameter_links=0
loop:     transactions=1 authorizations=0 applications=0 refusals=1 refusal_reason_mask=524288 transaction_ontology_parameter_links=1 application_ontology_parameter_links=0 last_ontology_class_hash=7199034902620903764
```

The gate also rejected the unrelated ontology observation during type checking
and returned `runtime_parity=5`, `ontology_cannot_authorize_loop=1`, and
`merge_ready=1`.

## D7 Boundary

The psychiatric-regime D7 work motivates occurrence-bound authority and the
distinction between observational receipts and functional state. Its runtime
receipts are public model evidence, not compiler capabilities. This compiler
slice imports and consumes none of them.

The source fixtures now declare `ExactBitwiseRebracketObligation` as a subclass
of `CompilerSemanticObligation` and use it in the optimized function signature.
The checker accepts that subsumption and rejects a
`ReceptorOccupancyObservation` in its place. A collection-only replay after the
authoritative typecheck preserves only the function, parameter position, and
class name as audit links in `IrOntologyTable`; it does not overwrite any
semantic class/property table. The cleanup receipt reports only a parameter-link
count and class hash. Cleanup asserts that the lowered function name remains
stable before making that lookup; a future symbol-renaming pass therefore fails
loudly instead of silently dropping the link. The modular witness places both
the ontology declaration and transformed function in the imported leaf, so its
receipt exercises link transport across the actual module merge. Link merge is
idempotent on `(function_name, parameter_index, class_name)`. This is a
function-level identity bridge, not a dataflow or authority bridge: the loop
witness carries the same ontology identity and is still refused by the CFG
certificate. D7 receipts remain adjacent observational evidence and are queried
only after the authority-owned cleanup decision; they never participate in
authority derivation or the mutation decision.

## Literature Compass

The design is informed by, but does not claim the guarantees of:

- George Necula, "Proof-Carrying Code," POPL 1997,
  <https://doi.org/10.1145/263699.263712>.
- Amir Pnueli, Michael Siegel, and Eli Singerman, "Translation Validation,"
  TACAS 1998, <https://doi.org/10.1007/BFb0054170>.
- Nuno P. Lopes et al., "Alive2: Bounded Translation Validation for LLVM,"
  PLDI 2021, <https://web.ist.utl.pt/nuno.lopes/pubs.php?id=alive2-pldi21>.
- Xavier Leroy, "Formal Verification of a Realistic Compiler," CACM 2009,
  <https://doi.org/10.1145/1538788.1538814>.
- Keith D. Cooper, Timothy J. Harvey, and Ken Kennedy, "A Simple, Fast
  Dominance Algorithm," 2001, <https://www.cs.rice.edu/~keith/EMBED/dom.pdf>.
- Ron Cytron et al., "Efficiently Computing Static Single Assignment Form and
  the Control Dependence Graph," TOPLAS 1991,
  <https://doi.org/10.1145/115372.115320>.

The nearest analogy is per-transformation validation: authority is derived for
one live rewrite rather than inferred from a global optimizer flag. Unlike PCC,
this slice carries no machine-checkable proof term. Unlike Alive2, it performs
no SMT refinement proof. Unlike CompCert, it provides no compiler-wide semantic
preservation theorem. The current forward-DAG slice does not implement the
general iterative dominator or SSA algorithms from Cooper-Harvey-Kennedy or
Cytron et al.; its exact path-exclusion test is valid only because every admitted
edge follows lexical block order.

## Claims Introduced

Only after the strict gate passes may this lane claim:

- Block L routes the exact ordered `i64` AND/OR/XOR constant-chain rewrite
  through a private occurrence-bound transaction.
- The transaction refuses replayed structure, a wrong occurrence, swapped
  operands, arithmetic operators, float markers, shared `c2` constants, a stale
  operator window, call-bearing IR, packed/implicit register encodings,
  non-dominating joins, malformed labels or targets, backward edges, intervening
  base writes, phi-bearing IR, and explicit float IR in its focused witnesses.
- A canonical parameter-consuming branch/label after an otherwise certified
  local tree no longer suppresses the transaction.
- One compiler-internal forward diamond is admitted when the inner block
  dominates the joined outer block and the exact no-kill/use obligations hold.
  The negative witnesses leave the candidate structurally unchanged and expose
  distinct refusal reasons for malformed CFG, non-dominance, backedges,
  intervening writes, and unsupported phi operands.
- Default native-v2 `-O` reaches the transaction once for the focused
  single-module witness and once after imported-module finalization, with both
  optimized executables preserving the witness result.
- Focused single-module and imported-leaf source diamonds each produce one
  cross-block application and preserve the executable result. The public
  receipt records zero same-region applications and one checked
  ontology-parameter link for those runs. In the modular witness, both the link
  and transformed function originate in the imported leaf.
- A branch-local definition that does not dominate its join is lowered through
  a join register and produces no candidate transaction. A separate source
  function containing a loop produces one unchanged reason-19 refusal even
  though its transaction carries the same ontology class identity as the
  admitted diamond.
- An unrelated `ReceptorOccupancyObservation` cannot discharge the compiler
  semantic obligation in the focused compile-fail witness.
- Public audit evidence cannot be consumed as mutation authority.

## Claims Forbidden

- General proof-carrying compilation.
- Compiler-wide or source-to-native semantic preservation.
- Float, GUM, interval, p-box, stochastic, or clinical rebracketing authority.
- Authorization from a D7 runtime receipt, PET observation, ontology label, or
  diagnostic fingerprint.
- Cryptographic unforgeability.
- Coverage of source shapes other than the focused same-block, forward-diamond,
  join-copy, and loop-refusal witnesses, of
  all optimization pipelines, or of compiler-wide runtime behavior.
- Loop, backedge, phi-edge, irreducible, or arbitrary-CFG authority. The
  certificate recognizes only same-block regions and canonical forward DAGs;
  it is not a reusable dominator tree or a general SSA reaching-definition
  analysis.
- General source-level cross-block reachability. The executable claim is limited
  to the exact single-module and imported/merged forward-diamond fixtures bound
  by the source-to-IR gate.
- A claim that Add/Mul legacy reassociation now satisfies this authority model.
- Merge readiness while the strict gate reports a blocked production smoke.

## Closed Compiler-Smoke Blocker

```text
Blocker-ID: BLK-20260715-REBRACKET-CURRENT-SOURCE-SMOKE
Status: closed
Severity: B3
Class: evidence-gap
Owner: Codex rebracketing-authority coordination lane
Lane: exact bitwise rebracketing authority
Worktree: /tmp/sounio-rebracketing-authority-compiler-20260715
Branch: codex/rebracketing-authority-binding-20260715
Files-Owned: self-hosted/compiler/main.sio; scripts/ci/exact_bitwise_rebracket_authority_gate.sh; docs/internal/concepts/rebracketing-authority.md
Files-Read-Only: self-hosted/check/*; self-hosted/ir/ir.sio; self-hosted/ir/egraph.sio; blocked issue #854 and SOIR stacks
Do-Not-Touch: contextual visibility semantics; IrFunction/SOIR capacity stack; legacy Add/Mul rewrite
Repro: SOUNIO_REBRACKET_COMPILER_BIN=<downloaded-artifact>/madaros SOUNIO_REBRACKET_EXPECTED_COMPILER_SHA256=9c46090f624363fee3fbc28c7c8d751018ace3c5ed87ce30185754821c9ecbfc SOUNIO_REBRACKET_REQUIRE_COMPILER=1 bash scripts/ci/exact_bitwise_rebracket_authority_gate.sh
Observed: the fresh compiler advertised the internal mode and emitted the exact 14-case receipt with merge_ready=1; the checked-in prebuilt still classifies separately as blocked-prebuilt-no-smoke
Expected: a compiler built from this branch advertises the focused mode and prints its exact PASS receipt
Acceptance-Gate: SOUNIO_REBRACKET_COMPILER_BIN=<current-source-foundry-madaros> SOUNIO_REBRACKET_EXPECTED_COMPILER_SHA256=<artifact-sha256> SOUNIO_REBRACKET_REQUIRE_COMPILER=1 bash scripts/ci/exact_bitwise_rebracket_authority_gate.sh
Evidence-Level: E4
Evidence: PR #1001 CI run 29436451446, job 87424483977, artifact 8351758740, compiler SHA-256 9c46090f624363fee3fbc28c7c8d751018ace3c5ed87ce30185754821c9ecbfc
Fallback-Path: none
Legacy-Kept: yes; legacy Add/Mul rewriting remains explicitly outside this authority claim
LLM-Offload: logged:.claude/llm_offload_log.md
Next-Action: completed by BLK-20260715-REBRACKET-DEFAULT-O-REACHABILITY below
```

## Closed Default-`-O` Reachability Blocker

```text
Blocker-ID: BLK-20260715-REBRACKET-DEFAULT-O-REACHABILITY
Status: closed
Severity: B3
Class: evidence-gap
Owner: Codex rebracketing-authority coordination lane
Lane: exact bitwise rebracketing authority
Worktree: /tmp/sounio-rebracketing-authority-compiler-20260715
Branch: codex/rebracketing-authority-binding-20260715
Files-Owned: self-hosted/ir/opt_cleanup.sio; self-hosted/compiler/main.sio; self-hosted/compiler/module_frontend.sio; self-hosted/compiler/module_native_driver.sio; tests/compiler/rebracket_authority_*; scripts/ci/exact_bitwise_rebracket_authority_gate.sh; docs/internal/concepts/rebracketing-authority.md
Files-Read-Only: scripts/lib/resolve_souc.sh; scripts/lib/resolve_madaros.sh; bin/souc; bin/madaros; .github/workflows/ci.yml
Do-Not-Touch: legacy Add/Mul rewrite; compact non-optimized imported path; resolver fallback policy; high-risk shared wrappers
Repro: SOUNIO_REBRACKET_COMPILER_BIN=<downloaded-artifact>/madaros SOUNIO_REBRACKET_EXPECTED_COMPILER_SHA256=0137f0be1c595a0d890076d2630af34e8b11508d35b92aaad48f574b343c6d45 SOUNIO_REBRACKET_REQUIRE_COMPILER=1 bash scripts/ci/exact_bitwise_rebracket_authority_gate.sh
Observed: no-opt emitted an exact disabled receipt and returned zero; single-module and imported/merged -O each emitted one transaction, one authorization, one application, combined constant 15, and returned zero
Expected: the default native-v2 frontend transports optimize intent and reaches authority-owned Block L after call resolution/finalization for both focused source shapes
Acceptance-Gate: strict exact_bitwise_rebracket_authority_gate.sh with an explicit current-source Foundry ELF and expected SHA-256
Evidence-Level: E4
Evidence: PR #1001 CI run 29439952545, job 87436301034, artifact 8353130310, compiler SHA-256 0137f0be1c595a0d890076d2630af34e8b11508d35b92aaad48f574b343c6d45; local strict receipt at source fef880b255d2329806e5b1407f526d4a574e9bc3
Fallback-Path: none
Legacy-Kept: yes; legacy Add/Mul, non-rebracketing AE/BG/BM consumers, and the non-optimized compact imported path remain
LLM-Offload: logged:.claude/llm_offload_log.md
Next-Action: preserve the focused claim boundary; broaden only with new typed dominance/use certificates and separate executable witnesses
```

## Semantic Lane Declaration

```text
Semantic-Lane-ID: rebracketing-authority-forward-cfg-v1
Owner: Codex coordinated compiler lane
Concept-IDs: SOUNIO-REBRACKETING-AUTHORITY; SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: parenthesization changes only under exact local authority; nonassociative and uncertain regimes remain ordered
Transformation: route one ordered i64 AND/OR/XOR constant-chain rewrite through private capture, revalidation, and mutation
Types-Changed: private optimizer occurrence, exact flow/use certificate with an injective inner/outer block binding, authority, and transaction audit; public compact probe and module audit receipts
Effects-Changed: none
IR-Changed: no new opcode or field; existing Block L mutation is gated
Claims-Introduced: focused exact-bitwise transaction exercised by the strict current-source internal smoke; one compiler-internal dominated forward diamond reaches the production cleanup pipeline; inherited same-block single-module and imported/merged default native-v2 -O witnesses remain executable
Claims-Forbidden: formal proof, float/GUM/clinical authority, D7 receipt consumption, source-level cross-block reachability, loops, phi-edge or arbitrary CFG authority, compiler-wide preservation, cryptographic sealing
Assumptions: the carrier is integer-only; both constants have local live IrLoadImm definitions; c2 has one globally modeled use; every instruction uses an admitted scalar or explicit control encoding; cross-block candidates require unique present labels, strict forward edges, entry reachability, exact path-exclusion dominance, and no lexical conflicting write; calls, phi lists, explicit float IR, packed or implicit operands, and all backedges cause conservative refusal
Write-Set: self-hosted/ir/opt_cleanup.sio; self-hosted/ir/rebracket_authority_self_test_runner.sio; self-hosted/compiler/main.sio; tests/compiler/rebracket_authority_*; scripts/ci/exact_bitwise_rebracket_authority_gate.sh; docs/internal/concepts/rebracketing-authority.md; docs/internal/concepts/registry.tsv
Read-Set: self-hosted/ir/ir.sio; self-hosted/ir/egraph.sio; self-hosted/check/*; scripts/ci/no_false_float_axioms.sh; tests/compiler/madaros_visibility_context/*
Positive-Witness: scalar kernel AND/OR/XOR masks; production probe application mask 7; same-region application with later parameter-consuming branch/label; dominated cross-block OR diamond; cleanup-pipeline cross-block AND diamond; inherited default native-v2 single and imported/merged same-block AND runtime witnesses
Negative-Witness: scalar refusal matrix; exact E175/E176 fixtures; production refusal mask 65535 including call-bearing, packed-register, explicit-float, non-dominating reason 21, backedge reason 19, duplicate-label reason 17, missing-target reason 18, base-kill reason 10, and phi reason 14 witnesses
Acceptance-Gate: strict exact_bitwise_rebracket_authority_gate.sh with a current-source Foundry compiler
Integration-Target: current-source Madaros optimizer and focused default native-v2 -O reachability
Authoritative-Only-If: the narrow internal-transaction and focused default-path claims use a recorded clean source SHA and explicit compiler hash with no fallback; any broader reachability claim requires a new executable gate
```

## Source-To-IR Semantic Lane Declaration

```text
Semantic-Lane-ID: rebracketing-authority-source-ir-v1
Owner: Codex source-to-IR compiler lane
Concept-IDs: SOUNIO-REBRACKETING-AUTHORITY; SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: source parenthesization changes only when lowering preserves a candidate whose live IR receives exact local authority
Transformation: expose same-region applications, cross-block applications, refusal reasons, and checked source ontology-parameter links in audit-only cleanup receipts; bind focused source programs to those receipts and native runtime parity
Types-Changed: OcpExactBitwiseRebracketAudit gains a private scope key; IrOntologyTable gains bounded function/parameter/class identity links; OcpCleanupModuleReceipt gains same-region count, cross-block count, refusal-reason mask, transaction/application ontology-parameter-link counts, and the last ontology class hash
Effects-Changed: none
IR-Changed: no opcode or executable instruction changes; audit-only ontology signature links are retained in IrOntologyTable and merged by the modular frontend
Claims-Introduced: focused single-module and imported-leaf source diamonds reach exactly one dominated cross-block transaction with one same-function ontology-parameter link; the imported receipt proves that link survives modular merge; a non-dominating source join forms no false candidate; a source loop candidate carries the same ontology identity and is refused as reason 19; an unrelated observation cannot discharge the obligation type
Claims-Forbidden: general source-to-native preservation, arbitrary source control flow, loop or phi authority, float/GUM/clinical reassociation, ontology-derived authority, compiler-wide reachability
Assumptions: the existing forward-DAG certificate remains unchanged; cleanup preserves the lowered function name and asserts that invariant before the audit lookup; receipt counters are evidence and cannot authorize mutation; the focused if-expression lowering uses explicit branch, copy, jump, and label instructions
Write-Set: self-hosted/ir/ir.sio; self-hosted/ir/opt_cleanup.sio; self-hosted/check/mod.sio; self-hosted/compiler/main.sio; self-hosted/compiler/module_frontend.sio; tests/compiler/rebracket_authority_*; scripts/ci/exact_bitwise_rebracket_authority_gate.sh; scripts/ci/exact_bitwise_rebracket_source_ir_gate.sh; docs/internal/concepts/rebracketing-authority.md; docs/internal/concepts/registry.tsv; docs/internal/concepts/bindings.tsv
Read-Set: self-hosted/ir/lower.sio; self-hosted/compiler/module_native_driver.sio; scripts/lib/resolve_souc.sh; bin/souc
Positive-Witness: rebracket_authority_cross_block_source.sio and the function defined in rebracket_authority_cross_block_imported_leaf.sio each require one cross-block application, one transaction/application ontology-parameter link with the ExactBitwiseRebracketObligation class hash, combined constant 15, and runtime exit zero
Negative-Witness: rebracket_authority_nondominating_source.sio requires zero transactions after join-copy lowering; rebracket_authority_loop_refusal_source.sio requires one ontology-parameter-linked refusal with reason-mask bit 19 and runtime exit zero without executing the historical loop body; rebracket_authority_unrelated_ontology_obligation.sio must fail type checking
Acceptance-Gate: strict exact_bitwise_rebracket_source_ir_gate.sh with a current-source Foundry compiler and expected SHA-256
Integration-Target: default native-v2 single-module and imported/merged optimized source paths
Authoritative-Only-If: the source and compiler SHAs are recorded, the inherited authority gate reports merge_ready=1, all five runtime controls return zero, and no fallback occurs
```

## Integration Receipt

```text
Semantic-Outcome: implementation-shaped hypothesis with executable scalar protocol, fresh-build-gated same-block and forward-DAG production transactions, focused default native-v2 single/imported reachability, and a current-source-validated audit-only source ontology identity link
Concept-Status-Before: unregistered
Concept-Status-After: hypothesis
Distinctions-Added: observed receipt != compiler authority; ontology obligation != compiler authority; diagnostic hash != scope identity; algebraic law != rewrite permission; explicit control-use model != flow authority; forward-DAG path-exclusion dominance != a general dominator tree; compiler-internal cross-block witness != source-level cross-block reachability; internal probe != default-path reachability; focused reachability != compiler-wide preservation
Distinctions-Preserved: parenthesization != formatting; compile success != runtime parity; formal model != empirical or clinical claim
Distinctions-Erased: none
Evidence-Run: local 11-case kernel; exact E175/E176 controls; no-false-float guard; prebuilt-no-smoke classifier; PR #1001 and #1013 current-source builds and inherited optimizer/default-path evidence; PR #1046 run 29552524219 job 87798137029 artifact 8396343656; strict current-source authority and source-to-IR gates at source 394934a4e7c2a7d84b2c222743e66608cc1c5aac with compiler SHA-256 6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88; five runtime controls, unrelated ontology rejection, imported-leaf link merge, and loop reason-19 refusal all passed without fallback
Fallback-Path: none
Legacy-Kept: legacy Add/Mul Block L path retained outside the new claim
Conflicting-Lanes: none; issue #854 and IrFunction/SOIR capacity stacks remain read-only and are no longer prerequisites for the focused smoke
Next-Semantic-Interface: loop/phi-capable dominance and reaching-definition certificate -> occurrence authority -> native-v2 receipt
```
