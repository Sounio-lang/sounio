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

Status: hypothesis. The scalar protocol, the production cleanup pipeline, and
the focused default native-v2 `-O` paths are executable under the strict
hash-bound gate. The executable claim covers one single-module witness and one
imported/merged witness; it is not compiler-wide semantic preservation.

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
6. admits only integer, straight-line canonical scalar operand encodings, refusing calls,
   control flow without a dominance certificate, phi lists, packed/wide
   registers, explicit float IR, and metadata-carried register operands;
7. requires the `c2` register to have exactly one observable consumer;
8. issues a private six-word occurrence capability;
9. recaptures the same indexed window through the same uninterrupted `&!`
   function reference;
10. recomputes both constants and the combined value from live IR; and
11. mutates only the `c2` definition and outer instruction.

Historically, pass-A1 Blocks AE, BG, and BM performed the same direct AND, OR,
and XOR constant-chain rewrites before Block L. That ordering made the guarded
transaction unreachable from lowered source even though its isolated probe
passed. Those three direct chain rewrites are now deferred to authority-owned
Block L. Their tracking tables and non-rebracketing identities remain intact.

The capability is private, is never returned or serialized, and is consumed in
the local transaction. Function scope is carried by the uninterrupted mutable
reference, not by a function-name hash. Audit fingerprints are computed only
after the structural check; they are diagnostic and cannot authorize mutation.

The six-word occurrence, eight-word transaction audit, five-word probe receipt,
and eight-word module receipt are deliberately below the aggregate boundary
exercised by the current compiler artifact. The module receipt is public audit
evidence only; it contains no private occurrence capability and cannot authorize
a mutation. The local investigation observed silent corruption when large model
records and nested instruction aggregates crossed that artifact's by-value
path. This observation motivates compact transport; it is not a general theorem
about all Sounio backends. Their exact counts are gate tripwires: record growth
must force an explicit review rather than silently changing the transport shape.

The 256-register admission limit is named in the implementation and mirrors the
existing fixed register-indexed tables in cleanup passes A2/B. It is a boundary
of this pass, not an asserted global compiler limit.

## Evidence Layers

`scripts/ci/exact_bitwise_rebracket_authority_gate.sh` separates five layers:

- The scalar Sounio kernel executes 11 protocol cases: valid AND/OR/XOR, replay, wrong
  occurrence, swapped tree, arithmetic, float marker, shared constant, stale
  epoch, and a diagnostic-hash collision across distinct scopes.
- Adjacent two-module fixtures require exact E175 and E176 diagnostics for the
  private issuer and private authority constructor.
- The production probe adds call-bearing, packed-register, uncertified
  control-flow, and explicit-float-IR refusal witnesses, for 14 focused cases and
  eleven unchanged refusals. A dedicated compiler mode calls that probe from
  inside the built Madaros binary.
- A cleanup-pipeline probe passes an AND chain through the real pass-A1/Block-L
  ordering and requires exactly one recorded transaction, authorization, and
  application. This prevents an earlier optimizer block from silently consuming
  the same source shape before the authority transaction.
- The default native-v2 executable layer requires an exact disabled receipt and
  correct runtime result without `-O`; with `-O`, it requires exactly one
  application and the same runtime result for both a single module and an
  imported/merged module.
- Static source anchors bind that protocol to Block L, the one-use and
  operand-encoding checks, float refusal, private declarations, compact records,
  the pass-A1 handoff, optimization-intent routing, and the exact operator set.

The operand scan is deliberately whole-function and fail-closed. One unsupported
opcode refuses bitwise rebracketing for that function, even outside the local
window; this sacrifices optimization coverage until use/dominance analysis is
certified. A pass-level precheck avoids repeating that known refusal for every
candidate, while the transaction still performs a live scan before mutation. It
does not route legacy Add/Mul through the new transaction.

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
```

The strict acceptance path must execute the internal production smoke and the
focused default native-v2 paths with a current-source compiler:

```bash
SOUNIO_REBRACKET_COMPILER_BIN=/path/to/current-source-madaros \
SOUNIO_REBRACKET_EXPECTED_COMPILER_SHA256=<sha256-of-that-elf> \
SOUNIO_REBRACKET_REQUIRE_COMPILER=1 \
bash scripts/ci/exact_bitwise_rebracket_authority_gate.sh
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

## D7 Boundary

The psychiatric-regime D7 work motivates occurrence-bound authority and the
distinction between observational receipts and functional state. Its runtime
receipts are public model evidence, not compiler capabilities. This compiler
slice imports and consumes none of them.

An eventual source-level ontology may describe rebracketing obligations and
transport them into IR. Until that interface exists, D7 and this compiler
transaction are adjacent evidence lanes, not one end-to-end proof.

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

The nearest analogy is per-transformation validation: authority is derived for
one live rewrite rather than inferred from a global optimizer flag. Unlike PCC,
this slice carries no machine-checkable proof term. Unlike Alive2, it performs
no SMT refinement proof. Unlike CompCert, it provides no compiler-wide semantic
preservation theorem.

## Claims Introduced

Only after the strict gate passes may this lane claim:

- Block L routes the exact ordered `i64` AND/OR/XOR constant-chain rewrite
  through a private occurrence-bound transaction.
- The transaction refuses replayed structure, a wrong occurrence, swapped
  operands, arithmetic operators, float markers, shared `c2` constants, a stale
  operator window, call-bearing IR, packed/implicit register encodings, control
  flow without a dominance certificate, and explicit float IR in its focused
  witness.
- Default native-v2 `-O` reaches the transaction once for the focused
  single-module witness and once after imported-module finalization, with both
  optimized executables preserving the witness result.
- Public audit evidence cannot be consumed as mutation authority.

## Claims Forbidden

- General proof-carrying compilation.
- Compiler-wide or source-to-native semantic preservation.
- Float, GUM, interval, p-box, stochastic, or clinical rebracketing authority.
- Authorization from a D7 runtime receipt, PET observation, ontology label, or
  diagnostic fingerprint.
- Cryptographic unforgeability.
- Coverage of source shapes other than the two focused native-v2 witnesses, of
  all optimization pipelines, or of compiler-wide runtime behavior.
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
Semantic-Lane-ID: rebracketing-authority-compiler-v0
Owner: Codex coordinated compiler lane
Concept-IDs: SOUNIO-REBRACKETING-AUTHORITY; SOUNIO-NONASSOCIATIVE-ORDER
Intent-Preserved: parenthesization changes only under exact local authority; nonassociative and uncertain regimes remain ordered
Transformation: route one ordered i64 AND/OR/XOR constant-chain rewrite through private capture, revalidation, and mutation
Types-Changed: private optimizer occurrence, authority, and transaction audit; public compact probe and module audit receipts
Effects-Changed: none
IR-Changed: no new opcode or field; existing Block L mutation is gated
Claims-Introduced: focused exact-bitwise transaction exercised by the strict current-source internal smoke and reached once by each of the single-module and imported/merged default native-v2 -O witnesses
Claims-Forbidden: formal proof, float/GUM/clinical authority, D7 receipt consumption, coverage beyond the focused native-v2 witnesses, compiler-wide preservation, cryptographic sealing
Assumptions: the admitted IrFunction slice is integer-only and straight-line and uses SSA-like register definitions within the inspected window; c2 has one observable use; every instruction uses the admitted canonical scalar operand encoding; calls, control flow, phi lists, explicit float IR, and packed or implicit operand forms cause conservative refusal
Write-Set: self-hosted/ir/opt_cleanup.sio; self-hosted/ir/rebracket_authority_self_test_runner.sio; self-hosted/compiler/main.sio; tests/compiler/rebracket_authority_*; scripts/ci/exact_bitwise_rebracket_authority_gate.sh; docs/internal/concepts/rebracketing-authority.md; docs/internal/concepts/registry.tsv
Read-Set: self-hosted/ir/ir.sio; self-hosted/ir/egraph.sio; self-hosted/check/*; scripts/ci/no_false_float_axioms.sh; tests/compiler/madaros_visibility_context/*
Positive-Witness: scalar kernel AND/OR/XOR masks; production probe application mask 7; cleanup-pipeline AND application; default native-v2 single and imported/merged AND runtime witnesses
Negative-Witness: scalar refusal matrix; exact E175/E176 fixtures; production refusal mask 2047 including call-bearing, packed-register, uncertified control-flow, and explicit-float IR
Acceptance-Gate: strict exact_bitwise_rebracket_authority_gate.sh with a current-source Foundry compiler
Integration-Target: current-source Madaros optimizer and focused default native-v2 -O reachability
Authoritative-Only-If: the narrow internal-transaction and focused default-path claims use a recorded clean source SHA and explicit compiler hash with no fallback; any broader reachability claim requires a new executable gate
```

## Integration Receipt

```text
Semantic-Outcome: implementation-shaped hypothesis with executable scalar protocol, fresh-build-gated production transaction, and focused default native-v2 single/imported reachability
Concept-Status-Before: unregistered
Concept-Status-After: hypothesis
Distinctions-Added: observed receipt != compiler authority; diagnostic hash != scope identity; algebraic law != rewrite permission; internal probe != default-path reachability; focused reachability != compiler-wide preservation
Distinctions-Preserved: parenthesization != formatting; compile success != runtime parity; formal model != empirical or clinical claim
Distinctions-Erased: none
Evidence-Run: local 11-case kernel; exact E175/E176 controls; no-false-float guard; prebuilt-no-smoke classifier; PR #1001 current-source builds; strict hash-bound internal/pipeline smoke; no-opt control; single-module and imported/merged default native-v2 -O receipts and runtime executions
Fallback-Path: none
Legacy-Kept: legacy Add/Mul Block L path retained outside the new claim
Conflicting-Lanes: none; issue #854 and IrFunction/SOIR capacity stacks remain read-only and are no longer prerequisites for the focused smoke
Next-Semantic-Interface: source ontology obligation -> typed dominance/use certificate -> occurrence authority -> native-v2 receipt
```
