<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r29-2026-08-01
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r29-2026-08-01
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R29 — the import wall falls, and R1 is falsified by its own line

**Date:** 2026-08-01
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `CLOSURE_WALKED__MODULE_CLOSURE_PASSES`
**Parents:** `self_falsifying_compilation_line_r1_2026-07-26.md` (measured the wall), `self_falsifying_compiler_spec_2026-07-25.md` §6 (stated the limitation)
**Harness:** `scripts/ci/self_falsifying_compilation_line_r29_gate.sh`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r29_gate.sh`

---

## 1. What was true until today

`claim_executor_verify` loaded exactly one file — the compilation's main source —
repopulated the claim registry from it, and verified the claims it found there.
The spec said so plainly (§6). R1 turned that sentence into a probe:
`self_falsifying_modclosure_main.sio` imports a module carrying a claim whose
gate always exits non-zero, and carries one green claim of its own. On
2026-07-26 the probe returned `VERIFY_CLAIMS_OK pass=1` with an ELF on disk. The
false claim one import away was invisible.

R1 recorded that as `MODULE_CLOSURE_BLOCKS` and said so in the token. It was a
true reading. It is now false, because the mechanism changed.

## 2. The change

`claim_executor_verify` now collects the module closure before verifying
anything, and verifies each module in it:

```
let closure_complete = module_frontend_collect_ast_closure_into(source_path, &!closure)
var scope_count: i64 = closure.node_count
...
while mod_idx < scope_count {
    let module_path = closure.paths[mod_idx]
    let main_prog = load_module_file(module_path)
    let total = ast_claim_count()
    ...
}
```

Two ordering facts make this the only shape that works, and both are why the
loop could not simply be nested inside the existing one:

* Every `load_module_file` resets the claim registry — `ast_reset_claims()` in
  `parser/mod.sio:92`, guarded by `GLOBAL_VAR_INIT_SUPPRESS_RESET`. Collection
  and verification therefore cannot interleave: the collector reloads every
  module it walks, so it must finish before the first verification load.
* The reset is also what keeps the counts honest. Because each module's load
  clears what the previous one left, `pass=` counts claims, not visits.

**Scope chosen: the full transitive closure, not depth-1.** The plan called for
depth-1. Depth-1 is available — `closure.edge_callers` filtered against
`source_path` — and was rejected: it would let a claim refuted two levels down
pass unseen, which is the exact hole this rung exists to close. The transitive
form is also the simpler code (iterate `closure.paths`, do not filter edges).
The cost of the wider scope is measured in D4 below and is nil.

When the closure comes back incomplete the executor prints
`VERIFY_CLAIMS_SCOPE_PARTIAL modules=N` before its total, so a reader is never
handed a count that quietly covers less than they will assume it covers.

## 3. Measurements

Binary: Madaros built from this tree, 102 216 722 bytes. All four run with
prefix-anchored greps — `print_int` newline-terminates, so `pass=2` never shares
a line with what follows it.

| clause | probe | expected | observed |
|---|---|---|---|
| **D1** | red imported claim, `--verify-claims` | build blocked, no ELF | `VERIFY_CLAIMS_SCOPE modules=2`, `CLAIM_PASS mcl_main_claim_that_holds`, `CLAIM_FAIL mcl_library_claim_that_is_false`, `VERIFY_CLAIMS_FALSIFIED fail=1`, rc=1, ELF absent — **PASS** |
| **D2** | green pair (`mcl_green_main.sio`) | both claims run, build proceeds | `VERIFY_CLAIMS_SCOPE modules=2`, two `CLAIM_PASS`, `VERIFY_CLAIMS_OK pass=2`, rc=0, ELF present — **PASS** |
| **D3** | same red pair, **no** `--verify-claims` | unaffected | rc=0, ELF present, zero `VERIFY_CLAIMS` lines — **PASS** |
| **D4** | `rupture_claims_verified.sio` (0 imports) | invariant | `VERIFY_CLAIMS_SCOPE modules=1`, 8 `CLAIM_PASS`, `VERIFY_CLAIMS_FALSIFIED fail=8`; three trials each side, 10/11/10 s pre-R29 against 11/11/10 s post — **PASS** |
| **D5** | chain: refuted claim **two** imports away | blocked | `VERIFY_CLAIMS_SCOPE modules=3`, importer and middle green, `CLAIM_FAIL mcl_chain_leaf_claim_false`, `fail=1`, no ELF — **PASS** |
| **D6** | diamond: two arms importing one leaf | leaf visited once | `VERIFY_CLAIMS_SCOPE modules=4`, `VERIFY_CLAIMS_OK pass=4` (not 5) — **PASS** |
| **D7** | one green import **and** one red import | green passes, red blocks, same run | `VERIFY_CLAIMS_SCOPE modules=3`, `CLAIM_PASS mcl_green_library_claim`, `CLAIM_FAIL mcl_library_claim_that_is_false`, `fail=1`, no ELF — **PASS** |

D2 exists because D1 alone cannot distinguish "the closure is verified" from
"anything imported fails". D4 is the invariance arm: a file with no imports has
a one-node closure and must behave exactly as before, in verdicts and in
wall-clock.

**D5 and D6 exist because three independent reviewers refused D1–D4.** Put to
xAI, DeepSeek and Z.AI under the repository's M3 offload policy, all three
returned the same BLOCKER: a two-module probe cannot tell a *closure* walk from
a walk over *direct imports*, so "transitively closed" was asserted on one-hop
evidence. They were right, and the answer was to measure rather than to soften
the sentence. D5 puts the refuted claim two hops away, where a depth-1 walk
would have emitted the ELF; D6 checks that a module reachable by two paths is
verified once. The same review also read D4's original single cold runs (30 s
against 33 s, taken while a build held the machine) as a 10 % regression. Three
warm trials each side dissolve it: 10/11/10 against 11/11/10, indistinguishable
at this resolution, which is the strongest statement three runs support.

**D7 came from the second round, and one reviewer was wrong in a useful way.**
The objection was that D2 cannot rule out a compiler that fails everything it
imports, since such a compiler would still report `pass=2` on two green claims —
which is not so: it would report them as failures. But the suggested control is
strictly better than the argument against it, so it was measured instead of
defended. One compilation importing a green-claimed module and a red-claimed
module reports the green one as a pass and the red one as a failure **in the same
run**, and blocks. Discrimination is now observed rather than deduced from D1
and D2 jointly.

The reviewers also asked for evidence behind "the fraction of the corpus under
obligation is unchanged". It is a census: outside these probe fixtures, claims
exist in exactly three files of the repository — the manifest and two tests — and
no library, compiler module or stdlib module carries one. The walk therefore
changes the reached set by zero today; what it changes is where a claim may be
put.

**Incidental finding, not caused by this change.** The 16-claim manifest is
8 green / 8 red at this tip — `ade_wildgen_mckay`, `cd_tower_nullity_histogram`,
`chingon_zd`, `e_series_semantic_germ`, `functor_f_g2_covariance`, `g2_zd_fibers`,
`trigintaduonion_zd`, `zd_fiber_spectra_count_law` all fail. The pre-fix baseline
records the identical eight (`sfcl_d4_baseline_prefix.txt`), so this rung neither
caused nor repaired them. They belong to other lanes and are recorded here only
so the next reader does not attribute them to the closure walk.

## 4. What this does to R1

R1's gate failed on its own, before anyone went looking:

```
C3: MODULE_CLOSURE_PASSES — imported claims now DO execute. This is a
real change in the mechanism, not a gate bug: update the spec's verdict token
from MODULE_CLOSURE_BLOCKS and re-derive R1's conclusions.
```

That message was written on 2026-07-26 by the same hand that recorded the wall,
against the possibility that the wall would one day fall. It fell, and the
instruction is followed here rather than worked around:

* R1's document keeps its measurement and its reasoning intact and carries a
  supersession notice. Nothing in its body was rewritten to hide the flip.
* R1's token moves to `BOUND_16__MODULE_CLOSURE_PASSES` **because the
  measurement moved**, which is the only reason a token in this line ever moves.
* R1's C3 clause is inverted on purpose, and now fails on a return to `BLOCKS` —
  with an explicit refusal in the failure text to let the next reader edit the
  token into agreement.

The binding half of R1 — 16 claims bound to real gates — is untouched and stands.

## 5. Claims downstream that are now stale

Three documents assert the wall as a finding and are **not** corrected by this
rung, because they are prose whose framing belongs to their author:

* `docs/papers/witness_based_compilation_2026-07-28.md:616` — cites
  `MODULE_CLOSURE_BLOCKS` as a limitation of the mechanism.
* `docs/papers/oopsla2027/paper.md:91` and `outline.md:81` — contribution row C2
  is stated as "the wall it hits: claims in **imported modules never execute**".
* `scripts/research/witness_based_compilation_paper_contract.py:127` binds the
  string `MODULE_CLOSURE_BLOCKS` to "the module-closure wall", and stays green
  only for as long as the paper keeps saying it.

C2 was a contribution built on a limitation that no longer exists. Whether that
becomes a stronger contribution — the guard propagates through the import graph —
or is retired, is a decision about the paper, not about the compiler.

## 6. Verdict

```
SELF_FALSIFYING_R29_VERDICT CLOSURE_WALKED__MODULE_CLOSURE_PASSES
```

A refuted premise in a module you import now refuses to let you build. And a
rung of this line was retired by the line itself, on schedule, by a gate that
was written to catch its own author.
