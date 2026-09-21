<!-- docs:meta
topic_id: repo.docs.architecture.ws-c-d-preflight-review-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.ws-c-d-preflight-review-2026-08-16
-->

# WS-C / WS-D Preflight Review — adversarial findings memo

**Status:** findings memo, review only. No `self-hosted/` edits; neither source doc modified.
**Reviewer:** fable-1 (lane `preflight-review`), 2026-08-16.
**Scope:** `docs/architecture/MIR_PORT_PLAN.md` (Route B, approved) and
`docs/architecture/MLI_DESIGN.md` (Option C, approved), reviewed against
`origin/main` @ `03416657fa` and `origin/canon/madaros-v2-sota` @ `97b525949`.
Every finding below carries a receipt command; all were run this session on this pod.

**Headline:** Route B is the right route — its central isolation claim survives
adversarial re-measurement, and the frontier `enir/` driver builds and emits
correctly under the **current main** seed, a stronger receipt than the study's.
But the PR stack as scoped is missing ~23 payload files, every gate script
carries a frozen-surface contract that cannot run on main unmodified, one enir
file does not parse under Madaros v0.80.0, and MLI_DESIGN's input contract
describes a MIR that Route B does not deliver — the S2/S3 ladder has an
unscoped dependency that should be named before S1 is dispatched.

---

## 1. What was re-verified and held (do not re-litigate)

| Claim (MIR_PORT_PLAN) | Re-measurement | Verdict |
|---|---|---|
| `enir/` imports `enir::*` only | all 36 `use` lines across 14 files enumerated: every one is `enir::…` | **HOLDS** |
| Divergence 189 / 2086, both-sides 115 (41 `self-hosted/`) | 189 / **2088** (main moved 2 commits), 115 / 41 unchanged; merge-base `a930c8ac72` | **HOLDS** |
| Driver builds under seed, emit OK | rebuilt with **current main's** `bin/souc-lean-single-x86_64` (not the frontier's): ELF 1,474,279 B, `emit` rc=0, artifact 2053 B, header `enir\|1\|1\|eisa_metron_shadow\|2` | **HOLDS, strengthened** |
| E200 `loop_closed` at `source_lower.sio:529`, "non-fatal" | reproduced — but see finding C5: it is a real bug, not noise | **Partially holds** |

MLI_DESIGN factual grounding also checked: all 11 referenced repo files exist on
main; `IrHyperMulQ/O/S`, `IrAssociator` present in `self-hosted/ir/ir.sio:137-190`;
`Epistemic { val: f64, variance: f64, confidence: i64 }` at
`stdlib/epistemic/knowledge.sio:62`. No fabricated references found.

Receipts:

```bash
F=origin/canon/madaros-v2-sota
for f in $(git ls-tree -r --name-only $F self-hosted/enir/); do git show $F:$f | grep -E '^\s*use '; done | sort -u
git rev-list --count origin/main..$F; git rev-list --count $F..origin/main
cd /workspace/.wt/mir-study && scripts/dev/souc-build-lock.sh \
  /workspace/.wt/fable-1/bin/souc-lean-single-x86_64 self-hosted/enir/driver.sio /tmp/enir.elf
```

---

## 2. WS-C findings (MIR_PORT_PLAN.md, Route B)

### C1 · HIGH — PR1–PR5 payload list is missing ~23 files the gates require

The gate scripts hard-reference frontier-only oracle and fixture files under
`tools/eisa/` that are **absent from main** and absent from the plan's PR stack
(§6: "enir/** + bin/madaros-enir + docs" / gate scripts). Measured missing set:
`eisa_enir_v1_oracle.sio`, `eisa_enir_v1_loop_oracle.sio`, and ~21
`eisa_enir_v1_*/v2_*.eisa` fixtures (add, mul, div, sqrt, sub, mem, mem_phi_*,
mem_poison, join_then/else, equal_*, frail, fuel, loop, rump_dd, rump_qd,
c2_rump, const_gate, emov).

```bash
comm -23 <(git ls-tree -r --name-only origin/canon/madaros-v2-sota tools/eisa | sort) \
         <(git ls-tree -r --name-only origin/main tools/eisa | sort)
```

**Consequence:** PR2–PR4 fail at file-not-found unless PR1 (or each gate PR)
also carries its `tools/eisa/` payload. **Fix:** enumerate the full transitive
file set per gate script and add it to the PR stack definition before PR1 opens.

### C2 · HIGH — every gate embeds a frozen-surface contract that cannot run on main as-is

Each E-gate begins with
`git diff --quiet "$BASE_REF" -- self-hosted/compiler/main.sio self-hosted/ir
self-hosted/native self-hosted/wasm self-hosted/gpu stdlib/runtime
[stdlib/eisa stdlib/math/dd64.sio stdlib/math/qd128.sio tools/eisa/eisa_evm_run.sio …]`
with `BASE_REF` defaulting to `origin/canon/madaros-v2-sota`. On main that diff
is never empty (thousands of lines), so **every gate fails at its shadow-discipline
check before doing any work**. The env override (`E1_BASE_REF` etc.) exists, but
re-anchoring is a design decision, not a search-replace: pinning to
`origin/main` makes the gate fail spuriously whenever any collocated lane has
in-flight edits to `ir/lower.sio` (constant on this pod), and pinning to `HEAD`
weakens the discipline the gate encodes. E3D itself uses
`E3C_BASE_REF=HEAD E3C_ALLOW_DOWNSTREAM_ENIR_EXTENSION=1` internally — precedent
exists, but PR2 must choose and document the anchor. This is an unbudgeted PR2
cost item, and the plan's "fix inside `enir/` only" boundary (Route B step 3)
**cannot hold for the gate scripts**.

### C3 · HIGH — shared oracle files have drifted on main, and WS-F is editing the same surface

The gates don't just pin the shared files — E2-class gates **compile the METRON
oracle live** (`souc-build-lock.sh $SEED $CORPUS $ORACLE`) and compare
observations against it. Measured drift frontier→main:

| File | Drift | Role in gates |
|---|---|---|
| `tools/eisa/eisa_evm_run.sio` | **+188/−5** | E1/E2 corpus + live oracle |
| `stdlib/math/qd128.sio` | 18/12 | pinned qd semantics (E2E/E3D) |
| `stdlib/eisa/` | differs (main adds `hypercomplex_zd.sio`) | pinned surface |

Porting the gates therefore means either carrying frontier versions of shared
files (a direct conflict with main's EISA evolution) or re-validating enir's qd
semantics against **main's** oracle — a semantic re-validation, not a re-run.
Additionally, **WS-F (grok-cli4) is concurrently porting EISA gates over the
same `tools/eisa/` + `stdlib/eisa/` files**; neither doc flags this cross-lane
conflict. Recommend an explicit coord claim boundary between WS-C PR2+ and WS-F
before either lands.

### C4 · HIGH — `mir_join.sio` does not parse under Madaros v0.80.0 (seed-only syntax)

Receipt: `bin/souc check self-hosted/enir/mir_join.sio` (default Madaros engine,
current main binary) → 8 parse errors at lines 476–489; `driver.sio` and
`mod.sio` fail transitively ("module failed to parse: …/mir_join.sio"). The
other 13 enir files parse clean under Madaros. Root causes at the failing lines:

- line 476: a **parenthesised `if`-expression used as a comparison operand**
  inside the `||` chain — `b.trap_kind != (if opcode >= EMIR_OP_ADD && … { … } else { … })`;
- line 481: **semicolon-joined `let` statements** on one line
  (`let a = …; let b = …`).

The seed (lean_single) accepts both. PR1's acceptance criterion is
"driver check/compile **under seed**" — as written it would merge code the
default engine cannot parse, and the discrepancy would surface later as a
mystery. **Fix:** cheap — rewrite the offending constructs in `mir_join.sio`
during the PR1 repair pass, and add a Madaros `souc check` receipt (green, or an
enumerated FAIL_HONEST list) to PR1's acceptance alongside the seed receipt.

### C5 · MEDIUM — the E200 `loop_closed` is a real semantic bug, not tolerable noise

`loop_closed` is **used** at `source_lower.sio:529`
(`if loop_closed && !is_load { return enir_lower_fail(module, line, 15) }`) but
**declared** at line 645 (`var loop_closed = false`) in a different function.
The plan records this as "non-fatal under this seed path". Adversarially: the
seed compiles the site as a guarded gate (build log: `gates[direct=2159
guarded=737]`), meaning the lower-fail-15 rule — "no non-load `let` after loop
close" — is **silently unenforced or trap-prone** on the seed path today, and
gate results that passed with it in place prove less than they appear to.
Madaros will hard-error on it once C4 is fixed. Fixing it (threading the flag or
deleting the check) **changes lowering semantics** and may shift E2-gate
expectations; budget it as a semantic fix with a gate re-run, not a syntax
papercut.

### C6 · LOW — WS-B ordering rationale doesn't actually cover ENIR

The focus plan justifies "SOIR gate before the port lands" as catching
serialisation drift mechanically. The SOIR gate covers `self-hosted/ir/*`; ENIR
has its **own** text format with its own canonical/roundtrip/hash checks inside
the E1 gate. SOIR-first is still fine sequencing, but it is not protection for
ENIR serialisation — don't let a WS-B slip hold PR1 hostage on a false safety
premise.

### C7 · PROCESS, HIGH — both approved docs exist only as uncommitted files

`MIR_PORT_PLAN.md` and `MLI_DESIGN.md` are **untracked** (`??`) in the shared
checkout `/workspace/sounio`, whose worktree is parked on a research branch
(`research/zd-fiber-antisymmetry-lemma-20260731`); the focus plan itself is a
locally-modified tracked file there. Founder-approved architecture is currently
one `git clean` / concurrent-agent collision away from loss — this exact
failure mode has occurred on this pod before. **Commit all three to a docs lane
targeting main immediately**; both docs also carry stale `docs:meta`
(`last_validated: 2026-03-07`) and will need the governance registry sync
(generated — never hand-edit) after landing.

---

## 3. WS-D findings (MLI_DESIGN.md, Option C)

### D1 · HIGH — the input contract describes a MIR that Route B does not deliver

MLI_DESIGN §3.2 assumes MIR arrives with SSA-ish values, explicit control flow,
and "typed operands map injectively into MLI kinds"; ladder S2 is
"`mir_to_mli` for **scalar R0 only (integers + f64)**". Measured against the
frontier MIR that Route B actually lands (`enir/mir.sio`):

- **10 opcodes total**: `CONST ADD SUB MUL DIV SQRT OBSERVE LOAD STORE MOVE`
  (`mir.sio:17-26`). **No integer ops, no call, no ret, no compare/branch ops.**
- **One type per module, enforced**: module verify requires `type_count == 1`
  and the single type to be exactly `{value_kind: f64, error_kind: qd128,
  uncertainty_kind: gum1, status_tracked, provenance_tracked}` (`mir.sio:306`).
  Every MIR value is an epistemic bundle, not a scalar.
- The post-E3D non-claims (general N-way SSA, loops-in-schema, alias, **ABI,
  MachineIR**) are precisely the pieces `mir_to_mli` needs.

**Consequence:** S2 cannot be fed from Route-B MIR as it exists, and the
Phase-2 golden `add1(x: f64) -> f64` is not even expressible in EMIR (no
function ABI, no ret). The real dependency of S2/S3 is a **post-E3D MIR
generalisation tranche that neither approved doc scopes or costs**. §3.2's own
fallback ("IR→MLI side door") covers this — but the ladder names `mir_to_mli`
as the only choke point. **Fix before S1 dispatch:** either (a) add the
generalisation tranche to WS-C as an explicitly costed follow-on, or (b)
re-anchor S2/S3 on the IR→MLI side door and record that Route-B MIR feeds R1
epistemic kinds, not the R0 scalar path. Option (b) is likely correct — note
Route-B MIR's bundles map naturally onto MLI's `Knowledge` kind (Gpu4-like
shape), which is an argument the design can use, not just a gap.

### D2 · MEDIUM — kind-model gaps to close in S1, cheap now, breaking later

1. **qd128 is not IEEE f128.** Frontier MIR's error lanes are qd128
   (double-double family); MLI's `Float { f128 }` (IEEE binary128, WS-G) must
   never be conflated with it. Add an explicit exclusion note or a distinct
   kind; a silent `f128 := qd128` mapping in O1 would be a semantic miscompile
   by construction.
2. **No `Int { bits: 128 }`** despite language-level i128/u128
   (wide-int source→ELF, 2026-06-05). State the exclusion explicitly so the
   direct-path-replacement story doesn't silently shrink language coverage.
3. **Flags liveness is unmodelled.** `Flags` exists as a kind, but on x86
   nearly every ALU op clobbers eflags; V-struct as specced ("no raw cross-kind
   moves") will not catch a `cmp`/`br_cc` pair separated by a flag-clobbering
   instruction. Either add a verify rule (a flags def must be immediately
   consumed) or make `br_cc` consume a bool vreg in R0 and let legalize fuse.
4. **Block params vs φ**: §3.2 says "φ resolved at MIR or early MLI", §4.4 has
   optional `params?`. Frontier MIR has neither in general form (E3D joins
   only). Pick one representation in S1 or the builder API churns in S5.

### D3 · MEDIUM — Phase-2 bit-identity is underspecified in two load-bearing ways

1. **Which direct path is the golden?** lean_single and Madaros native-v2 emit
   different bytes. O5 gestures at this; it must be resolved (pin the engine
   and the exact binary provenance — built this session, per focus-plan risk
   item 6) **before** S3 starts, not during.
2. **`imm f64` has no x86 encoding.** The worked example (`fadd v0, imm f64
   1.0`) is fine as IR, but byte-identity forces legalize to reproduce the
   existing emitter's exact constant-materialisation strategy (constant
   pool / rip-relative / movabs+movq), register choices, and scheduling.
   S3 is therefore "mimic the existing emitter's choices for one function",
   not a generic legaliser — estimate it as such, and don't let its smallness
   inflate confidence about general legalisation cost.

### D4 · LOW — the `MIR_*` name collision is now imminent, not theoretical

`native/machine_ir.sio` already owns the `MIR_` constant prefix
(`MIR_OPERAND_GPR64`, `MIR_MAX_INSTRS`, … — verified on main) while Route B
lands a different "MIR" in `enir/`. §5.4 treats the rename as optional-later;
once both trees coexist on main, every grep, gate log, and dispatch doc mixing
them will mislead (this plan cycle already produced one swapped-wording
incident). Recommend: promote the `X86_*` rename (or at minimum a
naming-disambiguation note in `native/machine_ir.sio`'s header) to a
precondition of the WS-C PR1 tranche.

---

## 4. Recommended pre-implementation actions (ordered)

1. **Commit the three planning docs to main** (C7) — before anything else.
2. **Amend MIR_PORT_PLAN PR stack**: add the `tools/eisa/` payload enumeration
   (C1), the BASE_REF re-anchoring decision (C2), a Madaros-check receipt in
   PR1 acceptance (C4), and the `loop_closed` semantic fix as a named PR1 task
   (C5). All are small edits to the plan, not re-litigation of Route B.
3. **Coordinate WS-C ↔ WS-F** on `tools/eisa/` + `stdlib/eisa/` ownership (C3)
   via explicit coord claims before either lands.
4. **Amend MLI_DESIGN §3.2/§7**: name the Route-B-MIR gap and pick the S2
   feed (D1); fold D2's four kind-model notes into §4; resolve O5 + golden
   engine pin before S3 (D3).
5. None of the above changes the approved decisions (Route B, Option C). Both
   survive this review; the findings are about making S1/PR1 land at the
   estimated cost instead of discovering these in-flight.

---

## 5. Receipts index

All commands runnable from `/workspace/.wt/fable-1` (main-based) with the
frontier via `origin/canon/madaros-v2-sota`; frontier worktree at
`/workspace/.wt/mir-study` @ `97b525949`. Session artefacts:
`/tmp/fable1-preflight-enir-seed.elf`, `/tmp/fable1-enir-emit.out` (2053 B),
`/tmp/fable1-enir-madaros-check.log` (mir_join parse failure),
`/tmp/fable1-enir-seed-build.log` (E200 + gates counts).

*End of preflight review. Reviewer: fable-1, 2026-08-16.*
