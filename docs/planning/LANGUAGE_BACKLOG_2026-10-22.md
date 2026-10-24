<!-- docs:meta
topic_id: repo.docs.planning.language-backlog-2026-10-22
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.planning.language-backlog-2026-10-22
-->

# Language backlog — 2026-10-22

**Status: index.** A decision that is not a tracked item gets lost on the way
from intent to code, the way semantic distance is lost on the way down the
compiler. This file is the scan surface: one line per item, each pointing at the
document that owns it. It owns nothing itself; when an item moves, the owning
document changes and this line follows.

Ordering rule: **silent wrong values first**, then refusals without a location,
then decisions owed, then capacity, then research. An item leaves when its
owning document records it closed with a receipt.

## 1. Silent wrong values (the program runs and the number is wrong)

| item | engine | owner |
|---|---|---|
| `.5` evaluates to 0 | lean_single | [receipt](../spec/2026-10/18-lexical-execution-receipt.md) §6, refused by PR #3019 (unmerged at the time of writing); ruling [D-33](../decisions/PENDING.md#d-33).2 decides whether it later means 0.5 |
| `affine`, `linear`, `study` accepted as names, read as 0 | lean_single | receipt §6; refused by [PR #3019](https://github.com/Sounio-lang/sounio/pull/3019), unmerged at the time of writing; the seed's misreading of a local `study` had reached Madaros (`check.sio`, `active_study_has_hypothesis_outcome`) |
| any token that starts no expression loads 0: `let var = 5; println(var)` prints 0, `println()` prints 0, `+a` is 0; reached by 166 files that compile today, including `stdlib/epistemic/knightian.sio:309` (p-box variance 0 under the seed) | lean_single | receipt §6, [`seed_fallback_sweep.tsv`](../audit/S01_LEXICAL_PROBES_2026-10-22/seed_fallback_sweep.tsv) |
| integer literals that do not fit wrap (`99999999999999999999` → `7766279631452241919`) | both | receipt; D-33.1 |
| unknown ASCII byte skipped, changing the expression (`1 $ 2` prints `1`) | lean_single | receipt |
| no f64 literal is correctly rounded at the midpoint probe | both | receipt; [D-27](../decisions/PENDING.md#d-27) |
| cross-call first-order variance not carried for general user calls (KL-11) | Madaros | [`KNOWN_LIMITATIONS.md`](../compiler/KNOWN_LIMITATIONS.md) |
| `Knowledge` propagation carries no rounding term at all | both | [interval proposal](../proposals/CERTIFIED_INTERVAL_PRIMITIVE_2026-10-22.md) §1 |

Fixed on the Madaros side by [PR #3016](https://github.com/Sounio-lang/sounio/pull/3016), **unmerged at the time of writing** (receipt §4): `\0` read as the digit
`0`, `\x41` read as `x41`, `\q` read as `q`, `'\q'` read as 113, an unterminated
`/*` swallowing the rest of the file.

## 2. Refusals without a location

Fixed on the Madaros side by PR #3016, unmerged at the time of writing (receipt §4): unknown character,
unterminated comment and string, invalid escape, line number after a multi-line
string, parse diagnostics printing two token numberings, `let x = @y` failing
with no message. Open: lean_single's misleading messages for an unterminated
string (`unbalanced braces`).

## 3. Decisions owed

[`PENDING.md`](../decisions/PENDING.md) D-01 – D-34. Added today: **D-33**, the
thirteen lexical rulings (one blocked by a measured collision: refusing an
unclosed character literal breaks `&'static` lifetimes in the standard library),
and **D-34**, the five decisions of the interval proposal.

## 4. Self-hosting and memory

| item | state | owner |
|---|---|---|
| Madaros is not fixed-point verified; `make build` verifies lean_single | stated in `CLAUDE.md` | [ADR-006](../decisions/adr-006-fixed-point-trust-anchor.md) |
| gen7 bit-identical self-compilation | target; gen2 = gen1 by founder report; gen3 unproven (CI: rc 181, `arena full`; founder: OOM) | [memory audit](../audit/MADAROS_MEMORY_FOOTPRINT_2026-10-22/README.md) §3 (triage) |
| gen3 self-build | **unproven**. CI's Gen2 Recovery Proof (run 39834416192) reaches gen2 `run`; gen2 compiling `main.sio` exits rc 181 with `arena full` in its never-reclaimed ~2 GiB runtime arena, on a 1.5 GiB stack. The ~15 GB OOM is a founder report, not reproduced. PR #3015's 8 MB vs 16 MB stack result is a capacity test, not a gen3 build | memory audit §3 |
| `Lowerer` (~4.5 MB) passed by value in 618 methods of `ir/lower.sio` | PR #3015; refactor to `&!Lowerer` sized at weeks | memory audit §2.2 (likely carrier of the unexplained BSS) |
| 3.8 GB static reservation; 1.23 GB explained by module arrays; seed gives every ≥1 MiB by-value aggregate a fresh block per site (reproduced) | measured | memory audit §2 |
| committed Madaros touches 1.25 GB to compile one line; source-built 230 MB | measured | memory audit §1.2 |
| phase arenas that never point upward; `munmap` per phase; `u32` indices | design | memory audit §4; complements [ADR-010](../decisions/adr-010-madaros-region-reclamation.md) |

## 5. Specification

The edition is [`docs/spec/2026-10/`](../spec/2026-10/) (source-research
standing). Executed receipts: chapter 15 (16 bounded cases, committed prebuilt)
and chapter 18 (lexical, 125 probes on both engines plus a source build). Every
other chapter's rules remain source-observed until a receipt executes them; the
next receipt should be chapter 01's `SYN` rules, using the same probe method.

## 6. Research lines (each with its kill test)

| line | first kill test | owner |
|---|---|---|
| certified interval primitive, Marco 0 | tightness criterion (an implementation returning `[−∞, +∞]` must fail) | [proposal](../proposals/CERTIFIED_INTERVAL_PRIMITIVE_2026-10-22.md) |
| epistemic non-interference (Q does not rise without an oracle) | Lean: monotonicity of Q under `reflect` | [research record](../research/EPISTEMIC_REASONING_AND_METACOGNITION_2026-10-22.md) §1 |
| rounding-induced σ in LLM decisions | partial R² < 0.05; Spearman < 0.5 across reduction orders | research record §2.4 |
| interoceptive organ (κ via duals) and SAN exit | VoI(m) ≈ 0 for every margin band (K1) | research record §3.4 |
| role-graded ontology algebra, typed `0_⊥`, semiring reasoner | provenance-semiring fixpoint tractable at `elplus.sio` scale | [algebra record](../research/ONTOLOGY_ALGEBRA_AND_COMPILER_TOWER_2026-10-22.md) §1–3 |
| compiler tower as information bottleneck | counted S_n, Φ_n on a log-log line | algebra record §4.2 |
| curvature-guided allocation | semantic proximity predicts co-access beyond liveness | algebra record §4.3 |
| sovereign targets; large models phases A–C | per-token KL and greedy-decode identity against an oracle | [roadmap](../roadmap/SOVEREIGN_TARGETS_AND_LARGE_MODELS_2026-10-22.md) |
