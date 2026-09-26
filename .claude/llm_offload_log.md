# LLM Offload Audit Log

Append-only. One entry per non-trivial offload attempt (caught a bug,
informed a design decision, blocked a commit) or per M1-mandatory review
that could not run, per `.claude/AGENT_OFFLOAD_POLICY.md`. Newest entries at
the bottom.

Format: `## <UTC timestamp> — <agent> — <what>` followed by outcome and
reasoning.

---

## 2026-09-26T20:20Z — Claude (session 3c9c1595) — M1 math-review, PBPK28 CN floor-clamp dispatch

| 2026-09-26 | xai (grok-4.6) ×2, qwen (OpenRouter Qwen 3 235B) | math-review | docs/audit/PBPK28_CN_FLOOR_CLAMP_MASS_INJECTION_DISPATCH_2026-09-26.md | PASS after fixes | Two independent providers; zai and local legs unavailable (see below), not counted as passes. |

**Trigger:** the dispatch makes math claims about CN stability, the exact discrete
mass identity, Bolley–Crouzeix positivity, TR-BDF2 quadrature weights and a
roundoff tolerance for a dissertation-path PBPK kernel (§M1).

**Legs run on the workspace (`/workspace/worktrees/claude-pbpk28-cn-mass`):**
- `bin/llm-offload -t math-review -p xai` on draft v1. Grok 4.6 flagged two items
  WRONG: the R(z) formula and the positivity factor. Both read h as dt; the kernel
  defines h = dt/2 (`tsit5_pbpk28.sio:77`) and every table value already used that
  convention. **Resolution:** the convention is now stated explicitly at both sites;
  the math is unchanged. Two TIGHTENABLE items were accepted: the AUC identity holds
  because CN's volume-weighted equations are the trapezoidal rule, not because the
  ringing "compensates"; and for γ = 2 − √2 both TR-BDF2 stages share one matrix.
- `bin/llm-offload -t math-review` (default fan-out) on draft v2. xai produced no
  WRONG items. Accepted: the Bolley–Crouzeix OVERREACH (split into two separate
  facts); "undamped" → "weakly damped, R ≈ −1"; the blood timescale is ≈ 24 s, not
  minutes; stiffness is set by Q, PS and V_v, while Kp and CL act through residues;
  "any trajectory" → "any CN trajectory"; "10 ulp" → "5 ulp".
  **zai: ERROR** (provider code 1313, account fair-usage rate limit). **local:
  ERROR** (litellm connection error, no `local-think` model group). Neither leg is
  counted.
- `-p deepseek`: **ERROR**, API key rejected as invalid. Not counted.
- `-p qwen` on draft v3: all claims OK. One OVERREACH and one note were **not
  accepted**, with reasons. (a) "without mentioning method linearity": the dispatch
  already says "any *linear* one-step method". (b) "BE positivity bound 2/λ appears
  incorrect": the dispatch states no BE bound. The 2/λ column is CN's bound, and
  Grok independently marked it OK.

**Flagged for re-review:** rerun the zai leg once the rate limit clears, and fix
the invalid DeepSeek key in `~/.sounio-keys.env`. Raw outputs from this session:
`/tmp/pbpk28cn/mathreview_{xai,fanout,deepseek,qwen}.txt` on the workspace
(ephemeral).

## 2026-09-26T21:05Z — Claude (session 3c9c1595) — M1 math-review, dispatch revision 2 (forced dosing, sensitivities, Hessian artefact)

| 2026-09-26 | xai (grok-4.6), qwen (OpenRouter Qwen 3 235B) | math-review | docs/audit/PBPK28_CN_FLOOR_CLAMP_MASS_INJECTION_DISPATCH_2026-09-26.md (v4) | PASS after fixes | Two independent providers. |

- **xai:** two items accepted. (1) "Kp/PS elasticity exactly 0" is exact for AUC(0–∞);
  at 168 h it holds up to M(T)/Dose ≈ 6e-18. The text now says so. (2) OVERREACH on
  "fu_plasma genuine by the same argument": resolved by citing the code. Both modules
  scale `cl_central` linearly by (fu+δ)/fu_ref (`epistemic_pbpk28_hessian.sio:117–119`,
  `epistemic_pbpk28.sio:203–207`), so fu's elasticities equal CL's.
- **qwen:** three items, **not accepted**, each a misreading. "Floors preserve mass
  conservation": the dispatch says the opposite. "TR-BDF2 guarantees positivity": the
  dispatch says it does not, and gives the −0.081 mg/L counter-example itself.
  "Specify two BE half-steps": already specified.
- **zai/local/deepseek:** not rerun. Status as in the previous entry (rate limit,
  endpoint down, invalid key).

## 2026-09-26T19:59Z — Claude (session 6620bb30, branch claude/pbpk28-cn-rannacher) — math-review, theta_pbpk28.sio

Re-recorded 2026-09-26: this log was gitignored when the entry was first written locally, and the
merge that made it tracked replaced the local copy.
- `-p xai` (grok-4.6): claims 1–5 and 7 OK (θ-step Schur coefficients, θ and TR-BDF2 mass
  identities, b-weights (w, w, d), Rannacher booking, positivity). Claim 6 OVERREACH: the
  `steps·1e-15` gate is a heuristic rounding budget, not a proven FP bound. The comments were
  reworded; the threshold is unchanged.
- Second opinions (policy): the default fan-out failed (zai 1313 rate limit, local unreachable),
  the deepseek key is invalid, and mistral errored. qwen3-235b: 7/7 OK.

## 2026-09-26 — Claude (session 6620bb30) — math-review, epistemic_pbpk28 TEST 5 analytic re-pin + Hessian values
- xai: 6/6 OK. qwen: 5/6 OK; one "WRONG" whose own correction restates the claim (a lost Kp
  column shows as an exact 0.0 share, and non-zero shares mean it is intact). Recorded as a misreading.

## 2026-09-26 — Claude (session 6620bb30) — math-review, Hessian ρ guard v1 + MC u_MC re-pin
- xai: 4/4 OK. qwen: 4/4 OK. mistral: 4/4 OK, plus a wording OVERREACH on "exact identity" (the
  identity is exact in ℝ; M(168 h)/Dose is an observation). No change.

## 2026-09-26 — Claude (session 6620bb30) — math-review, theta_pbpk28 input routing
- xai: 4/4 OK. qwen: 4/4 OK.

## 2026-09-26 — Claude (session 6620bb30) — external-facing review (`--raw xai gemini qwen`), 9 regenerated docs/dissertation/results pages
- gemini: every reply truncated (150–340 bytes), not counted. deepseek: invalid key.
- grok-4.6: substantive findings on all 9 pages (nominal-vs-sample bias labels; missing 12.4 vs
  12.7 L/h and MC-resolution qualifiers in "safe to cite"; causal overreach; Sobol S_i > S_Ti;
  budget-table closure; rel convention; legacy CV = 0.5807). All applied.
- qwen3-235b: minor clarifications applied. Its attribution comment (replace the "Claude Code"
  provenance) was rejected: that line is the AI disclosure.

## 2026-09-26 — Claude (session 6620bb30) — math-review, PR #2696 review round 3 (ρ undefined sentinel; TEST 9 order gate on C_brain(24 h); 24 h capture fix)
- xai (grok-4.6): 6/6 OK. qwen3-235b: 6/6 OK.
