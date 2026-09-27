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

## 2026-09-26 (recorded 2026-09-27T01:20Z) — Claude — M1 math-review, PR #2699 (venlafaxine implicit CYP2D6 sink + steady-state readout)

| 2026-09-26 | xai (grok-4.6), gemini-2.5-pro via OpenRouter | math-review | stdlib/darwin_pbpk/tsit5_pbpk28.sio (organ sink in the CN Schur solve), stdlib/darwin_pbpk/scenarios/venlafaxine_xr.sio (implicit formation, steady-state C_avg readout and its certified bound) | PASS after fixes; one disagreement recorded | Two independent providers. |

Transcribed from the `LLM-offload-review:` trailers on this PR's commits, which were the only record until now (Copilot review on #2699).

- **xai, organ sink (`1b61789b0`, `3e6b7e38e`):** claims 1–4 OK, with no coefficient error in pp/sb/rhs_v/rhs_t. Claim 5 was tightened: CN undershoot begins at h·k_s > 1.
- **xai, steady-state readout (`7399fd421`):** claims 1–3, 5 and 7 OK. Claim 4 was tightened: the bound uses the volume-weighted mass functional VᵀA ≤ 0, not raw column sums, and the ODV residual is analogous. The overreach in claim 6 was removed: no λ_fast/λ_slow factor is claimed.
- **gemini-2.5-pro (independent second leg, `d94b9ab40` review round):** claims 1–4, 6 and 7 OK. It **rejected claim 5** as "assumes F = 1".
  - **Disagreement, not accepted, with reasoning:** F ≤ 1, and all converted mass leaves via CL_odv, so m_p/CL_odv is an upper bound whether or not F = 1. xai rated claim 5 OK. The reasoning is also documented in the source.
- **Unavailable:** zai was rate-limited (provider code 1313), and the deepseek key was rejected as invalid. Neither leg counts.
