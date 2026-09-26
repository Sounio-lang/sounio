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

## 2026-09-26 — Claude (session 9e416846) — math-review, matrix_er K-P t^n helpers (docs/audit/MATRIX_ER_TRANSCENDENTAL_ACCURACY_DISPATCH_2026-09-26.md)
- Input: claims C1–C9. C1–C5 cover the old helpers (ln floor −4.959346, F floor 7.8825e−3, exp bias −x²/2048, the
  ln truncation for t > 4, and cap crossing 11.98637 → 12.00987 h). C6 is the q-tau non-telescoping leak (0.591 mg/dose →
  ≤1.3e−8 mg). C7–C8 are the pure.sio ln/exp/sqrt error bounds. C9 is the post-fix SS ratio residual.
  Route: chore/llm-offload-llmgateway-grok47 tooling (754c303cb) staged on the workspace.
- xai, Grok 4.7: the LLM Gateway leg errored, so it ran via the automatic xAI-direct `grok-4.7` fallback (OFFLOAD_TIMEOUT=1200).
  Verdicts: 8 OK and 1 TIGHTENABLE.
  - C5: drop "exactly" for t*. The dispatch never states it as exact.
  - C6: "about 7 orders" is 7.7 orders (4.5e7). Applied.
  - C9: sound under a narrow reading. 3.1e−12 is accumulated residual (~1e4 ulp), not one rounding, and the claim does
    not extend to the periodicity residual (still up to 1.7e−6 mg). Both points applied.
- qwen, qwen3-235b (OpenRouter): 9/9 OK. It is shallow, and it accepts "exactly" for C5, which Grok tightened.
- kimi, Kimi K3 (gateway), was NOT counted:
  - max=8192: finish_reason=length, with empty content (all reasoning).
  - max=32000: upstream 502 from scx-ai-gp.
  - max=24000: finish_reason=length again, with empty content (24000 reasoning tokens).
- zai and local: errored (rate limit and endpoint down), not counted.
- Two independent vendors: xAI + Alibaba/Qwen.
