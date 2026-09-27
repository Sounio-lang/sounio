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

## 2026-09-27T00:48Z — Claude (session 2d8d50e4) — math-review, darwin_pbpk local exp/ln helper accuracy dispatch

| 2026-09-26 | xai (Grok 4.7, `grok-4.7` via xAI direct after the gateway leg failed), qwen (OpenRouter `qwen/qwen3-235b-a22b`) | math-review | docs/audit/DARWIN_PBPK_LOCAL_EXP_LN_ACCURACY_DISPATCH_2026-09-26.md | PASS after fixes | Two independent vendors. Kimi K3 failed ×4, not counted. |

Route: `bin/llm-offload -t math-review` from `main` @ `ce93ea9534` (gateway tooling of #2701), keys from the workspace keys file.

- **Default fan-out (xai kimi zai local), 8192 tokens:** every leg failed. Grok 4.7 via the gateway and via xAI direct returned EMPTY at 180 s. Kimi K3 hit `finish=length` with empty content. zai returned 1313 (rate limit). local: connection error.
- **xai, rerun with OFFLOAD_MAX_TOKENS=32000 / OFFLOAD_TIMEOUT=1500:** the gateway returned `fetch_failed` (upstream xai), and the driver fell back to xAI direct `grok-4.7`, which completed. All load-bearing figures were marked OK: the expansion, the ppm table, the GMFE bias direction and the false-pass edges 2.0004694 / 3.0017691, and the printed deltas. **Four WRONG, all accepted and fixed:**
  - Q = 0 is reached at x = −2048, not −1024. The text now gives the exact-zero underflow band (x ≈ −529 to −1519, measured), Q → 0 at 914 286 h, and Q < 0 beyond.
  - The "< 0.01%" radius is ≈ 0.4525, not 0.45.
  - The −7.3e-13 ln error at 1e-6 is series tail at sx ≈ 1.2026, not e-rounding.
  - Newton sqrt from y = x converges; the problem is that 15 steps are too few for x ≳ 2³⁰.
  **Three TIGHTENABLE, applied:** 7-digit band edges, truncation bound 1.1547e-6, and ln_avg ≈ 0.242142 noted.
- **kimi (Kimi K3 via gateway):** four attempts, none produced content.
  - 8192 tokens, full doc: `length`.
  - 20000 tokens, full doc: `length`, 47k chars of reasoning.
  - 32000 tokens, math excerpt: `fetch_failed` (upstream novita).
  - 24000 tokens, math excerpt: `length` (together-ai).
  Not counted.
- **qwen, on the corrected text:** every math claim OK. One OVERREACH, accepted: "no verdict flips today" now says the false-pass bands are latent. Its gloss "even powers do not ensure positivity in floating point" is its own slip; the dispatch does not claim that.

**Flagged:** Kimi K3 needs a lower-reasoning mode or a longer gateway upstream timeout before it can review documents of this size. Raw outputs: `/workspace/worktrees/claude-transc-helpers/.transc/offload{1,2_xai,3_kimi,4_kimi,5_kimi,5_qwen}.log` (scratch worktree).
