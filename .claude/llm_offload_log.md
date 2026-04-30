# LLM Offload Audit Log

**Authority**: append-only audit trail of all non-trivial `bin/llm-offload` invocations, per `.claude/AGENT_OFFLOAD_POLICY.md`.

## Format

Each entry is a markdown table row. New entries go at the **top** (most-recent first):

| Date | Agent | Task | Provider | Target | Outcome | Commit |
|------|-------|------|----------|--------|---------|--------|

`Outcome` should be one of:

- **CAUGHT_BUG** — reviewer surfaced a real defect that was fixed
- **BLOCKED** — reviewer raised concern that delayed commit; explain in inline note
- **CONFIRMED** — reviewer endorsed correctness; useful for high-stakes claims
- **DISAGREEMENT** — reviewer flagged a non-issue; explain in inline note (do not silently dismiss)
- **NO_FINDINGS** — clean review, no issues
- **TIMEOUT / API_FAIL** — provider unavailable; substitution or deferral

## Entries

| Date | Agent | Task | Provider | Target | Outcome | Commit |
|------|-------|------|----------|--------|---------|--------|
| 2026-04-30 | Opus 4.7 | math-review | xai (Grok 4.1) | `formal/lean4/SounioVancomycinDosingSafety.lean` | NO_FINDINGS (3 TIGHTENABLE on `True` placeholders, all consciously deferred per file Status section) | (uncommitted) |
| 2026-04-30 | Opus 4.7 | math-review | xai (Grok 4.1) | `formal/lean4/SounioKnightian.lean` | CAUGHT_OVERREACH: `vacuous_widest` claim "widest possible gap" fails in `Float` edge case (`lo_mean < -1e18` representable). Renamed to `vacuous_widest_placeholder` with honest docstring deferring to semantic upgrade; 4 other TIGHTENABLE on `True` placeholders left as documented skeleton. Lean build green post-fix. | (uncommitted) |
| 2026-04-30 | Opus 4.7 | math-review | xai (Grok 4.1) | `formal/lean4/SounioApproxCausalKnowledge.lean` | CAUGHT_BUG ×2: (a) `mul` def's `approx_bound` missed `δx·δy` cross-term (mirror of `composed_effects.sio` bug); (b) docstring claimed "sketch with `sorry`" but proofs used `trivial` on `True` statements (dishonest per Sounio's no-`sorry` convention). Fixed: cross-term added in `mul` def + `variance` now full Gaussian-product (incl. `Var(X)·Var(Y)` term); 3 non-trivial theorems renamed to `*_placeholder` with honest docstrings; Lean build green post-fix. | (uncommitted) |
| 2026-04-30 | Opus 4.7 | math-review | xai (Grok 4.1) | `stdlib/epistemic/knightian.sio` | CAUGHT_OVERREACH: `pb_mul_variance` comment claimed "sound (an upper bound)"; delta-method actually omits `σ_x²σ_y²` term so it underestimates by ~1% when CV ≪ 1. Fixed: added the cross-term + rewrote comment honestly noting GUM convention vs exact moment; 9 OK findings on interval-extension arithmetic, vacuous-on-zero-straddle div, projection. 9/9 knightian tests pass post-fix. | (uncommitted) |
| 2026-04-30 | Opus 4.7 | math-review | xai (Grok 4.1) | `stdlib/epistemic/composed_effects.sio` | CAUGHT_BUG ×2: (a) `ck_mul`/`ck_div` `approx_bound` missed `δx·δy` cross-term (counter-example: a=b=0±1 → code says 0, true bound 1); (b) `ck_mul` comment said "alphas multiply" but code (correctly) sums — comment was wrong, code right. Fixed: cross-term added in mul AND div + variance now includes `Var(X)·Var(Y)` for honesty + comment rewritten "alphas SUM (independent-evidence pooling)". 11/11 composed_effect tests pass post-fix. | (uncommitted) |
| 2026-04-30 | Opus 4.7 (in-session) | math-review | xai (Grok 4.1) | `stdlib/clinical/vancomycin_pbpk.sio` `vp_cmin_point` monotonicity comment | CAUGHT_BUG: sign error on `dCmin/dVc`; corrected to `> 0` for all θ > 0; corner enumeration in `predict_cmin_knightian` flipped to `(Vc_lo, CL_hi)` for min and `(Vc_hi, CL_lo)` for max; pre-TDM band shifted `[11.30, 21.31]` → `[8.49, 24.29]` (correct, wider, REFUSE narrative strengthened); 4/4 vancomycin tests pass post-fix | (uncommitted at log creation) |
