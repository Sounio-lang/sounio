# Sci-notation modular-lexer gap — ported onto E008-cleared base; +26 PASS, 0 regr (2026-06-03)

Branch `parser/sci-notation-modular-e008` off `integration/e008-nested-store-complete` tip
`df8d1db36`. Commit `4d6913c53` (cherry-pick of `f77ae77b0`).

## Why now (the joint-block released)
The modular lexer's active number-emit path (`self-hosted/lexer/mod.sio`, feeds the parser via
`parser_set_token_flat`) consumed integer+fractional digits but DROPPED the exponent, so `1.0e5`
lexed as `FloatLit 1.0` + `Ident e5`. In let/value position the stray ident is absorbed
(false-pass); in `if`/`while` conditions it surfaces as `parse error: expected token`. This fix
(`f77ae77b0`) was validated +15 on its original base 2026-06-02 but NEVER pushed, so it was
absent from the integration line. The prior +0 parser-item lesson (const/kernel/extern) was an
E008 joint-block; that block is now cleared (E008 122→3 on integration/e008), so lexer/parser
fixes convert.

## Diagnosis (advisor-guarded, no 172-way workflow)
Re-derived the construct map off PRE-FAILURE lines (the reported "line N" is recovery-landing
noise — if/let/fn there are phantoms). Of 172 modular parse failures: **60/123 with-line fail on
a scientific-notation float line** (dominant), confirmed layout-stable (3 whitespace-perturbed
if-cond forms all parse-fail on mc, all pass on bin/souc; let-form false-passes — the documented
asymmetry, NOT miscompilation). Minimal repro: `fn main()->i32{ if 1.0e-15 > 0.5 {0} else {1} }`
→ bin/souc OK, mc parse error.

## Result (modular census, 504 run-pass, mc rebuilt via build lock)
| | PASS | CRASH | regressions |
|---|---:|---:|---:|
| integration/e008 (df8d1db36) | 209 | 0 | — |
| + sci-notation lexer fix | **235** | 0 | **0** |

**+26 FAIL→PASS, 0 PASS→FAIL.** All 26 wins are sci-notation-dense scientific tests: pbpk_*,
rapamycin_*, darwin_tmdd_*, dissertation_pbpk*_gum, ode_rk4_harmonic, special_functions,
sedenion_basics, clifford_spacetime, gum/pk uncertainty, prob_beta_basic, arima_levinson_ar2.

## Blast radius
Lexer-only change to the MODULAR path; `lean_single.sio` untouched ⇒ `bin/souc` unchanged
(`canonical_compiler_gate.sh` PASS, md5 `05348095`). x86; no a64 concern (lexer is arch-neutral).

## Remaining parser-class (post-fix)
Item-level: kernel (17, double-blocked by E072 "kernel must return unit"), ontology (9),
algebra (9), study (4) — novel top-level decl forms the modular item-parser lacks. Statement:
spawn{} (8, async — but `.await` parses then hits E012, likely joint-blocked), closure (5),
turbofish (3). Each is a separate self-hosted/parser production; verify flip-to-PASS per the
advisor before sizing (kernel is E072-double-blocked, so not a clean +17).
