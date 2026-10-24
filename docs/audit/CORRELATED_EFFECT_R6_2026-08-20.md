<!-- docs:meta
topic_id: repo.docs.audit.correlated-effect-r6-2026-08-20
authority: repo_only
audience: users
last_validated: 2026-09-01
validated_by: grok-cli4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.correlated-effect-r6-2026-08-20
-->

# R6 — `Correlated` effect (slot-identity detection)

**Rebased onto main 2026-09-01** after LOOM cutover cancel.

| Item | Value |
|---|---|
| Effect id | **29** `EffCorrelated` (`effect_named_id_max = 29`) |
| Diagnostic | **E236** (not E221/E230 — those are taken on main) |
| Detection | Knowledge binary arith + same `ExprIdent` (`m + m`) |
| Force control | `SOUNIO_FORCE_CORRELATED=1` |
| Not this PR | FO builtin `correlate(a,b,rho)` — already on main (#2137) |
| Not this PR | E230 anti-garbling noise-symbol independence — already on main |
| Mod | Still **HELD** on main (R5 #2059/#2067 closed unmerged) |

## Gates

`scripts/ci/correlated_effect_gate.sh` · ratchet frozen at 0 · workflow `.github/workflows/correlated-effect.yml`
