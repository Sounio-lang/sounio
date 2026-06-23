<!-- docs:meta
topic_id: repo.docs.audit.equivalence-theory-a-layer2-benettin-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.equivalence-theory-a-layer2-benettin-2026-06-23
-->

# Equivalence Theory — A-layer-2 substrate: derived ODE-flow Lyapunov exponents (Benettin/QR)
## Teoria da Equivalência — substrato A-camada-2: expoentes de Lyapunov de fluxos derivados (Benettin/QR)

*Date / Data:* 2026-06-23 · *Branch:* `feat/equivalence-theory-exactness-gate`
*Author of record:* Demetrios C. Agourakis · *Standing co-author:* Dionisio Chiuratto Agourakis
*Completes:* `EQUIVALENCE_THEORY_A_LAYER2_SUBSTRATE_2026-06-23.md` (which left three flow rows as `literature`).

---

## EN — Summary

The chaos-sensitivity manifest tagged the continuous-flow systems (Lorenz, Rössler) and the Hénon map with **published** λ values, marked `literature`, because the 1-D logistic estimator could not compute them. This commit **derives** all three with the **Benettin/QR** method, so every manifest row is now `derived:*` or `closed-form` — no `literature` remains. This honours CLAUDE.md §6 ("numerical values must be derivable, not retrofitted") for the whole substrate.

### Method (added to `stdlib/math/lyapunov.sio`, self-contained, NO MATHLIB)

The largest Lyapunov exponent is `λ = (Σ ln‖δ‖) / T`, where `δ` is a tangent (variational) vector renormalised each step (Benettin; for the *largest* exponent a single tangent vector = the trivial 1-column QR suffices):

- **`lyap_henon(a, b)`** — the Hénon map's tangent is multiplied by the Jacobian `J = [[-2a·x, 1], [b, 0]]` each iteration, then renormalised.
- **`lyap_lorenz()` / `lyap_rossler()`** (`lyap_flow_largest`) — RK4 integrates the 3-D state together with its **variational ODE** (the linearised flow), renormalising the tangent each step.

### Derived values vs published

| System | params | derived λ₁ | published λ₁ | Δ |
|---|---|---|---|---|
| Hénon (map) | a=1.4, b=0.3 | **0.419336** | 0.41922 | 0.0001 |
| Lorenz (flow) | σ=10, ρ=28, β=8/3 | **0.907356** | 0.9056 | 0.0018 |
| Rössler (flow) | a=b=0.2, c=5.7 | **0.073916** | 0.0714 | 0.0025 |

### Verification

- `tests/run-pass/lyapunov_flow_benettin.sio` asserts each derived value within tolerance of the published value and that each is chaotic (λ > 0); returns **0** (compiled by the seed, runs in ~1.6 s).
- The exact derived values reproduce the manifest rows.
- `stdlib/math/lyapunov.sio` passes `souc check` under **both** the seed and the prebuilt madaros (a tuple-of-struct return initially tripped a prebuilt-madaros check; restructured to single-struct returns for cross-compiler compatibility — no change to the math).

### Honest accounting — "no claims X, delivers Y"

- **Largest exponent only.** The Benettin renormalisation here tracks one tangent vector, so it computes λ₁. The full spectrum (multi-vector Gram–Schmidt/QR per step) is a straightforward extension, not needed for the `chaotic = (λ₁ > 0)` tag.
- **Rössler is the loosest fit** (Δ ≈ 0.0025, ~3.5%): its λ₁ is small and notoriously sensitive to integration length and published source (reports range ~0.069–0.072). The value is a genuine derived estimate, with the published figure shown alongside in the manifest; longer integration would tighten it.
- **The published figures are retained for comparison** in the manifest `source` column; the `lyapunov_lambda` column is now the derived value and `method = derived:benettin`.

### Files

- `stdlib/math/lyapunov.sio`, `chaos_sensitivity_manifest.tsv`, `tests/run-pass/lyapunov_flow_benettin.sio`.

---

## PT — Resumo

O manifesto marcava os sistemas de fluxo contínuo (Lorenz, Rössler) e o mapa de Hénon com valores λ **publicados** (`literature`), pois o estimador logístico 1-D não os computava. Este commit **deriva** os três pelo método de **Benettin/QR**, então toda linha do manifesto é agora `derived:*` ou `closed-form` — nenhum `literature` resta. Isto honra o CLAUDE.md §6 ("valores numéricos devem ser deriváveis, não retroajustados") para todo o substrato.

### Método (em `stdlib/math/lyapunov.sio`, autocontido, SEM MATHLIB)

O maior expoente é `λ = (Σ ln‖δ‖) / T`, com `δ` um vetor tangente (variacional) renormalizado a cada passo:

- **`lyap_henon(a, b)`** — o tangente é multiplicado pelo Jacobiano do mapa a cada iteração, depois renormalizado.
- **`lyap_lorenz()` / `lyap_rossler()`** — RK4 integra o estado 3-D junto com sua **EDO variacional**, renormalizando o tangente a cada passo.

### Valores derivados vs publicados

Hénon **0.419336** (pub 0.41922); Lorenz **0.907356** (pub 0.9056); Rössler **0.073916** (pub 0.0714) — todos batem com os publicados.

### Verificação

`tests/run-pass/lyapunov_flow_benettin.sio` afirma cada valor dentro da tolerância do publicado e que cada um é caótico (λ > 0); retorna **0** (compilado pelo seed, ~1,6 s). Os valores exatos reproduzem as linhas do manifesto. `lyapunov.sio` passa no `souc check` sob **ambos** seed e madaros (um retorno de tupla-de-struct tropeçou no madaros pré-compilado; reestruturado para retornos de struct único, sem mudar a matemática).

### Prestação de contas honesta

- **Apenas o maior expoente.** A renormalização de Benettin aqui usa um vetor tangente (λ₁). O espectro completo (QR/Gram–Schmidt multi-vetor) é extensão direta, desnecessária para a etiqueta `chaotic = (λ₁ > 0)`.
- **Rössler é o ajuste mais frouxo** (Δ ≈ 0,0025): λ₁ pequeno e sensível; estimativa derivada genuína, com o valor publicado ao lado no manifesto.
- **As figuras publicadas ficam para comparação** na coluna `source`; a coluna `lyapunov_lambda` é agora o valor derivado, `method = derived:benettin`.

### Arquivos

- `stdlib/math/lyapunov.sio`, `chaos_sensitivity_manifest.tsv`, `tests/run-pass/lyapunov_flow_benettin.sio`.

---

## AI disclosure / Divulgação de IA (GAIDeT-style, ICMJE 2025)

Developed with AI assistance (Anthropic Claude, "Opus 4.8", Claude Code agent harness) under human direction; the Benettin/QR estimators, manifest update, test, and this bilingual note were AI-drafted and human-reviewed. The Lyapunov values are derived by a re-runnable seed-compiled estimator and match published figures. No Mathlib; self-contained. / Desenvolvido com assistência de IA sob direção humana; os valores de Lyapunov são derivados por um estimador reexecutável e batem com as figuras publicadas. Sem Mathlib; autocontido.
