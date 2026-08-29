<!-- docs:meta
topic_id: repo.docs.audit.equivalence-theory-a-layer2-substrate-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.equivalence-theory-a-layer2-substrate-2026-06-23
-->

# Equivalence Theory — Feature A-layer-2: provisioning the chaos/sensitivity substrate
## Teoria da Equivalência — Recurso A-camada-2: provisão do substrato de caos/sensibilidade

*Date / Data:* 2026-06-23 · *Branch:* `feat/equivalence-theory-exactness-gate`
*Author of record:* Demetrios C. Agourakis · *Standing co-author:* Dionisio Chiuratto Agourakis

---

## EN — Summary

A-layer-2 (Lyapunov-gated reassociation, §3.3) was **blocked**: it needs to refuse f64 reassociation on a path that "integrates a vector field tagged `chaotic` in the manifest", but the repo's manifests are EEG/seizure spectral tables with no chaotic/Lyapunov column. This commit **provisions that substrate** — the data and the means to derive it — without yet implementing the gate that consumes it.

### What was provisioned

1. **`stdlib/math/lyapunov.sio` — a derivable Lyapunov estimator.** The largest Lyapunov exponent of a 1-D map, `λ = lim (1/N) Σ ln|f'(x_n)|`. `lyap_logistic(r)` (logistic map), `lyap_tent(μ)` (closed form `λ = ln μ`), `lyap_is_chaotic(λ) = λ > 0`. λ > 0 is exactly the `e^{λt}` amplification A-layer-2 must budget for. Crucially this makes the manifest values **derived, not retrofitted** (CLAUDE.md §6).

2. **`chaos_sensitivity_manifest.tsv` — the chaotic-tagged manifest** (schema `chaos_sensitivity.v1`). Each row tags a system with its λ and a `chaotic` flag (`= 1` iff `λ > 0`):
   - **Derived** (computed by `lyap_logistic`, reproducible): logistic at r = 3.2 (λ = −0.916290, periodic), 3.5 (−0.872507, periodic), 3.7 (+0.354391, chaotic), 3.9 (+0.496754, chaotic), 4.0 (+0.693149, chaotic). The r = 4.0 value matches the closed form `λ = ln 2 = 0.6931472` to 6 digits — the derivability proof.
   - **Closed-form:** tent μ = 2 (`λ = ln 2` exact).
   - **Literature** (ODE flows the 1-D estimator can't compute, cited to be replaced by a derived flow estimate): Lorenz (λ₁ ≈ 0.9056), Hénon (≈ 0.41922), Rössler (≈ 0.0714).

3. **`tests/run-pass/lyapunov_closed_forms.sio`** — verifies the estimator against the closed forms and the periodic/chaotic sign discriminator.

### Verification

`lyapunov_closed_forms.sio` returns **0** (logistic r=4 = ln2 within 1e-2; tent = ln2 within 1e-4; r=3.2 and r=3.5 classified non-chaotic; r=3.7, r=3.9 chaotic; derived r=4.0 value 0.693149 within tolerance), compiled by the seed. `lyapunov.sio` passes standalone `souc check`. The manifest is internally consistent: `chaotic = 1` iff `λ > 0` on every row.

### Honest accounting — "no claims X, delivers Y"

- **This provisions the substrate; it does NOT implement A-layer-2.** The gate itself — extending the carrier with a sensitivity/Lyapunov tag alongside GUM uncertainty, and making the e-graph refuse reassociation on a path whose system is `chaotic` (within an `e^{λt}` error budget) — is the next increment, now unblocked. No e-graph or checker code changed here.
- **Derived where computable, cited where not.** The 1-D logistic/tent λ are computed and validated; the ODE-flow λ (Lorenz/Hénon/Rössler) are published values, explicitly marked `literature`, to be replaced by a derived continuous-flow Lyapunov estimate (Benettin/QR method) in a follow-up.
- **Epistemic flag carried (§8):** using a Lyapunov exponent as a compiler optimisation parameter is a *claim to verify*, not established practice. The module header and the manifest both state this.

### Files

- `stdlib/math/lyapunov.sio`, `chaos_sensitivity_manifest.tsv`, `tests/run-pass/lyapunov_closed_forms.sio`.

---

## PT — Resumo

A-camada-2 (reassociação condicionada por Lyapunov, §3.3) estava **bloqueada**: precisa recusar reassociação f64 em um caminho que "integra um campo vetorial marcado como `chaotic` no manifesto", mas os manifestos do repositório são tabelas espectrais de EEG/convulsão sem coluna de caos/Lyapunov. Este commit **provisiona esse substrato** — os dados e o meio de derivá-los — sem ainda implementar a porta que os consome.

### O que foi provisionado

1. **`stdlib/math/lyapunov.sio` — estimador de Lyapunov derivável.** Maior expoente de Lyapunov de um mapa 1-D, `λ = lim (1/N) Σ ln|f'(x_n)|`. `lyap_logistic(r)`, `lyap_tent(μ)` (forma fechada `λ = ln μ`), `lyap_is_chaotic(λ) = λ > 0`. λ > 0 é exatamente a amplificação `e^{λt}` que A-camada-2 deve orçar. Torna os valores do manifesto **derivados, não retroajustados** (CLAUDE.md §6).

2. **`chaos_sensitivity_manifest.tsv` — o manifesto marcado por caos** (esquema `chaos_sensitivity.v1`). Cada linha marca um sistema com seu λ e um flag `chaotic` (`= 1` sse `λ > 0`):
   - **Derivado** (computado por `lyap_logistic`): logístico em r = 3.2 (λ = −0.916290), 3.5 (−0.872507), 3.7 (+0.354391), 3.9 (+0.496754), 4.0 (+0.693149 — bate com `ln 2 = 0.6931472` em 6 dígitos).
   - **Forma fechada:** tenda μ = 2 (`λ = ln 2`).
   - **Literatura** (fluxos de EDO que o estimador 1-D não computa, citados): Lorenz (≈ 0.9056), Hénon (≈ 0.41922), Rössler (≈ 0.0714).

3. **`tests/run-pass/lyapunov_closed_forms.sio`** — valida o estimador contra as formas fechadas e o discriminador periódico/caótico.

### Verificação

`lyapunov_closed_forms.sio` retorna **0** compilado pelo seed; `lyapunov.sio` passa no `souc check` isolado; o manifesto é consistente (`chaotic = 1` sse `λ > 0`).

### Prestação de contas honesta — "não prometer X e entregar Y"

- **Isto provisiona o substrato; NÃO implementa A-camada-2.** A porta em si — estender o portador com etiqueta de sensibilidade/Lyapunov junto à incerteza GUM e fazer o e-graph recusar reassociação em caminho `chaotic` (dentro de um orçamento de erro `e^{λt}`) — é o próximo incremento, agora desbloqueado. Nenhum código de e-graph/verificador mudou aqui.
- **Derivado onde computável, citado onde não.** Os λ de logístico/tenda são computados e validados; os λ de fluxos de EDO são valores publicados, marcados `literature`, a serem substituídos por uma estimativa derivada de fluxo contínuo (método de Benettin/QR) em sequência.
- **Flag epistêmico (§8):** usar um expoente de Lyapunov como parâmetro de otimização de compilador é uma *afirmação a verificar*, não prática estabelecida.

### Arquivos

- `stdlib/math/lyapunov.sio`, `chaos_sensitivity_manifest.tsv`, `tests/run-pass/lyapunov_closed_forms.sio`.

---

## AI disclosure / Divulgação de IA (GAIDeT-style, ICMJE 2025)

Developed with AI assistance (Anthropic Claude, "Opus 4.8", Claude Code agent harness) under human direction; the estimator, manifest, test, and this bilingual note were AI-drafted and human-reviewed. The Lyapunov values are derived by a re-runnable seed-compiled estimator validated against closed forms (λ = ln 2). / Desenvolvido com assistência de IA sob direção humana; os valores de Lyapunov são derivados por um estimador reexecutável compilado pelo seed, validado contra formas fechadas (λ = ln 2).
