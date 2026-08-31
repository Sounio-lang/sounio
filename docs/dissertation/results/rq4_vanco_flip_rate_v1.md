<!-- docs:meta
topic_id: repo.docs.dissertation.results.rq4-vanco-flip-rate-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.rq4-vanco-flip-rate-v1
-->

# Resultado — taxa de silenciamento do WARN na vancomicina em dois compartimentos (RQ4)

**Para o capítulo clínico** (*Sounio for Verified Clinical Decision Support: Vancomycin under
Knightian Uncertainty*), seção 6 — sugerido como **6.8 "O sinal da covariância decide o dano"**,
logo após 6.7 (compile-time confidence gate). Liga a Contribuição 2 (gates em tempo de
compilação: E230) à Contribuição 3 (orçamentos GUM/ISO) com um número clínico.

**Figura:** `docs/dissertation/figures/fig_rq4_flip_rate_2026-08-31.svg`
**Artefato executável:** `docs/research/sounio/rq4_vanco_two_compartment_flip.sio` (Sounio; Madaros
`bf1fe608`; determinístico; identidade `A/α + B/β = 2D/CL` verificada a erro 0 em 1ª ordem).
**Registro completo:** `docs/research/paper_A_rq4_two_compartment_flip_2026-08-31.md`.
**Twin executável do receipt em ambos os engines:** `examples/vancomycin_auc_affine.sio`.

---

## Parágrafo (português — versão para a dissertação)

Toda biblioteca de propagação de incerteza que a farmacometria usa — e o `ep_add`/`ep_mul` do
próprio `stdlib/epistemic/knowledge` — assume que os operandos de cada operação são independentes.
Numa cadeia farmacocinética essa hipótese é falsa por construção: peso, creatinina e depuração
entram mais de uma vez, e cada reencontro carrega uma covariância que a biblioteca omite. Pelo
Lema 1 da nossa formalização, a variância reportada erra exatamente em 2·Cov; o que ainda não se
sabia era quanto disso chega a uma decisão clínica. Medimos isso numa coorte determinística de
5 000 pacientes (peso 45–120 kg, creatinina 0,6–2,6 mg/dL, Q e Vp ±30 % da população; 500 mg
q12h), com a regra de decisão do exemplo canônico — WARN quando a estimativa pontual de AUC₀₋₂₄ é
terapêutica (≥ 400 mg·h/L) mas o limite inferior do IC95 cruza os 400 — propagada de três formas:
formas afins de primeira ordem que rastreiam as fontes de medição (a verdade, a mesma construção
`Aff` do cálculo mecanizado em Lean), a cadeia escalar `ep_*` como é (independência em toda
operação) e operandos exatos com uma única adição final independente, que isola o termo 2·Cov.
Em duas somas de fontes compartilhadas que um software de TDM realmente executa, o resultado tem
os dois sinais (Figura 6.x). Na soma de intervalos AUC(0–12) + AUC(12–24), ambos derivados da mesma
depuração (ρ = 1), a adição ingênua reduz a variância exatamente à metade — a contração de √2 no
desvio-padrão — e **silencia 311 dos 909 WARNs verdadeiros (34,2 %)**: um alerta em três desaparece,
e o paciente lê "terapêutico". Na decomposição da AUC em fases, A/α + B/β, a covariância entre as
fases é **negativa em 5 000 de 5 000 pacientes** — porque a AUC é invariante a Q e Vp e a
decomposição é uma partição desse invariante: o que Q e Vp movem para uma fase retiram da outra —
e a mesma hipótese de independência erra na direção oposta: sobrestima a variância em 20 % na soma
final e em **300 vezes** ao longo da cadeia inteira, produzindo **1 894 WARNs espúrios (38 % da
coorte)** e silenciando nenhum. É o outro dano clínico, a fadiga de alarme, produzido pelo mesmo
defeito. O sinal da covariância decide qual dano se obtém; o compilador não precisa conhecê-lo. A
disciplina de conjuntos de fontes (E230) recusa a adição com fontes compartilhadas nos dois casos, e
a propagação afim (`stdlib/epistemic/affine`) computa a soma correta nos dois — o que o teorema
`naive_add_understates_iff` (anti-garbling ⟺ Cov > 0) e seus corolários enunciam, e o que a coorte
confirma: sob a propagação exata, os 909 alertas são exatamente os 909 que deveriam existir.

## Legenda da figura

**Figura 6.x — Quantos alertas a hipótese de independência silencia — ou inventa.** Coorte
determinística de 5 000 pacientes; AUC₀₋₂₄ de vancomicina em dois compartimentos; WARN = estimativa
pontual ≥ 400 mg·h/L e limite inferior do IC95 < 400. A propagação afim (fontes de medição
rastreadas; verdade de 1ª ordem) produz 909 WARNs, a linha tracejada. **(A)** Soma de intervalos
AUC(0–12) + AUC(12–24) a partir da mesma depuração (ρ = 1): a adição que assume independência reporta
metade da variância (razão 0,500; Lema 1 com o termo 2·Cov > 0 omitido) e silencia 311 WARNs
(34,2 %). **(B)** Decomposição em fases A/α + B/β = 2D/CL: a covariância entre as fases é negativa em
todos os pacientes, a adição independente sobrestima (razão 1,204; +62 WARNs espúrios) e a cadeia
escalar `ep_*` completa sobrestima 300 vezes (2 803 WARNs; 1 894 espúrios, 38 % da coorte; nenhum
silenciado). Programa: `rq4_vanco_two_compartment_flip.sio`, Madaros `bf1fe608`, 2026-08-31;
identidade A/α + B/β = 2D/CL verificada a erro zero em primeira ordem.

## Paragraph (English — for the chapter outline / papers)

Every uncertainty-propagation library pharmacometrics uses — including this repository's own
`ep_add`/`ep_mul` — assumes the operands of each operation are independent. In a PK chain that
assumption is false by construction: weight, creatinine and clearance enter more than once, and each
re-entry carries a covariance the library omits. By Lemma 1 the reported variance is off by exactly
2·Cov; what was not known is how much of that reaches a clinical decision. We measured it on a
deterministic cohort of 5,000 patients under the canonical example's decision rule (WARN when the
point AUC₀₋₂₄ reads therapeutic but the lower 95 % bound crosses 400), propagated three ways:
first-order affine forms tracking the measurement sources (the truth; the `Aff` object of the
mechanized calculus), the shipped scalar `ep_*` chain, and exact operands with one
independence-assuming final add that isolates the 2·Cov term. In the interval sum
AUC(0–12) + AUC(12–24) from the same clearance (ρ = 1) the naive add halves the variance exactly and
**silences 311 of 909 true WARNs (34.2 %)**. In the phase decomposition A/α + B/β the phase covariance
is **negative in 5,000/5,000 patients** — AUC is invariant to Q and Vp, and the decomposition is a
partition of that invariant — so the same assumption errs the other way: 1.2× on the final add, 300×
across the whole chain, **1,894 spurious WARNs (38 % of the cohort)** and none silenced. The sign of
the covariance decides which harm you get; the compiler need not know it: E230 rejects the
shared-source add in both cases and affine propagation computes the sum correctly in both
(`naive_add_understates_iff`: anti-garbling ⟺ Cov > 0).

## Reproduzir

```bash
bin/souc run docs/research/sounio/rq4_vanco_two_compartment_flip.sio
# RQ4_FLIP n=5000 true_warn=909 silenced_sum=0 silenced_naive=0 spurious_naive=1894
#          var_ratio_sum_permille=1204 var_ratio_naive_permille=300662
#          B_true_warn=909 B_silenced=311 B_var_ratio_permille=500
bin/souc run examples/vancomycin_auc_affine.sio     # o receipt AUC 450 ± 44 → WARN, em ambos os engines
```
