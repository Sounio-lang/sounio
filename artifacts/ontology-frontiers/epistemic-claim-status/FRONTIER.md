# Fronteira: Status epistêmico verificável de claims em knowledge graphs

**Slug:** `epistemic-claim-status`
**Status:** em ataque
**Data:** 2026-08-02

## O problema aberto

Knowledge graphs biomédicos e organizacionais armazenam triples/claims sem
**status epistêmico machine-checkable**: não há contrato formal diga quanta
confiança um claim *derivado* merece a partir dos seus antecedentes, nem como
fusão de evidência independente deve se compor. A literatura recente (2025–
2026) trata o tema como infraestrutura em aberto: "epistemic infrastructure",
"belief graphs", resolução entrópica de claims — sem semântica formal
verificável da propagação.

## Evidência na literatura (via scite)

1. **"Belief Graphs with Reasoning Zones: Structure, Dynamics, and Epistemic
   Activation."** arXiv:2602.15353. DOI: `10.48550/arxiv.2602.15353`.
   - Propõe grafos de crenças com "zonas de raciocínio"; a ativação
     epistêmica é dinâmica, mas sem garantias formais de propagação de
     confiança.

2. **"Retrieval Is Not Enough: Why Organizational AI Needs Epistemic
   Infrastructure."** arXiv:2601.21116. DOI: `10.48550/arxiv.2601.21116`.
   - Argumenta que RAG sem rastreamento de status epistêmico é insuficiente;
   não oferece mecanismo verificável.

3. **"Entropic Claim Resolution: Uncertainty-Driven Evidence Selection for
   RAG."** arXiv:2604.11759. DOI: `10.48550/arxiv.2604.11759`.
   - Seleção de evidência guiada por incerteza para resolver claims;
   heurística, sem contrato formal.

4. **"AI-Assisted Engineering Should Track the Epistemic Status and Temporal
   Validity of Architectural Decisions."** arXiv:2603.28444.
   DOI: `10.48550/arxiv.2603.28444`.
   - Defende rastrear status epistêmico de decisões; reforça a demanda por
     propagação com garantias.

## A aposta Sounio

Modelar claims como valores com confiança e proveniência explícitas e provar
(formalmente) as duas regras de propagação usadas pelo protótipo:

- **Weakest link (derivação):** um claim derivado de premissas merece no
  máximo a confiança mínima da cadeia — e se toda premissa ≥ t, o derivado
  ≥ t (preservação de limiar em cadeias de comprimento arbitrário).
- **Fusão DS (fontes independentes):** Dempster-Shafer nunca fica abaixo da
  melhor fonte individual: `ds(a,b) ≥ max(a,b)` para confianças em [0,1].

O que é novo: as regras deixam de ser convenção de pipeline e viram teoremas
com prova mecanizada (Lean 4, sem Mathlib), mais um protótipo executável com
assertivas em runtime.

## Artefatos

- `claim_status.sio` — mini claim store: derivação weakest-link, fusão DS,
  classificação de status (alta/média/baixa confiança), invariantes
  verificados em runtime.
- `formal/OntologyClaimStatus.lean` — confianças em por-mil (Nat, 0–1000):
  teoremas de cadeia (limiar preservado, derivado ≤ cada premissa) e da
  fusão DS escalada (`dsNum(a,b) ≥ 1000·max(a,b)`).
- `interval_claims.sio` — extensão intervalar do claim store: claims com
  confiança `[lo, hi]` em [0,1]; derivação por mínimo pontual, fusão DS
  intervalar, invariantes verificados em runtime (intervalo válido, contenção
  do resultado pontual nos pontos médios, limiar preservado no lado `lo`,
  `lo` fundido ≥ melhor `lo` de origem).
- `formal/ClaimStatusInterval.lean` — intervalos por-mil `IConf = [lo, hi]`
  com validade `lo ≤ hi ≤ 1000`: preservação de validade para weakest-link e
  DS, contenção (soundness) dos resultados pontuais via monotonicidade de
  `dsNum` (provada a partir de `Nat.mul_le_mul`), preservação de limiar em
  cadeias (`chainIConf_lo_ge`) e `lo` fundido ≥ `max` dos `lo`s de origem.

## Lacunas e riscos

- O álgebra de propagação (min para derivação, DS para fusão) é uma escolha;
  alternativas (probabilística, Dempster-Shafer com conflito, fuzzy) ficam
  para trabalho futuro.
- ~~Confianças são escalares por-mil; incerteza de segundo grau não
  modelada.~~ **Fechado (2026-08-02):** incerteza de segundo grau agora é
  modelada por **intervalos de confiança** `[lo, hi]` por-mil — ver
  `interval_claims.sio` e `formal/ClaimStatusInterval.lean`. A derivação
  intervalar (mínimo pontual) e a fusão DS intervalar preservam validade,
  contêm o resultado pontual de quaisquer pontos interiores e preservam
  limiares no lado `lo`, tudo com prova mecanizada. Ficam como trabalho
  futuro: p-box completa e propagação GUM de segunda ordem — ver trilha GUM
  do repositório.
- A integração com um KG real (SPARQL/RDF) está fora de escopo.
