# Fronteira: Reparo epistêmico de alinhamentos de ontologias biomédicas

**Slug:** `epistemic-alignment-repair`
**Status:** em ataque
**Data:** 2026-08-02

## O problema aberto

Alinhar ontologias biomédicas (SNOMED CT, FMA, NCI, UMLS) produz *mappings*
com valores de confiança heurísticos (ex.: 0.06, 0.30). Quando o alinhamento
integrado gera consequências lógicas indesejadas — inconsistências ou violações
de conservatividade — o reparo clássico remove mappings de baixa confiança.
O problema: **a confiança dos mappings é um peso heurístico sem semântica
formal**. Não existe uma noção verificável de quanta confiança um entailment
retido *merece* depois do reparo, nem garantias de que o reparo preserva o
máximo de "massa epistêmica" sujeito à consistência.

## Evidência na literatura (via scite)

1. **Jiménez-Ruiz, Cuenca Grau, Horrocks (2011).** "Logic-based assessment of
   the compatibility of UMLS ontology sources." *Journal of Biomedical
   Semantics* 2(Suppl 1):S2. DOI: `10.1186/2041-1480-2-s1-s2`.
   - UMLS 2009AA contém erros detectáveis logicamente; o reparo proposto
     remove mappings com base em confiança (ex.: remove mappings μ₂ com
     confiança 0.06 e μ₃ com 0.30) — a confiança funciona como custo de
     remoção, sem semântica epistêmica formal.
   - SNOMED CT > 300.000 entidades; NCI ~79.000; FMA ~67.000 — reparo manual
     inviável; automação com garantias é necessária.

2. **Solimando, Jiménez-Ruiz, Guerrini (2016).** "Minimizing conservativity
   violations in ontology alignments: algorithms and evaluation." *Knowledge
   and Information Systems* 51(3):775–819. DOI: `10.1007/s10115-016-0983-3`.
   - Detecta e minimiza violações do princípio de conservatividade em
     alinhamentos do OAEI; trata mappings como pesos a minimizar, não como
     quantidades epistêmicas propagáveis.

3. **Rovai (2026).** "Open Ontologies: Tool-Augmented Ontology Engineering
   with Stable Matching Alignment." arXiv:2605.09184.
   DOI: `10.48550/arxiv.2605.09184`.
   - Estado da arte em alinhamento ainda atinge F1 = 0.832 na trilha Anatomy
     do OAEI — ou seja, ~17% dos mappings corretos escapam e mappings
     espúrios persistem; validação/reparo continua problema aberto.

## A aposta Sounio

Modelar cada mapping como `Knowledge<bool>` (valor + confiança + incerteza),
o grafo de conflitos como restrições lógicas, e o reparo como: **remover o
mapping de menor confiança de cada conflito até a consistência**, com
garantia em nível de tipo de que todo entailment retido tem confiança ≥ a
confiança mínima dos mappings retidos que o sustentam (weakest link).

O que é novo em relação à literatura: a confiança deixa de ser um peso de
custo e passa a ser uma quantidade com contrato formal — o reparo prova
(formal/Lean) que (i) o conjunto retido é livre de conflitos, (ii) todo
mapping removido conflita com um mapping retido de confiança ≥ a sua
(testemunha de maximalidade), e (iii) a propagação de confiança para
entailments é monotônica na confiança dos mappings.

## Artefatos

- `alignment_repair.sio` — protótipo: alinhamento UMLS-like em miniatura
  (mappings com confiança, conflitos explícitos), reparo epistêmico guloso
  com invariantes verificados em runtime (a sintaxe aspiracional de
  refinamentos `where` ainda não é aceita pelo parser Madaros atual).
- `formal/OntologyAlignmentRepair.lean` — formalização: grafo de conflito
  finito, reparo por dobra em ordem decrescente de confiança; teoremas de
  correção (livre de conflito, subconjunto, testemunha de remoção).
- `formal/OntologyRepairEquivalence.lean` — equivalência mecanizada entre o
  guloso pairwise drop-weaker do protótipo e a dobra prioritária `repair`
  (ver "Lacunas e riscos").
- `tie_repair_demo.sio` — protótipo do desempate determinístico: m0 (0.50) e
  m1 (0.50) empatam, ambos conflitam com m2 (0.30), e o próprio par empatado
  conflita; o empate é resolvido pelo id menor. Verifica determinismo (duas
  execuções idênticas), ausência de conflitos e testemunhas de maximalidade.
- `formal/OntologyRepairTies.lean` — extensão da equivalência para confianças
  ARBITRÁRIAS com desempate lexicográfico (confiança, id) (ver "Lacunas e
  riscos").

## Lacunas e riscos

- ~~A equivalência entre o guloso pairwise drop-weaker do protótipo e a dobra
  prioritária formalizada não estava mecanizada.~~ **Fechada (2026-08-02)**
  em `formal/OntologyRepairEquivalence.lean`: para confianças distintas,
  candidatos ordenados por confiança decrescente sem repetição, relação de
  conflito simétrica **e em grafo de clusters** (união disjunta de cliques;
  cobre a instância do protótipo, que é um conjunto de arestas disjuntas),
  `repair_iff_greedy` prova que o conjunto retido pelo guloso é exatamente o
  da dobra prioritária. A mecanização revelou que a hipótese de cluster é
  **necessária**: `cx_equivalence_fails` certifica por `native_decide` um
  contraexemplo num caminho de conflito de 3 vértices, onde o guloso remove
  um mapping cuja testemunha mais forte é depois removida por um terceiro
  mapping que não conflita com ele. Escopo residual: conflitos dados por
  oráculo (não derivados de OWL/EL++) e a equivalência cobre uma única
  passada sobre a lista de pares (como no protótipo).
- ~~Empates de confiança excluídos da equivalência guloso≡dobra.~~
  **Fechada (2026-08-02)** em `formal/OntologyRepairTies.lean`: a prioridade
  passa a ser a ordem lexicográfica em (confiança, id) — em empate de
  confiança vence o mapping de id **menor**, exatamente a regra
  `conf[i] >= conf[j]` do protótipo sobre pares ordenados por id
  (`greedyStep_prio_eq_sio`). A hipótese "confianças distintas" de
  `repair_iff_greedy` é substituída por "prioridades distintas", que vale
  sempre para ids distintos (`prio_injective_on`): `repair_iff_greedy_ties`
  prova a mesma equivalência para confianças ARBITRÁRIAS sob a hipótese de
  cluster, instanciando o teorema da rodada 2 com o encoding injetivo
  `prio m = conf m * (I + 1) + (I - id m)` (nenhum lema da rodada 2 precisou
  de cópia generalizada — as hipóteses de distinção são derivadas). Também
  provados: totalidade/antisimetria da prioridade lexicográfica
  (`outranks_total`, `outranks_antisymm`), determinismo do guloso
  (`greedyDrop_deterministic`) e uma instância `Fin 6` COM EMPATE (m0 e m1 a
  0.50 em clique com m2 a 0.30) computada por `native_decide` pelos dois
  algoritmos com o mesmo resultado {m0, m3, m5}. Protótipo executável:
  `tie_repair_demo.sio` (ALL PASS).
- A escala real (SNOMED CT com 300k+ entidades) está fora do escopo do
  protótipo; a formalização cobre o núcleo combinatório do reparo.
- A noção de "conflito" aqui é dada por um oráculo (relação simétrica);
  derivá-la de semântica OWL/EL++ completa é trabalho futuro.
  **Atualização (rodada 10, 2026-08-04):** o oráculo hardcoded do
  protótipo foi substituído pelo fecho EL+ role-aware verificado
  (`stdlib/ontology/elplus.sio`, variante densa; `elplus_fixpoint` +
  `elplus_derive_conflicts`), com uma camada de papel
  (`heart ⊑ ∃part_of.Organ`, `∃part_of.Organ ⊥ DrugClass`) que torna o
  conflito conceitual `conflict(heart, drugclass)` genuinamente
  role-derivado; os conflitos de mapping na instância compartilhada
  permanecem `{m0-m1, m2-m3}`. A derivação a partir de OWL completo (fora
  do fragmento EL+) continua fora de escopo.
