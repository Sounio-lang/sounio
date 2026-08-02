# Fronteira: Evolução de ontologias com consistência a priori verificada

**Slug:** `consistent-ontology-evolution`
**Status:** em ataque
**Data:** 2026-08-02

## O problema aberto

Ontologias biomédicas evoluem (SNOMED CT tem releases semestrais; UMLS
integra > 100 fontes). A verificação de consistência é tipicamente *a
posteriori*: a inconsistência é detectada depois que a nova versão já foi
publicada. Abordagens *a priori* existem, mas sem contrato formal
mecanizado de que toda versão alcançável da cadeia é consistente — a
garantia fica por conta de convenção de ferramenta.

## Evidência na literatura (via scite)

1. **Bayoudhi, Sassi, Jaziri (2018).** "Efficient management and storage of
   a multiversion OWL 2 DL domain ontology." *Expert Systems* 36(2):e12355.
   DOI: `10.1111/exsy.12355`.
   - Propõe geração de versões OWL 2 DL consistentes de forma a priori;
   reconhece que o estado da arte majoritário verifica inconsistência *após*
     a ocorrência ("a posteriori approach that checks inconsistency after
     its occurrence"), e que estratégias de armazenamento sacrificam
     semântica ou espaço.

2. **Jiménez-Ruiz, Cuenca Grau, Horrocks (2011).** "Logic-based assessment
   of the compatibility of UMLS ontology sources." *J. Biomedical
   Semantics* 2(S1):S2. DOI: `10.1186/2041-1480-2-s1-s2`.
   - O *consistency principle*: integração de ontologias bem estabelecidas
     não deveria introduzir inconsistências lógicas — violações são
     manifestação de erro de design e devem ser reparadas.

## A aposta Sounio

Modelar a cadeia de versões como uma transição de estado com guarda: um
edit só é aplicado se a versão resultante for consistente (oráculo de
conflito). A contribuição é tornar o invariante um **teorema mecanizado**:
por indução na construção da cadeia, *toda* versão alcançável é
consistente — não apenas a última, e não apenas quando verificado a
posteriori. O protótipo `.sio` executa o mesmo guarda com assertivas em
runtime sobre uma cadeia em miniatura (adição de axiomas, um edit
incoerente rejeitado).

### Extensão: remoção e repair-then-retry (2026-08-02)

A cadeia foi estendida com **remoção de axiomas** (cirurgia durante a
evolução), fechando a lacuna "apenas adição é modelada" e conectando esta
fronteira com `epistemic-alignment-repair`. Edits agora são
`add a | remove a`; a remoção sempre é aplicada e elimina *todas* as
ocorrências do axioma. Em `formal/OntologyEvolutionRepair.lean`:

1. **Lema geral de sublista** (`consistent_sublist`): qualquer sublista de
   uma versão consistente é consistente (indução na derivação de sublista).
2. **Remoção preserva consistência** (`consistent_removeAxiom`): corolário
   direto via `List.filter_sublist`.
3. **Invariante a priori generalizado** (`mem_versions2_consistent` /
   `consistent_evolve2`): toda versão da cadeia é consistente para scripts
   mistos de `add`/`remove`.
4. **Repair-then-retry** (`repair_retry`): se `add a` é rejeitado contra
   `v` com testemunha de conflito `k`, e `k` é o *único* parceiro
   conflitante de `a` em `v`, então após `remove k` o edit `add a` é
   aceito e a versão resultante é consistente.
5. **Instância concreta `Fin 6`**: `add 4` rejeitado contra {3,2,1},
   `remove 2`, re-adição de 4 aceita, versão final {4,3,1} consistente —
   verificado por `native_decide` e pelos teoremas gerais.

O protótipo `version_chain_removal.sio` executa o mesmo cenário com
assertivas em runtime (consistência checada após cada passo).

### Extensão: repair minimal contra MÚLTIPLOS parceiros (2026-08-02)

Fechada a lacuna restante: o teorema `repair_retry` exigia unicidade do
parceiro conflitante; agora o candidato `a` conflita com **vários**
parceiros da versão corrente. Como cada parceiro bloqueia `a`
independentemente, admitir `a` força a remoção de TODOS os parceiros — o
conjunto de remoção é unicamente determinado. A única escolha real é
binária, feita por **massa epistêmica** (confianças; convenção de massa:
SOMA das confianças dos parceiros, não o máximo):

- **ADMIT(a)** — remove todos os parceiros e adiciona `a`; massa
  contestada retida = `conf a`.
- **REJECT(a)** — mantém a versão; massa contestada retida = soma das
  confianças dos parceiros.
- **decide** — admite sse `conf a > soma`; empate rejeita (o candidato
  precisa superar estritamente a massa estabelecida que deslocaria).

Em `formal/OntologyMinimalRepair.lean`:

1. **ADMIT funciona** (`admit_succeeds`): após remover todos os
   parceiros, `a` não conflita com nada e a versão admitida é consistente
   (via o lema geral de sublista) — sem hipótese de unicidade.
2. **REJECT é consistente** (`reject_consistent`): rejeitar mantém `v`.
3. **Otimalidade** (`decide_optimal`): `decide` retém massa ≥ a da outra
   opção (análise de casos na comparação estrita).
4. **Necessidade / minimalidade única** (`partner_not_mem_of_admissible` /
   `admissible_sublist_partnerfree`): nenhum parceiro pode sobreviver em
   qualquer subconjunto mantido que admita `a`; o restante livre de
   parceiros é o ÚNICO conjunto mantido admissível maximal — dualmente,
   o conjunto de parceiros é o ÚNICO conjunto de remoção minimal.
5. **Instância concreta `Fin 7`**: candidato 4 conflita com DOIS
   parceiros (1 e 3) de {3,2,1}; perfil ADMIT (`conf 4 = 900 > 400+400`)
   e perfil REJECT (`conf 4 = 300 < 400+400`), ambos computados por
   `native_decide`, mais a direção de necessidade (remover apenas UM
   parceiro ainda bloqueia o candidato).

O protótipo `minimal_repair_demo.sio` executa os dois ramos (admit com
0.90 > 0.80; reject com 0.30 < 0.80) e a assertiva de necessidade, com
checagem de consistência por pares (ALL PASS).

## Artefatos

- `version_chain.sio` — cadeia de versões com guarda de consistência;
  invariantes verificados em runtime (todas as versões consistentes, edit
  incoerente rejeitado, histórico preservado).
- `formal/OntologyEvolution.lean` — modelo da transição com guarda;
  teorema de invariante: toda versão da cadeia é consistente; teorema de
  preservação: rejeitar um edit mantém a versão anterior.
- `version_chain_removal.sio` — extensão com remoção: add 1,2,3; add 4
  rejeitado; remove 2; re-adição de 4 aceita; versão final {1,3,4};
  consistência checada após cada passo (ALL PASS).
- `formal/OntologyEvolutionRepair.lean` — edits `add | remove`; lema
  geral de sublista consistente; invariante a priori para scripts mistos;
  teorema repair-then-retry (remover o único parceiro conflitante
  desbloqueia o edit rejeitado); instância `Fin 6` por `native_decide`.
- `minimal_repair_demo.sio` — repair minimal contra múltiplos parceiros:
  candidato 4 conflita com 1 e 3; ramo ADMIT (0.90 > 0.80) remove ambos e
  adiciona 4; ramo REJECT (0.30 < 0.80) mantém a versão; necessidade
  (remover um só parceiro ainda bloqueia) verificada (ALL PASS).
- `formal/OntologyMinimalRepair.lean` — decisão binária admit/reject por
  massa (soma das confianças dos parceiros); teoremas admit_succeeds,
  reject_consistent, decide_optimal e minimalidade única do conjunto de
  remoção; instância `Fin 7` por `native_decide`.

## Lacunas e riscos

- O oráculo de consistência é abstrato (relação simétrica de conflito);
  ligá-lo a um reasoner OWL/EL++ real é trabalho futuro.
- ~~Apenas adição de axiomas é modelada~~ **FECHADA (2026-08-02)**:
  remoção/cirurgia de axiomas modelada em
  `formal/OntologyEvolutionRepair.lean` + `version_chain_removal.sio`,
  com teorema repair-then-retry que conecta formalmente com a fronteira
  `epistemic-alignment-repair`.
- ~~Repair minimal com múltiplos parceiros conflitantes~~ **FECHADA
  (2026-08-02)**: `formal/OntologyMinimalRepair.lean` +
  `minimal_repair_demo.sio` — decisão admit/reject por massa epistêmica
  com otimalidade provada e minimalidade única do conjunto de remoção
  (todos os parceiros devem sair). Resta como trabalho futuro: empates de
  massa com mais de duas opções e oráculos de conflito com custo de
  remoção não uniforme.
- Compactação de armazenamento multi-versão (o problema de espaço de
  Bayoudhi et al.) não é tratada.
