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

## Lacunas e riscos

- O oráculo de consistência é abstrato (relação simétrica de conflito);
  ligá-lo a um reasoner OWL/EL++ real é trabalho futuro.
- ~~Apenas adição de axiomas é modelada~~ **FECHADA (2026-08-02)**:
  remoção/cirurgia de axiomas modelada em
  `formal/OntologyEvolutionRepair.lean` + `version_chain_removal.sio`,
  com teorema repair-then-retry que conecta formalmente com a fronteira
  `epistemic-alignment-repair`. Resta como trabalho futuro: repair
  *minimal* (escolher qual axioma remover quando há vários parceiros
  conflitantes — hoje o teorema exige unicidade do parceiro).
- Compactação de armazenamento multi-versão (o problema de espaço de
  Bayoudhi et al.) não é tratada.
