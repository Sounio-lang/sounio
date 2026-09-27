# Rodada 13 — fecho EL+ role-aware em MÚLTIPLAS raízes GO e novas ontologias OBO

> **Round 15 (2026-08-06):** ChEBI (H=218 253) + PATO (H=1 887) closed under
> the same self-validating sparse multi-target engine — see
> [`CHEBI_PATO_RESULTS.md`](CHEBI_PATO_RESULTS.md). Gate driver:
> `chebi_pato_elplus_driver.sio`.

**Data:** 2026-08-05 · **Lane:** `kimi-cli2/elplus-scale-multi-20260805`
· **Compilador:** `bin/souc` (Madaros), branch
`research/zd-fiber-antisymmetry-lemma-20260731`.

A rodada 12 rodou o fecho EL+ role-aware sobre o GO go-plus completo
(38.245 classes, 92 papéis). Esta rodada escala em DUAS direções:

1. **Múltiplas raízes GO** — os três cones de topo do GO
   (`GO:0008150` biological_process, `GO:0005575` cellular_component,
   `GO:0003674` molecular_function), rodados separadamente e comparados
   com a rodada 12 por **identidades de decomposição exatas**;
2. **Novas ontologias OBO** — **CL** (cell ontology, 66 MB) e **UBERON**
   (anatomia, 96 MB), baixadas de `purl.obolibrary.org`, extraídas sob a
   política namespace-only da rodada 12 e fechadas com o mesmo motor.

Tudo nesta rodada vive em `artifacts/ontology-frontiers/multi-ontology/`
(os diretórios `real-data/` e o README/gate das rodadas 1-12 estavam sob
claims ativas de outras lanes — ver §7).

## 1. Método (reuso validado + uma generalização)

- Espelho python: `bitmask_reduce` da rodada 12
  (`real-data/scale/gen_go_full_data.py`), validado contra o fixpoint
  geral de conjuntos no slice da rodada 11 — reusado SEM alteração.
- Cones GO: cortados da extração round-12
  (`../real-data/go_full_elplus_tbox.txt`), sem re-parse do go-plus.
  Axiomas sub/exsub/disj restritos ao cone; conjunto de papéis RO-fechado
  (política das rodadas 11-12).
- CL/UBERON: `parse_go`/`parse_ro` de `extract_tbox.py` (genéricos),
  política namespace-only (`/CL_`, `/UBERON_`), `owl:deprecated`
  excluídas, papéis RO-fechados a partir de `ro.owl`.
- **Cross-check novo (contador agrupado de conflitos):** como `pm[c]` é
  FUNÇÃO de `epm[c]` (união dos bits parceiros sobre os bits de `epm`),
  classes com a mesma máscara de endpoints contribuem uniformemente:
  `conf = Σ_v1 Σ_{v2: pm(v1)&v2} n(v1)·n(v2) − Σ_{v: pm(v)&v} n(v)` sobre
  as máscaras DISTINTAS. Assertado igual ao contador O(nact²) do espelho
  em todos os 5 alvos.
- **Generalização do driver (necessária para UBERON):** as máscaras de
  endpoints do conflito passam de 2 palavras fixas (128 endpoints, round
  12) para `NEPW` palavras (`epm[c*NEPW+w]`) — UBERON tem 589 pares disj
  (822 endpoints distintos). A tabela de fecho de papéis `rclos` passa de
  96×96 para `NRC×NRC` (UBERON usa NR=128 papéis após o RO-fecho).
- **Dados auto-validantes:** o arquivo packed de cada alvo carrega no
  header de 13 inteiros as 7 contagens de axiomas E os 6 valores do
  espelho (arestas atômicas, arestas de papel, conflitos, rodadas, 2
  ablações) — substitui os módulos `expected_*.sio` da rodada 12.

## 2. Resultados (espelho python == driver Sounio, todos os números)

| alvo | H | NR | sub | exsub | disj | roleSub | roleComp | arestas atômicas | arestas de papel (= alvos existenciais) | conflitos | rodadas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GO:0008150 BP | 24.129 | 32 | 40.863 | 12.597 | 28 | 36 | 31 | 298.203 | 1.480.543 | 21.144.668 | 4 |
| GO:0005575 CC | 4.075 | 7 | 4.693 | 2.158 | 21 | 6 | 6 | 23.943 | 105.887 | 8.621.578 | 4 |
| GO:0003674 MF | 10.041 | 28 | 12.268 | 433 | 3 | 37 | 26 | 73.793 | 45.685 | 4.522 | 3 |
| **CL** | 3.335 | 29 | 4.664 | 477 | 35 | 29 | 14 | 37.926 | 146.188 | 1.071.098 | 5 |
| **UBERON** | 14.975 | 128 | 19.607 | 17.080 | 589 | 87 | 36 | 150.515 | 2.343.535 | 25.001.610 | 7 |

Ablações (arestas de papel sem a família; contribuição entre parênteses):

| alvo | sem roleComp | contrib. roleComp | sem roleSub | contrib. roleSub | dominante |
|---|---|---|---|---|---|
| GO BP | 1.316.520 | 164.023 (11%) | 370.247 | 1.110.296 (75%) | roleSub |
| GO CC | 79.082 | 26.805 (25%) | 57.459 | 48.428 (46%) | roleSub |
| GO MF | 45.098 | 587 (1%) | 10.365 | 35.320 (77%) | roleSub |
| CL | 110.394 | 35.794 (24%) | 36.587 | 109.601 (75%) | roleSub |
| UBERON | 1.507.927 | 835.608 (36%) | 1.036.281 | 1.307.254 (56%) | roleSub |
| (ref. GO completo, r12) | 1.883.813 | 251.394 (12%) | 597.305 | 1.537.902 (72%) | roleSub |

**roleSub domina em todos os alvos** — no GO completo a hierarquia RO
profunda já dominava (72%); nos cones e em CL/UBERON o padrão se repete.
A amplificação papel/estatuído varia 17× entre alvos: CL 146.188/477 =
**306×** (hierarquia de papéis rasa sobre poucas restrições estatuídas,
mas cones ancestrais densos), UBERON 137×, GO BP 118×, GO CC 49×,
GO MF 105×.

## 3. Decomposição exata dos cones GO vs a rodada 12

Os três cones **particionam** o GO completo (24.129 + 4.075 + 10.041 =
38.245 = H; medido: sem sobreposição, sem classes órfãs de pai, resto 0).

- **Arestas atômicas:** 298.203 + 23.943 + 73.793 = **395.939 =
  exatamente o total da rodada 12** (assertado no gerador).
- **Conflitos:** o contador agrupado, rodando SÓ sobre as máscaras dos
  cones (levantadas de volta aos ids globais), reproduz **792.814.846 =
  exatamente o total da rodada 12** — validação independente do número da
  rodada 12 por um segundo algoritmo. Destes, **763.044.078 (96,24%) são
  pares ENTRE cones** (as 3 grandes disjunções MF×BP×CC, únicas pares
  disj cross-cone — medido: cross-disj = 2 por cone = as 3 arestas entre
  as raízes); só 29.770.768 são intra-cone.
- **Arestas de papel:** a soma dos cones (1.632.115) é menor que o total
  (2.135.207) porque 3.603 restrições estatuídas **cruzam cones**
  (1.860 BP + 813 CC + 930 MF — ex. fillers UBERON-internalizados no GO
  caem fora pela política namespace-only da rodada 12, mas restrições
  cross-namespace GO→GO existem de verdade: medido, não assumido). A
  identidade de decomposição de arestas de papel é portanto
  intencionalmente NÃO assertada; o déficit (503.092) é o fecho das
  restrições cross-cone omitidas.

## 4. Verificação

1. Driver == espelho em TODOS os números (7 contagens de axiomas,
   arestas atômicas, arestas de papel, conflitos, rodadas, 2 ablações)
   para os 5 alvos: `go_roots_elplus_driver.sio` e
   `obo_elplus_driver.sio`, ambos `ALL PASS` via `./bin/souc run`.
2. Contador agrupado de conflitos == contador de varredura do espelho
   (assertado nos 5 alvos).
3. Identidades de decomposição da §3 (assertadas no gerador).
4. Sonda de forma (CL/UBERON): restrições no lado superclasse
   (`∃r.F ⊑ C`): 1 em cada arquivo — a MESMA axioma,
   `∃RO:0000053.PATO:0010006 ⊑ CL:0000000`; restrições em
   equivalentClass: 2 (CL) / 15 (UBERON), todas anônimas/aninhadas
   (sem onProperty+someValuesFrom nomeado). **Correção da math-review
   (xai, 2026-08-05):** a minha formulação original ("restrições no lado
   superclasse não podem mudar estatísticas atom-level") é FALSA no
   sistema de 8 regras em geral — transitividade sobre S através de um
   nó existencial (`A ⊑ ∃r.F`, `∃r.F ⊑ B` ⇒ `A ⊑ B`) produz subsunção
   átomo-átomo; é por isso que o teorema de perfil das rodadas 11-12
   EXIGE 0 tais axiomas. Para ESTAS duas extrações o axioma omitida é
   comprovadamente inerte sob a política namespace-only: o filler
   (PATO) nunca é internado, logo o existencial não existe no universo
   e nenhum átomo o alcança; no caso UBERON até o alvo (CL:0000000) está
   fora do namespace. Os números reportados são exatos para a TBox
   extraída; a completude atom-level contra a semântica OWL completa
   NÃO é garantida pelo teorema de perfil (ver §8).

## 5. Reprodução

```bash
cd artifacts/ontology-frontiers/multi-ontology
# downloads: cl.owl, uberon.owl de purl.obolibrary.org; ro.owl copiado de
# ../real-data/downloads/
python3 gen_multi_data.py                # espelho + packed + drivers
cd /workspace/sounio
./bin/souc check artifacts/ontology-frontiers/multi-ontology/go_roots_elplus_driver.sio
./bin/souc run   artifacts/ontology-frontiers/multi-ontology/go_roots_elplus_driver.sio   # ALL PASS
./bin/souc run   artifacts/ontology-frontiers/multi-ontology/obo_elplus_driver.sio        # ALL PASS
bash scripts/ci/ontology_multi_ontology_gate.sh   # gate standalone
```

## 6. Arquivos (todos sob artifacts/ontology-frontiers/multi-ontology/,
salvo o gate)

| arquivo | papel |
|---|---|
| `gen_multi_data.py` | gerador: cones GO, extração OBO, espelho bitmask, contador agrupado, template do driver |
| `downloads/{cl,uberon,ro}.owl` | dados (66 MB + 96 MB + 1,2 MB) |
| `{go_bp,go_cc,go_mf,cl,uberon}_packed.txt` | dados runtime auto-validantes |
| `{cl,uberon}_{classes,roles}.tsv`, `{cl,uberon}_elplus_tbox.txt` | registro da extração |
| `go_roots_elplus_driver.sio` | 3 cones GO num run (gerado) |
| `obo_elplus_driver.sio` | CL + UBERON num run (gerado) |
| `scripts/ci/ontology_multi_ontology_gate.sh` | gate standalone dos 2 drivers |

## 7. Coordenação de lanes (por que diretório próprio)

Na data desta rodada, `real-data/*.tsv|*.txt|downloads/` e
`scripts/ci/ontology_frontiers_gate.sh` estavam sob claim ativa da lane
`kimi-cli1/elplus-scale-more-20260805`, e `real-data/scale/` +
`artifacts/ontology-frontiers/README.md` sob
`kimi-cli1/elplus-optimize-20260805`. Para não editar em paralelo
(contrato do repo), a rodada 13 inteira ficou em `multi-ontology/` com
gate próprio (`scripts/ci/ontology_multi_ontology_gate.sh`), e a lane
irmã foi notificada via `bin/sounio-coord send`. Integração no README/gate
principais fica para o shepherd após o release das claims.

## 8. Limitações honestas

- Estatísticas em NÍVEL ATÔMICO (mesmo escopo da rodada 12): arestas com
  fonte existencial e células S sobre o universo internado não são
  computadas.
- Política namespace-only exclui fillers externos (CL→UBERON/GO etc.) —
  em CL só 3.335 das 18.880 classes declaradas são CL; as 477 restrições
  CL→CL são uma fração das 19.815 estatuídas no arquivo (a maioria aponta
  para fillers externos). Os números são exatos para a TBox extraída,
  não para a ontologia OWL completa.
- Interseções (definições lógicas) e restrições em equivalentClass não
  entram na extração (contadas); o teorema de perfil das rodadas 11-12
  cobre o formato extraído.
- Os 3 cones GO cobrem o GO inteiro, mas a soma das arestas de papel dos
  cones difere do total da rodada 12 pelas 3.603 restrições cross-cone
  (medido; §3).
- **Completude atom-level de CL/UBERON (apontada pela math-review
  xai):** os arquivos brutos NÃO satisfazem a premissa do teorema de
  perfil (cada um tem 1 restrição no lado superclasse +
  equivalentClass restrictions aninhadas). Os axiomas omitidas são
  inertes sob a política namespace-only (filler PATO não internado; no
  caso UBERON o alvo CL:0000000 também está fora), logo os números são
  exatos para a TBox extraída — mas contra a semântica OWL completa a
  extração pode perder arestas átomo-átomo (transitividade via nó
  existencial). GO go-plus satisfazia a premissa exatamente (0 tais
  axiomas, sondado na rodada 11).

## 9. Revisão LLM-offload (política do repo)

`bin/llm-offload -t math-review -p xai` sobre as 5 claims da rodada
(partição/decomposição, contador agrupado, déficit de arestas de papel,
teorema de perfil para CL/UBERON, máscaras NEPW) → **4 [OK], 2
correções incorporadas**: (i) typo aritmético no meu texto
(29.770.678 → 29.770.768 intra-cone); (ii) Claim 4 refutada em geral e
re-enquadrada como em §4/§8. Input em `MATH_REVIEW_INPUT.md`; log em
`agent_logs/multi_ontology_offload_2026-08-05.md`.
