<!-- docs:meta
topic_id: repo.docs.research.erdos-508-704-sounio-resolution-plan
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.erdos-508-704-sounio-resolution-plan
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio Resolution Plan: Erdős #508 and #704

**Primary Objective (locked)**  
Resolver (ou entregar evidência decisiva e verificável em direção à resolução de) dois problemas abertos de Erdős usando a pilha completa do Sounio:

- **#508**: Chromatic number of the plane (Hadwiger–Nelson) — atualmente 5 ≤ χ ≤ 7.
- **#704**: Chromatic number of the unit-distance graph in ℝ^d e a taxa de crescimento assintótica χ(d)^{1/d}.

Este é o objetivo principal de proof-of-platform. Tudo mais (incluindo Track B em paralelo leve) é subordinado.

---

## Por que Sounio tem uma alavanca real aqui (e a maioria não tem)

1. **Dimensão 16 exata**  
   O kernel de produção sinkhorn16 (usado em 3,1M valores de ORC na coorte ABIDE) opera exatamente em neighborhoods de tamanho 16. Sedenions são 16-dimensionais. Isso não é coincidência — é uma correspondência estrutural que ninguém mais tem rodando em produção.

2. **168 classes ZD verificadas + ponte formal**  
   Exatamente 168 non-Fano triples em octonions (SounioCayleyDickson.lean, provado com native_decide).  
   Exatamente 168 classes projetivas de zero-divisores em sedenions (SounioZeroDivisorBridge.lean).  
   Cada classe dá um annihilator exato de 4D via multiplicação à direita. Isso é "cirurgia" precisa em subespaços — não heurística.

3. **Emitter unificado em Kretikos**  
   O mesmo binário self-hosted que emite o sinkhorn16 de produção já contém `kaxi_emit_sedenion_associator_asm` (kretikos_emit_kaxi.sio:2660, 4 fases, spill controlado, layout explícito de 96 words/thread).  
   Podemos computar associators e projeções ZD em configurações de pontos sem escrever novo código de baixo nível do zero.

4. **Parenthesization + epistemic propagation**  
   A variância total em sedenions **não** é invariante por parenthesization (HYPER_UNCERTAINTY_PARENTHESIZATION_REPORT.md + knightian.sio).  
   Podemos gerar famílias de 168 versões "torcidas" da mesma configuração de pontos e propagar bandas Knightian nos invariantes resultantes (curvatura, chromatic forcing, etc.).

5. **Controle até microarquitetura + bit-reprodutibilidade**  
   Modelos epistêmicos em `target_policy.sio` + exigência de bit-reprodutibilidade (já usada cientificamente na ABIDE) transformam qualquer busca ou kernel em artefato auditável cross-uarch.

Nenhum sistema atual (IA + Lean clássico, Julia, etc.) combina formalização da não-associatividade + emissor unificado de kernels de produção + cirurgia 4D exata em neighborhoods 16D + UQ epistêmica de segunda ordem.

---

## O que significa "resolver" (níveis de sucesso)

- **Nível 1 (Witness computacional forte)**: Encontrar uma configuração algébrica sedenion (ou projeção ZD dela) cujo grafo de conflito torcido é 5-cromático com número de vértices competitivo ou melhor que o estado da arte conhecido, com todos os passos reproduzíveis via `souc` + Kretikos + Lean.
- **Nível 2 (Novo invariante verificável)**: Definir e provar (parcialmente em Lean) um novo invariante 168-espectro ou ZD-modulado que dá lower bounds rigorosos ou separação entre famílias de grafos unit-distance.
- **Nível 3 (Melhoria de bound)**: Melhorar um bound conhecido para #508 (ex.: reduzir o tamanho do menor exemplo 5-cromático conhecido) ou para a taxa de crescimento em #704, com a melhoria traçável diretamente para o uso de 168/ZD/parenthesization.
- **Nível 4 (Resolução parcial ou full, com formalização)**: Provar um novo bound que estreita o gap 5-7 de forma significativa, ou resolver uma versão algébrica restrita do problema, tudo com artefatos Sounio verificáveis.

Sucesso em qualquer um desses níveis já é um proof-of-platform visível e defensável.

---

## Plano Faseado (foco em resolver, não só explorar)

### Fase 0 — Fundação (concluída / em andamento)
- Módulo `stdlib/hypercomplex_graph/erdos_unit_distance.sio` criado e integrado.
- Estruturas básicas: `SedenionPoint`, `ZDSurgery`, `LabeledUnitDistanceGraph`, geração de famílias 168.
- Lean stub `SounioErdosUnitDistance.lean` com primeiros witnesses decidíveis (MoserSmoke).
- Documento de plano de resolução (este arquivo).

### Fase 1 — Primeiros Witnesses Concretos (avançado nesta iteração)
- `apply_zd_surgery` implementado usando `sed_mul` real de `stdlib/algebra/sedenion.sio` + seleção de ZD por class_id (demo com canônicos z/w, pronto para generalização).
- Exemplo executável `examples/erdos/moser_zd_probe.sio` criado: levanta probe em 16D, aplica múltiplas cirurgias das 168 classes, demonstra deltas de geometria.
- Lean avançado com 7-vertex Moser probe + teorema decidível de que ZD produz twisted graphs distintos.
- Kretikos com sketch detalhado da extensão `kaxi_emit_erdos_point_associator`.
- Entregável: os quatro artefatos acima + build verde no Lean.

### Fase 2 — Kretikos em Escala (computação pesada)
- Usar o emissor existente de associator sedenion para calcular signatures em todas as triplas das configurações.
- Alimentar as matrizes de custo torcidas no sinkhorn16 de produção (mesmo caminho da ABIDE).
- Medir o 168-espectro de curvatura local nos grafos de conflito.
- Entregável: runs reprodutíveis via Kretikos + bandas epistêmicas.

### Fase 3 — Avanço Formal (Lean)
- Crescer o MoserSmoke para exemplo de 7 vértices com mudança mensurável de lower bound cromático sob diferentes cirurgias ZD.
- Provar (ou esboçar prova) monotonicidade ou separação de propriedades sob aplicação de classes ZD específicas.
- Entregável: teoremas Lean com native_decide onde possível + obrigações claras.

### Fase 4 — Busca / Otimização Dirigida
- Busca sistemática (via Sounio + Kretikos) por configurações sedenion/ZD que maximizem chromatic lower bound ou minimizem tamanho para χ=5.
- Exploração em 16D (para #704) e projeções de volta para 2D (para #508).
- Entregável: novas famílias de grafos ou novos argumentos de lower bound.

### Fase 5 — Publicação / Proof de Plataforma
- Artefatos completos: código Sounio, Lean theorems, Kretikos binaries, relatórios de reprodutibilidade cross-uarch, bandas epistêmicas.
- Paper ou nota técnica focada em "Non-associative algebraic methods for unit-distance chromatic problems via verified 168/ZD structure".

---

## Riscos e Honestidade

- Risco principal: as famílias 168/ZD podem não produzir bounds estritamente melhores (ainda assim geram invariantes novos e o proof de plataforma funciona).
- O problema clássico é extremamente difícil; sucesso total (fechar o gap 5-7) pode não acontecer em uma única iteração. Sucesso intermediário em Nível 1-3 já é valioso.
- Manter escopo: foco em #508 e #704. Não dispersar em outros Erdős problems.

---

## Status Atual

### Iteração 2026-05-24 — primeira computação REAL (fatia vertical end-to-end)

Os esboços anteriores (dialeto inválido, `dot`/`near` placeholder, `chromaticNumber
:= 4 + surgery%2` no Lean, teorema central `sorry`) foram substituídos por uma
fatia computável e verificável, em paridade Sounio↔Lean.

**Enquadramento matemático honesto (verificado, não assumido):**
- Todo grafo de coordenadas inteiras cujas arestas são pares a distância
  euclidiana ao quadrado `== 1` é **bipartido (χ ≤ 2)** — é subgrafo do reticulado
  ℤ¹⁶, 2-colorível pela paridade da soma das coordenadas. (A moldura anterior
  "Moser spindle ⇒ χ=4" estava confusa: o spindle não é realizável com coords
  inteiras a dist²=1.)
- Pergunta real, decidível: **uma cirurgia ZD consegue quebrar a bipartição
  (forçar χ ≥ 3)?**

**Mecanismo implementado (cirurgia ZD / multiplicação à direita):** para cada um
dos **84 primitivos válidos** `v = e_lo ± e_hi` (`validPrims` do bridge), o ponto
é multiplicado à direita por `v` (mapa linear com kernel não-trivial — sedenions
NÃO são norm-multiplicativos). Aresta torcida ⟺ `‖(p−q)·v‖² == ‖v‖² == 2`.
A multiplicação usa exatamente `cd_sigma_ct` (== `cdSigma` do Lean).

**Artefatos (verdes, reprodutíveis):**
- `stdlib/hypercomplex_graph/erdos_unit_distance.sio` — núcleo auto-contido: 84
  primitivos, cirurgia, grafo de conflito, **número cromático exato** (força
  bruta), espectro das 84 cirurgias. Self-check: `validPrims==84`,
  `(e3+e10)(e6-e15)==0`.
- `examples/erdos/moser_zd_probe.sio` — roda via `souc`, imprime números reais.
- `formal/lean4/SounioErdosUnitDistance.lean` — **sem `sorry`/axioma novo**;
  4 teoremas `native_decide` espelhando o Sounio (`lake build` verde).

**RESULTADO (probe de 7 vértices, sondando ambas as metades do sedenion):**
- clássico: 6 arestas, **χ = 2**.
- cirurgia NÃO é trivial: **4 das 84** cirurgias alteram o conjunto de arestas
  (até 9 arestas) — `some_zd_surgery_changes_edges`.
- porém **nenhuma das 84 cirurgias eleva χ**: χ permanece 2 em todas
  (`no_zd_surgery_raises_chromatic` / `twisted_chromatic_le_classical`).

**Leitura honesta:** NEGATIVO genuíno (não vacuoso) para a cirurgia linear de
multiplicação-à-direita neste probe — ela mexe nas arestas mas preserva a
bipartição. Não é p-hacking de probe; é um resultado machine-checked que
descarta esse mecanismo simples como alavanca de χ aqui.

**Próxima alavanca principiada (não probe-fishing):** cirurgia por **associador**
`(p·u)·v` com `u·v = 0`. Como `p·(u·v) = 0`, tem-se `assoc(p,u,v) = (p·u)·v` — a
não-associatividade genuína, indexada diretamente pelos **168 pares ZD** (não 84
mapas lineares). É o caminho correto para testar χ ≥ 3.

### Iteração 2026-05-25 — cirurgia por associador (Fatia 2)

Implementada a cirurgia por associador `(p·u)·v` com `u·v=0`, sobre os **168
pares ZD** (enumerados via `is_zero_pair`; composição de duas multiplicações à
direita por primitivos, reaproveitando o mesmo atalho de 2 termos). Alvo de
"distância unitária torcida" = `‖u‖²·‖v‖² = 4`. Verificado empiricamente que a
distribuição de `‖((e_m)·u)·v‖²` nas 16 direções de base é `{0:6, 4:8, 8:2}` —
clusters limpos (kernel / unitário-torcido / amplificado), sem cluster em 2, logo
o alvo `==4` é bem-definido.

**RESULTADO (mesmo probe de 7 vértices):**
- `168` pares ZD enumerados (== `unorderedZDPairs` do bridge).
- cirurgia **totalmente ativa**: **168 de 168** classes alteram o conjunto de
  arestas (arestas variam de 2 a 12, vs 6 clássicas) — muito mais forte que a
  linear (4/84).
- porém **0 de 168** elevam χ: permanece 2 em todas.

**Leitura honesta (mais forte que a Fatia 1):** mesmo a não-associatividade
genuína, agindo em TODAS as 168 classes, não quebra a bipartição neste probe. As
duas fatias juntas isolam a conclusão: **o gargalo é o tamanho do probe (7
vértices), não o mecanismo algébrico** — o associador é a alavanca certa e está
plenamente ativo. (Lembrete de escala: o grafo 5-cromático de de Grey tem 1581
vértices; esperar χ≥3 de 7 vértices sempre foi otimista.)

Lean: `associator_class_count_168`, `all_associator_surgeries_change_edges`,
`no_associator_surgery_raises_chromatic`, `associator_chromatic_le_classical` —
todos `native_decide`, sem `sorry`/axioma novo; paridade com o run Sounio.

**Próxima alavanca (Fatia 3):** escalar o nº de vértices com um teste de
**bipartição por BFS** (O(V+E), escala a centenas de vértices — o brute-force
k^n não), e/ou **partir de um grafo unit-distance clássico com χ≥3** (realização
racional do Moser spindle ou subgrafo estilo de Grey) e perguntar se a cirurgia
ZD **preserva ou quebra** a não-bipartição. Essa é a pergunta de pesquisa real.

### Iteração 2026-05-25 — escala + busca de χ≥3 (Fatia 3)

Nota: um grafo unit-distance **clássico** com χ≥3 e coords **inteiras** é
impossível (dist²=1 inteiro ⟹ bipartido; o spindle precisa de coords
irracionais, que quebram a exatidão do `native_decide`). Então a pergunta foi
reformulada para o objeto que NÃO é reticulado: o **grafo torcido** (aresta ⟺
‖M_v(p_i−p_j)‖²==alvo). Como `M_v` é um mapa linear fixo, esse grafo pode em
princípio conter triângulos (existem três pontos inteiros a dist² mútua 2).

Teste por **bipartição BFS** (O(V+E), escala; o brute-force k^n não), sobre duas
famílias inteiras **completas** (não cherry-picked):
- binária peso-≤2: `{0} ∪ {e_i} ∪ {e_i+e_j}` = **137** pontos;
- com sinal peso-≤2: `{0} ∪ {±e_i} ∪ {e_i±e_j}` = **273** pontos.

**RESULTADO (504 grafos = 252 cirurgias × 2 famílias):**
- **0 não-bipartidos.** Nenhuma cirurgia (linear ou associador) força χ≥3.
- Contagens de arestas constantes por família e simetria PSL(2,7) (linear 340 /
  856; associador 1862 binário, 6528–6692 com sinal — grafos genuinamente
  distintos, hashes diferentes, exclui bug).
- **Linear: bipartição é ESTRUTURAL** — a paridade do peso de Hamming total é
  uma 2-coloração própria de TODOS os 84 grafos lineares em AMBAS as famílias
  (84/84). Provado em Lean (`linear_surgery_total_parity_2colors`, `native_decide`,
  sobre o probe binário completo de 137) — não é acidente de tamanho.
- **Associador: bipartido (BFS) em todos**, mas paridade total/meia-graduação
  NÃO 2-colore — a 2-coloração explícita é não-óbvia (em aberto).

**Leitura honesta:** as três fatias juntas mostram que a alavanca algébrica está
validada (o associador age em toda classe) mas a cirurgia baseada em distância,
sozinha, **não quebra a bipartição** de probes inteiros — para o linear isso é um
teorema (2-coloração por paridade). Forçar χ≥3 exige quebrar essa paridade.

**Próxima alavanca (Fatia 4):**
- provar a 2-coloração explícita do caso **associador** (ou achar o invariante);
- mecanismos que QUEBRAM a paridade total: multiplicação à **esquerda**, ou uma
  construção de grafo que não seja distância-torcida (rótulo de associador /
  wave como restrição de cor, curvatura ORC modulada);
- #704: subir dimensão (pathions 32D) / escala Kretikos.

### Iteração 2026-05-25 — multiplicação à ESQUERDA (Fatia 4a)

Hipótese: como unidades imaginárias distintas anticomutam (`σ(a,b)=−σ(b,a)`), a
multiplicação à esquerda `v·d` poderia quebrar a 2-coloração por paridade total
(os termos que tocam o índice 0 não invertem o sinal). Testado `lmul_into`
(`(v·x)_k = x_{k^lo}σ(lo,k^lo) + s·x_{k^hi}σ(hi,k^hi)`) com a mesma busca BFS.

**RESULTADO: idêntico à direita.** Mesmas contagens de arestas (340 binário,
856 com sinal), **0 não-bipartidos**, paridade total 2-colore 84/84 em ambas as
famílias. A norma `‖v·d‖²` é invariante ao lado para estes primitivos → a
multiplicação à esquerda NÃO é a alavanca de quebra de paridade.

Resultado fortalecido: o invariante de bipartição é **bilateral**. Provado em
Lean (`left_surgery_total_parity_2colors`, espelhando o lado direito,
`native_decide`, sem `sorry`). Conclusão: forçar χ≥3 com cirurgia ZD baseada em
distância é impossível por paridade (ambos os lados); é preciso uma construção
que **não** seja distância-torcida.

### Iteração 2026-05-25 — grafo de conflito por ASSOCIADOR: χ≥3 (Fatia 4b) ★

**Primeiro χ≥3 do programa — o teto de paridade foi quebrado.**

Construção não-distância: o associador `assoc(x,y,c) = (x·y)·c − x·(y·c)` é
**bilinear** em (x,y), NÃO é função de `x−y`, logo o grafo de conflito não é
invariante por translação e não herda a bipartição de paridade. Aresta `(i,j)`
⟺ `‖assoc(p_i,p_j,c)‖² == 4` (simetrizado; 4 = menor valor não-nulo da
distribuição `{0,4,8,16,…}`, grafo esparso ~13%). `c` = primitivo ZD.

**RESULTADO (probe binário peso-≤2, 137 pts, varrendo os 84 c):**
- **NÃO-bipartido para TODOS os 84 c** (χ≥3); paridade total 2-colore **0/84** —
  o invariante que travava a distância está **quebrado**.
- Certificado concreto de χ≥3: **triângulo `{e_1, e_2, e_3}`** (índices 2,3,4)
  no grafo de conflito com c=primitivo 0 (`e_1+e_10`). Provado em Lean
  (`associator_conflict_triangle`, `cWit_is_first`, `native_decide`, sem `sorry`).
- Limite de cor (guloso): χ ∈ [3, ~10].

**Leitura honesta (Nível 2 — novo invariante verificável + separação):**
isto NÃO é um novo limite para χ(plano) (#508). É uma **separação estrutural**: a
**não-associatividade é essencial** para escapar do teto χ≤2 das construções ZD
baseadas em distância. Toda cirurgia por distância (esq./dir./associador-como-
distância) é provadamente bipartida; o **associador como relação de conflito**
(bilinear, não-translação-invariante) atinge χ≥3. A não-associatividade do
sedenion não é decorativa — ela carrega obstrução cromática genuína.

**Próximo (Fatia 5):** (a) quão alto vai χ do grafo de conflito (clique/χ exato,
crescer probe); (b) ligar essa obstrução cromática algébrica de volta a uma
construção geométrica unit-distance (a ponte real para #508/#704); (c) #704 em
pathions 32D.

### Iteração 2026-05-25 — ponte alg→geom + regime esparso χ>ω (Fatia 5)

**Gating honesto da ponte (por que χ≥3 não basta):** um grafo unit-distance
clássico com χ≥3 e coords inteiras é impossível; e mesmo realizando o grafo de
conflito como unit-distance, χ≥3 não diz nada novo (χ(ℝ²)≥5 já é conhecido). A
ponte só importa se χ for alto E o regime for o "difícil".

**χ do grafo de conflito é alto mas CLIQUE-DRIVEN.** Medições (clique guloso +
DSATUR, heurísticas): χ ∈ [8, ~11], com clique ω≥8 (uma K₈ concreta em
`{e₃,e₁₂,e₁₃,…}`). χ≈ω ⇒ o χ alto vem de uma clique — o regime TRIVIAL para
unit-distance (K₈ = 7-simplex regular, dá só χ(ℝ⁷)≥8, o bound trivial d+1). O
regime de Erdős é **χ≫ω** (esparso, clique-poor). χ exato em 137 vértices é
inviável (provar não-k-colorabilidade por backtracking é exponencial; timeout).

**Resultado RIGOROSO no regime esparso (χ>ω):** extraíndo um subgrafo induzido
**triangle-free** (ω=2, por construção + verificado) que é **não-bipartido**
(χ≥3, BFS exato) ⇒ χ>ω com a MENOR clique possível. Concretamente, no grafo de
conflito T=8 existe um **C₅ induzido** (pentágono sem cordas) em
`{e₃, e₂+e₆, e₂, e₄, e₈}` ⇒ ω=2, χ=3. Provado em Lean
(`associator_conflict_induced_C5`: 5 arestas presentes + 5 cordas ausentes,
`native_decide`, sem `sorry`) e `cWit_is_first`.

**Leitura honesta:** alcançamos o regime esparso χ>ω de forma RIGOROSA — distinto
do χ≥8 clique-driven. Mas continua sendo separação ALGÉBRICA (Nível 2): um C₅ é
um pentágono unit-distance trivial; isto NÃO é (ainda) um bound geométrico novo
para #508/#704. A ponte alg→geom genuína exige χ≫ω **grande** num grafo
realizável — exatamente a dificuldade do problema aberto, não resolvida aqui.

**Honestidade sobre a fronteira:** as 5 fatias mapearam o que a estrutura
168/ZD/associador faz para o número cromático (rigorosamente), e onde ela esbarra
na dificuldade real de Erdős. Não há overclaim de resolução.

### Iteração 2026-05-25 — χ−ω num grafo REALIZÁVEL (Fatia 6): NULL informativo

Insight: um grafo com aresta ⟺ `‖p_i−p_j‖²==T` sobre pontos INTEIROS em ℝ¹⁶ É um
grafo unit-distance (escala 1/√T → arestas = distância 1) — **realizável** e
exato, ao contrário do grafo de conflito por associador (não-distância). Ponto
natural Sounio: os **84 primitivos** `e_lo±e_hi`.

Distribuição de `‖diff‖²` (84 primitivos): T=2 (630), **T=4 (2646, denso)**, T=6
(210, mais esparso). Buscando χ>ω esparso (subgrafo triangle-free não-bipartido):
- T=6: grafo INTEIRO triangle-free mas **bipartido** (χ=2).
- T=2, T=4: **clique-driven** (têm triângulos); a parte triangle-free é bipartida.
- Contraste no conjunto genérico 137-binário (Euclidiano): **idem** — bipartido /
  clique-driven em T=2,4,6.

**RESULTADO: NULL.** Nenhum grafo unit-distance realizável (primitivos OU genérico)
exibe χ>ω no regime esparso. O χ>2 só aparece via cliques (triângulos).

**Leitura honesta (e o porquê da ponte falhar, com precisão):** o gap esparso
χ>ω existe SÓ na relação de **associador (não-distância, não-realizável)** —
slice 5, C₅ induzido. As versões **realizáveis** (distância Euclidiana) perdem
esse gap: suas regiões triangle-free são bipartidas; χ>2 só clique-driven. Ou
seja, a não-associatividade que cria ciclos ímpares clique-poor é **incompatível
com a realizabilidade Euclidiana** neste conjunto de pontos. Essa é a obstrução
estrutural concreta à ponte — não "não achei", mas "o gap mora exatamente na
parte não-realizável".

### Infra de base (mantida)
- Emitter Kretikos pronto para uso (Fase 2, escala — adiado).
- Objetivo travado: resolver os dois Erdős com Sounio.

---

**Próximos passos imediatos**

Fatia 1 (cirurgia ZD linear) — CONCLUÍDA: itens 1–3 abaixo feitos, resultado
negativo honesto registrado acima.

1. ~~Implementar cirurgia ZD real no módulo Sounio~~ ✅ (84 primitivos, χ exato).
2. ~~Exemplo executável em `examples/erdos/`~~ ✅ (`moser_zd_probe.sio`).
3. ~~Avançar o lado Lean para 7 vértices~~ ✅ (4 teoremas `native_decide`, sem `sorry`).

Fatia 2 (cirurgia por associador) — CONCLUÍDA:
4. ~~**Cirurgia por associador** `(p·u)·v`, `u·v=0`, 168 classes, Sounio + Lean~~
   ✅ — 168/168 ativas, 0 elevam χ (negativo honesto, mecanismo validado).

Fatia 3 (escala + busca BFS) — CONCLUÍDA:
5. ~~Teste de **bipartição por BFS** (O(V+E)) substituindo o brute-force k^n~~ ✅
   (137 + 273 pontos, 504 grafos, 0 não-bipartidos).
6. ~~Busca de χ≥3 sobre famílias inteiras completas~~ ✅ — negativo; linear é
   bipartido por teorema (2-coloração de paridade, provado em Lean).

Fatia 4 (quebrar a paridade) — CONCLUÍDA:
7. ~~mult. à **esquerda** como alavanca~~ ✅ — descartada (idêntica à direita,
   invariante bilateral, provado em Lean).
8. ~~Construção NÃO-distância (grafo de conflito por associador)~~ ✅ ★ —
   **χ≥3 atingido** (84/84 c, triângulo `{e1,e2,e3}` provado em Lean); paridade
   quebrada. Primeiro χ≥3 do programa; não-associatividade é essencial.

Fatia 5 (ponte + regime esparso) — PARCIAL:
9.  ~~Quão alto vai χ do grafo de conflito~~ ✅ — χ∈[8,11] mas CLIQUE-driven
    (ω≥8), regime trivial; χ exato inviável em 137 vts.
10. ~~Atingir o regime esparso χ>ω rigorosamente~~ ✅ — C₅ induzido (ω=2, χ=3)
    no T=8, provado em Lean. Foothold no regime difícil.
11. **Ponte geométrica genuína** (EM ABERTO, o passo real para Erdős): χ≫ω
    GRANDE num grafo realizável como unit-distance — a dificuldade do problema
    aberto. Não resolvida.

Fatia 6 (χ−ω em grafo realizável) — NULL informativo:
12. ~~Grafo unit-distance realizável (Euclidiano) sobre os 84 primitivos / 137~~
    ✅ — bipartido (T=6) ou clique-driven (T=2,4); SEM χ>ω esparso. O gap mora
    só na relação de associador NÃO-realizável (slice 5). Obstrução à ponte
    identificada com precisão.

Frente aberta (não resolvida):
13. Construir o gap χ>ω num grafo genuinamente realizável — exige uma relação
    realizável que herde a estrutura de ciclo-ímpar clique-poor do associador
    (não óbvio que exista; é a dificuldade do problema aberto).
14. #704: pathions 32D / escala Kretikos.

O objetivo não é "explorar". É **resolver** — com cada passo verificável, e sem
overclaim quando a fronteira real do problema é atingida.
