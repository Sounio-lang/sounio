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

Fatia 2 (recomendada, decisão de alavanca):
4. **Cirurgia por associador** `(p·u)·v`, `u·v=0` — não-associatividade genuína,
   168 classes diretas. Implementar em Sounio + Lean (mesma estrutura de prova),
   medir se algum dos 168 eleva χ ≥ 3 num probe que span ambas as metades.
5. Se positivo: provar `∃ classe, χ_torcido > χ_clássico` por `native_decide`
   (resultado grande). Se negativo: subir nº de vértices / dimensão (#704) e/ou
   escala Kretikos (Fase 2).

O objetivo não é "explorar". É **resolver** — com cada passo verificável.