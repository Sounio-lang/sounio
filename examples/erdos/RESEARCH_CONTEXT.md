# Contexto de Pesquisa — Lane Erdos

## 1. O problema

O **problema de Hadwiger–Nelson** pergunta: qual é o número mínimo de cores necessário para colorir o plano euclidiano `ℝ²` de modo que **nenhum par de pontos à distância 1** tenha a mesma cor? Esse número é chamado de **número cromático do plano**, denotado `χ(ℝ²)`.

Equivalentemente: qual o menor `k` tal que existe uma `k`-coloração própria do grafo de distância unitária do plano?

## 2. Estado da arte (2025/2026)

Os limites clássicos são:

```
5 ≤ χ(ℝ²) ≤ 7
```

- **Lower bound 4**: Moser spindle (1961).
- **Lower bound 5**: Aubrey de Grey (2018).
- **Upper bound 7**: coloração por hexágonos regulares (Isbell, 1950).

**Não existe prova de `χ(ℝ²) ≥ 6` para o plano euclidiano padrão.** O problema está aberto há mais de 70 anos.

## 3. A quebra de de Grey (2018)

Aubrey de Grey construiu o primeiro grafo de distância unitária com número cromático 5:

- Grafo original: **20.425 vértices**.
- Redução publicada: **1.581 vértices** (grafo de De Grey).
- Outras reduções posteriores:
  - Mixon: 1.585 → 1.577 vértices.
  - Marijn Heule (SAT solver / cube-and-conquer): grafos ainda menores.
  - Jaan Parts: atualmente o menor grafo 5-cromático conhecido tem **509 vértices** (Parts graph, 2020).

O método de de Grey usa **spindling**: combinar cópias rotacionadas de um grafo base para forçar restrições cromáticas. O verificador é computacional (geralmente SAT solver com prova DRAT/LRAT).

## 4. Por que `χ ≥ 6` é tão difícil?

Se `χ(ℝ²) = 6` ou `7`, deve existir um grafo de distância unitária 6-cromático (ou 7-cromático) no plano. Pritikin (1998) provou que qualquer grafo 7-cromático unit-distance deve ter pelo menos **6.198 vértices**.

Para `χ ≥ 6`, espera-se que um grafo 6-cromático unit-distance deva ser **muito grande** — provavelmente ordens de magnitude acima dos ~500 vértices do grafo 5-cromático atual. A busca exaustiva por grafos desse tamanho é computacionalmente impraticável sem heurísticas muito fortes.

## 5. Variantes com respostas mais fortes

Restringindo o tipo de coloração, conseguimos limites melhores:

| Variante | Definição | Limites conhecidos |
|----------|-----------|-------------------|
| `χ(ℝ²)` | coloração arbitrária | `5 ≤ χ ≤ 7` |
| `χ_mes(ℝ²)` | coloração mensurável | `5 ≤ χ_mes ≤ 7` (Falconer, 1981) |
| `χ_map(ℝ²)` | coloração tipo mapa (fronteiras contínuas/Jordan) | `6 ≤ χ_map ≤ 7` |
| `χ_poly(ℝ²)` | regiões poligonais | `6 ≤ χ_poly ≤ 7` |

Isso mostra que, se exigirmos regularidade topológica na coloração, 6 cores são necessárias. O obstáculo para `χ(ℝ²) ≥ 6` são colorações patológicas/irregulares do plano.

## 6. Trabalhos recentes relevantes

### Mundinger, Pokutta, Spiegel, Zimmer (2024/2025)
- Paper: *Extending the Continuum of Six-Colorings*.
- Usaram uma técnica de *deep annealing* (machine learning) para encontrar novas 6-colorações do plano.
- Expadiram o range de distâncias `d` para as quais existe uma 6-coloração evitando distância 1 nas 5 primeiras cores e distância `d` na sexta.
- **Não prova `χ ≥ 6`**, mas mapeia o espaço de 6-colorações.

### Fractional chromatic number
- Melhor lower bound publicado: `χ_f(ℝ²) ≥ 3.8991` (Bellitto, Pêcher, Sédillot).
- Melhor lower bound não publicado: `χ_f(ℝ²) ≥ 3.9898` (Jaan Parts).
- Upper bound: `χ_f(ℝ²) < 4.36` (Croft).

### "Almost coloring"
- Pokutta et al. (2025): **96.29%** do plano pode ser 5-colorido sem conflitos de distância unitária.
- Isso mostra que, se `χ = 6`, a parte problemática é muito pequena.

## 7. Métodos computacionais usados

A lane Erdos parece usar uma combinação de:

1. **SAT encoding** de grafos de distância unitária.
2. **Cube-and-conquer / cube sieving**: dividir o espaço de busca em cubos (restrições parciais) e refutar cada um.
3. **DRAT/LRAT proofs**: certificados verificáveis de não-5-colorabilidade.
4. **Spindling e Moser spindle-like constructions**.
5. **Scouts / campaigns**: busca heurística por candidatos a grafos 6-cromáticos.
6. **Reflexão Lean**: traduzir certificados para provas formais em Lean 4.

## 8. Implicações para a lane Erdos

### O que a lane provavelmente está tentando fazer

Dado o nome `chi6_candidate_search`, `cube_sieve`, `cube_cover`, `k65`, `g529`, `degrey`, a lane Erdos parece estar em uma de duas frentes:

1. **Buscar um grafo unit-distance 6-cromático** via computação (SAT + spindling + busca heurística).
2. **Provar que nenhum candidato em uma classe restrita é 6-cromático** — ou seja, refutar uma família de candidatos.

### Riscos e realismo

- Provar `χ(ℝ²) ≥ 6` é um problema em aberto de alto risco. A comunidade matemática não tem evidência forte de que seja verdade ou falso.
- A busca computacional por um grafo 6-cromático unit-distance pode exigir grafos com milhares ou dezenas de milhares de vértices.
- A estratégia mais promissora atualmente é provar resultados em **variantes restritas** (mensurável, map-type, poligonal) ou desenvolver novas técnicas de spindling.

### Oportunidades concretas para a lane

1. **Replicar e formalizar a prova de `χ ≥ 5`** em Lean 4 (já em andamento, vide `CHI5_REAL_AXIOM_AUDIT_2026-05-30.md`).
2. **Explorar `χ_map ≥ 6` ou `χ_poly ≥ 6`**: esses já são teoremas; formalizá-los em Lean pode ser um marco alcançável.
3. **Melhorar lower bounds de fractional chromatic number** via LP/SDP computacional.
4. **Desenvolver invariantes de coloração** que descartem classes grandes de candidatos.
5. **Usar a estrutura do Moser spindle e de Grey como seed** para spindling automatizado em busca de `K₆`-minor ou 6-cromaticidade.

## 9. Referências-chave

- de Grey (2018): *The chromatic number of the plane is at least 5*.
- Exoo & Ismailescu (2020): simplificações do grafo de de Grey.
- Heule: reduções SAT para grafos menores.
- Parts (2020): 509-vértice graph.
- Mundinger, Pokutta, Spiegel, Zimmer (2024): *Extending the Continuum of Six-Colorings*.
- Falconer (1981): lower bound mensurável.
- Pritikin (1998): lower bound no tamanho de grafos 7-cromáticos unit-distance.

---

*Documento criado em 2026-06-15 para contextualizar o trabalho da lane Erdos.*
