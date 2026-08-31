# Varredura de problemas abertos no cone de luz Sounio — 2026-08-31

Busca sistemática (alphaXiv, prioridade recência) por problemas abertos **declarados**
nos últimos ~18 meses, nos três cones: álgebras de Cayley-Dickson, metrologia/propagação
de incerteza, e sistemas de tipos graduais/graduados. Quatro papers lidos na íntegra ou
por extração dirigida; IDs arXiv citados.

---

## 1. O achado principal: cluster Guterman–Zhilina, 27–28/08/2026 (há 3 dias)

Quatro papers do grupo de Moscou (Guterman & Zhilina), publicados **esta semana**,
exatamente no território do nosso resultado ker L_z = ker R_z (memória:
`project_sedenion_ker_twosided_witness_2026-08-23`):

| arXiv | Conteúdo | Estado |
|---|---|---|
| 2608.26903 | *Relation graphs of the sedenion algebra* — componentes de Γ_O(𝕊) ↔ retas em Im(𝕆) (bijeção, Thm 4.15); diâmetro de cada componente = 3 (Thm 4.11); cintura 4; cliques máximas = 2 | **Conjecture 6.8 ABERTA** |
| 2608.28176 | Part I — hexágonos de zero-divisores duplamente alternativos em CD reais arbitrárias; tabela de multiplicação do double hexagon | resultados fechados |
| 2608.28163 | Part II — para dim ≥ 16, grafo de ortogonalidade em pares de base ≅ ⟺ álgebras ≅; algoritmo que recupera os parâmetros γ do grafo | resultados fechados |
| 2608.26893 | Zero-divisores duplamente alternativos sobre corpo F arbitrário, char ≠ 2; dim de aniquiladores ≡ 0 (mod 4) | classificação geral declarada "not trivial, except particular cases" |

### 1a. Conjecture 6.8 (2608.26903) — alvo imediato

> *O diâmetro do subgrafo Γ_C^Z(𝕊) do grafo de comutatividade dos sedenions
> (vértices = elementos cuja parte imaginária é zero-divisor) é igual a 3.*

Eles provam 3 ≤ diam ≤ 4 (Props 6.6/6.7) e conjecturam 3 **com base em experimentos
no Wolfram Mathematica** — ou seja, evidência numérica flutuante, sem certificado.

**Por que é nosso:** a prova do limite inferior deles usa exatamente
O_𝕊(a,b) ∩ O_𝕊(b,a) = 0 — a mesma álgebra linear do nosso witness exato
(rank 12 / dim 4, dois lados, base de 4 geradores com suporte disjunto).
A Prop. 4.6 deles (ortogonalizador de qualquer ZD tem dim 4, via Moreno) é o
**teorema clássico por trás do nosso "exactly 4/side" medido no 240-scan** — nosso
resultado mecânico agora tem âncora bibliográfica direta, e nossa exatidão sobre ℚ
+ Lean é o que falta a eles. Fechar a conjectura exige exibir, para pares
(a,b),(a′,b′) arbitrários, um caminho de comprimento 3 no grafo de comutatividade —
um enunciado de existência sobre subespaços de dim finita, candidato natural a:
1. varredura exata sobre ℚ com o pipeline do 240-scan (falsificação ou evidência total);
2. redução por Aut(𝕆) (transitivo em pares de ZD, Khalil–Yiu) a um número finito de
   configurações canônicas — depois certificado em Lean.

Risco: os autores podem fechá-la eles mesmos (é a conjectura deles). Mitigação: nosso
ângulo — witness exato + mecanização — é diferenciado mesmo se a prova clássica sair
primeiro; e a infraestrutura é 100 % reutilizável no item 1b.

### 1b. Componentes de Γ_O(M_n) para n ≥ 5 — aberto e maior

O Example 4.3 de 2608.26903 mostra que a classificação por componentes **quebra em
M₅** (existem ZDs com ab, cd linearmente independentes na mesma relação de
ortogonalidade — a bijeção com retas falha). A estrutura das componentes conexas de
Γ_O(M_n), n ≥ 5, fica aberta. Nossa escada de erasure L4–L11 (medida + Lean,
memória `project_cayley_dickson_erasure_ladder`) é literalmente uma máquina de scan
para M₅…M₁₁ — ninguém mais tem isso mecanizado.

### 1c. Corpo arbitrário (2608.26893)

Classificação de zero-divisores **não** duplamente alternativos e seus aniquiladores:
declarada não-trivial e aberta. Conexão direta com o ExactlyPrivate (ker L_z como
projeção de esquecimento): a tabela do double hexagon (Thm 5.2 de 2608.26903) dá uma
base canônica contendo qualquer ZD — candidata a alimentar a API de certificados.

---

## 2. Metrologia / propagação de incerteza

Nenhum "Problem 1" declarado, mas o estado do campo confirma que **a caracterização
geral de quando a soma ingênua é conservadora não tem dono**:

- 2606.30105 (*Interval belief structures + imprecise copulas*, NN verification):
  dependência entre incertezas tratada por cópulas imprecisas — sem critério de sinal
  de covariância estrutural; nosso lema da partição (invariância ⟹ Cov ≤ 0) não tem
  contraparte lá.
- 2605.15789 (*Gaussian overbounds*): overbounding conservador por convolução,
  correlação tratada empiricamente.
- 2510.24313 (revisão HEP): incertezas sistemáticas correlacionadas documentadas como
  problema foundacional **em aberto na prática da física de altas energias** — dor
  nomeada pela comunidade, sem teoria de decomposição com certificado.

**Pergunta aberta fabricável (nossa):** caracterizar a classe completa de decomposições
de uma soma para as quais a invariância do total implica Cov ≤ 0 entre as partes
(generalização do lema da partição de dce9f/178ee). RQ4 já é a base experimental.

## 3. PL / tipos graduais e graduados

- 2604.05246 (*A Gradual Probabilistic Lambda Calculus*, 04/2026): gradual +
  probabilístico acabou de nascer — semântica de medida, **não** momentos/GUM.
- 2607.20801 (*Imprecise probabilistic programming via graded monads*, 07/2026):
  credal sets como grading — adjacente a Knowledge<T>, sem budgets de segundo momento.
- 2606.28042 (*Same Coeffect, Different Base*, 06/2026): unificação das duas tradições
  de coeffects graduados — o andaime algébrico pronto.

**Teorema não reivindicado:** *gradual guarantee* para incerteza metrológica —
"adicionar anotações Knowledge<T> nunca degrada o certificado", com budgets de
variância como semiring de coeffects. Está na interseção de dois papers de 2026 e
nenhum dos dois o enuncia. Só existe se a Sounio existir.

## 4. Lean como fonte de problemas abertos (meta-achado)

- 2605.13171 *Formal Conjectures* — benchmark aberto e crescente de conjecturas de
  pesquisa formalizadas em Lean.
- 2608.11941 *OEIS Open* — 492 conjecturas abertas do OEIS formalizadas em Lean.
- 2603.15617 *HorizonMath* — progresso de IA rumo a descoberta com verificação automática.

Estoques curados de problemas abertos **já machine-checkable** — o formato exato do
nosso workflow humano+IA+Lean. Minerável quando quisermos um alvo externo com juiz
automático.

---

## Ranking (alavancagem × frescor × reutilização)

1. **Conjecture 6.8** — 4 dias de idade, nossa máquina já construída, fechável.
2. **Γ_O(M₅) componentes** — aberto de verdade, escada L4–L11 pronta, sem competição mecanizada.
3. **Gradual guarantee epistêmica** — teorema Sounio-nativo, venue PL, zero competição.
4. **Generalização do lema da partição** — metrologia foundacional, RQ4 como laboratório.

Arquivos-fonte (extração): `tool-results/mcp-claude_ai_alphaXiv-get_paper_content-17881888{41608,93286,94568}.txt`
no diretório da sessão.
