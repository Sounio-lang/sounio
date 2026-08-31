# Ataque à Conjecture 6.8 (Guterman–Zhilina, arXiv:2608.26903) — log 2026-08-31

**Alvo:** diam Γ_C^Z(𝕊) = 3 (provado 3 ≤ d ≤ 4 no paper; o 3 é conjecturado a partir
de experimentos Mathematica em ponto flutuante). Contexto e ranking:
`open_problems_scan_2026-08-31.md`.

## A redução (o que tornou a conjectura computável)

1. Comutação depende só da parte imaginária ⟹ vértices reduzem a ZDs puros.
2. Para ZD puro x: Im C(x) = ℝx ⊕ O(x), dim 5 (Lemma 6.2(2) + Moreno dim 4).
3. d=1 ⟺ [x,x′]=0 (exato); d=2 ⟺ (ℝx⊕O(x)) ∩ (ℝx′⊕O(x′)) ≠ 0 (posto exato);
   d=3 ⟺ o kernel racional exato do mapa bilinear T: U⊗V → Im 𝕊, (u,w) ↦ [u,w]
   contém um tensor real de posto 1 (k mᵀ). Contagem dimensional ingênua diz que
   genericamente NÃO deveria existir (dim ker ≥ 9 em ℝ⁵ˣ⁵; posto-1 tem codim
   demais) — quando existe sempre, é estrutura octoniônica, não acaso.
4. Um par sem tensor posto-1, com U∩V=0 e [x,x′]≠0, é contraexemplo (diam 4).

## Perna primária (Sounio sob Madaros, ADR-008)

`tests/run-pass/sedenion_conj68_basis_probe.sio` — canônico x₀=(e₁,e₂) vs swap
(e₂,e₁) + os 84 ZDs de base (aproveitando: no subgrafo de base, O(x′) = span dos
4 parceiros de base = vizinhos do double hexagon; VERIFICADO por aniquilação
bilateral exata i64, independência por posto de Bareiss inteiro).

Resultado (Madaros v0.80.0 COMMITTED, ELF md5=bf1fe608):

```
PARTNERS4 OK          # 84 ZDs de base, exatamente 4 parceiros cada (Descr. 4.18)
URANK5 OK             # dim(ℝx₀⊕O(x₀)) = 5 exato
D1 4  D2 37  D3W 43  D3U 0
SWAP_PROP67_AND_WITNESS OK   # Im C(x)∩Im C(x̃)=0 exato (Prop 6.7) + witness d=3
CONJ68_BASIS_SWEEP OK
```

**Todos os 85 alvos têm caminho de comprimento ≤ 3.** A conjectura sobrevive ao
sweep de base completo com certificados exatos nos degraus lineares.

## As três classes de witness descobertas (a estrutura do problema)

1. **Combinatória (mesma componente de Γ_O, paridade ímpar)** — e.g. o swap e
   (e₁,−e₂): o witness é um caminho puro de ortogonalidade dentro do double
   hexagon; aparece como fatia (i,j) inteiramente ZERO do tensor inteiro
   [uᵢ,wⱼ] — detectada exatamente, sem float. A iteração numérica cega tem
   bacia minúscula aqui (por isso os "unresolved" das primeiras rodadas).
2. **Contínua genérica** — witness (k,m) denso, achado pela iteração alternante
   (potência inversa em AᵀA 5×5) + polimento Gauss–Newton.
3. **Cruzada não-ortogonal (a família dura)** — pares como (e₁,e₂) vs (e₂,±e₃)
   (componente octoniônica compartilhada): o witness tem k₀ = m₀ = 0, i.e.
   u ∈ O(x₀), w ∈ O(x′) — ZDs de **componentes distintas de Γ_O** (invariantes
   e₃ vs ∓e₁) que **comutam sem serem ortogonais** (caso w = λu + q impossível
   de ortogonalidade: componentes distintas ⟹ uw ≠ 0). A busca z-parametrizada
   de witnesses ortogonais (caso-(i), u = √n(z)·x₀ + L(z) afim em z) devolve
   residual 2.6 — confirmando que NÃO há witness ortogonal para esses pares.
   Resolvido na perna souc com fase `lock0` (subespaço k₀=m₀=0 imposto
   exatamente via linhas-identidade nos sistemas normais).
   Nota: os witnesses de (e₂,+e₃) e (e₂,−e₃) são permutação/negação um do outro
   com o MESMO m — forte indício de família uniforme sob a simetria do hexágono.

## Oráculo diferencial (busca apenas; não define pass/fail — ADR-008)

`scripts/research/sedenion_commutativity_diameter_probe.py` — mesma redução em
Python/numpy; reproduziu D1/D2 idênticos e forneceu os witnesses numéricos da
classe 3 que orientaram o `lock0`. Convenção de multiplicação verificada
idêntica à Table 1 do paper e ao `cd_sigma` do repo (par canônico
(e₁+e₁₀)(e₇+e₁₂)=0 ✓).

## Rodada 2 (mesmo dia): os witnesses são INTEIROS — o float era muleta

Kernel exato sobre ℚ do bilinear restrito ao setor k₀=m₀=0 (O(x₀)⊗O(x′) → Im 𝕊,
matriz inteira 16×16) para os pares duros: **dim ker = 13 de 16** — o mapa é
massivamente degenerado (estrutura octoniônica, não acaso) — e o kernel contém
elementos de posto 1 **inteiros**:

- (e₁,e₂) vs (e₂,+e₃): u = (e₄,−e₇) ∈ O(x₀), w = (e₆−e₄, e₇+e₅) ∈ O(x′),
  [u,w] = 0 exato. (E o espelho u = (e₄,−e₇)+(e₆,−e₅), w = (e₄,−e₅).)

Consequência implementada: `exact_combo_witness` — varredura de combinações
com suporte ≤ 2 e coeficientes ±1 (25 direções projetivas por lado, 625 checks
de comutador inteiro por par). Resultado no sweep completo:

```
D1 4  D2 37  D3W 43  D3EXACT 43  D3U 0     ← float fallback NUNCA executado
CONJ68_BASIS_SWEEP OK
```

**Todo witness de comprimento 3 no sweep de base é uma combinação inteira
±1 de no máximo 2 geradores por lado.** A perna primária é agora aritmética
i64 decidível de ponta a ponta (o hunt f64 permanece só como fallback
documentado para o futuro sweep racional denso). Isso também aponta o caminho
da prova: um certificado FINITO e combinatório por classe de pares — formato
ideal para Lean.

## Rodada 3 (mesmo dia): sweep racional off-basis — leis de posto

Runner: `examples/research/conj68_rational_sweep.sio` (100 % Sounio, i64 +
triagem f64 portada; mod-p só para postos, com direção de erro favorável:
posto 10 mod p CERTIFICA posto exato 10).

ZDs inteiros aleatórios x′ = (k·a, a·t), n(t) = k² (parametrização densa da
variedade; entradas pequenas por limite de i64 — declarado, não silencioso).
**Forma fechada do ortogonalizador descoberta e verificada: os parceiros de
(A,B) são (N·c, c·z), z = AB, N = n(A), c ⊥ {1,A,B,z}** — FORMULA_FAIL = 0
em 60/60 (verificação por aniquilação bilateral exata, não assunção).

```
SWEEP 60  D1 0 D2 2 W1 0 W2 8 W3 4 UNRES 46
WFLOAT 46 CAND_D4 0            ← todos os UNRES têm witness contínuo
RANKT 11 ALL 58 UNRES 46       ← rank(T) = 11 UNIFORME (ker dim 14)
RANKT44 7 N 7 · RANKT44 9 N 51 ← setor O×O: posto ≤ 9, bimodal {7,9}
```

Três fatos novos:

1. **O fenômeno inteiro ±1 NÃO persiste off-basis** (46/60 sem combo pequena) —
   witnesses genéricos são contínuos/algébricos. A estrutura inteira da base
   era o caso especial, não a regra.
2. **Conjecture 6.8 sobrevive 145/145** (85 base + 60 racionais): nenhum
   candidato a d=4.
3. **Lei de posto**: rank(T) = 11 em todos os pares d≥3 amostrados; o quociente
   é um mapa bilinear ℝ⁵×ℝ⁵ → ℝ¹¹. Deficit de Segre = 3 nos dois setores.
   Stiefel–Hopf NÃO obstrui mapas nonsingulares 5×5→11 (intervalo da condição
   binomial é vazio), logo o posto baixo sozinho não basta: a conjectura
   equivale a (a) provar rank(T) ≤ 11 estruturalmente (candidato: identidades
   de flexibilidade/Moufang da recursão CD) e (b) provar que ESTE quociente
   sempre tem zero real não-trivial — as "3 degenerescências escondidas" são o
   coração do problema. Formato ideal: (a) é álgebra linear mecanizável em
   Lean; (b) é um lema de topologia real sobre uma família explícita.

Sub-pergunta aberta interna: caracterizar os 7 pares com rank(T|₄ₓ₄) = 7
(mais degenerados; provável relação com subálgebras quaterniônicas comuns).

## Próximos degraus

1. **Sweep racional geral** (além da base): amostragem densa de pares ZD
   racionais via b = a·t, t ⊥ 1,a com n(t) quadrado (parametrização completa da
   variedade ZD) — perna souc dedicada fora do run-pass (custo maior).
2. **Construção uniforme**: minerar a estrutura dos witnesses classe 3
   (mesma-m, permutação-k) → candidato a fórmula fechada u(x,x′), w(x,x′);
   se fechar, a prova da conjectura vira verificação de identidades algébricas.
3. **Lean**: identidades da construção uniforme sobre a recursão CD (pipeline
   EpistemicEffectsNS: `lean --threads=1`, sem `decide` em não-Decidable).
4. **Witnesses exatos da classe 3**: identificar algebricidade de k
   (0,−0.7794,−0.1183,0.3879,0.4776) — PSLQ/candidatos quadráticos.

## Risco de corrida

Conjectura dos próprios autores (grupo ativo, Moscou). Nosso diferencial:
certificados exatos nos degraus lineares + perna mecanizável em Lean. Mesmo se
a prova clássica deles sair primeiro, a infraestrutura ataca o alvo #2 do scan
(componentes de Γ_O(M₅), genuinamente aberto).
