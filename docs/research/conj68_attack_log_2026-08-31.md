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

## Rodada 4 (mesmo dia): O TEOREMA DE ESTRUTURA — rank(T) ≤ 11 explicado e mecanizado

Runner: `examples/research/conj68_rank_structure.sio`. Leg Lean (kernel-verified,
Mathlib-free, sem sorry): `formal/lean4/SounioConj68RankBound.lean` (LEAN_OK,
`lean --threads=1`, v4.33.0).

**Correção honesta:** a rodada anterior desta sessão reportou brevemente
"rank-7 ⟺ quatérnion compartilhado (QOVERLAP 4)" — era ARTEFATO de um buffer
overflow no extrator 4×4 (`[0;192]` para índices até 255, sem bounds-check no
Madaros). Corrigido; dados refeitos.

### O teorema de estrutura (medido 145/145; prova em esboço; canônico em Lean)

Para ZDs x = (a,b), x′ = (a′,b′):

> **im(T) ⊥ span{ x, x′, x̃, x̃′ }, onde x̃ = (b,−a) é o companheiro de
> hexágono (Lemma 3.6 do paper). Logo rank(T) ≤ 15 − 4 = 11 — e os dados
> dizem que o limite é JUSTO (= 11 em todos os pares genéricos).**

Prova (3 linhas, cada peça verificada no par canônico pelo kernel do Lean):

1. φ(u,w,v) = ⟨[u,w],v⟩ é **cíclica** (⟨a,bc⟩ = ⟨ac̄,b⟩ em elementos puros).
2. x ⊥ im T: φ(u,w,x) = ⟨[x,u],w⟩ = 0 pois u ∈ C(x). Idem x′.
3. **Lemma B:** [x̃, u] ∈ ℝ·(0,1) = ℝ·e₈ para todo u ∈ Im C(x)
   ([x̃,x] = 4f̃₀ pela Table 2; x̃ comuta com os 4 geradores de O(x)).
   Como todo elemento de Im C(x′) é **duplamente puro** (⊥ e₀ e e₈ — ZDs são
   doubly pure, Moreno), ⟨[x̃,u],w⟩ = 0. Idem x̃′.

As direções-mistério foram identificadas lendo os pontos de rede inteiros do
complemento (scan exato, suporte ≤ 3) em três pares representativos: em todos,
o complemento é exatamente {e₀, x, x′, x̃, x̃′}.

Legs Lean (native_decide sobre ℤ): frames aniquilam bilateralmente; Lemma B
canônico (suporte de [x̃₀,u] só em e₈); dupla-pureza dos frames; as 4 direções
⊥ aos 25 comutadores no par duro (e₁,e₂)vs(e₂,e₃); independência linear das 4.
Falta (futuro): redução por transitividade Aut(𝕆) para o caso geral.

### Estratos do setor O×O (degrau 2, dados corrigidos)

HIST44 = {2:6, 3:9, 7:45, 9:83} sobre 143 pares. Condição **necessária**
para rank44 = 7: **A ⊥ span{a₀,b₀}** (⟨A,e₁⟩=⟨A,e₂⟩=0): tabela 45/0
(P&r7=45, notP&r7=0); não suficiente (P&no7=55). Caracterização completa dos
estratos: aberto interno.

### O que resta para a Conjecture 6.8 inteira

Com rank(T) ≤ 11 estrutural, falta UMA peça: provar que o mapa bilinear
quociente ℝ⁵×ℝ⁵ → im(T) (dim ≤ 11) sempre tem zero real não-trivial.
Stiefel–Hopf não decide (sem obstrução genérica em q=11) — a prova precisa
usar a estrutura específica: candidatos (i) contagem de grau/Euler sobre a
família explícita, (ii) a simetria f ↔ f̃ do double hexagon agindo no kernel
de dim 14, (iii) redução Aut(𝕆) a uma família de 8 parâmetros + análise real.

## Rodada 5 (mesmo dia): o assalto à peça (b) — mapa fino, sem fechamento

Runners: `conj68_rank_structure.sio` (estendido), `conj68_hidden_identity_probe.sio`.

**Negativos honestos (hipóteses mortas com dados):**
- A estrutura complexa R: (p,q)↦(−q,p) (mult. por (0,1); o til do Lemma 3.6)
  **preserva O(x) em 143/143 pares** ✓, mas **não intertwina T**: as quatro
  identidades candidatas ([Ru,Rw]±[u,w]=0, [Ru,w]∓[u,Rw]=0) falham fora de 6
  pares degenerados. A rota "complexificar o quociente" morre aí.
- **Claim C é falsa como lei universal**: ⟨[u,w],(z±z′,0)⟩=0 vale em ~50 % dos
  pares (com frames verificados por aniquilação), o sinal − sozinho NUNCA vale,
  e os dois slots (z+z′,0)/(0,z+z′) sempre concordam — é UMA lei, valendo num
  sub-estrato próprio que **não coincide** com o estrato rank-7
  (cross-tab hold×rank44 = 38/47/7/51, todos os quadrantes vivos).
- Nenhum invariante linear simples (⟨A,A′⟩,⟨A,B′⟩,⟨B,A′⟩,⟨B,B′⟩,⟨z,z′⟩) prediz
  o hold (tabela PAIRDIAG).

**Positivos:**
- Forma fechada SEM ambiguidade de sinal do parceiro, derivada das equações
  puras (a,b)(c,d)=0 ⟺ ac=−db ∧ da=bc:  **d(c) = −(bc)a / n(a)** —
  conferida nos parceiros canônicos; mais prova-amigável que a forma via c·z
  (que carrega sinal por gerador).
- Par t=28 ((e₁,e₂) vs (e₃,e₄), rank-7): complemento setorial 9-dim explícito
  em rede: {x,x̃,x′,x̃′,(z+z′,0),(0,z+z′),(e₅,−e₆),(e₆,e₅)} + e₀ — os
  rank-7 têm colapso lattice-visível; os rank-9 genéricos têm as 2 direções
  extras FORA da rede pequena (por isso o scan não as vê).

**Estado da peça (b):** rank(T|₄ₓ₄) ≤ 9 uniforme (medido; ainda sem prova das
2 relações setoriais extras em forma geral). Próximo degrau técnico definido:
kernel exato ℚ do setor para pares racionais genéricos (via bigrat ou lifting
mod-p com reconstrução racional) → forma fechada das 2 direções em função do
par → aí sim o argumento topológico de zero real tem os dados que precisa.

## Rodada 6 (mesmo dia): A LEI DOS FANTASMAS — o teorema setorial completo

Runner novo: `examples/research/conj68_dictionary_kernel.sio` — pipeline
dicionário algébrico (11 vetores × 2 slots) → kernel mod-p → **reconstrução
racional** (Euclides estendido, |num|,den < √p) → **verificação exata sobre ℤ**.
Soundness total: o mod-p é bússola, o veredito é inteiro.

**Resultado:** as duas direções-fantasma do complemento setorial genérico são,
em TODOS os pares (25/25 no pipeline com VERIFY_FAIL=0; 143/143 no runner de
estrutura com o teste dedicado — nohold = 0 em ambos os estratos):

> **im(T|O×O) ⊥ (n(A′)·z + n(A)·z′, 0) e (0, n(A′)·z + n(A)·z′)**
> — a soma dos invariantes de Γ_O **normalizados pelas normas**: z/n(A) + z′/n(A′).

Corolários imediatos:
1. **rank(T|₄ₓ₄) ≤ 9 explicado**: complemento setorial = e₀ ⊕
   span{x, x̃, x′, x̃′, v₅, v₆} — as 6 direções puras, todas agora nomeadas.
2. **O estrato ~50 % da Rodada 5 resolvido**: a lei (z+z′) crua era o caso
   n(A) = n(A′). Nenhum mistério restante nos dados.
3. Estratos especiais (NSDIM 10) têm relações ADICIONAIS com caras
   (2z + ab′, 0), (−2z + bb′, 0) — estrutura extra dos rank-7, a caracterizar.

Lean atualizado (`SounioConj68RankBound.lean`, LEAN_OK): teorema
`ghost_orthogonal_canonical` — os fantasmas do par canônico (e₁+e₃, e₉+e₁₁)
⊥ aos 16 comutadores setoriais, kernel-verified.

**Estado da peça (b) após a rodada:** o quociente setorial é agora um mapa
bilinear explícito ℝ⁴×ℝ⁴ → ℝ⁹ com TODAS as relações lineares nomeadas em
forma fechada. O que falta é apenas (i) provar as duas identidades-fantasma
algebricamente (candidato: expandir ⟨[u,w], (n′z+nz′,0)⟩ com u=(Nc,·),
w=(N′d,·) via Moufang — agora um alvo concreto de ~1 página), e (ii) o
argumento de zero real para o mapa ℝ⁴×ℝ⁴ → ℝ⁹ específico — com a observação
de que 9 = dim genérico e Stiefel–Hopf não obstrui em (4,4,9), o zero forçado
virá das identidades, não da topologia genérica: provavelmente da estrutura
de composição n(ab)=n(a)n(b) que as normalizações z/n sugerem.

## Rodada 7 (mesmo dia): AS IDENTIDADES-FANTASMA PROVADAS — rank(T|₄ₓ₄) ≤ 9 é teorema

Prova em papel (registrada também em `SounioConj68RankBound.lean`, com a
instância normalizada n=1 vs n′=2 kernel-verified: `ghost_normalized_n2`).

Normalização: n(a)=n(b)=n(a′)=n(b′)=1; frames u=(c,cz), c⊥{1,a,b,z};
w=(g,gz′), g⊥{1,a′,b′,z′}; φ(u,w,v)=⟨[u,w],v⟩ cíclica (como no teorema
principal).

**Ghost 1** (Z=(z+z′,0)): φ(u,w,Z)=⟨[Z,u],w⟩.
[(z,0),u] = (−2cz,−2c) (z⊥c ⟹ zc=−cz; alternatividade (cz)z=−c);
[(z′,0),u] = ([z′,c], 2(cz)z′). Isometria-direita: ⟨(cz)z′,gz′⟩=⟨cz,g⟩ mata o
cruzado; ⟨c,gz′⟩=−⟨cz′,g⟩; soma = ⟨cz′+z′c, g⟩ = −2⟨c,z′⟩·Re(g) = **0 pela
pureza de g**. ∎

**Ghost 2** (Z₂=(0,z+z′)): [(0,z),u]=(−2c,2cz); [(0,z′),u]=([cz,z′],−2z′c)
(flexibilidade). ⟨cz, gz′+z′g⟩ = −2⟨g,z′⟩Re(cz)=0 (cz puro); reflexão
z′cz′ = c − 2⟨c,z′⟩z′ dá ⟨z′c,gz′⟩ = −⟨c,g⟩+2⟨c,z′⟩⟨z′,g⟩; soma =
−4⟨c,z′⟩⟨z′,g⟩ = **0 porque g ⊥ z′** (estrutura do frame). ∎

Escala não-normalizada ⟹ exatamente a lei medida (n(A′)z + n(A)z′).

**Consequência: o TEOREMA SETORIAL COMPLETO está provado.** As seis direções
puras do complemento — x, x′ (comutação), x̃, x̃′ (Lemma B + dupla-pureza),
v₅, v₆ (ciclicidade + pureza + g⊥z′) — todas com prova de poucas linhas, cada
peça verificada pelo kernel do Lean em instância canônica. rank(T|O×O) ≤ 9 e
rank(T) ≤ 11, ambos justos nos dados (145 pares).

Ingredientes usados (todos clássicos): multiplicação CD, flexibilidade,
isometria da multiplicação por unidade (composição), reflexão quaterniônica,
dupla-pureza de ZDs (Moreno), ciclicidade da forma trilinear.

**O que resta para a Conjecture 6.8, agora com precisão cirúrgica:** provar
que o mapa bilinear EXPLÍCITO T̄: ℝ⁴×ℝ⁴ → ℝ⁹ (o quociente pelas 6+1 relações
agora todas conhecidas) tem zero não-trivial para toda configuração (x,x′).
Um único lema de existência real, sobre um objeto totalmente explícito.

## Rodada 8 (mesmo dia): dois mecanismos de redução eliminados — o lema mora no mapa completo

Runner: `examples/research/conj68_sylvester_scan.sio` (scan fraco + caça forte
alternante/GN com lock0 + z-busca caso-(i) no leque comum).

**Medições (30 pares genéricos, caça forte):**
- `STRONG_FULL = 0.000000` em todos — witnesses no espaço 5×5 completo SEMPRE
  existem (consistente com 145/145 anteriores).
- `STRONG_SECTOR` com pisos 0.05–0.9 mesmo com 120 restarts pesados:
  **witnesses setoriais (O×O) NÃO existem para pares genéricos** — as
  componentes-x são essenciais (k₀ dos witnesses: 0.35–0.87). O ansatz de
  Sylvester setorial (c ⊥ z′, confirmado nos witnesses INTEIROS dos pares
  duros) não é o mecanismo universal.
- `CASE1_ZFAN` (witness ortogonal com invariante comum z no leque 3-dim
  {z ⊥ 1,a,b,a′,b′}, u(z) = s·x + L(z) afim): pisos 0.08–0.72 na maioria —
  **caso-(i) tampouco é universal**. |uw| dos witnesses reais varia de ~0 a 1:
  a variedade de witnesses é positivo-dimensional e mista (nem ortogonal, nem
  setorial, nem alinhada).

**Estado do lema final após a rodada:** a existência de zero vive
irredutivelmente no mapa completo T: ℝ⁵×ℝ⁵ → ℝ¹¹ (posto 11 provado, kernel 14,
deficit de Segre 3). Os dois caminhos de redução natural estão eliminados com
dados; o que sobra é o argumento global — de posse, agora, de TODAS as
relações lineares em forma fechada e da anatomia dos witnesses (k₀,m₀ ≠ 0
genéricos, mistura contínua entre os casos). Candidatos restantes: (α) grau
ímpar/classe de Euler de uma seção construída das 14 relações do kernel;
(β) argumento de deformação: a variedade de witnesses é conexa-dimensional
positiva nos dados — provar que não pode colapsar a vazio ao variar (x,x′)
no espaço conexo de configurações, via um invariante topológico da família.

## Rodada 9: ROTA α — A OBSTRUÇÃO DE EULER É NÃO-NULA

A peça que faltava para transformar o deficit-3 em empate exato: **três
relações PONTUAIS** da seção comutador, universais para u,w puros, cada uma
com prova de uma linha:

- **P1/P2**: ⟨[u,w],u⟩ = ⟨[u,w],w⟩ = 0 — φ(u,w,v) = ⟨[u,w],v⟩ é antissimétrica
  em (u,w) E cíclica (Rodada 4) ⟹ **totalmente alternada**.
- **P3**: ⟨[u,w], uw+wu⟩ = 0 — para u,w puros, wu = conj(uw) ⟹
  n(uw) = n(wu), e ⟨[u,w],u∘w⟩ = n(uw) − n(wu).

Verificação: 2000 amostras inteiras exatas (P1=P2=P3=0 violações,
`conj68_pointwise_relations_probe.sio`) + polarização multilinear completa
sobre a base pura no kernel do Lean (`p12_polarized` 15³ instâncias;
`p3_bilinear` wu=conj(uw) em 15²).

**Consequência topológica:** com as 4 relações lineares provadas, a seção
s(u,w) = T(u,w) vive num fibrado de **posto 8 = dim(ℝP⁴×ℝP⁴)**:
E ≅ ℝ¹¹ ⊖ γ₁ ⊖ γ₂ ⊖ γ₁γ₂, seção bilinear ⟹ twist γ₁γ₂. A obstrução mod 2:

    e = Σᵢ wᵢ(E)·(α+β)^{8−i},  w(E) = [(1+α)(1+β)(1+α+β)]⁻¹
    em ℤ/2[α,β]/(α⁵,β⁵):  **coeficiente de α⁴β⁴ = 1**

(computado em `conj68_euler_class.sio`, conferido à mão — o termo W₈ = α⁴β⁴
contribui sozinho, os demais cancelam em pares — e kernel-verified em
`SounioConj68EulerLeg.lean::euler_obstruction_nonzero`).

**⟹ uma seção sem zeros é impossível ⟹ toda configuração admite witness ⟹
d(x,x′) ≤ 3 para todo par ⟹ CONJECTURE 6.8 — módulo a lapidação dos loci de
degeneração.**

### O que a lapidação precisa fechar (honestidade cirúrgica)

O argumento do fibrado usa E de posto constante 8; as linhas perpendiculares
{proj_{W₁₁}u, proj w, proj Im(u∘w)} podem colapsar em loci especiais
(e.g. u = x, onde proj u = 0; Im(u∘w) ∈ span{proj u, proj w}). O passo
restante é topologia diferencial padrão mas real: mostrar que uma seção sem
zeros induziria uma seção sem zeros do fibrado honesto fora dos loci +
argumento de extensão/posição geral (ou blow-up), OU reformular via
obstrução equivariante que não precise do posto constante. Só isso separa
"esqueleto completo" de "teorema".

Lição de infra: o pod limita o número de `native_decide` por unidade de
tradução (thread cap) — daí o split em dois arquivos Lean; e `Poly` precisa
ser `abbrev` (não `def`) para herdar `BEq` de List — o erro manifestava como
"failed to create thread" espúrio.

## Rodada 10: RETRATAÇÃO PARCIAL DA RODADA 9 — P3 é vácua; o argumento precisa de uma 3ª relação genuína

**Erro encontrado na lapidação (antes de qualquer publicação):** para u,w
puros, wu = conj(uw) implica u∘w = uw + wu = 2Re(uw)·e₀ — SEMPRE REAL.
Logo Im(u∘w) ≡ 0 e "P3" (⟨T, u∘w⟩ = 0) é trivialmente verdadeira mas
VÁCUA como restrição: o probe passou porque T é puro e u∘w é real, não
porque exista uma terceira linha perpendicular. Só há DUAS relações
pontuais genuínas (P1, P2: φ alternada). Consequência: o fibrado é de posto
11 − 2 = 9 > 8 = dim base, a obstrução primária vive em H⁹ = 0, e **o
argumento de Euler da Rodada 9 colapsa como enunciado**. A aritmética
α⁴β⁴ = 1 (souc + Lean) permanece correta — para um fibrado cuja existência
dependia da relação vácua.

Estado corrigido da rota α: falta UMA relação pontual genuína (com o twist
certo) para o empate posto = dimensão. A caça: dicionário de formas
cúbicas/bilineares ⟨[u,w], V(u,w)⟩ ≡ 0 com V ∈ {[u,w,w], (uw)w, w(wu),
Jacobiano de Malcev, ...} — verificação exata sobre pares puros aleatórios
(conj68_pointwise_relations_probe.sio, estendido). Se existir → recomputar a
classe com o twist de V; se não → a rota precisa de topologia deficit-1
(grau relativo/equivariante) ou de outra ideia.

Lição (a de sempre, e ainda assim): entusiasmo não substitui o cheque de
vacuidade — testar uma identidade sem testar que ela RESTRINGE é meio teste.

## Rodada 11: A ROTA α RENASCE MAIS FORTE — relações de ASSOCIADOR + obstrução primária em H⁷

A caça pós-retratação achou as relações verdadeiras, e desta vez com o
dever de casa completo ANTES do anúncio:

**As quatro relações pontuais genuínas** (T = [u,w], u,w puros):
1. T ⊥ u, T ⊥ w — φ totalmente alternada (twists γ₁, γ₂).
2. **T ⊥ [u,w,w] e T ⊥ [w,u,u]** — os ASSOCIADORES (twists γ₁, γ₂).
   *Prova* (não só amostras): ⟨[u,w],[u,w,w]⟩ = ⟨[u,w],(uw)w⟩ (P1 + w² = −n(w));
   o termo ⟨uw,(uw)w⟩ morre por auto-negação via ⟨a,bc⟩ = ⟨ac̄,b⟩; o termo
   ⟨wu,(uw)w⟩, com v = uw e wu = conj v, reduz a ⟨v²,w⟩ = 2Re(v)⟨uw,w⟩ e
   ⟨uw,w⟩ = n(w)⟨u,1⟩ = 0. ∎ Simétrico para [w,u,u].
   *Não-vacuidade* (a lição da R10): rank{u,w,[u,w,w],[w,u,u]} = 4 em
   2000/2000 amostras inteiras exatas.

**Consequência topológica corrigida:** E₇ ≅ ℝ¹¹ ⊖ 2γ₁ ⊖ 2γ₂, posto 7 < 8;
para posto < dim, a obstrução PRIMÁRIA a uma seção sem zeros vive em H⁷ e
sua não-nulidade sozinha força zeros:

    w₇(E₇ ⊗ γ₁γ₂) = Σᵢ wᵢ(E₇)(α+β)^{7−i},
    w(E₇) = (1+α²+α⁴)(1+β²+β⁴)
    ⟹  w₇ = α⁴β³ + α³β⁴  ≠ 0

— computado à mão, em souc (`conj68_euler_class.sio`, INVERSE7_CHECK OK,
ambos os coeficientes = 1) e no kernel do Lean
(`euler7_primary_obstruction_nonzero`, LEAN_OK).

**⟹ toda configuração admite witness ⟹ Conjecture 6.8 — módulo a lapidação,
agora com a lista fechada:**
(i) independência genérica das QUATRO projeções em W₁₁ (medida em ℝ¹⁶;
falta o passo proj_{W₁₁} — codimensão favorável, medir);
(ii) os loci de degeneração ([u]=[x], [w]=[x′], associadores caindo no span);
o argumento padrão: seção sem zeros ⟹ seção sem zeros do fibrado honesto
fora dos loci + extensão/posição geral ou blow-up.
Um bônus da versão posto-7: mesmo onde UMA relação degenera, sobram 3 e o
posto local ≤ 8 = dim — a folga do deficit-(-1) torna a lapidação mais
robusta que a versão original.

## Rodada 12: O LOCUS MEDIDO + MANUSCRITO

`conj68_loci_probe.sio`: 40 configurações × ~60 amostras (u,w) cada:

```
RANK9_HIST 8 N 85 · RANK9_HIST 9 N 2312
DROPS: at_parallel 85  other 0
TARGETED_DEGENERATE CONFIRMED
```

**Z = Z₁ ∪ Z₂ exatamente** ({[u]=[x]} ∪ {[w]=[x′]}, codim 4 cada), todo drop
é de exatamente 1 (posto local 8 = dim, dentro da folga do posto-7), e s é
canonicamente não-nula perto de Z quando d ≥ 3 (s(x,w) = 0 daria witness
trivial d ≤ 2). A lista da lapidação virou UM item: o passo de excisão
relativa L(iii) (codim 4, sequência exata, classe relativa ↦ w₇).

**Manuscrito iniciado**: `conj68_manuscript_draft.md` — estrutura de paper
com status por claim ([T]/[K]/[M]/[L]), todas as provas das rodadas 4-11,
o argumento de obstrução, o Technical Lemma L com a evidência, o apêndice de
artefatos e as retratações registradas como parte do método.

## Rodada 13: o locus não se dissolve — ele CARREGA a prova

**Seis relações novas descobertas** (0 violações em 2397 amostras de
configuração, `conj68_loci_probe.sio` DEEPCAND): ⟨T,[u,x,x]⟩ = ⟨T,[u,x̃,x̃]⟩ =
⟨T,[w,x′,x′]⟩ = ⟨T,[w,x̃′,x̃′]⟩ = ⟨T,[u,w,x′]⟩ = ⟨T,[u,w,x̃′]⟩ = 0 (e as
espelhadas [w,x,x], [u,x′,x′], [u,w,x], [u,w,x̃] NÃO são relações — assimetria
interessante). Provas pendentes; violação exata zero.

**O swap falhou** (SWAP_HIST idêntico: 85 drops nos paralelos): a troca
σ₁ → P([u,x̃,x̃]) não remove o locus porque **[x,x̃,x̃] = 0** — computado na
Table 2: (x,x̃) geram com f₀,f̃₀ uma subálgebra associativa do hexágono; todo
candidato u-linear construído dos vetores do próprio hexágono morre em u=x.

**A descoberta topológica da rodada (feita à mão):** na sequência
H⁷(B, B∖Z) → H⁷(B) → H⁷(B∖Z), a imagem do primeiro mapa (classes de Thom,
normal trivial) é α⁴·H³(ℝP⁴) ⊕ β⁴·H³(ℝP⁴) = **span{α⁴β³, α³β⁴} — exatamente
w₇**. Ou seja: w₇ morre em H⁷(B∖Z); **a obstrução é inteiramente suportada
no locus**. Não é patologia — é o mecanismo: os zeros forçados vivem perto de
Z, onde a seção degenera CANONICAMENTE ao pullback
(T(x+εq, w) = T(x,w) + ε·T(q,w), bilinearidade — q-independente em ordem
dominante).

**O endgame reformulado com precisão:** M = B ∖ N(Z), bordo
∂M ≅ ℝP⁴×S³ ⊔ S³×ℝP⁴; a trivialização de bordo de s é homotópica ao pullback
de T(x,·) pelo fator w (4-complexo ⟹ as classes de diferença em grau ≥ 5
morrem por dimensão). A conjectura segue de:

> **e_rel(E₇⊗γ₁γ₂; s₀-pullback) ≠ 0 em H⁷(M, ∂M)** — equivalentemente, a
> contagem mod 2 dos zeros de QUALQUER seção que estenda o dado canônico de
> bordo é 1. Computável com uma seção-modelo explícita cujos zeros se contam
> à mão.

O plano da próxima sessão: construir a seção-modelo (candidata: interpolação
linear entre T e uma seção algébrica com zeros conhecidos nos pares de base,
onde D3EXACT já deu os witnesses inteiros) e computar e_rel.

## Rodada 14: a curva medida, o invariante real morto, e a rota COMPLEXA

`conj68_witness_count.sio` (novo): caça exaustiva + clustering + Jacobiano +
traçador de curva por continuação (tangente = autovetor mínimo de JᵀJ +
penalidade nas escalas; passo + re-polimento GN) + censo de componentes.

**Leis novas medidas:**
1. **A variedade de witnesses é SEMPRE uma curva**: Jacobiano de posto
   exatamente 7 (= posto do fibrado — transversalidade genérica!) em 320/320
   witnesses de 8 configurações genéricas; nulidade 3 = 2 escalas + 1 tangente.
2. **Γ ∩ Z = ∅ rigorosamente** (u=x ⟹ T(x,w)=0 ⟹ w ∈ ImC(x)∩V = 0 p/ d≥3 —
   uma linha de álgebra, agora teorema).
3. **Censo de componentes da CONF#1**: 3 componentes fechadas, classes
   (−,−), (+,+), (−,−) no double cover — **TOTAL [Γ] = 0 em H₁**. A predição
   PD [Γ] = a+b FALSIFICADA (como devia: E₇ não é fibrado sobre B todo; e
   H₁(B∖Z) → H₁(B) é iso, então o cobordismo em B∖Z não obstrui nada).
   Terceiro invariante topológico real a morrer no dia.

**A ROTA COMPLEXA (o programa correto, nascido das cinzas):**
- Sobre ℂ, as 4 relações pontuais são identidades algébricas ⟹ a variedade
  de witnesses complexa Γ_ℂ ⊂ ℂP⁴×ℂP⁴ tem dim_ℂ ≥ 1 e é **não-vazia por
  teoria de dimensão** — a existência complexa é de graça.
- Γ_ℂ é conjugação-invariante (equações reais). **Se o bidegree (d_a, d_b)
  de Γ_ℂ for ímpar em um slot, cortar com um hiperplano REAL genérico dá um
  0-ciclo conjugação-invariante ímpar ⟹ ponto fixo real ⟹ WITNESS REAL.**
  Argumento clássico, sem topologia real sutil.
- A parte ingênua do bidegree: c₇((E₇)_ℂ(1,1)) — nossa w₇ = α⁴β³+α³β⁴ é a
  sombra mod 2 ⟹ **naive degree ímpar nos dois slots** ✓ já computado.
- Falta UMA peça: a paridade da **correção de excesso** (Fulton) suportada
  em Z_ℂ = duas cópias de ℂP⁴ com normal trivial — computação finita de
  interseção excedente. Se a correção for PAR ⟹ bidegree ímpar ⟹ conjectura.

O dia terminou com o problema transposto do mundo real-topológico (onde três
invariantes morreram) para o algébrico-complexo (onde a não-vacuidade é grátis
e falta uma paridade de classe de Segre). É a formulação mais promissora até
agora — e a mais computável.

## Rodada 15: TODAS as paridades se anulam — e a curva revela sua anatomia: RETAS

**Fechamento do círculo das obstruções (por dados próprios):** a paridade dos
cruzamentos de cada componente com um hiperplano real = ε da componente
(validado: (1,1)/(2,2)/(1,1) ↔ (−,−)/(+,+)/(−,−)); d_a mod 2 de Γ_ℂ =
#cruzamentos reais mod 2 = componente-a de [Γ] = **0**. Logo o bidegree
complexo é PAR e **a rota do grau ímpar (R14) morre pelos nossos próprios
dados de censo** — a quarta e última esperança de obstrução de paridade.
Conclusão estrutural do dia: a existência de witnesses NÃO é forçada por
nenhum invariante topológico/enumerativo de paridade; ela é
ALGÉBRICO-CONSTRUTIVA.

**E a anatomia da curva aponta a construção:**
- Bidegree real da CONF#1: **(4,4)** = componentes de grau (1,1), (2,2), (1,1).
- Nuvens de pontos: **KRANK = MRANK = 2 nas componentes ímpares** — são
  RETAS PROJETIVAS nos dois fatores: **(1,1)-curvas = pênceis lineares de
  witnesses**: u(t) = u₀+tu₁, w(t) = w₀+tw₁ com T(u₀,w₀) = T(u₁,w₁) = 0 e
  **T(u₀,w₁) + T(u₁,w₀) = 0**. A componente par é uma cônica (rank 3).
- Estrutura da CONF#1: **duas retas de witnesses + uma cônica**.

Teste exato negativo (registrado): os dois witnesses INTEIROS do par duro
(e₁,e₂)vs(e₂,e₃) NÃO formam pêncil entre si (cross ≠ 0, exato) — o pêncil de
uma config específica deve ser extraído da reta traçada (fit racional), não
de pares arbitrários de witnesses.

**O PROGRAMA FINAL (algébrico, nascido de 4 obstruções mortas):**
> Provar: toda configuração admite um PÊNCIL de witnesses — um sistema
> (u₀,u₁,w₀,w₁) com as três equações acima. Alvo intermediário: extrair o
> pêncil exato da CONF#1 (rationalizar a reta traçada), ler sua forma
> octoniônica, e generalizar — o caminho de RQ→lei→prova que já funcionou
> três vezes hoje (ghost law, associadores, forma fechada do parceiro).

## Rodada 16: O GOLPE — o primeiro pêncil exato de witnesses

Pipeline do golpe (`conj68_witness_count.sio` + `conj68_pencil_extract.sio`):

1. **Pontos de 9 dígitos** em duas posições da reta traçada (polimento GN a 1e-15).
2. **Rationalização à mão** (resíduos 0–2 × 10⁻⁹): as retas têm equações
   INTEIRAS nas coordenadas de frame:
   - U₂ = span{(3,1,1,−5,0), (0,0,1,−1,1)} (frame de x₀)
   - W₂ = span{(3,−1,1,0,−4), (0,−1,0,1,1)} (frame de x₂ = (−2e₂, e₃+e₄−e₆−e₇))
3. **Scan de coeficientes pequenos**: 8 witnesses inteiros no produto das retas,
   todos obedecendo UMA lei de Möbius δ = (β+6)/(β+2).
4. **Base canônica**: ŵ₀ = w₀+3w₁, ŵ₁ = w₀+w₁; w(β) = 2ŵ₀ + βŵ₁; as três
   identidades do pêncil verificadas EXATAS em i64:
   [u₀,ŵ₀] = 0, [u₁,ŵ₁] = 0, [u₀,ŵ₁] + 2[u₁,ŵ₀] = 0
   ⟹ **T(u₀+βu₁, 2ŵ₀+βŵ₁) ≡ 0 ∀β — família a 1 parâmetro de witnesses em
   forma fechada.** (Erro de normalização no primeiro teste — peso 2 no cross
   — pego e corrigido; endpoints já eram exatos.)

Vetores inteiros do pêncil (sedenion coords):
U0 = 3e₁+e₄+e₅−5e₆+3e₁₀+5e₁₃+e₁₄−e₁₅; U1 = e₅−e₆+e₇+e₁₂+e₁₃+e₁₄;
WH0 = −6e₂+12e₃−8e₄+2e₅+6e₆−2e₇+6e₉+2e₁₂−10e₁₃+2e₁₄−12e₁₅;
WH1 = −6e₂−4e₄+2e₅+2e₆−6e₇+6e₉+2e₁₂−2e₁₃−6e₁₄−4e₁₅.

**Programa pós-golpe:** extrair o segundo pêncil (COMP#3) e o de outras
configurações; ler a forma octoniônica dos (u₀,u₁,ŵ₀,ŵ₁) em termos do par
(dicionário, como na ghost law) → construção geral → prova de existência do
pêncil para toda configuração → **Conjecture 6.8**. O objeto agora é
algébrico, exato, e generalizável — o formato que rendeu teorema três vezes
neste log.

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

## Rodada 17: O PÊNCIL É ARTEFATO DE SIMETRIA — genericidade mata a reta

O revisor forte apontou o buraco antes de eu ajustar qualquer dicionário: toda a
decomposição em retas (R14 censo, R15 anatomia, R16 pêncil) foi medida numa
ÚNICA config — CONF#1 = `(e₁,e₂)`, a MAIS simétrica da família (a=e₁, b=e₂,
z=ab=e₃ todos alinhados à base Cayley–Dickson ⟹ maior estabilizador). Reducibilidade
de uma (4,4) NÃO é genérica; uma (4,4) irredutível é. Retas sinalizam simetria extra
do ponto especial. Alvo do programa ("toda config admite pêncil") tinha suporte n=1.

**Correções do revisor incorporadas (2 erros meus):**
- O "peso 2" no cross e o "6" de Möbius são GAUGE, não sinal: pondo v₀=2ŵ₀ as três
  identidades viram a forma simétrica [u₀,v₀]=[u₁,ŵ₁]=0, [u₀,ŵ₁]+[u₁,v₀]=0; toda
  (1,1)-curva é grafo de Möbius por definição (PGL₂ automático). Fork "gerador do
  pêncil" morto.
- O conteúdo invariante de um pêncil não é o vetor u₀ (definido a menos de GL₂ na
  reta) mas os 2-planos U⊂Im C(x), W⊂Im C(x′) e a iso ψ:U→W. dim(U∩O(x))=1 NÃO é
  informativo (2-plano em ℝ⁵ ∩ hiperplano = 1-dim genericamente).

**Experimento (witness_count.sio estendido para bateria; 1 config/processo por causa
do vazamento de handles 2²² nas funções quentes do trace):**
- CONTROLE POSITIVO — a config EXATA das retas do R16 (x1=(e₁,e₂),
  x2=(−2e₂, e₃+e₄−e₆−e₇)): traçador reproduz a anatomia R15 EXATA — COMP#1
  KRANK=MRANK=2 CROSS(1,1) [RETA], COMP#2 KRANK=MRANK=3 CROSS(2,2) [CÔNICA],
  COMP#3 KRANK=MRANK=2 [RETA] = duas retas + uma cônica. Máquina VALIDADA: sabe
  achar reta quando existe.
- 8 CONFIGS GENÉRICAS (ambos ZD amostrados), ~24 componentes traçadas:
  **0 componentes (1,1) genuínas (KRANK=2 MRANK=2); 20 cônicas (KRANK=MRANK=3).**
  Única exceção — seed 12345678 (a=2e₃−2e₄, só 13 witnesses) — deu componentes
  DEGENERADAS (2,1)/(1,2): rank-2 num fator, rank-1 (ponto FIXO) no outro = "reta
  de u com w fixo", não pêncil (1,1). CROSS é probe ruidoso (KRANK é o detector
  confiável: u(t)=u₀+tu₁ ⟹ rank afim exatamente 2; rank 3 ⟹ cônica).

**VEREDITO:** o pêncil linear do R16 é a DEGENERAÇÃO cônica→par-de-retas no ponto de
simetria máxima. Genericamente as componentes ímpares do curve de witnesses são
CÔNICAS, não retas. O programa "provar existência de pêncil" está mal-fundado como
enunciado — mais uma rota de forma-fechada-linear morta pelos próprios dados.

**O que sobrevive:** witnesses existem em TODA config testada (40,13,37,40,… sempre
>0) — a conjectura continua de pé empiricamente. Mas o objeto uniforme não é uma reta:
ou é uma CÔNICA de witnesses (o pêncil sendo seu limite degenerado), ou — seguindo a
necessidade MÍNIMA da conjectura (UM witness, não uma família) — um único witness
algébrico de grau 2 sobre o corpo da configuração (consistente com os k da classe 3:
−0.7794,−0.1183,0.3879,0.4776, candidatos a PSLQ/quadráticos).

**Próximo degrau (a decidir):** (a) caçar a CÔNICA uniforme — relação quadrática
(u(t),w(t)) grau 2 — e ler sua forma octoniônica; ou (b) abandonar a família e
atacar UM witness via forma fechada grau-2 + PSLQ dos k. A obstrução topológica
(w₇=α⁴β³+α³β⁴, R11/Teorema 6.1) continua sendo a única prova de EXISTÊNCIA que não
depende de construção — o manuscrito §6 já a carrega; a Lemma L(iii) (excisão
relativa codim-4) é o gargalo formal restante, independente do colapso do pêncil.

### R17b — robustez: o cap de 40 não fabricou o negativo (checado a pedido do revisor)

Furo metodológico apontado: 6/8 seeds saturavam `WITNESSES 40` (cap `nwit<40`) e o
census só traça a partir dos witnesses retidos — uma reta cujos witnesses caíssem fora
seria invisível. Correção: cap 40→300 (`wit[3000]`, `visited[300]`), componentes 6→8,
e LEITURA do campo MAXED (antes ignorado).

- Witnesses são CONTÍNUOS (curvas): configs saturam mesmo em 300 (`WITNESSES 300`,
  MAXED>0). Contagem é sem sentido — os 13/24/74/139/300 são artefatos de cluster.
- TESTE DE SENSIBILIDADE (o decisivo): controle positivo (config das retas do R16) com
  os MESMOS settings de cap alto ainda pega as duas retas — COMP KRANK=MRANK=2 CROSS(1,1),
  MAXED 19. Logo o método É sensível a retas mesmo saturando; "só cônicas" nas genéricas
  não é truncamento.
- Sweep com caps altos: 20 cônicas (KRANK=3 MRANK=3), 0 retas (1,1) genuínas
  (KRANK=2 MRANK=2). As únicas rank-2 (5 componentes, todas na config especial
  a=2e₃−2e₄, seed 12345678, MAXED 0 com 24 witnesses honestos) são DEGENERADAS (2,1)/(1,2)
  = "reta de u com w FIXO", não pêncil (1,1). **R17 sobrevive; agora robusto.**

### R17c — a topologia dura: w₇ MORRE em B∖Z (a obstrução primária não prova sozinha)

Fazendo a excisão na ponta do lápis (confirmado pelo revisor): H⁷(B;ℤ/2) tem só 2
monômios (α⁴β³, α³β⁴). Thom: Z_i codim 4, Z₁∩Z₂ = 1 ponto (codim 8, invisível a H⁷) ⟹
im(H⁷(B,B∖Z)→H⁷(B)) = span{α⁴β³,α³β⁴} = TODO H⁷(B). Por exatidão i*: H⁷(B)→H⁷(B∖Z) = 0,
logo **w₇|_{B∖Z} = 0**. Mas E₇ só existe sobre B∖Z (salta a rank 8 em Z). A obstrução
primária do fibrado honesto sobre o aberto B∖Z é ZERO — como tem de ser (obstruções de
grau-topo sobre variedades abertas penteiam-se para os fins). ⟹ **Teorema 6.1 (w₇≠0) NÃO
prova sozinho a conjectura.** w₇∈im(j*) é check de consistência necessário, não derivação.

A prova, SE existir, é a obstrução RELATIVA o_rel(s) ∈ H⁷(B,∂N(Z)) ≅ H³(Z₁)⊕H³(Z₂),
usando L(ii) (s≠0 em Z = dado de bordo). o_rel = w₇ ⟺ grau local de s numa fatia normal
4-dim a cada Z_i ÍMPAR. o_rel PODE ser 0 (graus pares) — aí a seção nowhere-zero existiria
e o argumento não fecha. Invariante = classe H³(Z₁)=ℤ/2⟨[ℝP³]⟩, NÃO contagem. Manuscrito
§6/Cor 6.2 reescritos honestamente condicionais.

**Endgame único restante:** computar deg_local(s; Z₁) mod 2 (winding de T num S³ ligando
o locus paralelo u∥x). Rotas paralelas vivas: (b) positividade do discriminante da
quadrática de witness (coefs explícitos em n(a),n(a′),⟨z,z′⟩, ghost-law) — problema de
álgebra nas identidades que já temos, subiu de prioridade agora que o primário morreu no
aberto.

## Rodada 18: A ROTA TOPOLÓGICA ESTÁ FECHADA (negativo completo, não "em preparação")

Consultei Grok (xai) + revisor forte no ponto duro. Grok errou ($\pi_3(S^7)\cong\mathbb{Z}/2$
— FALSO, $\pi_3(S^7)=0$), e o erro dele iluminou a verdade. Conclusão, no grupo CERTO:

**O invariante de forçamento é a classe de Euler RELATIVA**, não a absoluta sobre o aberto:
  e_rel(E₇, s|_∂M) ∈ H⁷(M,∂M) ≅ H₁(M;ℤ/2),  M = B∖N̊(Z).
Remover Z (codim 4) de B (dim 8) não muda π₁ ⟹ H₁(M;ℤ/2) ≅ H₁(B;ℤ/2) = (ℤ/2)², e o
census R14 mediu EXATAMENTE essa classe: **[Γ] = 0**. Ou seja, o invariante que forçaria
existência é medido-NULO. (Meu argumento anterior "w₇|_{B∖Z}=0 ⟹ sem forçamento" era
non-sequitur: grupo errado. A restrição H⁷(B)→H⁷(B∖Z) não diz nada sobre H⁷(M,∂M).)

**Kill dimensional independente:** estender seção nowhere-zero por um locus de codim k
com esfera-fibra S^{r-1} tem primeira obstrução em π_{k-1}(S^{r-1}), não-nula só se k≥r.
Aqui k=4 (codim de Z_i), r=7 (rank de E₇): **4 < 7 ⟹ NENHUMA obstrução local em Z, para
qualquer seção.** A esfera de linking S³ mapeia em S⁶/S⁷ e π₃(S⁶)=π₃(S⁷)=0.

**A RAIZ DO ERRO QUE MANTEVE §6 VIVO POR 4 RODADAS (R11→R17):** R13 concluiu
"im(H⁷(B,B∖Z)→H⁷(B)) = span{α⁴β³,α³β⁴} = w₇ ⟹ obstrução suportada em Z". Isso é
COINCIDÊNCIA DE COHOMOLOGIA PEQUENA, não localização geométrica: H⁷(B;ℤ/2) TEM só 2
monômios (α⁴β³, α³β⁴), e os dois pushforwards de Thom de [Z₁],[Z₂] são exatamente essa
base ⟹ im(j*) = TODO H⁷(B) automaticamente. TODA classe de grau 7 "aparece suportada em
Z". R13 carregou ZERO informação sobre s. Lição mais transferível do dia: num H^n de
dimensão baixa, "está na imagem da excisão" é vácuo, não evidência.

**Status honesto (duas direções de escopo):**
(i) É "ESTA rota, como construída, está morta" — NÃO "não existe prova topológica".
(ii) e_rel = [Γ] pressupõe que o census achou TODAS as componentes; R17b mostrou que ele
é handle-limitado (~5/config, nem sempre fecha). Medido 0, não provado 0. Mas isso não
resgata o forçamento: componentes não-achadas teriam de carregar classe ÍMPAR enquanto as
achadas já cancelam — sem razão para isso.

**Corolário 6.2 RETRATADO** (era condicional em o_rel≠0; o_rel medido = 0). §6 reescrito
como negativo completo: a obstrução relativa foi computada e VANISHES; rota fechada.
Uma rota fechada é contribuição real (mata o approach de característica para este alvo).

**PIVÔ TOTAL PARA A ROTA CONSTRUTIVA (b):** a existência do witness NÃO é topológica —
é algébrica. Programa concreto (primeiro deliverable): racionalizar uma CÔNICA das configs
genéricas do R17 (como o R16 fez com a reta), ler a relação grau-2 em coords de frame,
extrair a forma quadrática do witness, expressar o discriminante em n(a),n(a′),⟨z,z′⟩ e as
grandezas da ghost-law, e provar positividade contra as identidades que já temos. Cônica é
o que os dados genéricos REALMENTE mostram (≠ pêncil).

## Rodada 19: rota construtiva (b) — confirmação EXATA da estrutura + fronteira de ferramenta

Correção do usuário incorporada: float/numpy não é a prova nem a ferramenta de
descoberta certa aqui ("se numpy resolvesse já teriam resolvido"). Pivô para o
idioma EXATO do projeto (mod-p → reconstrução racional → i64/Lean).

**Setup exato:** frames inteiros exatos emitidos pela máquina Sounio validada
(FRAME_U/FRAME_W, novo no census); tensor comutador C[a][i][j]=[f^U_i,f^W_j]_a
construído em ℤ exato (cd_sigma Cayley-Dickson). Witness: Σ C[a][i][j]k_i m_j = 0.
Dados: docs/research/data/conj68_frames_13370001.txt (config a=2e₂, distinto b).

**Confirmado EXATAMENTE (mod-p, p=32003, SEM float):**
- M(k) (16×5, linear em k) tem rank genérico 5 (colunas [u,g_j] independentes);
  200/200 random k. Simétrico p/ N(w).
- 0/20000 random k com rank<5 ⟹ u-locus é codim ~3 em ℙ⁴ = CURVA.
- Codim 3 = exatamente (7-4)(5-4) do determinantal de uma matriz de rank-EFETIVO
  7 ⟹ **confirma exatamente a lei rank-7 dos §§2-5** (não via SVD float, via mod-p).
- degrevlex GB do ideal bilinear (chart afim k0=m0=1) fecha RÁPIDO (curva tratável
  mod p); lex/eliminação-de-m ESTOURA (>2min) — é o núcleo computacional pesado.

**Critério de existência, exato:** witness real ⟺ alguma componente cônica tem
ponto real ⟺ sua forma quadrática 3×3 (racional) é INDEFINIDA. Cônicas genéricas
têm pontos irracionais quadráticos ⟹ SEM parametrização racional estilo R16 (por
isso o R16 deu inteiros só na config super-simétrica). O objeto exato é a EQUAÇÃO
da cônica; extração precisa de FGLM (após slice 0-dim) ou CAS (Singular/msolve/
Macaulay2) — NENHUM instalado aqui. sympy lex estoura; degrevlex fecha mas
change-of-order limpo não cabe em script de 2min.

**Fronteira honesta:** este é o núcleo duro/aberto (por isso a conjectura está em
disputa). Ferramenta certa p/ o próximo golpe: (a) pipeline mod-p+CRT do próprio
Sounio (o que fez a ghost law) sobre o ideal determinantal, OU (b) instalar
Singular/msolve e usar degrevlex+FGLM. O ALVO exato: a forma quadrática Q da
cônica em coords de frame, e a prova de que disc/assinatura é indefinida uniforme
para d≥3 — expressa em n(a),n(a′),⟨z,z′⟩ e ghost-law. Ainda o núcleo aberto.

## Rodada 20: O GRAU ÍMPAR — possível PROVA via Singular (rota exata do usuário)

Instalei Singular+msolve (sudo apt). Tensor comutador verificado EXATO (2e-8 num
witness real do census; casa com [u,w] direto). O u-locus de witness =
{[u]∈ℙ⁴ : rank M(u)≤4} = ideal dos MENORES 5×5 da matriz comutador M(u) (16×5→15
linhas não-nulas, entradas lineares em u). primdecGTZ sobre ℚ (char 0, exato).

**DESCOBERTA (8/8 configs):** o u-locus é uma curva REDUZIDA de **degree 7 (ÍMPAR)**,
sempre com uma componente de grau ÍMPAR:
- Genéricas (6 seeds: 13370001,99998887,60651729,50000917,22222223,33333331):
  componentes (deg) = 2,2,**3** = 2 cônicas + 1 **CÚBICA**. Total 7.
- "Planas" (12345678 [a=2e₃−2e₄], 11111119): 1,2,2,2 = 1 **RETA** + 3 cônicas. Total 7.
Mais: sempre 1 par de pontos COMPLEXOS conjugados (dim1, k3²+2k3k4+2k4², disc<0) +
origem — irrelevantes (sem pontos reais / triviais).

**MECANISMO DE EXISTÊNCIA (candidato a prova):**
1. Witness real ⟺ ponto real do determinantal D={rank M(u)≤4}, curva de degree 7.
2. Um hiperplano real genérico corta D em 7 pontos; conjugação complexa os pareia
   ⟹ #real ímpar ≥ 1 ⟹ **D tem ponto real** (u real).
3. u real ⟹ M(u) real de rank≤4 ⟹ ker≠0 sobre ℝ ⟹ **w real** ⟹ WITNESS REAL.
4. ⟹ diam Γ_C^Z(𝕊)=3. QED (módulo rigor abaixo).

**RESSUSCITA "c₇ ímpar" do R14 no framework CORRETO:** a paridade que força
existência NÃO é a classe de Euler (c₇, morta em R18 porque vivia em B∖Z), mas o
**GRAU da curva de witness** (um número c₃/Thom-Porteous de interseção, config-
independente por ser enumerativo/topológico). Degree 7 ímpar ⟹ ponto real. A
intuição de paridade estava certa; o objeto era o grau, não a obstrução.

**Rigor restante (a fechar):**
(a) degree = 7 UNIFORME: ou (i) Thom-Porteous/c₃ do determinantal de rank-efetivo
    7 (7×5) — cálculo de Chern, config-independente; ou (ii) constância do grau em
    família plana conexa de pares ZD (medido 7 em 8 pontos). Nota: rank≤4 de 15×5
    genérico teria codim 11; a codim-3 observada vem da estrutura rank-7 dos §§2-5,
    então o Porteous é no fibrado EFETIVO E₇, não no 15×5 cru.
(b) D reduzida (para o argumento de seção-hiperplana): primdecGTZ deu componentes
    PRIMAS (reduzidas) em todas testadas; falta o geral.
(c) d≥3: a existência de witness vale p/ todo par; d≥3 é só o enunciado do que o
    witness significa (caminho de comprimento 3).

Dados exatos: docs/research/data/conj68_ulocus_decomp_13370001.txt (equações das
componentes, incl. a cúbica ternária 16k2³+...), conj68_ulocus_degrees_8configs.txt.
Ferramenta: Singular (minor(M,5) + primdecGTZ). Frames exatos via census Sounio.

### R20b — rigor do revisor forte incorporado (3 checks; o resultado sobrevive)

1. **REDUÇÃO (o check que podia matar):** eu tinha impresso pd[i][2] (o RADICAL/primo),
   não a componente primária pd[i][1] — logo não tinha prova de que E é reduzido.
   Verificado direto: deg(E)=deg(radical(E))=7, MESMO grau. A não-redução (reduce=6≠0)
   está só nas componentes EMBEBIDAS de dim menor (par complexo + origem); a curva
   top-dimensional (dim 1 proj) é REDUZIDA grau 7. Seção-hiperplana aplica-se. SOBREVIVE.
2. **PORTEOUS ERRADO:** "Thom-Porteous no fibrado efetivo E₇" NÃO serve — E₇ é exatamente
   o fibrado que §6/R18 mostrou não existir sobre Z (reintroduziria o objeto morto). O
   argumento certo de uniformidade é **constância do grau em família PLANA conexa** de
   pares ZD (conexidade via transitividade de [GZ26]); grau localmente constante onde
   dim D=1 ⟹ ≡7. GAP real = **locus ruim** onde dim D salta p/ 2 (aí o grau nada diz).
3. **CAÇA AO LOCUS RUIM (falsificador):** check mod-p rápido (minor+std+dim, sem primdec)
   em ~17 configs distintas (8 primdec ℚ + 9 mod-p novas): TODAS dim(D)=2 (codim 3), deg=7.
   ZERO anomalias. Não achei o par com codim<3; descartá-lo é o que resta em aberto.

**d≥3 NUNCA usado:** a prova dá witness p/ TODO par (força, não buraco); d≥3 entra só na
leitura do witness como caminho de comprimento 3. Manuscrito §8 reescrito como o
"argumento de grau ímpar" (rota viva; substitui o placeholder construtivo/discriminante —
o grau ímpar dispensa provar disc≥0 de cada cônica). Rigor restante delimitado: (a)
deg D=7 uniforme via flatness+conexidade, fechando o locus-ruim; (b) redução top-dim geral.

## Rodada 21: o locus ruim MAPEADO — grau REDUZIDO top-dim ímpar (correção do "deg=7")

Caça ao locus ruim (Fix 2 do revisor) em 31 configs de base ALINHADAS
(e_i,e_j)v(e_k,e_l) + super-simétrica + par duro. Achados:
- Grau NÃO é uniformemente 7. Histograma (dim,mult_esquema): curvas (dim1 proj):
  (2,7)×13, (2,9)×4 [grau 9 também ocorre]; estratos maiores: (3,3)×7 [dim2,deg3],
  (4,1)×4 [hiperplano], (5,1)×2 [D=tudo], **(4,2)×1 = config 6**.
- **Config 6 = (e₁,e₂)v(e₂,e₁): mult_esquema=2 (PAR)** — parecia falsificador. MAS
  decomposição: comp1 = hiperplano {k₀=0} REAL dim3 deg1 (com MULTIPLICIDADE 2 no
  esquema → mult 2) + comp2 quadric complexa k2²+k3²=0 (real-point-free) + origem.
  O grau REDUZIDO top-dim = 1 (ÍMPAR). Witnesses reais existem (o hiperplano). NÃO
  é contraexemplo. O "grau par" era artefato de não-redução (esquema vs reduzido).

**CORREÇÃO do §8:** "deg D = 7 uniforme" estava ERRADO. O grau do esquema varia
(1,2,3,7,9…) e pode ser par por multiplicidade. A afirmação CORRETA e uniforme é:
**o grau REDUZIDO da parte top-dimensional de D é sempre ÍMPAR** (medido: 7 e 9 nas
curvas d≥3 genéricas; 3 e 1 nos estratos alinhados). Grau reduzido ímpar (qualquer
dim) ⟹ ponto real (seção linear real de dim complementar → 0-ciclo ímpar conjugação-
estável → ponto real) ⟹ witness real. Cobre TODOS os estratos.

**Papel de d≥3 (afinal essencial):** os estratos de dim D grande (hiperplano/tudo)
são configs alinhadas degeneradas, provavelmente d≤2 (witnesses abundantes, fora do
escopo). Config 6 tem [x1,x2]≠0 (d≥2). Hipótese: d≥3 ⟺ dim D=1 (curva grau 7 ou 9,
ímpar); d≤2 ⟺ dim D≥2. Se verdade, na curva-estrato (o escopo) o grau ímpar 7/9 fecha
tudo, e os estratos grandes (d≤2) são triviais. FALTA: caracterizar d por dim D.

**Gaps restantes p/ prova completa (honesto):**
(a) grau REDUZIDO top-dim ímpar UNIFORME — provavelmente um número característico
    mod-2 (grau mod-2 do determinantal), config-independente = 1. Não provado.
(b) correlação d≥3 ⟺ dim D=1 (para localizar o argumento no escopo). Não provado.
(c) redução top-dim (para a seção-hiperplana). Medido, não geral.
Progresso real; núcleo ainda aberto. Ferramenta exata (Singular) foi decisiva.

## Rodada 22: GZ verificado — a formulação do witness está CORRETA (restrição ZD é vácua)

Li as definições exatas de GZ (arXiv 2608.26903, via alphaXiv). Pontos:
- Vértices de Γ_C^Z(S) = P(R+Z(S)) = linhas de elementos cuja PARTE IMAGINÁRIA é ZD
  (parte real livre). Adjacência: [a]~[b] ⟺ ab=ba. Conjectura 6.8: diam Γ_C^Z(S)=3.
- C_S(x) = R ⊕ Rx ⊕ O_S(x) (Lemma 6.2(2)), dim 6; Im C_S(x)=Rx⊕O_S(x), dim 5. ✓
- Prop 4.1: 1,a,b,c,d,ab,ac,ad ortonormais.

**CHECAGEM QUE PODIA MATAR (resolvida a favor):** o path x~u~w~x' exige u,w VÉRTICES
(Im ZD). Como u∈Im C(x) é puro, u é vértice ⟺ u∈Z(S). PROVEI que é AUTOMÁTICO: para
u=k₀x+k₁(c,d)+k₂(d,-c)+k₃(ac,-ad)+k₄(ad,ac), usando ortonormalidade (Prop 4.1):
n(u₁)=n(u₂)=Σkᵢ² (idênticos), Re(u₁)=0, ⟨u₁,u₂⟩=−k₁k₂+k₂k₁+k₃k₄−k₄k₃=0. Por Lemma 4.12
(GZ), (u₁,u₂)∈Z(S). **Todo u∈Im C(x) é zero divisor** ⟹ restrição de vértice é VÁCUA ⟹
D={rank M(u)≤4} É exatamente o locus de witness, SEM constraint extra. Formulação do §1
CONFIRMADA correta.

**Distância:** vizinho comum y de x,x' ∈ C_S(x)∩C_S(x') = R⊕(Im C(x)∩Im C(x')) = R
(interseção medida =0). R é central (não-vértice). ⟹ SEM vizinho comum ⟹ todos os pares
testados (incl. par duro) são d≥3. Logo o locus ruim (dim D≥2, ex. par duro dim3) contém
pares d≥3 GENUÍNOS ⟹ gap (b) "d≥3⟺dimD=1" FALSO. O argumento deve cobrir todos estratos.

**Nota:** o mult mod-p=9 em algumas alinhadas era multiplicidade/embebido; grau REDUZIDO
da curva via primdec ℚ é 7 (não achei curva reduzida grau 9). O grau reduzido top-dim
ímpar (7 curvas; 1/3 estratos maiores) continua sendo a afirmação uniforme medida.

**Estado honesto:** formulação verificada correta contra a fonte; mecanismo de grau-ímpar
sólido no estrato-curva (d≥3 genérico, deg 7); estratos degenerados (par duro) têm
componente linear real (existência ok ali também). NÚCLEO ABERTO: provar grau reduzido
top-dim ímpar UNIFORME (paridade via conjugação, NÃO número característico único — o grau
varia). Problema genuinamente aberto; progresso real, sem prova fabricada.
