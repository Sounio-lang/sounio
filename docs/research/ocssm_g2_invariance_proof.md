<!-- docs:meta
topic_id: repo.docs.research.ocssm-g2-invariance-proof
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.ocssm-g2-invariance-proof
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# G₂-Invariância do Conjunto de Divisores-de-Zero em 𝕊

**Status:** v1.0, 2026-04-23. Componente §5 do skeleton do preprint O-CSSM. Formalização do Fato 1 usado na construção da transformação natural `η : F_e ⇒ F_{e'}`.

**Afirmação a provar:** O conjunto `Z = {(a, b) ∈ 𝕊 × 𝕊 : a ≠ 0, b ≠ 0, a·b = 0}` é invariante sob a ação de `G₂ = Aut(𝕆)` estendida a 𝕊 via construção Cayley–Dickson.

Esta invariância é pré-requisito para que a afirmação (iii) do paper ("rupturas estão na pré-imagem de configurações ZD *up to G₂-orbit equivalence*") seja formulada como transformação natural entre functores, e não como waiver informal.

---

## 1. Preliminares algébricas

### 1.1 Construção Cayley–Dickson

Seja `(𝔸, ·, ∗)` uma álgebra-∗ com involução `∗` (conjugação) satisfazendo:
- `(a∗)∗ = a`
- `(a·b)∗ = b∗ · a∗`
- `a + a∗ ∈ ℝ · 1` (real) e `a·a∗ ∈ ℝ_{≥0} · 1` (norma).

A **duplicação Cayley–Dickson** produz `𝔸' = 𝔸 ⊕ 𝔸` com multiplicação
```
(a, b) · (c, d) = (a·c − d∗·b,  d·a + b·c∗),
```
e involução
```
(a, b)∗ = (a∗, −b).
```

Escreve-se simbolicamente `(a, b) = a + b·ℓ` onde `ℓ² = −1` e `ℓ` anticommuta com a parte imaginária do nível inferior.

A cadeia `ℝ → ℂ → ℍ → 𝕆 → 𝕊 → ...` é obtida por iteração. Em cada passo perde-se uma propriedade: `ℂ` perde auto-conjugação, `ℍ` perde comutatividade, `𝕆` perde associatividade, `𝕊` perde a propriedade-divisão (emergem ZDs).

### 1.2 Octônios e G₂

**Definição.** `G₂ := Aut(𝕆)`, o grupo de automorfismos da álgebra `𝕆`. Isto é, `g : 𝕆 → 𝕆` tal que:
- `g` é ℝ-linear,
- `g(1) = 1`,
- `g(x·y) = g(x)·g(y)` para todos `x, y ∈ 𝕆`.

**Fatos clássicos (Baez 2002; Schafer 1966):**

1. `G₂` é um grupo de Lie compacto, conexo, simples, de dimensão 14, exceptional.
2. `G₂ ⊂ SO(7)` pela ação nos 7 imaginários `e_1, ..., e_7`.
3. Todo `g ∈ G₂` preserva a norma: `|g(x)| = |x|` para todo `x ∈ 𝕆`.
4. Todo `g ∈ G₂` preserva a conjugação: `g(x∗) = g(x)∗`.
   - *Prova:* `x + x∗ = 2·Re(x) ∈ ℝ·1`, e `g(1) = 1` ℝ-linear ⟹ `g(Re(x)) = Re(x)` (parte real preservada) ⟹ `g(x∗) = g(2Re(x) − x) = 2Re(x) − g(x) = g(x)∗`. ∎

### 1.3 Divisores-de-zero em 𝕊

**Definição.** `Z := {(a, b) ∈ 𝕊 × 𝕊 : a ≠ 0, b ≠ 0, a · b = 0}`.

**Fatos clássicos (Conway-Smith 2003; Moreno 1998):**

1. `𝕆` não tem ZDs (é álgebra-divisão; `|a·b| = |a|·|b|`).
2. `𝕊` tem ZDs. Uma contagem padrão enumera **84 pares ZD primitivos** (aqueles que envolvem uma única linha-de-Fano via doubling). Pares compostos são obtidos por combinações lineares.
3. Todo ZD de 𝕊 tem componentes não-triviais em ambas as cópias da decomposição `𝕊 = 𝕆 ⊕ 𝕆·ℓ`. *Corolário:* se `a = a_0 + a_1·ℓ` e `a_1 = 0`, então `a · b ≠ 0` para todo `b ≠ 0` — a cópia 𝕆 pura não contribui para ZDs.
4. `𝕊 \ {0}` não é um grupo de Moufang; a propriedade-divisão falha junto com a alternativa.

---

## 2. Extensão de G₂ a Aut(𝕊)

### 2.1 Lema (extensão CD de automorfismos)

**Lema 1.** Seja `g : 𝔸 → 𝔸` um automorfismo da álgebra-∗ `𝔸`, isto é, `g` preserva produto E conjugação. Seja `𝔸' = 𝔸 ⊕ 𝔸` a duplicação CD. Então a aplicação
```
ĝ : 𝔸' → 𝔸',    ĝ(a, b) := (g(a), g(b))
```
é um automorfismo de `𝔸'` (preserva produto E conjugação).

**Prova.**

*(produto)* Para `(a, b), (c, d) ∈ 𝔸'`:
```
ĝ((a, b)·(c, d))
  = ĝ(a·c − d∗·b,  d·a + b·c∗)                    [def. produto em 𝔸']
  = (g(a·c − d∗·b),  g(d·a + b·c∗))                [def. ĝ]
  = (g(a)·g(c) − g(d∗)·g(b),  g(d)·g(a) + g(b)·g(c∗))
                                                    [g preserva produto, ℝ-linear]
  = (g(a)·g(c) − g(d)∗·g(b),  g(d)·g(a) + g(b)·g(c)∗)
                                                    [g preserva ∗]
  = (g(a), g(b)) · (g(c), g(d))                    [def. produto em 𝔸']
  = ĝ(a, b) · ĝ(c, d).                              [def. ĝ]
```

*(conjugação)*
```
ĝ((a, b)∗) = ĝ(a∗, −b) = (g(a∗), −g(b)) = (g(a)∗, −g(b)) = (g(a), g(b))∗ = ĝ(a, b)∗.
```

*(identidade)* `ĝ(1, 0) = (g(1), 0) = (1, 0) = 1_{𝔸'}`.

*(ℝ-linearidade)* Herdada de `g`. ∎

### 2.2 Corolário: G₂ ↪ Aut(𝕊)

Como todo `g ∈ G₂` preserva produto e conjugação em `𝕆` (§1.2), aplica-se o Lema 1 com `𝔸 = 𝕆`, `𝔸' = 𝕊`. Obtém-se monomorfismo de grupos
```
ι : G₂ ↪ Aut(𝕊),    ι(g)(a, b) := (g(a), g(b)).
```

*Injetividade:* se `ι(g) = ι(g')` então `g(a) = g'(a)` para todo `a ∈ 𝕆` (tomando `b = 0`), logo `g = g'`.

*Observação.* `ι` não é sobrejetor em geral; `Aut(𝕊)` pode ter componentes extra-G₂ oriundas de simetrias da doubling. Irrelevante para a afirmação do paper: afirmamos apenas invariância sob `ι(G₂)`.

---

## 3. Teorema principal

**Teorema (G₂-invariância de Z).** Para todo `g ∈ G₂` e todo `(a, b) ∈ Z`:
```
(ι(g)(a),  ι(g)(b)) ∈ Z.
```

**Prova.**

Seja `ĝ := ι(g)`. Pelo Lema 1, `ĝ` é automorfismo de 𝕊.

(1) `ĝ(a) · ĝ(b) = ĝ(a · b) = ĝ(0) = 0` (usando que `ĝ` é homomorfismo e ℝ-linear).

(2) `ĝ(a) ≠ 0` pois `ĝ` é bijeção (automorfismo) e `a ≠ 0`. Análogo para `b`.

Portanto `(ĝ(a), ĝ(b))` satisfaz os três critérios de pertinência a `Z`. ∎

**Corolário (invariância orbital).** Para cada par `(a, b) ∈ Z`, a órbita
```
G₂ · (a, b) := { (ĝ(a), ĝ(b)) : g ∈ G₂ } ⊂ Z.
```

Em particular, `Z` é união disjunta de `G₂`-órbitas. As 84 classes de ZDs primitivos de `𝕊` se organizam em `G₂`-órbitas (decomposição explícita em Moreno-Pérez-Izquierdo 2004).

---

## 4. Consequência para a transformação natural η

**Contexto (skeleton §5.2).** Dois encoders `e, e' : U → S^{15}` relacionados por mudança de rotulagem Fano satisfazem `e'(u) = g · e(u)` para algum `g ∈ G₂` (aplicado via `ι(g)`). Induzem dois functores `F_e, F_{e'} : C_dial → C_𝕊`.

Definimos `η : F_e ⇒ F_{e'}` por `η_τ := ι(g)` aplicada pontualmente a cada `h_t ∈ F_e(τ)`.

**Predicado ZD.** Para cada trajetória `τ` com ruptura anotada em turno `t*`:
```
P_e(τ) := "∃ (a, b) ∈ G₂·(h_{t*−1}, e(u_{t*}))  tal que  a · b = 0".
```

**Lema 2 (invariância do predicado).** `P_e(τ) ⟺ P_{e'}(τ)` para qualquer `e, e' = ι(g)·e`.

**Prova.** Pelo Teorema, `Z` é `G₂`-invariante. Mas `P_e(τ)` afirma exatamente que `(h_{t*−1}, e(u_{t*})) ∈ G₂·(pré-imagem em Z)`, ou equivalentemente que a `G₂`-órbita intersecta `Z`. Aplicar `η_τ = ι(g)` leva toda a configuração para a correspondente `F_{e'}(τ)`, e `ι(g)(G₂·x) = G₂·ι(g)(x)` (ação de grupo). Portanto `P_{e'}(τ)` é obtido de `P_e(τ)` por `G₂`-transporte, e `Z`-pertinência é preservada pelo Teorema. ∎

**Consequência final.** A afirmação (iii) do paper — "rupturas estão na pré-imagem de configurações ZD up to G₂-orbit equivalence" — é bem-definida sobre o quociente `C_𝕊 / ι(G₂)`, onde escolha de representante de base Fano é invisível. A crítica "você cherry-picked a base" falha como erro de categoria: o paper afirma a propriedade sobre a classe de equivalência, e exibe um representante apenas para visualização.

---

## 5. Teste empírico associado

**Protocolo de verificação do framework.** Treinar `e` duas vezes sobre o mesmo corpus:
- Corrida A: rotulagem Fano canônica (e.g., Baez convention).
- Corrida B: rotulagem Fano obtida por `g ∈ G₂` explícito fixado (e.g., rotação de 120° preservando uma linha).

**Predição:** `e_B(u) = ι(g) · e_A(u)` dentro de tolerância bit-idêntica Sounio `τ_nat = 10^(−11)` para todo `u` no corpus de teste, exceto variação introduzida por aleatoriedade de inicialização (que deve ser controlada com semente fixa).

**Falha pré-registrada:** `‖ e_B(u) − ι(g)·e_A(u) ‖ > τ_nat` para fração significativa de `u` ⟹ encoder fora de `E_adm`, framework inaplicável sob aquele `e`. Este é um teste da conformidade do encoder com {R1..R7}, não da teoria em si.

---

## 6. Notas honestas

1. **Moreno-Pérez-Izquierdo (2004)** provê a decomposição explícita dos 84 pares ZD em órbitas de `G₂` (ou de um subgrupo discreto de `G₂`). Não reproduzo a contagem aqui — cita-se e, se relevante para o paper, verifica-se em código-Sounio como teste de unidade da aritmética CD.

2. **Aut(𝕊) vs ι(G₂).** `Aut(𝕊)` pode conter automorfismos não oriundos de `𝕆` via doubling (e.g., troca dos fatores da doubling se simétrica). Para a afirmação do paper basta `ι(G₂)`. Se um revisor exigir caracterização completa de `Aut(𝕊)`, resposta: além do escopo; consultar literatura de álgebras CD superiores (Flaut, Shestakov).

3. **ℝ-linearidade de `g`.** Assumida implicitamente no Lema 1. `G₂` automaticamente opera por ℝ-linearidade (fato padrão; ver Schafer §III). Não requer hipótese extra.

4. **Moufang e alternatividade.** Preservadas sob `ĝ` porque são identidades polinomiais satisfeitas por `𝕆` e por hipótese-em-𝔸, e a extensão CD preserva satisfação de identidades polinomiais de grau ≤ 3 que envolvam apenas a cópia `𝕆`. Para identidades envolvendo o fator `ℓ`, verificar caso a caso — irrelevante para o Teorema principal, relevante para as falsificações R5/R6 no §6 do skeleton.

---

## Referências

- Baez, J. C. (2002). *The Octonions.* Bull. AMS 39, 145–205. [§4: G₂]
- Schafer, R. D. (1966). *An Introduction to Nonassociative Algebras.* Academic Press. [§III: automorphisms of composition algebras]
- Conway, J. H., Smith, D. A. (2003). *On Quaternions and Octonions.* A K Peters.
- Moreno, G. (1998). *The zero divisors of the Cayley–Dickson algebras over the real numbers.* Bol. Soc. Mat. Mex. 4(1), 13–28.
- Moreno, G., Pérez-Izquierdo, J. M. (2004). *Totally real forms of the Cayley–Dickson algebras.* (decomposição orbital explícita dos 84 ZDs).
- Flaut, C. (2006). *Some equations in algebras obtained by the Cayley–Dickson process.* An. Şt. Univ. Ovidius 14(2), 59–68.

---

**Próximos passos associados (skeleton §10):**
- Verificação em Sounio: teste unitário que exibe todos os 84 pares ZD primitivos, aplica `ι(g)` para `g ∈ G₂` discreto (168 elementos de `PSL(2,7)`), e confirma que `Z` é preservado bit-identicamente. Este é um teste mecânico da prova em §3.
- Derivar rotulagem Fano canônica a usar no paper; referenciar Baez convention ou justificar desvio.
