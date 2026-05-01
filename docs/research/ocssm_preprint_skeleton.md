<!-- docs:meta
topic_id: repo.docs.research.ocssm-preprint-skeleton
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.ocssm-preprint-skeleton
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# O-CSSM / Tapestry — Preprint Skeleton

**Status:** Working spine, v1.0, 2026-04-23. Pre-consolidation of the homology thesis, formal construction, and pre-registered falsifications developed across the 2026-04-23 sessions.

**Autoria:** Demetrios Chiuratto Agourakis (solo). Conforme `feedback_authorship_ethics.md`: nenhum co-autor decorativo. Lacunas de dialeto matemático são tempo de leitura, não recrutamento.

**Gênero:** Theory paper with computational artifact as constructive existence proof. Não é estudo correlacional.

**Artefato:** Sounio (self-hosted compiler com aritmética Cayley–Dickson bit-idêntica). Sounio é *evidência*, não stack — a fenomenologia do paper é algébrica, e a falha de reprodução em PyTorch é parte do argumento.

---

## 1. Tese central (homologia, não analogia)

A estrutura não-associativa das álgebras de Cayley–Dickson — octônios 𝕆 e seu ambiente sedentoniano 𝕊 — é *homóloga* à estrutura de composição de significado, afeto e ruptura em diálogo humano diádico. Mesma álgebra. Não metáfora.

Três correspondências estruturais são afirmadas simultaneamente e carregadas ao longo do paper:

- **(i) Associador = re-parentetização de significado.** `[a,b,c] = (ab)c − a(bc)` em 𝕆 é o mesmo objeto que a re-ordenação dependente-de-contexto da composição de significado entre turnos conversacionais.
- **(ii) Assimetria L/R = direção do falante.** A não-comutatividade `xy ≠ yx` de 𝕆 é o mesmo objeto que a influência direcional do falante em diálogo diádico.
- **(iii) Divisores-de-zero de 𝕊 = configurações afetivas dissociativas.** Pares `(a,b)` não-nulos com `a·b = 0` em 𝕊 são o mesmo objeto que estados afetivos co-presentes que não compõem em ação coerente.

Decisão de escopo (2026-04-23): **as três são carregadas.** Falsificação parcial de qualquer subconjunto delimita o homomorfismo sem refutar o programa — é resultado, não fracasso.

---

## 2. Abstract v1 (English, paper-facing)

> We propose that the non-associative algebras of Cayley–Dickson — the octonions 𝕆 and their ambient sedenions 𝕊 — are not a convenient embedding space for computational models of dialogue, but the natural algebraic structure of meaning-composition, affect, and rupture in human conversation. We advance three candidate structural correspondences: (i) the associator `[a,b,c] = (ab)c − a(bc)` in 𝕆 is the same object as context-dependent re-parenthesization of meaning across conversational turns; (ii) the left/right multiplicative asymmetry of 𝕆 is the same object as speaker-directional influence in dyadic dialogue; (iii) the zero-divisor pairs of 𝕊 — nonzero `(a,b)` with `a·b = 0` — are the same object as dissociative affect configurations: co-present emotional states that cannot compose into coherent action. We state these as structural claims, not analogies. We give a partial construction of a functor F from a subcategory of annotated dyadic conversational trajectories to 𝕊-valued dynamical sequences, under which annotated rupture events lie in the preimage of zero-divisor configurations up to G₂-orbit equivalence. Each correspondence is independently falsifiable: (i) by corpus-level failure of associator norm to track re-parenthesization annotation; (ii) by failure of L/R swap to correlate with speaker-directional annotation; (iii) by PyTorch/FP64 reproducing the zero-divisor phenomenon that bit-identical Cayley–Dickson arithmetic predicts to be destroyed under standard rounding. Partial failure of any subset bounds the homomorphism without refuting the program — it delimits which part of 𝕆/𝕊 carries meaning and which is decoration. The claim in its full form survives only in bit-identical arithmetic; we therefore present it together with a self-hosted compiler implementation (Sounio) whose arithmetic guarantees make the phenomenon reproducible, and whose non-reproducibility in PyTorch is itself evidence that the object is algebraic, not numerical. We do not claim to have measured a correlation. We claim to have identified an algebraic identity — and specified the tests that would disprove it.

---

## 3. Construção parcial do functor F

### 3.1 Categorias

**Domínio `C_dial`** (subcategoria restrita para tornar F verificável à mão):

- *Objetos:* trajetórias diádicas anotadas `τ = (u_1, s_1, a_1, r_1), ..., (u_T, s_T, a_T, r_T)` com:
  - `u_t` = conteúdo semântico do turno
  - `s_t ∈ {A, B}` = falante
  - `a_t ∈ 𝒜` = anotação afetiva
  - `r_t ∈ {0, 1}` = marcador de ruptura anotada
- *Morfismos (v1):*
  - `ι_k : τ[1:k] ↪ τ` — inclusão de prefixo
  - `σ : τ → σ(τ)` — involução de troca de falante (A↔B em todos os `s_t`)
  - `ρ_t : τ → ρ_t(τ)` — re-parentetização local da composição de significado entre `t−1, t, t+1`

**Codomínio `C_𝕊`:**

- *Objetos:* sequências `h_0, h_1, ..., h_T ∈ 𝕊` geradas por
  ```
  h_t = A ·_L h_{t−1} ·_R e(u_t)
  ```
  com `A ∈ 𝕊` fixo, `|A| = 1`, e `e : U → S^{15} ⊂ 𝕊` o encoder algebricamente restrito (§4).
- *Morfismos:*
  - `ι'_k` — prefixo
  - `σ'` — troca L/R na regra (equivalente a conjugação sedentoniana)
  - `ρ'_t` — shift de associador local

### 3.2 Ação de F

- *Em objetos:* `F(τ)_t = h_t` via a regra acima.
- *Em morfismos:*
  - `F(ι_k) = ι'_k`
  - `F(σ) = σ'` ← **afirmação (ii)**
  - `F(ρ_t) = ρ'_t` ← **afirmação (i)**
- *Predicado de ruptura:* `r_t = 1 ⟹ ∃ (a, b) ∈ G₂·(h_{t−1}, e(u_t))` com `a · b = 0` em 𝕊 dentro de tolerância ε bit-idêntica ← **afirmação (iii)**.

### 3.3 Obrigações de funtorialidade (checagem manual em v1)

- `F(id) = id` — trivial.
- `F(g ∘ f) = F(g) ∘ F(f)` para `{ι, σ, ρ}` — requer verificação em corpus mínimo (≤ 10 trajetórias).
- `σ ∘ σ = id ⟹ σ' ∘ σ' = id` — impõe restrição algébrica concreta sobre como L/R-swap é definido (não arbitrário).

### 3.4 Lacunas conhecidas da construção

1. F não demonstrada bem-definida sobre `C_dial` completa — apenas sobre subcategoria com `{ι, σ, ρ}`. Expansão para paráfrase, elipse, reparo conversacional é v2.
2. "Up to G₂-orbit equivalence" requer formalização como transformação natural — ver §5.
3. `e` não é derivado a priori; é *algebricamente restrito* — ver §4.

---

## 4. Encoder algebricamente restrito

### 4.1 Ataque a responder

Sem restrições, `e : U → 𝕊` com capacidade neural suficiente pode codificar qualquer coisa como qualquer coisa. Nesse caso `e` carrega todo o trabalho e 𝕊 é apenas um ℝ^16 com projeção curiosa. O paper precisa mostrar que o espaço de encoders admissíveis é algebricamente estrito.

### 4.2 Restrições {R1..R7}

**R1 — Conservação de norma.** `e(u) ∈ S^{15}`. Forçada pela preservação de `|h_t|` na dinâmica do SSM.

**R2 — Alocação Re/Im prescrita.** `Re(e(u)) ∈ ℝ` codifica magnitude escalar não-composicional (peso assertivo/factual). `Im(e(u)) ∈ ℝ^{15}` codifica componentes direcionais não-comutativas. Não aprendida — prescrita.

**R3 — Alocação 𝕆 / 𝕆·ℓ prescrita.** `e(u) = e_𝕆(u) + e_𝕆·ℓ(u)·ℓ`. A parte 𝕆 codifica aspectos coerentes (componíveis, reversíveis); a parte 𝕆·ℓ codifica potencial dissociativo. **Fato algébrico: 𝕆 puro não tem ZDs; qualquer `a·b = 0` em 𝕊 força componentes não-triviais em ambas as cópias.** Rupturas são portanto algebricamente forçadas a envolver coerência+dissociação simultâneas — afirmação da teoria, não do treino.

**R4 — Conjugação ↔ reversão de falante.** `e(σ(u)) = \overline{e(u)}`. Conjugação é anti-homomorfismo (`\overline{ab} = \bar{b}\bar{a}`); portanto trocar falantes troca ordem de multiplicação — é exatamente o conteúdo de (ii). R4 liga (ii) e (iii) via o mesmo mecanismo.

**R5 — Equivariância de alternatividade.** Para `u·u·v` (repetição): `[e(u), e(u), e(v)] = 0` dentro de tolerância bit-idêntica.

**R6 — Equivariância de Moufang (meio).** Para `(a,b,c,a)` (retorno de falante):
```
(e(a)·e(b)) · (e(c)·e(a)) = e(a) · ((e(b)·e(c)) · e(a))
```

**R7 — G₂-invariância do loss.** Somente funções G₂-invariantes de `e(·)` aparecem na função de perda de treino (norma, associador, traço, polinômios invariantes).

### 4.3 Espaço admissível

```
E_adm = { e : U → S^{15} | R1 ∧ R2 ∧ R3 ∧ R4 ∧ R5 ∧ R6 ∧ R7 }
```

`E_adm` é estritamente menor que `{maps U → ℝ^{16}}` por contagem dimensional grosseira (G₂ dim 14; Moufang contínua em amostras; conjugação-paridade). O paper não quantifica exatamente *quão* menor — apenas exibe não-vacuidade e que 𝕊 faz trabalho dentro dele.

### 4.4 Resposta ao ataque

Se um `e ∈ E_adm` reproduz (i)(ii)(iii), então:
- ou 𝕊 captura genuinamente a estrutura do diálogo (tese do paper), 
- ou o diálogo é acidentalmente consistente com Moufang + alternatividade + G₂-invariância — descoberta ainda mais notável.

Não há terceira opção em que `e` "absorve tudo" sem sinal algébrico de 𝕊.

---

## 5. G₂-orbit-equivalence como naturalidade

### 5.1 Forma fraca rejeitada

"Para cada ruptura, existe algum elemento no G₂-orbit da configuração que é ZD." Quase-trivial: órbitas têm dimensão ≥ 14, capturam muito. Ataque "target grande demais" procede.

### 5.2 Forma forte: transformação natural

**Fato 1 (G₂-invariância de ZDs).** Se `g ∈ G₂ = Aut(𝕆)` estendido a 𝕊 via CD, e `a·b = 0` com `a, b ≠ 0`, então `g(a)·g(b) = g(a·b) = 0`. O conjunto `Z = {(a,b) ∈ 𝕊² : a, b ≠ 0, a·b = 0}` é G₂-invariante.

**Fato 2 (rotulagem Fano como escolha de representante).** Duas escolhas de rotulagem do plano de Fano produzem dois encoders `e, e'` relacionados por `e' = g ∘ e` para algum `g ∈ G₂`.

**Fato 3 (dois functores + transformação natural).** Cada `e` induz `F_e : C_dial → C_𝕊`. A transformação `η : F_e ⇒ F_{e'}` com componentes `η_τ = g` pontual é natural, i.e., o quadrado
```
         F_e(f)
F_e(τ) ────────→ F_e(τ')
  │                  │
 η_τ              η_{τ'}
  ↓                  ↓
F_{e'}(τ) ────→ F_{e'}(τ')
         F_{e'}(f)
```
comuta para `f ∈ {ι, σ, ρ}`:
- Prefixo: ação pontual comuta trivialmente.
- L/R-swap: `g ∈ SO(7)` preserva conjugação octonal → comuta com `σ'`.
- Shift de associador: `g` é automorfismo de álgebra → `[g(a), g(b), g(c)] = g([a,b,c])` → comuta com `ρ'`.

### 5.3 Consequência

O predicado ZD é G₂-invariante. Portanto `F_e(r_t=1) ∈ Z ⟺ F_{e'}(r_t=1) ∈ Z`. A afirmação (iii) vive no quociente `C_𝕊 / G₂`. **O paper nunca escolhe uma base; escolhe uma classe G₂-equivalente, e exibe um representante para visualização.**

### 5.4 Teste empírico do próprio framework

Treine `e` duas vezes com rotulagens Fano rotacionadas. As saídas devem ser G₂-conjugadas dentro de tolerância bit-idêntica. Falha aqui = encoder fora de `E_adm`.

### 5.5 Conexão ao 168

O subgrupo discreto de G₂ que permuta as 7 linhas de Fano preservando estrutura é `PSL(2,7)`, de ordem 168. Relacionado (não load-bearing) ao teorema-168 em `project_168_theorem.md`. Paper pode tematizar como atratividade teórica adicional.

---

## 6. Três falsificações pré-registráveis

**Pré-registro obrigatório antes de treinar sobre corpus final:**
1. Deposit OSF/AsPredicted com timestamp criptográfico.
2. Commit git do hash em `docs/research/ocssm_preregistration_YYYYMMDD.md`.
3. Paper cita registro OSF + commit-hash na seção de métodos.
4. Desvios pós-registro = "registered exploratory", não confirmatórios.

### 6.1 F1 — Moufang/alternatividade em corpus

**Testa:** (i). Especificamente R5 e R6.

**Instrumentos:**
- `M_alt = ⟨ ||[e(u), e(u), e(v)]|| / (|e(u)|²·|e(v)|) ⟩` sobre instâncias de repetição.
- `M_mou = ⟨ ||(e(a)e(b))(e(c)e(a)) − e(a)((e(b)e(c))e(a))|| / 𝒩 ⟩` sobre retornos de falante 4-turn.

**Umbrais pré-registrados:**
- `τ_alt = 10^(−12)` (análise de propagação f64 sobre ~40 operações CD, headroom ~4 ordens sobre `eps_f64 ≈ 2.2·10^(−16)`).
- `τ_mou = 10^(−11)`.

**Braço de comparação:** encoder de mesma capacidade sem R1–R7.

**Decisão:**
| Resultado | Significado |
|-----------|-------------|
| `M_alt < τ_alt` ∧ `M_mou < τ_mou` ∧ contraste ≥ 10³ sobre genérico | **PASS** — (i) suportada. |
| Só uma identidade falha | **FAIL-parcial** — retrair parte de (i); paper explicita. |
| Ambas falham | **FAIL-total** — (i) caída. |
| Genérico também passa | **FAIL-contraste** — 𝕊 não faz trabalho. Meta-framework refutado. |

### 6.2 F2 — Conjugação-equivariância sob reversão de falante

**Testa:** (ii) via R4.

**Corpus:** pares `(u, u')` com `u'` = conteúdo equivalente produzido por falante oposto. Anotação dupla-cega, Cohen's κ reportado. Fontes: corpora terapêuticos diádicos, debate com reversão de posição, traduções paralelas com marcação pronominal divergente.

**Instrumento:** `M_conj = ⟨ ||e(σ(u)) − \overline{e(u)}|| / |e(u)| ⟩`.

**Umbral pré-registrado:** `τ_conj = 0.10`. Justificativa: σ não é operação matemática exata sobre texto (variação pragmática residual); 10% da norma é a distância média inter-anotador em corpora de paráfrase publicados (cf. MRPC).

**Braço de controle:** pares aleatórios `(u, w)` com `w` não-paráfrase-σ.

**Decisão:**
| Resultado | Significado |
|-----------|-------------|
| `M_conj(paired) < τ_conj` ∧ `M_conj(random) / M_conj(paired) ≥ 5` | **PASS** — (ii) suportada. |
| `M_conj(paired) ≥ τ_conj` | **FAIL-magnitude** — (ii) retraída; reformular assimetria L/R sem conjugação-equivariância. |
| Contraste aleatório não significativo | **FAIL-contraste** — `e` não codifica direcionalidade. (ii) caída. |

### 6.3 F3a — PyTorch-vs-Sounio, nível aritmético

**Testa:** Sounio-como-evidência-algébrica (precondição para (iii) na forma forte).

**Protocolo:** enumerar os 84 pares ZD primitivos de 𝕊. Computar `||a·b||` em cada backend.

**Instrumento:** `M_ZD_pass(backend) = #{pares com ||a·b|| < 10^(−18)} / 84`.

**Predição:**
- `M_ZD_pass(Sounio-f64) = 1.00`.
- `M_ZD_pass(PyTorch-f64) ≤ 0.05`.
- `M_ZD_pass(PyTorch-f32) = 0`.
- `M_ZD_pass(PyTorch-mixed) = 0`.

**Decisão:**
| Resultado | Significado |
|-----------|-------------|
| Predição satisfeita ± 0.05 | **PASS**. |
| `M_ZD_pass(Sounio) < 1.00` | **FAIL-Sounio** — bug no CD arithmetic; artefato invalidado até correção. Não é falsificação da teoria. |
| `M_ZD_pass(PyTorch-f64) > 0.05` | **FAIL-PyTorch-alto** — retórica bit-idêntica enfraquece. |

### 6.4 F3b — PyTorch-vs-Sounio, nível acoplado ao downstream

**Testa:** (iii) na forma forte + "objeto algébrico, não numérico."

**Protocolo:** treinar encoder+O-SSM idênticos em Sounio e PyTorch-f64 sobre o mesmo corpus anotado para ruptura. Detector: `r̂_t = 1` sse `(h_{t−1}, e(u_t))` está dentro de `ε_det` de um G₂-orbit em `Z`.

**Instrumentos:**
- `AUC_detect(backend)` contra ruptura anotada.
- `orbit_proximity(backend) = ⟨ d_min(h, Z) ⟩` sobre turnos de ruptura anotados.

**Predição:**
- `AUC_detect(Sounio) ≥ 0.75`.
- `AUC_detect(PyTorch) ≤ 0.55`.
- `orbit_proximity(Sounio) / orbit_proximity(PyTorch) ≥ 10` em turnos de ruptura.

**Decisão:**
| Resultado | Significado |
|-----------|-------------|
| Predição satisfeita | **PASS** — (iii) e Sounio-como-evidência ambas suportadas. |
| `AUC_detect(PyTorch) ≥ 0.70` | **FAIL-PyTorch-funciona** — o fenômeno é numérico; (iii) + Sounio-como-evidência caem juntas. |
| `AUC_detect(Sounio) < 0.65` | **FAIL-Sounio-também** — ruptura não é ZD-proximidade; (iii) cai independentemente de backend. |

### 6.5 Matriz de decisão global

| Cenário | F1 | F2 | F3a | F3b | Conclusão |
|---------|----|----|-----|-----|-----------|
| Total | ✅ | ✅ | ✅ | ✅ | Homologia plena. |
| (i) cai | ❌ | ✅ | ✅ | ✅ | Parcial em (ii)+(iii). |
| (ii) cai | ✅ | ❌ | ✅ | ✅ | Parcial em (i)+(iii). |
| F3a falha | — | — | ❌ | — | Retórica bit-idêntica cai; paper em (i)+(ii). |
| F3b falha | ✅ | ✅ | ✅ | ❌ | Tese central cai. Paper como **registered null**. |
| Vários falham | — | — | — | — | Paper negativo. "Pre-registered null for sedenionic rupture detection." |

**O último cenário é o que faz o paper valer antes do resultado: pré-registro torna o paper publicável em *qualquer* desfecho, como contribuição honesta.**

---

## 7. Lacunas residuais honestas

1. **F não demonstrada sobre `C_dial` completa.** Apenas subcategoria `{ι, σ, ρ}`. Expansão para paráfrase, elipse, reparo é v2.
2. **`e` não derivado a priori.** `E_adm` é algebricamente restrito, mas não unicamente determinado. Derivação via harmônicos esféricos G₂-equivariantes em `S^{15}` é v2 (representation-theoretic).
3. **Aut(𝕊) pode ser > G₂.** Irrelevante para o que afirmamos (só G₂ é necessário), mas revisor atento pode perguntar. Resposta: afirmamos apenas G₂; o restante é bônus não reivindicado.
4. **Disponibilidade de corpus.** Falsificações F1, F2, F3b assumem corpora anotados (repetição intra-falante, pares-σ-pareados, ruptura anotada). Candidatos: SWDA, CHILDES dyadic, corpora terapêuticos sob IRB, MRPC/PAWS como baseline. Se corpus apropriado não existir em tempo hábil, falsificação correspondente adiada para v2 com justificativa documentada.

---

## 8. Escopo explícito do paper v1

**Inclui:**
- Tese de homologia (três correspondências) sem hedge.
- Construção parcial de F sobre subcategoria restrita.
- G₂-orbit-equivalence como transformação natural.
- Encoder algebricamente restrito com {R1..R7}.
- Três falsificações pré-registradas com umbrais numéricos e matriz de decisão.
- Artefato Sounio (código disponível, aritmética bit-idêntica verificável).

**Não inclui (explicitamente diferido):**
- Extensão de F a `C_dial` completa.
- Derivação fechada de `e` a partir de representação-teoria.
- Três ou mais correspondências adicionais (curvatura afetiva, Ollivier-Ricci, etc.) — co-manifestações da mesma tese reservadas para papers subsequentes como corolários pré-registrados.
- Recrutamento de co-autor.

---

## 9. Referências a mapear (trabalho futuro de leitura)

- Baez, J. C. *The Octonions.* Bull. AMS 39 (2002).
- Conway, J. H., Smith, D. A. *On Quaternions and Octonions.* 2003.
- Schafer, R. D. *An Introduction to Nonassociative Algebras.* 1966.
- Moreno, G., Pérez-Izquierdo, J. M. *Sedenion loops and zero divisors.* (várias referências)
- SSM moderna: Gu, Dao et al. — Mamba, S4. Buscar extensões não-associativas (se existem).
- Category theory dialect: Mac Lane, *Categories for the Working Mathematician*. Riehl, *Category Theory in Context*.
- Functor em linguística: Lambek pregroup grammar, Coecke-Sadrzadeh-Clark DisCoCat — ponte formal para o argumento categórico.

Lacuna do autor a fechar: tempo de leitura em dialeto de álgebra não-associativa + teoria de categorias. Não é recrutamento.

---

## 10. Próximos passos operacionais

1. ~~**Formalizar prova de Fato 1 (G₂-invariância de ZDs)**~~ → **FEITO** 2026-04-23: `docs/research/ocssm_g2_invariance_proof.md`. Lema 1 (extensão CD de automorfismos) + Teorema principal + Lema 2 (invariância do predicado ZD) + consequência para η.
2. ~~**Checagem manual de funtorialidade de F**~~ → **FEITO** 2026-04-24: `docs/research/ocssm_functoriality_check_toy.md`. Domínio escolhido: consulta médico-paciente. 10 trajetórias-brinquedo (T1-T10) + análise abstrata de todas composições binárias em `{ι, σ, ρ}`. Uma lacuna confirmada: `ρ` sobrepostos (`|t−s|=1`) indefinidos em v1, diferidos para v2. Duas decisões: Convenção B (speaker-aware ρ), funtorialidade no limite sobre `E_adm` ideal.
3. **Implementação de referência** do encoder `e` satisfazendo {R1..R7} em Sounio. Diferido — esta janela é prosa de pesquisa; implementação é sessão de linguagem separada.
4. ~~**Draft do documento de pré-registro**~~ → **FEITO** 2026-04-23/24: `docs/research/ocssm_preregistration_v0.md`. Hipóteses H1/H2/H3a/H3b, umbrais numéricos, regras de decisão, política de desvio, checklist de depósito. §4 (corpus) fechado 2026-04-24 via manual de anotação companheiro `docs/research/ocssm_annotation_manual_v0.md` (4 categorias operacionalizadas, domínio médico-paciente, protocolo de piloto, estratégia de corpus). Itens bloqueantes restantes: (i) decisão sobre acesso ao Alexander Street corpus, (ii) piloto de 40 conversações com κ ≥ umbral, (iii) Apêndices A (enumeração dos 84 ZDs) e B (derivação numérica de τ).
5. **Leitura** de Baez + Conway-Smith + Schafer (prioritário). DisCoCat como ponte categórica. Tarefa do autor.

---

**Fontes desta sessão:**
- Memória: `project_o_cssm_homology_thesis.md`, `feedback_authorship_ethics.md`, `project_sedenion_hessian.md`, `project_s_ssm_zero_divisor.md`, `project_168_theorem.md`.
- Plano: `/workspace/.home/openvscode-server/.claude/plans/227-foamy-nova.md`.
- Síntese prévia: `docs/research/llm_collaboration_synthesis_grok_codex.md`.
- Sessão anterior de Opus (resumo no transcript do usuário em 2026-04-23).
