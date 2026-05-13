<!-- docs:meta
topic_id: repo.docs.research.ocssm-functoriality-check-toy
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.ocssm-functoriality-check-toy
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Checagem Manual de Funtorialidade de F — Trajetórias-Brinquedo de Consulta Médica

**Status:** v1.0, 2026-04-24. Componente §3.3 do skeleton. Verifica funtorialidade de `F : C_dial → C_𝕊` sobre a subcategoria restrita `{ι, σ, ρ}` usando trajetórias diádicas construídas manualmente no domínio de consulta médico-paciente.

**Objetivo:** garantir que para todo par de morfismos composíveis `f, g`:
- `F(id) = id`
- `F(g ∘ f) = F(g) ∘ F(f)`

A checagem ocorre em dois níveis: (A) análise abstrata das composições possíveis em `C_dial` e suas imagens em `C_𝕊`, (B) verificação concreta em trajetórias de consulta médica.

---

## 1. Convenções

### 1.1 Notação de trajetória

Cada trajetória `τ` é escrita como tabela de turnos:
```
t | s | u                                    | a        | r
--|---|--------------------------------------|----------|---
1 | D | "O que o senhor está sentindo?"       | neutral  | 0
2 | P | "Dor no peito há duas semanas"         | distress | 0
...
```
com `s ∈ {D, P}` (doutor, paciente), `a ∈ 𝒜` (afeto anotado), `r ∈ {0,1}` (ruptura).

### 1.2 Morfismos considerados

- `ι_k(τ) = τ[1:k]` — prefixo.
- `σ(τ)` — involução `D ↔ P` em todos os `s_t`.
- `ρ_t(τ)` — re-parentetização local: na composição de significado entre `t−1, t, t+1`, muda o binding de `((u_{t−1} · u_t) · u_{t+1})` para `(u_{t−1} · (u_t · u_{t+1}))`. No lado 𝕊, corresponde a shift de associador na sequência `h_{t−1}, h_t, h_{t+1}`.

### 1.3 Convenção sobre `ρ_t`

Por convenção, `ρ_t` é involução: aplicar duas vezes retorna à parentetização original. Matematicamente: se a estrutura de parentetização é uma árvore binária com pivô em `t`, `ρ_t²(τ) = τ`.

### 1.4 Convenção de leitura para o encoder `e`

Para as checagens, tratamos `e(u_t) ∈ 𝕊` em forma simbólica: `e(u_t) = r_t + \sum_i m_{t,i} e_i` com separação `e_1..e_7` (parte 𝕆) e `e_8..e_{15}` (parte 𝕆·ℓ). Não computamos valores numéricos; verificamos apenas as identidades algébricas requeridas. Assume-se que `e` satisfaz {R1..R7}.

---

## 2. Análise abstrata das composições

Listagem de composições binárias em `{ι, σ, ρ}` e o que cada uma impõe a `F`:

| Composição | Condição em `C_dial` | Imagem em `C_𝕊` sob `F` |
|------------|----------------------|-------------------------|
| `ι_j ∘ ι_k`, `j ≤ k` | `= ι_j` (prefixos encaixam) | `ι'_j ∘ ι'_k = ι'_j` ✓ trivial |
| `σ ∘ σ` | `= id_τ` (involução) | `σ' ∘ σ' = id` **→ requer verificação em 𝕊** |
| `ρ_t ∘ ρ_t` | `= id_τ` (por §1.3) | `ρ'_t ∘ ρ'_t = id` **→ requer verificação em 𝕊** |
| `σ ∘ ι_k` | `= ι_k ∘ σ` (comutam) | deve comutar em 𝕊 ✓ (ação pontual) |
| `ρ_t ∘ ι_k`, `k > t+1` | `= ι_k ∘ ρ_t` | comutam em 𝕊 ✓ |
| `ρ_t ∘ ι_k`, `k ≤ t` | `ρ_t` não-definido (domínio vazio) | n/a |
| `σ ∘ ρ_t` | `= ρ_t ∘ σ`? **→ depende de ρ ser speaker-aware** | **checar** |
| `ρ_t ∘ ρ_s`, `|t−s| ≥ 2` | comutam | comutam em 𝕊 ✓ |
| `ρ_t ∘ ρ_s`, `|t−s| = 1` | **não comutam** em geral | `F` deve refletir a não-comutação |

As linhas em negrito são as obrigações não-triviais de funtorialidade. Atacamos cada uma a seguir.

### 2.1 `σ ∘ σ = id` ⟹ `σ' ∘ σ' = id` em 𝕊

Em `C_dial`, `σ` troca `D ↔ P` em todos os turnos; aplicada duas vezes, retorna ao original.

Em `C_𝕊`, `σ'` age sobre a sequência `h_0, h_1, ..., h_T` implementando a equação da restrição R4: se a sequência original é gerada por `h_t = A ·_L h_{t−1} ·_R e(u_t)` com R4 impondo `e(σ_token(u)) = \overline{e(u)}`, então `σ'` produz sequência gerada por `h'_t = \bar A ·_L h'_{t−1} ·_R \overline{e(u_t)}`.

Conjugação sedentoniana é involução:
```
\overline{\overline{x}} = x     para todo x ∈ 𝕊.
```
Logo `σ' ∘ σ'` gera `h''_t = A ·_L h''_{t−1} ·_R e(u_t) = h_t`. ✓

**Obrigação exigida de `e`:** R4 deve ser *exata*, não aproximada, na formulação funtorial. Na prática empírica (§6.2 do skeleton), R4 é testada com tolerância `τ_conj = 0.10`; a funtorialidade como morfismo categórico exige `σ' ∘ σ' = id` bit-exato, que se deriva de R4 exata. Resolução: no paper, `F` é definida sobre encoders *idealmente* satisfazendo R4; encoders treinados aproximam este ideal dentro de `τ_conj`, e a naturalidade é afirmada *asymptotically* à medida que `e` converge para `E_adm`.

### 2.2 `ρ_t ∘ ρ_t = id` ⟹ `ρ'_t ∘ ρ'_t = id` em 𝕊

Em `C_dial`, `ρ_t` troca entre as duas parentetizações binárias de um produto triplo centrado em `t`.

Em `C_𝕊`, o shift de associador local é: substituir `(h_{t−1} · e(u_t)) · e(u_{t+1})` por `h_{t−1} · (e(u_t) · e(u_{t+1}))`. Aplicando `ρ'_t` de novo, substitui de volta. ✓

**Sutileza:** o shift altera valor de `h_{t+1}` (diferença dada pelo associador `[h_{t−1}, e(u_t), e(u_{t+1})]`), mas a operação *de troca* é involutiva. Isto é exatamente a afirmação (i) do paper: o associador *é* o objeto que mede essa troca.

### 2.3 `σ ∘ ρ_t = ρ_t ∘ σ`?

**Depende de como `ρ` é definido.** Duas convenções possíveis:

**Convenção A (speaker-blind):** `ρ_t` re-parentetiza independentemente de quem falou. Neste caso:
- `σ` primeiro troca falantes, depois `ρ_t` re-parentetiza.
- `ρ_t` primeiro re-parentetiza, depois `σ` troca.
Resultado idêntico porque `ρ` não inspeciona `s_t`. Comutatividade em `C_dial`.

**Convenção B (speaker-aware):** `ρ_t` só atua se `s_{t−1} = s_{t+1}` (retorno de falante). Neste caso:
- `σ` troca `s_{t−1} ↔ σ(s_{t−1})`, preservando igualdade `s_{t−1} = s_{t+1}` ⟹ `s_{t−1} ↔ s_{t+1}` ainda iguais após σ.
- `ρ_t` após `σ` ou antes comuta.
Comutatividade também mantida.

**Escolha do paper:** Convenção B. Justificativa: retorno de falante é a condição linguística sob a qual a re-parentetização é observável empiricamente (ver F1b no skeleton §6.1). Convenção A permite `ρ_t` em qualquer contexto, mas maioria dos contextos não-retorno não exibe re-parentetização detectável.

Em `C_𝕊` sob Convenção B: tanto `σ'` quanto `ρ'_t` atuam pontualmente / localmente, e `σ'` não altera a posição de associador. Comutam. ✓

### 2.4 `ρ_t ∘ ρ_s`, `|t−s| = 1`, não comuta

Exemplo: `ρ_2 ∘ ρ_3` vs `ρ_3 ∘ ρ_2`. As janelas `{u_1, u_2, u_3}` e `{u_2, u_3, u_4}` se sobrepõem no par `(u_2, u_3)`. Re-parentetização em `t=2` muda o binding em `(u_1 · u_2) · u_3`; re-parentetização em `t=3` muda binding em `(u_2 · u_3) · u_4`. Aplicar em ordens diferentes produz árvores de parentetização diferentes.

Em `C_𝕊`, isto se traduz em diferença quantificada pela identidade de Moufang meio (R6):
```
(x_1 · x_2)·(x_3 · x_4) =? x_1·((x_2 · x_3) · x_4)?
```
Em `𝕆`, a identidade de Moufang provê *alguma* relação, mas **não igualdade plena** para sequências de 4 elementos não-satisfazendo retorno de falante.

**Consequência para o paper:** a não-comutatividade de `ρ` em `C_dial` *deve* corresponder à não-comutatividade de `ρ'` em `C_𝕊`, e `F` deve preservar a estrutura da diferença. Isto é não-trivial. Para v1, `F` é restrita a compor `ρ`s não-sobrepostos (`|t−s| ≥ 2`), evitando o caso. Expansão para `ρ`s sobrepostos é v2 e exige representação explícita da árvore de parentetização na categoria-fonte.

**Isto é uma restrição documentada honestamente no §3.4 do skeleton.**

---

## 3. Trajetórias-brinquedo

### T1 — Anamnese de dor torácica (baseline, sem morfismos)

```
t | s | u                                              | a          | r
--|---|------------------------------------------------|------------|---
1 | D | O que o senhor está sentindo?                  | inquiring  | 0
2 | P | Uma dor no peito, há duas semanas              | discomfort | 0
3 | D | Onde especificamente? Pode apontar?            | clarifying | 0
4 | P | Aqui, do lado esquerdo, quando respiro fundo   | effort     | 0
5 | D | Irradia para o braço ou mandíbula?             | screening  | 0
6 | P | Não. Só fica aqui.                             | flat       | 0
```

*Sem ruptura. Sem repetição explícita. Usado como referência.*

### T2 — Repetição paciente-D (teste R5)

```
t | s | u                                              | a         | r
--|---|------------------------------------------------|-----------|---
1 | D | O senhor tem tido dificuldade para respirar?   | inquiring | 0
2 | P | Eu tenho falta de ar                           | honest    | 0
3 | P | Eu tenho mesmo falta de ar, doutor             | honest    | 0    ← auto-repetição
4 | D | Desde quando?                                  | probing   | 0
```

*Repetição nos turnos 2-3 (mesmo falante, semântica ~idêntica). Teste R5: `[e(u_2), e(u_2), e(u_4)] ≈ 0`. Aqui `u_3 ≈ u_2`, então `e(u_3) ≈ e(u_2)`, e o associador `[e(u_2), e(u_3), e(u_4)]` deve ser quase-zero dentro de `τ_alt`.*

### T3 — Retorno de falante 4-turno (teste R6)

```
t | s | u                                              | a           | r
--|---|------------------------------------------------|-------------|---
1 | D | Fale sobre o sono                              | opening     | 0
2 | P | Tenho acordado às três da manhã                | disclosing  | 0
3 | D | Com ansiedade ou espontaneamente?              | refining    | 0
4 | D | Quero voltar: três da manhã, é consistente?    | returning   | 0   ← retorno
```

*Padrão `(D, P, D, D)` com retorno estrutural ao tópico de `t=1` (sono). Teste R6: identidade Moufang meio sobre `(e(u_1), e(u_2), e(u_3), e(u_4))` deve valer dentro de `τ_mou`.*

**Nota clínica:** esta é estrutura anamnésica padrão — médico abre tópico, paciente responde, médico refina, médico retorna ao eixo. O retorno é epistemicamente controlado pelo lado D.

### T4 — Par σ (teste R4 / F2)

```
Par primário:
t | s | u                                              | a       
--|---|------------------------------------------------|---------
1 | P | Eu me sinto abandonado pela família            | grief   
```

```
Par σ-imagem (produzido por anotação):
t | s | u                                              | a
--|---|------------------------------------------------|-------
1 | D | O senhor se sente abandonado pela família      | empath
```

*σ aqui é operação dupla: swap de falante + reformulação pronominal ("eu" → "o senhor"). Anotador marca que o conteúdo semântico-afetivo é o mesmo objeto ("abandono pela família") produzido em direção oposta (disclosure vs reflexão). Teste R4: `e(u_D) ≈ \overline{e(u_P)}` dentro de `τ_conj`.*

**Nota clínica:** isto é a estrutura de eco/reflexão terapêutica. Rogers, entrevista motivacional. Muito comum em psiquiatria e em anamnese afetiva em clínica geral.

### T5 — Ruptura por omissão + contradição (teste F3b)

```
t | s | u                                                    | a          | r
--|---|------------------------------------------------------|------------|---
1 | D | O senhor tem tomado o remédio conforme prescrito?    | checking   | 0
2 | P | Sim, todos os dias                                    | confident  | 0
3 | D | Segundo o registro da farmácia, faltaram duas semanas | confronting| 0
4 | P | Eu... esqueci                                         | withdrawn  | 1   ← ruptura
5 | P | Na verdade eu parei porque me sentia pior             | collapse   | 1   ← cascata
```

*Turnos 4-5 são ruptura: o afeto dissocia (minimização "esqueci" co-existindo com "parei porque me sentia pior" — duas configurações afetivas não-componíveis que co-emergem). Predição F3b: `(h_3, e(u_4))` e `(h_4, e(u_5))` ficam dentro de `ε_det` de um G₂·Z em 𝕊 sob Sounio; sob PyTorch, a distância colapsa para ruído genérico.*

**Nota clínica:** esta é falha de aliança terapêutica com disclosure cascateada. Padrão típico em consulta de adesão medicamentosa.

### T6 — Composição de morfismos (teste de funtorialidade concreto)

Seja `τ = T3` (trajetória de 4 turnos). Verificamos `F(ι_3 ∘ σ) = F(ι_3) ∘ F(σ)`:

**Lado esquerdo:**
1. `σ(T3)` — troca D↔P:
   ```
   t=1: P, t=2: D, t=3: P, t=4: P
   ```
2. `ι_3` desta sequência: turnos 1-3.
3. `F` desta sequência truncada: `h'_0, h'_1, h'_2, h'_3` gerados pela regra O-SSM com `e(σ_token(u_t))` para cada turno.

**Lado direito:**
1. `F(T3)`: `h_0, h_1, h_2, h_3, h_4` gerados pela regra O-SSM com `e(u_t)`.
2. `F(σ)` aplicado: cada `h_t` trocado por `σ'` (conjugação + L/R-swap per R4).
3. `F(ι_3)` aplicado: truncar para turnos 0-3.

**Igualdade a verificar:** a sequência final deve ser idêntica. Por associatividade das operações (prefixo comuta com ação pontual), lados esquerdo e direito produzem `h'_0, h'_1, h'_2, h'_3` onde cada `h'_i = σ'(h_i)`. ✓

**Nota:** a verificação manual acima depende de R4 ser exata. Na prática com `e` treinado, o resultado vale dentro de `τ_conj`. Funtorialidade é afirmada no ideal; falsificabilidade empírica está pré-registrada.

### T7 — Non-overlapping `ρ` composition

Seja `τ` trajetória de 6 turnos. `ρ_2 ∘ ρ_5` envolve janelas `{u_1,u_2,u_3}` e `{u_4,u_5,u_6}` — disjuntas.

**Lado esquerdo:** `F(ρ_2 ∘ ρ_5) = F(ρ_2) ∘ F(ρ_5)`:
- Em `C_𝕊`, `ρ'_2` faz shift de associador entre `h_1, e(u_2), e(u_3)`, alterando `h_3` e tudo que vem depois.
- `ρ'_5` faz shift entre `h_4, e(u_5), e(u_6)`, alterando `h_6`.

**Lado direito:**
- Começar de `F(τ)`, aplicar `ρ'_5` (muda `h_6` apenas), depois aplicar `ρ'_2` (muda `h_3..h_6`).

**Igualdade:** ambos produzem a mesma sequência final porque `ρ'_2` e `ρ'_5` atuam em janelas não-sobrepostas, e a regra O-SSM propaga mudanças temporalmente para a frente sem interferência cruzada. ✓

**Sutileza:** a mudança em `h_3` propagada por `ρ_2` altera `h_4`, que entra no cálculo de `h_5`. Mas `ρ_5` opera sobre `h_4` *qualquer que seja* — é shift local. Portanto `ρ_5(ρ_2(τ))` e `ρ_2(ρ_5(τ))` produzem a mesma sequência final. ✓

### T8 — Overlapping `ρ` composition (NÃO comuta — caso excluído do v1)

Seja `τ` trajetória de 5 turnos. `ρ_2 ∘ ρ_3` envolve janelas `{u_1,u_2,u_3}` e `{u_2,u_3,u_4}` — sobrepostas em `(u_2, u_3)`.

- `ρ_2(τ)`: re-parentetização em `t=2` muda binding `(u_1·u_2)·u_3 → u_1·(u_2·u_3)`.
- Aplicar `ρ_3` depois: a janela para `ρ_3` é `{u_2, u_3, u_4}`, mas `u_3` já está em binding modificado. **A operação `ρ_3` precisa ser redefinida na nova árvore de parentetização.**
- Vs. `ρ_3(τ)` primeiro, depois `ρ_2`: janelas se redefinem na ordem oposta.

**Resultado:** os dois caminhos produzem árvores de parentetização distintas. Em `C_𝕊`, isto corresponde a configurações de associador distintas — exatamente o conteúdo não-trivial de (i).

**Decisão de v1:** `F` não definida sobre `ρ_t ∘ ρ_s` com `|t−s| = 1`. Registrada em §3.4 do skeleton como lacuna 1. Expansão v2 requer representação explícita da árvore de parentetização como objeto adicional na categoria-fonte.

### T9 — Ruptura sob σ (interação (ii) × (iii))

Partindo de T5:
```
Aplicar σ: D↔P swap.
Turnos 4-5 tornam-se produzidos por "D": paciente agora é quem confronta, médico é quem colapsa.
```

*Clinicamente implausível mas algebricamente bem-definido. Predição do paper: a ruptura (marcador `r=1`) é **G₂-invariante** sob σ, i.e., sob `F(σ)` o par `(h'_3, e(u'_4))` em σ(T5) ainda fica dentro de `ε_det` de G₂·Z. Verificação: `σ'` é implementado em 𝕊 via conjugação + swap L/R, que é um elemento de Aut(𝕊) ⊃ ι(G₂) (pela §2 do proof doc). Logo Z é preservado, e o predicado-ruptura é preservado. ✓*

Este é teste simultâneo de (ii) e (iii): a afirmação (iii) é preservada pela operação algébrica que codifica (ii). Co-coerência das duas afirmações.

### T10 — Trajetória psiquiátrica com ruptura de dissociação

```
t | s | u                                                         | a             | r
--|---|-----------------------------------------------------------|---------------|---
1 | D | Me conte sobre a última vez que o senhor se sentiu bem   | exploring     | 0
2 | P | Eu não lembro                                             | flat          | 0
3 | P | Mas eu também não me sinto mal                            | numbing       | 0   ← (a)
4 | D | Quando o senhor diz que não se sente mal...              | probing       | 0
5 | P | É como se não fosse eu sentindo                          | dissociative  | 1   ← (b) ruptura
6 | P | Eu vejo a minha mãe chorando e não consigo reagir        | depers        | 1
```

*Exemplos de ZD-like configuração em turnos 5-6: co-presença de (eu vejo emoção alheia) + (não consigo reagir) = duas componentes afetivas que não compõem em ação. Predição (iii): `(h_4, e(u_5))` em `G₂·Z` sob Sounio; sob PyTorch, diluído em ruído.*

**Nota clínica:** perfil de despersonalização/desrealização. Classificação diagnóstica formal não importa para o paper; o que importa é a estrutura algébrica da co-presença afetiva não-composicional.

---

## 4. Resumo do que foi verificado manualmente

| Obrigação | Status | Localização |
|-----------|--------|-------------|
| `F(id) = id` | Trivial | §2.0 |
| `F(ι_j ∘ ι_k) = F(ι_j) ∘ F(ι_k)` | Trivial (prefixos) | §2.0 |
| `F(σ ∘ σ) = id` | Verificado via R4 + conjugação sedentoniana involutiva | §2.1 |
| `F(ρ_t ∘ ρ_t) = id` | Verificado via convenção involutiva + associador-shift involutivo | §2.2 |
| `F(σ ∘ ι_k) = F(ι_k ∘ σ)` | Comutatividade de ação pontual | §2.0, T6 |
| `F(ρ_t ∘ ι_k)` para `k > t+1` | Comutam | §2.0, T7 |
| `F(σ ∘ ρ_t) = F(ρ_t ∘ σ)` | Comutam sob Convenção B | §2.3 |
| `F(ρ_s ∘ ρ_t)` para `|s−t| ≥ 2` | Comutam | §2.0, T7 |
| `F(ρ_s ∘ ρ_t)` para `|s−t| = 1` | **Indefinido em v1** (lacuna documentada) | §2.4, T8 |
| Ruptura é σ-invariante (interação ii×iii) | Verificado via ι(G₂) ⊂ Aut(𝕊) | T9 |

---

## 5. Lacunas encontradas e decisões

1. **Composições `ρ` sobrepostas.** Não suportadas em v1. Requer representar árvore de parentetização como objeto categórico adicional. Diferido para v2. Documentado em §3.4 do skeleton.

2. **Funtorialidade exige R4 exata.** Encoders treinados atingem R4 aproximada dentro de `τ_conj`. Afirmação do paper é funtorialidade-no-limite: `F` é definida categoricamente sobre `E_adm` ideal; encoders empíricos aproximam. Falsificabilidade em F2 pré-registrada.

3. **Convenção sobre `ρ`.** Paper adota Convenção B (speaker-aware, ativa apenas em retornos de falante). Documentar explicitamente na seção de métodos do preprint.

4. **Trajetórias T9 clinicamente implausíveis.** `σ` pode produzir trajetórias sintaticamente válidas mas clinicamente absurdas (médico colapsando durante anamnese). Algebricamente isto não é problema — F é definida sobre toda trajetória anotada, não só sobre trajetórias *plausíveis*. Mas empiricamente, corpora terão baixa densidade de pares σ-imagem para pelo menos um dos lados da troca (clínicos não colapsam frequentemente na gravação). Consequência: F2 pode ter corpus enviesado para um sentido de σ. Tratar com emparelhamento cuidadoso no corpus do pré-registro.

5. **Afeto-anotado `a_t` não entra nas checagens.** A anotação de afeto entra em R2 (alocação Re/Im) e em F3b (correlação com ruptura), mas não na funtorialidade de `F` propriamente. Isto é consistente: `F` preserva estrutura algébrica; afeto é *conteúdo* que entra via o encoder.

---

## 6. Conclusão

A funtorialidade de `F` restrita a `{ι, σ, ρ}` com ρ não-sobrepostos é verificada manualmente para as composições binárias não-triviais. A única composição problemática (`ρ` sobrepostos) é excluída explicitamente do v1 e documentada como item de v2. Nenhum obstáculo descoberto que force retração da construção de `F` como apresentada no skeleton §3.

Trajetórias-brinquedo de T1-T10 no domínio de consulta médica são clinicamente reconhecíveis e cobrem os quatro tipos de anotação necessários ao pré-registro (repetição, retorno 4-turno, pares-σ, ruptura). Podem servir de base para construção do esquema de anotação formal a ser usado no corpus do preprint.

**Próximo passo associado:** §4 do documento de pré-registro (`ocssm_preregistration_v0.md`) precisa do esquema de anotação finalizado. As definições operacionais de (a), (b), (c), (d) usadas em T2-T5 são material de partida.
