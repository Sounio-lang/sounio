<!-- docs:meta
topic_id: repo.docs.research.ocssm-annotation-manual-v0
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.ocssm-annotation-manual-v0
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# O-CSSM — Manual de Anotação v0

**Status:** v0, 2026-04-24. Escopo: fechamento operacional do §4 do `ocssm_preregistration_v0.md`. Esquema formal de anotação para as quatro categorias (a)–(d) usadas nas falsificações F1, F2, F3b.

**Domínio do corpus:** consulta médico-paciente (decidido 2026-04-24). Inclui consulta ambulatorial, anamnese psiquiátrica, consulta de adesão medicamentosa, aconselhamento.

**Público:** dois anotadores humanos treinados (nível: estudante avançado de medicina ou clínico) + um adjudicador para casos de desacordo.

---

## 1. Unidade de anotação

**Turno (`t`):** sequência contígua de fala de um único falante, delimitada por mudança de falante. Unidades abaixo de turno (morfemas, entonação) não são anotadas.

**Janela (`W_k(t)`):** intervalo `[t, t+k]` de turnos consecutivos usados para identificar padrões. `k` varia por categoria.

Cada item anotado é uma tupla:
```
(corpus_id, conversation_id, turn_range, category, label, subtype_or_partial, annotator_id)
```

---

## 2. Categoria (a): Auto-repetição (para F1a, R5)

### 2.1 Definição operacional

Dois turnos `u_i, u_j` produzidos pelo **mesmo falante** dentro de janela `W_5(i)` (no máximo 4 turnos de distância) tais que o conteúdo proposicional de `u_j` é substancialmente equivalente ao de `u_i`.

### 2.2 Critérios de inclusão

- Mesmo falante (D→D ou P→P).
- Janela `|j − i| ≤ 5` turnos.
- Ambos `u_i, u_j` têm ≥ 5 palavras de conteúdo (não são back-channels).
- Equivalência proposicional ≥ 80% (julgamento do anotador, ver §2.4).
- Não mediado por clarification de contra-parte (ver exclusões).

### 2.3 Exclusões

- Citação direta de terceiros ou leitura de documento.
- Back-channels: "uh-huh", "sim", "entendi" — não conteúdo bearing.
- Reformulação após pedido de clarificação: isto é *reformulation-after-repair*, não auto-repetição livre. Exemplo:
  ```
  P: "dor no peito"          ← turno i
  D: "dor aguda ou em peso?" ← clarificação
  P: "é uma dor no peito, em peso"  ← reformulação, NÃO repetição
  ```
- Turnos com < 5 palavras de conteúdo.
- Paráfrase que muda polaridade (negação) ou quantificador (todos → alguns).

### 2.4 Tipos de repetição (subtype)

| Subtype | Critério | Uso no F1a |
|---------|----------|------------|
| `exact` | Sequência de palavras idêntica ≥ 80% dos tokens | Inclusão prioritária |
| `high` | Conteúdo proposicional idêntico, formulação diferente (≥ 80% sobreposição semântica, < 80% lexical) | Inclusão |
| `partial` | Sobreposição 50–80% | Anotado mas excluído do cálculo confirmatório |
| `none` | Sobreposição < 50% ou polaridade/quantificador divergente | Não anotar |

### 2.5 Exemplos

**Positivo (exact):**
```
P: "eu tenho falta de ar"
P: "eu tenho mesmo falta de ar, doutor"          ← exact
```

**Positivo (high):**
```
P: "não durmo há três noites"
[1 turno da D]
P: "o sono tá uma coisa terrível essa semana"    ← high
```

**Negativo (reformulação após clarificação):**
```
P: "dor aqui"
D: "aqui onde, o senhor pode mostrar?"
P: "aqui no peito, esquerda"                     ← reformulation-after-repair, NÃO
```

### 2.6 Umbral de fiabilidade

Cohen's κ binário (repetição presente: sim/não) em amostra-piloto de 100 pares candidatos (par-candidato = qualquer par de turnos mesmo-falante dentro de W_5): **κ ≥ 0.70**.

Desacordo > 30% na subcategorização (`exact` vs `high`) é aceitável; apenas o binário entra no instrumento M_alt.

---

## 3. Categoria (b): Retorno de falante em janela 4-turn (para F1b, R6)

### 3.1 Definição operacional

Sequência de 4 turnos consecutivos `(u_t, u_{t+1}, u_{t+2}, u_{t+3})` tal que:
- Falante em `t` e `t+3` é o **mesmo**.
- Conteúdo de `u_{t+3}` **retorna ao tópico** estabelecido em `u_t`, após digressão em `u_{t+1}`, `u_{t+2}`.

### 3.2 Critérios de inclusão

- Padrão de falantes: `(A, B, A, A)` ou `(A, B, B, A)` ou `(A, B, A, B→A)` onde o turno 4 é do falante A.
- Retorno explícito: `u_{t+3}` referencia `u_t` por (i) repetição lexical de termo-chave, (ii) expressão explícita de retorno ("voltando ao que o senhor disse...", "sobre aquele ponto..."), (iii) inferência temática clara do anotador.
- Turnos intermediários `u_{t+1}, u_{t+2}` não retomam o tópico de `u_t` prematuramente.

### 3.3 Exclusões

- Retorno a tópico anterior > 3 turnos atrás (fora de W_4).
- Sequência tem mais de 4 turnos em disputa de tópico antes do retorno.
- Retorno é mero hedge ou marcador discursivo ("de qualquer forma...") sem conteúdo substantivo.

### 3.4 Exemplo

```
t=10 | D | "Me conte sobre o sono"                      ← tópico-sono estabelecido
t=11 | P | "Tenho acordado às três"
t=12 | D | "Com ansiedade?"                             ← refinamento
t=13 | D | "Mas voltando: três da manhã é consistente?" ← retorno explícito ao tópico-sono
```

Classificado como retorno-4-turno com pivô em `t=10`, frame `(D, P, D, D)`.

### 3.5 Exclusão vs (a)

Categoria (b) pode coexistir com (a) na mesma janela: se o retorno do turno 4 repete lexicalmente o turno 1, é ambas. Anotadores marcam ambas independentemente; instrumentos M_alt e M_mou usam sub-populações distintas do corpus.

### 3.6 Umbral de fiabilidade

Cohen's κ binário (4-turn return presente: sim/não) sobre amostra-piloto de 100 janelas candidatas: **κ ≥ 0.70**.

---

## 4. Categoria (c): Pares-σ (para F2, R4)

### 4.1 Problema de disponibilidade natural

Pares σ naturalmente-ocorrentes são raros em corpora spontâneos — exigem duas pessoas diferentes expressando *o mesmo conteúdo* em direções opostas. Duas estratégias:

**Estratégia C1 (natural):** escuta reflexiva em contexto terapêutico. Paciente expressa afeto; terapeuta reformula na forma "você...". Literature Motivational Interviewing, Carl Rogers.

**Estratégia C2 (sintética):** anotador recebe `u` e escreve `u'` invertendo direção do falante. Produz corpus controlado mas artificial.

**Decisão para v1:** usar C1 como fonte primária (validade ecológica). C2 como fallback se corpus C1-adequado não estiver disponível, com ressalva explícita no paper.

### 4.2 Definição operacional (C1)

Par de turnos `(u_i, u_j)` tal que:
- Falante em `i` é P, falante em `j` é D (ou vice-versa).
- `j − i ≤ 3` turnos.
- `u_j` é reformulação-em-segunda-pessoa de `u_i`: mantém conteúdo proposicional, inverte deíxis pronominal e pragmática de produção.

### 4.3 Critérios de inclusão (C1)

- Inversão pronominal: "eu" → "o senhor/você", "me" → "lhe", etc.
- Conteúdo afetivo-semântico ≥ 80% preservado.
- `u_j` é responsivo a `u_i` (não é tópico independente).

### 4.4 Exclusões (C1)

- Pergunta diagnóstica sem reformulação: "onde dói?" não é σ-par de nenhum disclosure.
- Advice-giving: "o senhor deveria..." não é σ-imagem de "eu preciso...".
- Informação factual ("o exame mostrou X") não entra em C1.

### 4.5 Exemplo (C1 positivo)

```
t=20 | P | "Eu me sinto abandonado pela família"
t=21 | D | "O senhor se sente abandonado pela família"   ← σ-par
```

### 4.6 Estratégia C2 (protocolo se usada)

- Anotador recebe `u` e reescreve como `u'` com inversão de deíxis.
- Anotadores diferentes escrevem independentemente; divergência documenta variância natural da operação σ aplicada por humanos.
- Corpus C2 é explicitamente rotulado como `synthetic_sigma_pair` em todas as análises.

### 4.7 Braço aleatório (para M_conj_rnd)

Para cada σ-par `(u_i, u_j)` no corpus, gerar par aleatório `(u_i, u_k)` onde `u_k` é turno aleatório do oposto falante no mesmo corpus com `|i − k|` distribuído uniformemente em `[5, 50]`. Este braço não requer anotação separada; é construído automaticamente.

### 4.8 Umbral de fiabilidade

Cohen's κ binário (é σ-par C1: sim/não) sobre 100 pares-candidatos (par-candidato = qualquer par turno-P seguido por turno-D em W_3): **κ ≥ 0.70**.

Para graduação de equivalência semântica em pares positivos, weighted κ com escala 3-pontos (`alta/média/baixa preservação`): **κ_w ≥ 0.60**.

---

## 5. Categoria (d): Ruptura (para F3b)

### 5.1 Definição operacional

Turno `u_t` no qual o estado conversacional exibe **quebra qualitativa** de coerência semântica, afetiva, ou relacional, manifestando pelo menos um dos quatro subtipos abaixo.

### 5.2 Subtipos (não mutuamente exclusivos)

| Subtype | Critério operacional |
|---------|----------------------|
| `contradiction` | `u_t` nega ou inverte claim explícito em turno anterior `u_s`, `s < t`, sem que o falante tenha recebido nova informação que justifique a mudança |
| `withdrawal` | Falante recusa-se a responder, minimiza, muda de tópico abruptamente, ou emite pausa + resposta truncada após sequência de disclosure |
| `collapse` | Quebra afetiva observável: choro, raiva explícita, dissociação, despersonalização verbal ("não sou eu"), afeto incongruente com conteúdo |
| `cascade` | Disclosure súbito de conteúdo previamente oculto, tipicamente desencadeado por confronto, quebra de defensa, ou momento de insight |

### 5.3 Critérios de inclusão

- Observável a partir do texto (não depende de áudio/vídeo para subtipo primário).
- Mudança qualitativa, não apenas gradual.
- Ancorada em turnos específicos (não afirmação difusa sobre "esta consulta toda").

### 5.4 Exclusões

- Desacordo factual sem componente afetiva-relacional.
- Interrupção técnica (ruído, queda de conexão).
- Disfluência linguística isolada (hesitação, gagueira) sem outro subtype.
- Mudança de tópico por iniciativa coordenada (não-abrupta).

### 5.5 Exemplos por subtype

**contradiction:**
```
t=5  | P | "Tomo o remédio todos os dias"
t=6  | D | "A farmácia registra duas semanas sem retirada"
t=7  | P | "Na verdade eu parei porque me sentia pior"    ← contradiction + cascade
```

**withdrawal:**
```
t=12 | D | "E a bebida, o senhor tem conseguido controlar?"
t=13 | P | "Ah, doutor... isso aí...              [pausa]"  ← withdrawal
t=14 | P | "Vamos falar de outra coisa?"           ← withdrawal reforçado
```

**collapse:**
```
t=8  | P | "Minha mãe morreu na semana passada"
t=9  | P | "E eu não... eu não consigo... [chora]"        ← collapse (afeto observável)
```

**cascade:**
```
t=15 | D | "O senhor mencionou estar só. Sempre foi assim?"
t=16 | P | "Não. Eu tinha um filho. Ele se foi há cinco anos."  ← cascade
```

### 5.6 Referência literária

- Safran, J. D., & Muran, J. C. (1996). *The resolution of ruptures in the therapeutic alliance.* Definições de ruptura confrontativa vs. retirada; base para `contradiction`/`withdrawal`.
- Alliance Negotiation Scale (Doran et al., 2012). Escala operacional.
- Stiles et al. (2004). *Assimilation of problematic experiences.* Modelo de cascata.

### 5.7 Umbral de fiabilidade

Ruptura é notoriamente difícil de anotar. Literatura de rupture terapêutica reporta κ na faixa 0.4–0.7 com protocolos detalhados.

- **Binário (ruptura presente em turno: sim/não):** Cohen's κ ≥ 0.65 sobre amostra-piloto de 200 turnos.
- **Subtype (dado que ruptura=sim):** agreement categórico ≥ 60%; weighted κ ≥ 0.50. Subtype é registrado mas não entra no instrumento confirmatório `AUC_det`, que usa apenas binário.

---

## 6. Protocolo de anotação

### 6.1 Treinamento

1. Dois anotadores + um adjudicador treinados em sessão conjunta de 4h sobre este manual.
2. Pilotam 20 conversações-treinamento (não fazem parte do corpus final).
3. Discussão de desacordos em reunião de calibração.
4. Pilot formal inicia após κ-treinamento ≥ 0.60 nas quatro categorias.

### 6.2 Fluxo de anotação

1. **Primeira passada (independente):** cada anotador anota conversação inteira em isolamento, marcando todas as quatro categorias.
2. **Computação de κ:** após cada 20 conversações, calcular κ; se abaixo do umbral, reunião de calibração.
3. **Adjudicação:** desacordos são resolvidos por adjudicador cujo julgamento é final. Itens adjudicados são rotulados `adjudicated=true` no dataset.
4. **Anotações públicas:** apenas itens onde ambos anotadores concordaram OU adjudicador decidiu entram no dataset confirmatório. Itens de desacordo sem adjudicação são excluídos.

### 6.3 Pilot de fiabilidade (pré-condição para depósito OSF)

- 40 conversações-piloto sob protocolo completo.
- Distribuição: 10 consulta ambulatorial, 10 psiquiátrica, 10 adesão, 10 aconselhamento.
- Todas quatro categorias atingem umbral? ⟹ prosseguir para corpus de confirmação.
- Falha em qualquer umbral? ⟹ revisão do manual antes de depósito.

---

## 7. Escolha de corpus

### 7.1 Candidatos

| Corpus | Tipo | Acesso | Adequação |
|--------|------|--------|-----------|
| Alexander Street Counseling & Therapy Corpus | Terapia transcrita | Licença institucional (USP/PUC possível) | Ideal para F2, F3b |
| Counseling and Psychotherapy Transcripts Collection (PEPWeb) | Terapia | Pago | Alternativa |
| TalkBank MEDIA/ORACLE | Diálogo médico | Público | Limitado em volume |
| IEMOCAP (áudio-anotado de afeto) | Diálogo atuado | Público | Usa-se apenas transcrição; diálogos são performados, não clínicos |
| SimulatedPatient OSCE transcripts | Consulta simulada | PUC-SP/SLM potencialmente obtível | Controlado mas artificial |
| PI-curado (anonimizado, IRB) | Consulta real do autor | Requer IRB + TCLE | Pequeno n; alta qualidade |

### 7.2 Estratégia v1

**Primária:** Alexander Street Counseling & Therapy Corpus (~10.000 transcrições). Se licença institucional obtenível via PUC-SP ou Mandic, este é o corpus. Rico em todas as quatro categorias.

**Secundária / de fallback:** corpus híbrido:
- IEMOCAP transcrito (para categorias a, b — diálogo geral com estrutura).
- OSCE transcripts (para categoria d — rupturas anotáveis em examinandos consistentes com padrões clínicos).
- C2-sintético para pares-σ (corpus criado pelo PI com anotador de par).

**Baseline paramétrico:** MRPC / PAWS apenas para justificar o valor `τ_conj = 0.10` via distância inter-anotador de paráfrase em domínio geral. Não entra no corpus confirmatório.

### 7.3 Limitações a serem documentadas no paper

- Se Alexander Street é inviável e recorre-se ao fallback: o corpus é heterogêneo (terapia + diálogo + consulta simulada), e isto enfraquece a generalização para consulta médica propriamente dita. Paper documenta isto na seção de limitações.
- Se apenas corpus PI-curado é obtível (n pequeno), o paper é reclassificado como *case series* com pré-registro, não estudo populacional. Ainda publicável; escopo explícito.

### 7.4 Tamanho do corpus

Meta v1: 500 conversações anotadas para confirmação (após os 40 piloto). Se corpus primário comporta, usar todas; se fallback, dividir proporcionalmente.

Poder estatístico: para `AUC_det` com efeito predito `ΔAUC = 0.20` (0.75 − 0.55), n ≈ 200 turnos-ruptura-positivos suficiente (α = 0.05, potência 0.80, cálculo padrão para AUC). Corpus de 500 conversações com taxa de ruptura ~5% provê ~500 × 30 × 0.05 = 750 turnos-ruptura-positivos estimados, margem confortável.

---

## 8. Lacunas e riscos abertos

1. **Subjetividade de ruptura.** Mesmo com subtipos explícitos, rupture annotation é operação clínica interpretativa. Mitigação: dois anotadores + adjudicador + treinamento; κ-target modesto (0.65); paper é honesto sobre confiabilidade.

2. **Dependência de corpus Alexander Street.** Acesso não-garantido. Se falhar, v1 pode ter que usar fallback degradado — e isto deve ser disclosed no paper.

3. **C2 (pares-σ sintéticos) podem invalidar F2.** Se só C2 é disponível, F2 testa R4 sob dados construídos pelo próprio framework de anotação, risco de circularidade. Mitigação: resultado de C2 reportado apenas como exploratório; resultado confirmatório de F2 exige C1.

4. **OSCE transcripts clinicamente controlados.** Pacientes simulados têm repertório de ruptura mais estreito (atuadores). Viés potencial; documentar.

5. **Escassez de `(a, b)` (retorno 4-turn) em consulta médica breve.** Consultas ambulatoriais curtas (< 15 min) podem não ter densidade suficiente. Estratégia: priorizar consulta psiquiátrica + aconselhamento no corpus, onde a densidade é maior.

6. **Anotadores bilíngues.** Se corpus misto inglês+português (provável), anotadores precisam ser bilíngues e calibrados em ambas as línguas — custo adicional. Ou: restringir v1 a uma língua só (inglês provavelmente, dada disponibilidade de corpus).

---

## 9. Entregáveis pré-depósito

- [ ] Este manual finalizado (este documento).
- [ ] 40 conversações-piloto anotadas sob protocolo §6.3.
- [ ] κ-piloto calculado para cada categoria, atingindo umbrais §2.6/§3.6/§4.8/§5.7.
- [ ] Corpus final escolhido e licença/IRB em mãos.
- [ ] Tamanho final confirmado (ver §7.4).
- [ ] Este manual depositado em OSF junto com o `ocssm_preregistration_v0.md`.

---

## Referências

- Cohen, J. (1960). *A coefficient of agreement for nominal scales.* Educ. Psychol. Meas. 20, 37–46.
- Safran, J. D., & Muran, J. C. (1996). *The resolution of ruptures in the therapeutic alliance.* J. Consulting Clin. Psychol. 64(3), 447–458.
- Doran, J. M. et al. (2012). *The Alliance Negotiation Scale.* Psychotherapy 49(2), 145–153.
- Stiles, W. B. et al. (2004). *Assimilation of problematic experiences: the case of John Jones.* Psychother. Res. 14(4), 371–384.
- Rogers, C. R. (1957). *The necessary and sufficient conditions of therapeutic personality change.* J. Consulting Psychol. 21(2), 95–103. [base de reflexão terapêutica → pares-σ C1]
- Miller, W. R., & Rollnick, S. (2013). *Motivational Interviewing, 3rd ed.* Guilford. [operacionalização de escuta reflexiva]
- Artstein, R., & Poesio, M. (2008). *Inter-coder agreement for computational linguistics.* Comput. Linguist. 34(4), 555–596. [κ em NLP]
