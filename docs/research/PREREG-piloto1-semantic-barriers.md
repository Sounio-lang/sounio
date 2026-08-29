<!-- docs:meta
topic_id: repo.docs.research.prereg-piloto1-semantic-barriers
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.prereg-piloto1-semantic-barriers
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Pré-registro — Piloto 1: existem barreiras em campos semânticos reais?

**Autor:** Demetrios Chiuratto Agourakis
**Data:** 20 de julho de 2026
**Estatuto:** pré-registro. Escrito antes de rodar. Nenhum dado tocado.
**Ética:** nenhum sujeito humano, nenhum dado clínico, nenhum dado de rede social. Corpus público arquivístico. **Não requer CEP** — e não licencia nenhuma afirmação clínica.

> **Nota de registro (2026-07-20).** Os únicos `[FILL]` bloqueantes — termos de uso do arquivo e
> identificação das séries por tamanho — foram resolvidos *antes* deste registro e *antes* de qualquer
> contato com os dados, por verificação direta em dreambank.net. Ficam abaixo, em §2. Os `[FILL]`
> remanescentes são parâmetros de execução (modelo de LM, custo de permutação) fixados aqui e reportados
> tal como fixados. Este documento é o carimbo temporal: nenhuma análise foi executada até seu merge.

---

## 1. A pergunta, e o que ela decide

$$\textbf{Campos semânticos reais têm barreira?}$$

Formalmente: existe $c$ tal que o subnível $\{s\le c\}$ seja **desconexo**, acima do que se obtém por acaso?

### Por que esta pergunta e não outra

O teorema de conexidade (registro §1.4) mostrou que em $\mathbb S$ **todos** os subníveis são conexos, logo $c^*(A,B)=\max\{s(A),s(B)\}$ e não há passo de montanha. Isso demoliu, *naquele objeto*, a distinção entre sofrimento necessário e gratuito.

A defesa foi que campos clínicos e de decisão "quase certamente têm barreira" — não se chega da evitação à recuperação sem atravessar distresse. **Essa defesa é conjectura e nunca foi testada.** Ela sustenta as §A.2 e §A.3 do registro, isto é, o núcleo formal inteiro do Mercyful Learning.

| desfecho | consequência |
|---|---|
| há desconexão acima do nulo | passo de montanha existe em domínio real; $c^*$ é estimável; a distinção necessário/gratuito ganha referente empírico pela primeira vez |
| tudo conexo, em todas as definições de campo | **a distinção é vacuosa em geral, não só em $\mathbb S$**; o núcleo formal perde a sua peça central e o arcabouço tem de ser reformulado sem ela |

É o falseador mais forte que resta ao programa, e custa uma tarde.

---

## 2. Corpus

**DreamBank** (dreambank.net) — relatos oníricos públicos, arquivísticos, com codificação padronizada Hall/Van de Castle.

**Termos de uso (verificados em dreambank.net, 2026-07-20):** todo o conteúdo do DreamBank.net está licenciado sob **Creative Commons BY-NC-SA 4.0** — *"You are welcome to use the data… you must give attribution to dreambank.net, and commercial use is prohibited."* Uso de pesquisa **não-comercial com atribuição** é explicitamente permitido; este piloto é não-comercial e arquivístico e cita a fonte. Compatível.

Escolha motivada, não de conveniência: relatos de sonho exibem a estrutura formal isolada na discussão anterior — conteúdo gerado internamente e experimentado como externo, com falha de monitorização da fonte. O sonho como modelo natural de psicose é linha estabelecida. Mas ver §7: **isto não é psicose**, e o piloto não afirma nada sobre ela.

Duas amostras (identificadas pelo inventário de séries do arquivo, por tamanho):

- **A — série longa de sujeito único.** **Barb Sanders**, $n = 3{,}116$ relatos (em inglês; a série de
  sujeito único mais estudada da tradição Hall/VdC/Domhoff, com codificação padronizada e maior *baseline*
  documentada). Densidade de amostragem do espaço semântico de um único gerador. *Alternativa de robustez,
  maior:* Izzy (*all*), $n = 4{,}329$ — home dream series adolescente contínua; se A der positivo, repetir
  em Izzy separa "propriedade de Barb Sanders" de "propriedade de séries longas de sujeito único".
- **B — corpus agrupado entre séries.** **Hall/VdC Norms** combinadas (Female $490$ + Male $491$ = $981$
  relatos) — o corpus normativo canônico, muitos geradores. Testa se a resposta depende de ser um só gerador.

Corpus de comparação, se houver tempo: qualquer corpus narrativo não-onírico de tamanho comparável (por exemplo prosa de domínio público — Project Gutenberg — segmentada em unidades equivalentes), para verificar se a resposta é específica do material ou genérica de texto.

---

## 3. Definições de campo

**A resposta pode depender da definição.** Isso não é problema a esconder: é resultado a reportar. Três campos, todos computados, e a robustez da conclusão entre eles é parte do achado.

### 3.1 Déficit de informação mútua (primário)

$$s_{\text{PMI}}(u_{t}) \;=\; -\log\frac{P(u_{t}\mid u_{t-1})}{P(u_{t})}$$

Alto quando o contexto anterior **destrói** a previsibilidade da unidade seguinte, e não apenas quando não a informa. É o análogo textual mais próximo de $\sigma_{\min}$: mede falha de composição, não distância.

> **Razão de ser o primário.** A condição necessária (registro §A.6) diz que sob composição **aditiva** a aniquilação é estruturalmente inacessível. *Embeddings* de sentença compõem por soma/pooling — um campo construído sobre distância euclidiana entre embeddings herda essa aditividade e não pode exibir aniquilação por construção. PMI é razão de probabilidades, não soma de vetores. É a única das três definições que não está morta antes do experimento por esse motivo.

Implementação (fixada): LM causal aberto **GPT-2** (`gpt2`, 124M, via `transformers`), rodando em CPU;
$u_t$ = **sentença** (unidade idêntica à dos nós, §4). $P(u_t\mid u_{t-1})$ e $P(u_t)$ estimados como
verossimilhança da sentença sob o LM com e sem a sentença anterior no contexto (média por token,
comprimento-normalizada). Modelo e versão reportados no resultado; a robustez a um segundo LM aberto
(p.ex. `distilgpt2`) entra como checagem se houver tempo.

### 3.2 Curvatura de Ollivier–Ricci (secundário)

Sobre o grafo semântico construído em §4. Já validado no seu programa contra transições tipo-psicopatologia; entra como campo com precedente empírico próprio.

### 3.3 Salto semântico local (controle de campo trivial)

$$s_{\text{jump}}(u_t)=1-\cos\big(e(u_t),e(u_{t-1})\big)$$

Incluído **explicitamente como campo aditivo/trivial**. Previsão: não deve exibir barreira. Se exibir, é sinal de que o método de conectividade está detectando artefato de esparsidade e não estrutura — ver §6.

---

## 4. Construção do grafo

- Nós: unidades embutidas — **sentença** como unidade (idêntica à de §3.1), embutida por um modelo de
  *sentence embedding* aberto (`all-MiniLM-L6-v2`, 384-d). Unidade e embedding **idênticos entre os três
  campos**; só o valor nodal $s$ muda.
- Arestas: $k$-vizinhos mais próximos no espaço de embedding, mútuos. $k\in\{5,10,20\}$ — **varrer, não fixar**; a conectividade de grafo kNN depende de $k$ e a conclusão tem de sobreviver à varredura.
- Grafo idêntico para os três campos. Só o valor nodal muda.

---

## 5. Estatística

Varredura de limiar com union-find (reaproveitar o código do #1254):

Para cada $c$ crescente sobre o suporte de $s$:

- $N_{\text{comp}}(c)$ — número de componentes conexas de $\{s\le c\}$;
- $\rho(c)=\dfrac{|\text{maior componente}|}{|\{s\le c\}|}$ — **estatística primária**;
- $c_{\text{merge}}$ — limiar em que $\rho$ atinge $1$ (todas as componentes fundidas).

**Assinatura de barreira:** $\rho(c)$ permanece substancialmente abaixo de 1 ao longo de uma **faixa** de $c$, com componentes grandes (não muitas e minúsculas), fundindo-se abruptamente em $c_{\text{merge}}$.

**Assinatura de ausência:** $\rho(c)\approx1$ desde que o subnível tenha tamanho apreciável.

---

## 6. O nulo — e o confundidor que ele existe para matar

### 6.1 O confundidor

**Desconexão por esparsidade não é barreira.** Para $c$ pequeno o subnível tem poucos nós, e poucos nós num grafo kNN desconectam-se trivialmente. Sem nulo, qualquer campo produz "barreira" perto do mínimo.

### 6.2 Nulo primário: embaralhamento do campo

Permutar os valores de $s$ entre os nós, mantendo o grafo intacto.

Preserva exatamente: a topologia do grafo, a distribuição marginal de $s$, o tamanho de cada subnível em cada $c$.
Destrói apenas: **a correspondência entre valor do campo e posição no grafo.**

É o mesmo princípio do *orientation scramble* do probe: preservar todo o conteúdo local, destruir só a correspondência geométrica. É o nulo condicional do mecanismo.

- $B=200$ permutações.
- Estatística pareada: $\Delta(c)=\rho_{\text{obs}}(c)-\operatorname{mediana}_b\rho_b^{\text{perm}}(c)$.
- $p$ empírico $=\dfrac{1+\#\{\rho_b^{\text{perm}}(c)\le\rho_{\text{obs}}(c)\}}{B+1}$, com correção para a varredura sobre $c$ **e** sobre $k$ (o nulo realiza a mesma varredura e o mesmo mínimo).

### 6.3 Nulos secundários

- **Grafo embaralhado** (rewiring preservando grau) com campo intacto — separa estrutura do campo de estrutura do grafo.
- **Campo trivial** (§3.3) — deve dar nulo. Se der barreira, o pipeline está errado.

---

## 7. Previsões pré-registradas e falseador

**Confirmatório:**

1. $\Delta(c)<0$ significativo numa faixa contígua de $c$, sob $s_{\text{PMI}}$;
2. estável na varredura de $k$;
3. componentes grandes, não pulverizadas;
4. presente na amostra A **e** na B;
5. **ausente** no campo trivial $s_{\text{jump}}$.

**Falseador — qualquer um basta:**

- $\rho_{\text{obs}}$ dentro do envelope do nulo permutado em todo $c$, para as três definições de campo → **não há barreira**; a distinção necessário/gratuito é vacuosa em campos semânticos reais e o núcleo formal do Mercyful Learning tem de ser reescrito sem ela;
- barreira presente também em $s_{\text{jump}}$ → artefato de esparsidade ou erro de pipeline, resultado descartado;
- barreira em A e não em B (ou vice-versa) → não é propriedade do campo, é do gerador ou da amostragem; reportar como tal e não generalizar;
- desaparece ao variar $k$ → artefato de construção de grafo.

**Reportar o negativo como negativo.** São oito até aqui e todos valeram mais que os positivos.

---

## 8. Escopo — o que este piloto NÃO estabelece

- **Não é sobre esquizofrenia.** Sonho não é psicose. Nenhuma inferência clínica é licenciada.
- **Não é sobre sofrimento.** O campo mede falha de composição informacional, não estado afetivo nem distresse.
- **Não estabelece que trajetórias reais atravessem a barreira.** Existência de barreira no campo $\ne$ necessidade de travessia por trajetórias observadas. Essa é a segunda pergunta, exige desfecho, e exige CEP.
- **$n$ de séries é pequeno.** A conclusão é sobre a geometria destes campos, não sobre campos semânticos em geral.

O que ele estabelece, se positivo: que a premissa geométrica do arcabouço tem **ao menos um** referente empírico fora de $\mathbb S$ — o que hoje não tem.

---

## 9. Dívida paralela, também sem ética envolvida

Independente deste piloto e ainda em aberto desde o O-SSM: **demonstrar recuperação do instrumento em dado com estrutura injetada por construção.** O nulo em ABIDE e em dados de depressão foi conclusivo para os dados e não para a hipótese exatamente por essa falta. Continua sendo pré-requisito de qualquer alegação clínica futura, e é inteiramente sintético.

---

## 10. Entregável

Nota metodológica curta, com o negativo (se for o caso) em destaque e não em rodapé. Se positivo, é a §A.2 do registro deixando de ser conjectura — e a pergunta afiada que justifica uma submissão ao CEP.

---

## Apêndice — Referências (verificadas 2026-07-21)

> **Nota de verificação.** As entradas abaixo fundamentam afirmações de literatura do corpo do
> pré-registro e foram **acrescentadas após o registro** (não fazem parte do corpo carimbado de 20/07).
> Verificação por checagem independente de título+autor: **existência, autoria, venue e DOI confirmados**;
> retração/correção checada via scite (`retraction_notices` ausente = sem retração) para as entradas 1–3.
> O DOI da entrada 3 foi **corrigido** nesta verificação — uma inferência inicial apontava para
> `10.1038/s41598-019-40871-5` (*Sci Rep*), que o registro real resolve para um artigo não relacionado;
> o correto é *Nat Commun* abaixo.

1. **Sonho como modelo de psicose** (§2, §7) — Scarone S, Manzone ML, Gambini O, Kantzas I, Limosani I,
   D'Agostino A, Hobson JA. The dream as a model for psychosis: an experimental approach using bizarreness
   as a cognitive marker. *Schizophr Bull*. 2008;34(3):515–22. doi:10.1093/schbul/sbm116. *(sem retração)*
2. **Falha de monitorização da fonte** (§2) — Johnson MK, Hashtroudi S, Lindsay DS. Source monitoring.
   *Psychol Bull*. 1993;114(1):3–28. doi:10.1037/0033-2909.114.1.3. *(sem retração)*
3. **Curvatura de Ollivier–Ricci com precedente em psicopatologia** (§3.2) — Farooq H, Chen Y, Georgiou TT,
   Tannenbaum A, Lenglet C. Network curvature as a hallmark of brain structural connectivity.
   *Nat Commun*. 2019;10:4937. doi:10.1038/s41467-019-12915-x. *(sem retração)* — **Ressalva:** o precedente
   firme é conectividade estrutural/envelhecimento e transtornos do neurodesenvolvimento (p.ex. TEA), **não**
   depressão. Ler §3.2 como "precedente em psicopatologia geral", não "em depressão".
4. **Normas Hall/Van de Castle e disponibilização no DreamBank** (§2) — Domhoff GW. *The Scientific Study of
   Dreams: Neural Networks, Cognitive Development, and Content Analysis*. Washington, DC: American
   Psychological Association; 2003.

## Divulgação de uso de IA (GAIDeT / ICMJE 2025)

- **Claude Opus 4.8 (Claude Code)** — verificação dos termos de uso e do inventário de séries do DreamBank
  por tamanho (resolução do `[FILL]` bloqueante de §2), fixação dos parâmetros de execução (§3.1, §4, §6.2),
  formatação e registro deste pré-registro. Implementação e execução do pipeline (pós-registro) sob a mesma
  divulgação.
- **Claude Fable 5 (Claude Code)** — verificação bibliográfica do Apêndice de Referências (2026-07-21) via
  scite MCP + busca independente por WebSearch; correção do DOI da entrada 3. Nenhuma citação foi gerada sem
  recuperação do registro real.

O autor revisou, verificou e assume responsabilidade integral pelo conteúdo, incluindo todos os resultados numéricos e sua interpretação.
