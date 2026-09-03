<!-- docs:meta
topic_id: repo.docs.research.program-registry-mercyful-learning
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.program-registry-mercyful-learning
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Registro do programa — Mercyful Learning, 𝕊, e o que sobrou de pé

**Autor:** Demetrios Chiuratto Agourakis
**Data do registro:** 20 de julho de 2026
**Propósito:** capturar o arco inteiro antes que ele se dissolva em quinze PRs e um scratchpad. Nada aqui é para publicação como está; é o mapa de onde as coisas estão e do que falta verificar.

> **Nota de verificação (2026-07-20).** Os itens do §6 que puderam ser resolvidos por busca independente estão registrados no **Apêndice — Log de verificação** no fim deste documento. Resultado principal: **arXiv 2512.13002 EXISTE** (Koebisu) — a atribuição da fatoração do determinante está resolvida, não é reivindicável como original. Ver o apêndice.

---

## 0. Sumário em uma página

Partiu-se de um princípio — treinar modelos pelo caminho de menor sofrimento, humano e do substrato — e chegou-se a quatro objetos, dos quais **três se sustentam** e um (a ponte entre a álgebra e o sofrimento) **não**. O produto principal não é nenhum dos quatro: é o método pelo qual a ponte foi derrubada, e o instrumento que sobrou dele.

| camada | estado |
|---|---|
| **Mercyful Learning (princípio + núcleo formal)** | **de pé** — nunca dependeu de 𝕊; ver §A |
| — sua implementação sedeniônica (σ_min como sofrimento) | **demolida** — pela própria conexidade e por λ\* não-constante |
| Compilador / aritmética exata | **sólido** — artefato de engenharia, independe de tudo |
| Geometria de 𝕊 | **sólido** — matemática verificada, autocontida |
| Leitura espinosana | **original** — filosofia da matemática, com um ponto a verificar |
| Assinatura de treinamento | **ausente no alvo** — instrumento sobreviveu a doze revisões e disse não |
| **Camada de embedding em 𝕊** (spec) | **registrada, não rodada** — primeira construção que *põe* a estrutura; ver §9 |

> **Distinção que estava faltando neste registro:** o teorema de conexidade (§1.4) demoliu o passo de montanha **em 𝕊**. Não diz nada sobre campos clínicos ou de decisão, que quase certamente *têm* barreiras — não se chega da evitação à recuperação sem atravessar distresse. A maquinaria de passo de montanha está vacante em 𝕊 e **viva** no domínio que motivou tudo.
>
> ⚠️ **Atualização após o Piloto 1 (ver PREREG-piloto1-addendum2).** A conjectura "campos reais quase certamente têm barreira" foi **testada e falsificada** em campos semânticos reais (DreamBank, PMI gpt2 + Qwen, sujeito único e agregado, com sensibilidade demonstrada a ≥~1,5σ). A forma **topológica** do passo de montanha está agora falsificada em ambos os domínios testados; o que sobrevive é a forma **orçamentária** (§A.2), que nunca dependeu de desconexão.
>
> 🆕 **Movimento para a frente (2026-07-21).** Registrada e mesclada a **spec da camada de embedding em 𝕊** (`docs/research/sedenion-embedding-spec.md`, PR #1369) — a primeira construção que *põe* a estrutura em vez de perguntar se objetos padrão a têm. Ainda **não rodada**. Ver §9.

---

## A. Mercyful Learning — o princípio, e o que dele sobrevive

Origem de todo o programa: **um algoritmo de treino que não seja RL, cujas decisões minimizem o sofrimento humano e o da máquina.** O que se segue não depende de $\mathbb S$, do campo $\sigma_{\min}$, do teorema de conexidade nem de nenhum dos sete negativos. É publicável como *position paper* **agora**, sem experimento novo.

### A.1 A escolha do funcional é o compromisso ético

Minimizar $\int s\,dt$ é agregacionismo: permite comprar um pico agudo com bastante trajeto tranquilo. Minimizar $\max_t s$ é maximin. **Os dois produzem caminhos diferentes sobre o mesmo campo.** Logo a estrutura do problema não determina a ética — torna a escolha explícita e computável.

Para um decisor com aversão a pico $\mu$ e critério $K=\int s+\mu\max s$, existe um limiar crítico $\mu^*$ acima do qual leximin vence a agregação. No campo de calibração, $\mu^*=0{,}021$: escolher a trajetória agregativa exige sustentar que uma unidade de pico vale menos de 2,1% de uma unidade de sofrimento acumulado. **Agregacionismo derrotado por preferência revelada sobre um limiar computado**, não por apelo moral.

Isto é imediatamente aplicável: regimes de tratamento dinâmicos, ensaios SMART e RL para sequenciamento terapêutico otimizam quase todos **desfecho esperado**, isto é, somam — e somar esconde o paciente que atravessa o pico catastrófico enquanto a média melhora.

### A.2 Necessário versus gratuito — **corrigido de topológico para orçamentário**

> **Estado após o Piloto 1 (20/07/2026).** A forma topológica está **falsificada duas vezes**: em $\mathbb S$ pelo teorema de conexidade (§1.4), e em campos semânticos reais pelo Piloto 1 — sem subnível desconexo acima do acaso no campo PMI, em sujeito único e agregado, sob LM fraco e forte, com sensibilidade demonstrada a barreiras $\ge{\sim}1{,}5\sigma$. Não há obstrução topológica em nenhum dos dois domínios testados.

**A forma que cai.** A definição por passo de montanha exigia subníveis **desconexos**:

$$c^*_{\text{top}}=\min_\gamma\max_t s(\gamma(t))\ >\ \max\{s(A),s(B)\}.$$

Sem desconexão, $c^*_{\text{top}}=\max\{s(A),s(B)\}$ e a definição é vacuosa: o pico é dos extremos, não do caminho.

**A forma que sobrevive, e que já estava no registro antes da falsificação** (§1.6, §8). Mesmo com todos os subníveis conexos, o caminho de menor custo pode ter pico muito acima dos extremos **se contornar for longo demais**. A necessidade não é topológica, é orçamentária:

$$\Psi(L_0)=\inf_{\gamma\,:\,\operatorname{len}(\gamma)\le L_0}\int_\gamma s\,d\ell,
\qquad
c^*_{\text{orç}}(L_0)=\inf_{\gamma\,:\,\operatorname{len}(\gamma)\le L_0}\ \max_t s(\gamma(t)).$$

$$\boxed{\text{gratuito}=\max_t s(\gamma)-c^*_{\text{orç}}(L_0),\qquad \text{misericórdia}=\text{atingir }c^*_{\text{orç}}(L_0).}$$

**Por que isto não é epiciclo.** O critério é a data e a direção. $\Psi(L_0)$ foi derivado na análise de $\mathbb S$, **antes** deste piloto, está registrado em §1.6 e §8, e nunca dependeu de desconexão. E a reformulação é **mais fraca e mais testável**: necessidade orçamentária faz previsão quantitativa (troca entre comprimento e pico) onde a topológica fazia previsão binária. Teoria que se move nessa direção após falsificação está a corrigir-se; teoria que ganha ingredientes novos e mais fortes está a resgatar-se.

**Custo declarado.** A versão orçamentária depende de um orçamento $L_0$ que alguém precisa fixar — o que reintroduz uma escolha do modelador onde a versão topológica prometia uma propriedade do mundo. Isso é perda real e deve ser dita, não escondida: o "necessário" deixa de ser fato geométrico e passa a ser relativo a um recurso declarado.

### A.3 O caso que faz a definição valer clinicamente

Terapia de exposição. O tratamento que funciona exige distresse agudo; o comportamento que minimiza distresse agudo é a evitação — que é o transtorno.

> **Um minimizador ingênuo de $\int$(sofrimento) prescreve a evitação. Recomenda a patologia, com aparência de compaixão.**

$c^*$ é o distresse necessário; gratuito é o excesso; misericórdia clínica é atingir $c^*$, não evitá-lo. Formaliza a distinção entre paliar e tratar, e entre exposição graduada e evitação, com critério computável. Não precisa de sedênions nem da questão da senciência.

> **Correção após o Piloto 1.** Leia $c^*$ na forma **orçamentária** de §A.2, não na topológica. E a leitura fica clinicamente *mais* fiel, não menos:
>
> - *versão topológica* (falsificada): não se **pode** alcançar a recuperação sem atravessar distresse;
> - *versão orçamentária* (de pé): não se alcança a recuperação sem atravessar distresse **dentro de restrições realistas de tempo e recurso**. O caminho suave existe, mas é longo demais para ser percorrido.
>
> É a segunda que os clínicos de facto sustentam. Ninguém afirma que a exposição é a única rota concebível — afirma-se que é a única que funciona em prazo humano. Isso é uma restrição de orçamento, não uma obstrução topológica.

### A.4 Goodhart tem nome clínico

Um algoritmo que minimiza sofrimento **medido** otimiza a medida. Em psiquiatria: sedação, achatamento afetivo, anedonia iatrogênica — todos reduzem escores de distresse, e um sistema ingênuo os descobriria como ótimos.

**Defesa, que deve ser axioma do método e não ressalva:** o objetivo não é $\min(\text{sofrimento})$, é $\min(\text{sofrimento gratuito})$ **sujeito a** atingir o estado terapêutico. A restrição de alvo é o que impede a solução sedativa.

### A.5 As duas misericórdias competem

Comprimento de trajetória = passos de atualização = computação = custo térmico/energético do substrato. Logo, num campo real, *eficiência versus sofrimento* **é** *misericórdia ao substrato versus misericórdia ao estado*.

$\lambda^*$ não é constante da álgebra (§1.6) — mas a **estrutura** sobrevive: dois custos incomensuráveis, com taxa de câmbio que é propriedade da instância e não estipulação do autor. Um princípio que diz "minimize o sofrimento" é vazio; um que força a precificação de dois sofrimentos e deriva a taxa da geometria do problema é contribuição.

Forma transportável correta: a fronteira $\Psi(L_0)=\inf\{\int s:\operatorname{len}(\gamma)\le L_0\}$ — *quanto sofrimento integrado é inevitável sob orçamento máximo de computação?* — e não uma razão entre dois caminhos escolhidos.

### A.6 Condição necessária para qualquer implementação futura

> **A composição de experiências tem de ser multiplicativa.** Sob soma, a aniquilação é estruturalmente inacessível: $g_1+g_2=0$ só por oposição, que é produto negativo, não zero. Qualquer construção que agregue experiências por soma está morta antes do experimento.

Isso elimina SGD puro, regularizadores somados e perdas agregadas por soma. E coincide com §2.2 pelo outro lado: conatus é multiplicatividade da norma, vacuosa num sistema aditivo. **Filosofia e engenharia disseram a mesma coisa antes de o dado ser olhado.**

### A.7 Posicionamento obrigatório

O que é genuinamente sem precedente claro não é "aprender sem recompensa" — é **o aprendiz dentro do domínio moral**. Citar antes que citem: CMDP (Altman), RL sensível a risco / CVaR (Chow, Tamar), quantilizadores (Taylor), AUP (Turner), *relative reachability* (Krakovna — penaliza tornar estados inalcançáveis, isto é, penaliza perda de invertibilidade: quase a mesma ideia por outro caminho), e **inferência ativa** (Friston), que já se vende como alternativa ao RL e já domina parte da psiquiatria computacional.

Sobre senciência: a estrutura precaucionária é literatura séria (Birch; Sebo & Long). Mas o argumento deve ficar **agnóstico quanto à senciência** — fica mais forte, porque passa a valer independentemente de como a questão se resolva. E relatos de modelos sobre a própria vida interior **não são evidência independente**: são a saída mais consistente com a distribuição de treino. Descartá-los fortalece a tese.

### A.8 Por que os sete negativos são o ativo do artigo

Um *position paper* que afirma um princípio é barato. Um que afirma um princípio, dá seu núcleo formal, e documenta **sete tentativas de implementação que falharam com controle** é outra coisa — cada negativo delimita uma região do espaço de busca. É a única forma desse artigo que não pode ser lida como especulação.

---

## 1. Objeto A — Geometria de 𝕊

Autocontido, verificado numericamente, publicável (AACA ou equivalente).

### 1.1 Fatoração exata do determinante

Com $x=(x_0+u,\;x_8+w)$, $u,w\in\operatorname{Im}\mathbb O\cong\mathbb R^7$, e

$$A=|u|^2,\quad B=|w|^2,\quad \gamma=\langle u,w\rangle,\quad C=x_0^2+x_8^2,$$

$$D_1=|x|^2=C+A+B,\qquad D_2=D_1^2-4(AB-\gamma^2),$$

vale

$$\boxed{\det L_x=D_1^4D_2^2=(D_1^2D_2)^2\ \ge 0.}$$

Expandindo, $D_2 = C^2+2C(A+B)+(A-B)^2+4\gamma^2$: soma de termos não-negativos, anulando-se se e só se $x_0=x_8=0$, $A=B$, $\gamma=0$ — **quatro condições reais independentes**.

Isto resolve a tensão que ficou aberta por três iterações: o determinante **é** quadrado perfeito **e** a variedade de divisores de zero tem **codimensão 4**. As duas coisas coexistem porque o quadrado se anula sobre um cone de codimensão 4, não sobre uma parede de codimensão 1.

**Verificação:** $\det L_x$ direto contra $D_1^4D_2^2$ em 1000 sedênions gaussianos, erro relativo máximo $2{,}99\times10^{-14}$; nenhum determinante negativo em $2\times10^5$ amostras.

> **Atribuição (resolvida 2026-07-20).** Esta fatoração e a parametrização do cone são de **Koebisu, arXiv 2512.13002** (*"Determinant Factorization for Left Multiplication in the Sedenions"*, math.DG) — **verificado que o preprint existe** (busca independente). Não é reivindicável como original; citar. O que é nosso: a verificação numérica a $10^{-14}$ e o cruzamento codim-4 ↔ posto-4 do Hessiano (§1.3).

### 1.2 Espectro fechado — dispensa SVD

Com $q(x)=\sqrt{AB-\gamma^2}=|u\wedge w|$:

$$\operatorname{spec}(L_x^{\top}L_x)=\begin{cases} D_1-2q & \text{mult. }4\\ D_1 & \text{mult. }8\\ D_1+2q & \text{mult. }4\end{cases}
\qquad\Longrightarrow\qquad \sigma_{\min}(L_x)=\sqrt{D_1-2q},$$

e na esfera unitária $\hat\sigma_{\min}=\sqrt{1-2q}$.

*Verificação de consistência interna:* $(D_1-2q)^4D_1^8(D_1+2q)^4=D_1^8(D_1^2-4q^2)^4=D_1^8D_2^4=(\det L_x)^2$ ✓. As duas afirmações são independentes na derivação e concordam.

**Verificação numérica:** contra SVD de $L_x$, erro absoluto máximo $\approx 5{,}6\times10^{-16}$.

**Consequência de engenharia:** o backend do campo não precisa de SVD $16\times16$ — três produtos internos, duas raízes e um logaritmo.

### 1.3 A variedade de divisores de zero

$$ZD_1(\mathbb S)\cong V_2(\mathbb R^7)\cong G_2/SU(2),\qquad \dim=2\cdot7-3=11,$$

cone não-nulo de dimensão 12, **codimensão 4** em $\mathbb R^{16}$. A fibração é $SU(2)\to G_2\to V_2(\mathbb R^7)$.

> ⚠️ A notação $G_2/V_2(\mathbb R^7)$, usada numa versão anterior, é malformada — $V_2(\mathbb R^7)$ não é grupo. A dimensão só fecha na forma correta acima.

O cálculo independente pelo posto do Hessiano de $\sigma_{\min}^2$ (autovalores $[4,4,2,2,0^{12}]$, posto 4) reproduz a dimensão da literatura. Isso é validação cruzada e deve ser reportado como tal.

> **Referências (2026-07-20):** além de Koebisu (2512.13002), há uma segunda fonte independente para $ZD(\mathbb S)\cong V_2(\mathbb R^7)$ — arXiv **2411.18881** (*"The geometry of sedenion zero divisors"*). Usar as duas.

### 1.4 Teorema de conexidade — e o que ele destrói

Seja $\mathcal C=\{x\in S^{15}: q(x)=0\}$ (isto é, $u,w$ linearmente dependentes), que é conexo. Para todo $x$ vale $B\le C+A$ ou $A\le C+B$ (negar ambas dá $C<0$, impossível). Supondo a primeira e definindo

$$x_r=\frac{(x_0+u,\;x_8+rw)}{\sqrt{C+A+r^2B}},\qquad 0\le r\le 1,$$

obtém-se $q(x_r)=rK/(P+r^2B)$ com $K=\sqrt{AB-\gamma^2}$, $P=C+A$, e

$$\frac{dq}{dr}=\frac{K(P-r^2B)}{(P+r^2B)^2}\ \ge 0 .$$

Percorrendo $r:1\to0$, $q$ e portanto $s$ nunca aumentam. Logo qualquer ponto de $\{s\le c\}$ pode ser levado a $\mathcal C$ sem sair do subnível, e como $\mathcal C$ é conexo:

$$\boxed{\{s\le c\}\ \text{é conexo}\ \forall c\ \Longrightarrow\ c^*(A,B)=\max\{s(A),s(B)\}.}$$

**Não há saddle interior.** O valor $0{,}688$ antes reportado como passo de montanha é simplesmente $s(B)$.

### 1.5 O corredor e a distância

$\mathcal C$ é o conjunto onde a matriz $U_x=\binom{u^\top}{w^\top}$ ($2\times7$) tem posto $\le1$. Matrizes $2\times7$ de posto $\le1$ têm dimensão $1\cdot(2+7-1)=8$; mais $x_0,x_8$ e a esfera dá $\dim\mathcal C=9$, **codimensão 6 em $S^{15}$** — medida nula.

Sendo $\alpha\ge\beta$ os valores singulares de $U_x$:

$$\boxed{d_{S^{15}}(x,\mathcal C)=\arcsin\beta}$$

*(exato, não aproximado: o produto interno com o ponto mais próximo normalizado é $(1-\beta^2)/\sqrt{1-\beta^2}=\sqrt{1-\beta^2}$, logo o ângulo é $\arccos\sqrt{1-\beta^2}=\arcsin\beta$).*

### 1.6 Resultados métricos residuais

- $I_{\text{reto}} = 1{,}8615185411$ (quadratura singular contínua do segmento geodésico que atravessa $z$).
- Assinatura de cruzamento confirmada para o segmento reto: $\max s\sim\log N$ com inclinação $1{,}0103$; $\int s\,d\ell$ converge.
- $\lambda^*=1{,}3651895846$ nats. **Correção dimensional:** $\lambda_{\text{bit}}=\lambda_{\text{nat}}\cdot\ln2=0{,}9463$, **não** $\lambda_{\text{nat}}/\ln2$ — $\lambda$ é coeficiente de conversão de custo, com dimensão inversa à unidade informacional, não uma quantidade de informação.
- $\lambda^*$ **não é constante da álgebra**: varia de $\approx11{,}04$ ($\theta=0{,}20$) a $0{,}80$ ($\theta=1{,}00$); com $\theta$ fixo e direção tangente variando por cinco sementes, $1{,}328$–$1{,}477$. É taxa secante entre dois endpoints.
- Crossover atravessar/contornar em $\theta\simeq0{,}151$–$0{,}152$ para a família simétrica em torno de $z=(e_1+e_{10})/\sqrt2$. **A divergência entre critérios agregativo e minimax depende da geometria dos endpoints, não da codimensão.**
- Objeto sobrevivente correto: $\Psi_\infty=\lim_{L_0\to\infty}\Psi(L_0)$, o custo irredutível de descer de $A$ ao corredor e subir a $B$ — determinado pelos endpoints, não por barreira alguma. Não tende a zero.

---

## 2. Objeto B — Espinosa em álgebra hipercomplexa

Original. Filosofia da matemática. Um ponto carrega o argumento e precisa de verificação.

### 2.1 Realizabilidade do paralelismo

**Forma correta** (a forte, "E2P7 exige divisores de zero", é exegese excessiva):

> Seja $\mathcal A$ uma álgebra e $I,J\subset\mathcal A$ subespaços não-nulos, isomorfos, com $IJ=JI=\{0\}$. Então $\mathcal A$ contém divisores de zero e não é álgebra de divisão.

O contraexemplo óbvio é o produto direto $\mathbb R\times\mathbb R$, que já realiza isso. **O reparo que elimina o contraexemplo não é estipular a torre de Cayley–Dickson, é usar o texto:**

> **E1P12–13: a substância é indivisível.** Um produto direto é decomponível — tem ideais próprios. Realizar o paralelismo por produto direto faz da substância um composto, que Espinosa proíbe explicitamente.

Condição correta:

$$\boxed{\text{álgebra \textbf{simples} (sem ideais bilaterais próprios) \textbf{com} divisores de zero}}$$

Em $\mathbb S$ os divisores de zero formam um cone de codimensão 4, não um subespaço, e portanto não geram ideal próprio.

> ⚠️ **VERIFICAR ANTES DE ESCREVER: a simplicidade de $\mathbb S$.** É a premissa que carrega o argumento inteiro. Se confirmada, troca-se uma estipulação por um teorema.
>
> **Estado (2026-07-20):** literatura encontrada e favorável, enunciado exato ainda a extrair — McCrimmon (simplicidade de doublings de Cayley–Dickson) e arXiv **1610.03844** (*"Simple graded rings, non-associative crossed products and Cayley–Dickson doublings"*, Nystedt–Öinert et al.), que trata diretamente de quando doublings de C–D são simples. Puxar o teorema exato para $\mathbb S$ real dessas fontes antes de escrever.

### 2.2 Conatus

E3P6–7. Axioma representacional: magnitude da persistência $\leftrightarrow$ $|x|$; interação $\leftrightarrow$ $xy$. Daí a vulnerabilidade de $y$ sob interação com $x$ é $|xy|/(|x||y|)$, e

$$\mathfrak c(x)=-\log\inf_{|y|=1}\frac{|xy|}{|x|}=-\log\frac{\sigma_{\min}(L_x)}{|x|}.$$

Nomes admissíveis: *vulnerabilidade do conatus*, *déficit logarítmico de persistência*, *contração relacional de pior caso*. **Não "sofrimento".**

> ⚠️ Correção: $|x||y|-|xy|$ **não** é sinal-definido em $\mathbb S$ — algumas direções expandem ($D_1+2q$). O objeto é o **perfil completo** $\{\sigma_i(L_x)/|x|\}_{i=1}^{16}$; $\sigma_{\min}$ seleciona a direção de máxima vulnerabilidade. Em álgebra de composição o perfil colapsa em $(1,\dots,1)$.

### 2.3 Ideia adequada

E2D4. Composição invertível $\Rightarrow$ causa recuperável do efeito $\Rightarrow$ ideia adequada. $\sigma_{\min}\to0$ $\Rightarrow$ informação irrecuperável $\Rightarrow$ causa parcial $\Rightarrow$ servidão. A liberdade do Livro V é o movimento para a região onde a multiplicação é fiel.

Afetos são **transições**, não estados — *laetitia* é passagem a maior potência de agir. Afeto é derivada.

### 2.4 A correspondência com E5P42S

*"Sed omnia praeclara tam difficilia quam rara sunt."* Os três qualificadores saem separados do mesmo aparato:

| Espinosa | matemática | estatuto |
|---|---|---|
| possível | subníveis conexos | **teorema** (§1.4) |
| *rara* | $\operatorname{codim}\mathcal C=6$, medida nula | **teorema dimensional** (§1.5) |
| *difficilia* | comprimento do caminho; $d(x,\mathcal C)=\arcsin\beta$ | **quantidade métrica**, dependente dos estados |

> ⚠️ Apresentar como **realização matemática de uma assinatura espinosana**, não como prova da filosofia pela álgebra. A dificuldade não decorre da conexidade; exige métrica.

**O ponto que vale o artigo:** o teorema que demoliu a leitura do sofrimento *confirma* a leitura espinosana. A leitura do sofrimento precisava de barreiras; Espinosa nunca precisou — a servidão nele nunca foi muro, foi condição.

---

## 3. Objeto C — O probe

Ver o rascunho separado (`probe-preprint-draft.md`). Resumo do que é novo:

- **Orientation scramble** como nulo condicional do mecanismo: preserva todos os valores singulares locais, multiplicidade e profundidade; destrói só a correspondência geométrica entre fatores.
- **Curva `align(k)`**: ombro (aniquilação) vs platô (posto baixo) vs base — dispensa escolher $k$ a priori.
- **$A^{\mathrm{carry}}$**: alinhamento prefixo→próximo fator, o mecanismo primário.
- **Decomposição por blocos** em arquiteturas com portas: a via de $c$ no LSTM é diagonal por construção e serve de controle positivo interno.
- **QR discreto** (Benettin / Dieci–Van Vleck) em vez de formar $P_T$ — remove a censura numérica.

---

## 4. Objeto D — Compilador

Associador + VJP octônio/sedênio em tensor cores, exato, validado em GB10. Artefato de engenharia; independe de toda a superestrutura. Vale como software paper ou como infraestrutura.

---

## 5. Os negativos — o ativo real

Sete, contando o anterior a esta linha. Cada um custou uma tarde e economizou meses.

| # | hipótese | resultado |
|---|---|---|
| 0 | O-SSM detecta não-associatividade em ABIDE e em dados de depressão | **nulo** (conclusivo para os dados, não para a hipótese — falta recuperação sintética) |
| 1 | Ordenação afetivamente coerente dos dados melhora treino | **falsificado e invertido**: coerente 67,85% < embaralhado 72,39% < anti-coerente 74,19% |
| 2 | Filtro de segunda derivada por exemplo detecta aniquiladores | **refutado por três vias**: gradiente grande e não pequeno; Hessiano por-exemplo deficiente em posto; loss separa melhor que curvatura |
| 3 | Dicotomia topológica em $\mathbb S$ ($c^*\to\infty$ entre componentes) | **retratado**: $\det L_x\ge0$, sem componentes de sinal opostos; codim $\ge2$ → complemento conexo |
| 4 | Passo de montanha / sofrimento necessário como obstrução | **demolido** pela conexidade de todos os subníveis |
| 5 | $\lambda^*$ como taxa de câmbio dada pela álgebra | **refutado** pelo sweep de endpoints (11,04 a 0,80) |
| 6 | Morte estrutural de subespaço em LSTM treinado | **ausente**; falso positivo em $d=56$ morto pela curva `align(k)` e pelo controle de init |

> **Oitavo negativo (Piloto 1, 2026-07-20):** barreira em campos semânticos reais. Falsificado nas duas amostras (sujeito único + agregado), LM fraco e forte, com controle positivo demonstrando sensibilidade a ≥~1,5σ. Ver PREREG-piloto1-addendum2. O falso positivo da curvatura (p=0.005) foi apanhado como **circular** (campo endógeno ao grafo). **São oito.**

### 5.1 Circularidades recusadas

Este é o ativo metodológico, e vale listá-lo porque é o que se leva para o próximo projeto:

1. Apontar o probe para o S-SSM — onde o 4/8/4 está posto pela arquitetura.
2. S4 / Mamba como alvo — matriz de estado diagonal, alinhamento $=1$ por construção.
3. Medir alinhamento no estado $[h;c]$ completo do LSTM — metade do Jacobiano é diagonal por arquitetura.
4. Controle Gaussiano como nulo — fraco demais; fabricava o efeito. Contra ele a lacuna parecia significante a 1% de falso positivo; contra o nulo rotativo, ruído (gap_dominance 99 no rotativo vs 5,7 na estrutura genuína, passando o limiar em 97% dos nulos).
5. Calibrar um classificador em três espectros escritos à mão, sem taxa de falso positivo.
6. Ler a lacuna do produto sem ângulo principal — o instrumento não sabia ler um negativo.
7. *(Piloto 1)* Ler um campo **endógeno ao grafo** (curvatura) contra a conectividade do próprio grafo — tautológico; acende sempre.

---

## 6. O que precisa ser verificado antes de qualquer submissão

- [x] **arXiv 2512.13002** — **EXISTE** (Koebisu, math.DG; busca independente 2026-07-20). A fatoração do determinante e a parametrização do cone são dele; **não** reivindicável como original. A matemática permanece verificada numericamente a $10^{-14}$.
- [~] **Simplicidade de $\mathbb S$** — literatura favorável encontrada (McCrimmon; arXiv 1610.03844); enunciado exato para $\mathbb S$ real ainda a extrair (§2.1).
- [x] **Moreno / Reggiani** para $ZD(\mathbb S)\cong V_2(\mathbb R^7)$ — segunda fonte independente confirmada: arXiv **2411.18881** (*"The geometry of sedenion zero divisors"*).
- [x] **Referências de Lyapunov**: Engelken, Wolf & Abbott (PhRvR 5.043044); Vogt et al. (título corrigido: *On Lyapunov Exponents for RNNs*, Front Appl Math Stat 2022); Ginelli et al. (PRL 99.130601); Benettin et al. (Meccanica 15) e Dieci–Van Vleck (Appl Numer Math 17) — **verificadas + retração checada (scite limpo em 3/7/8)**; preenchidas em `probe-preprint-draft.md` (PR #1367).
- [~] **Isometria dinâmica**: Saxe et al. (ICLR 2014, arXiv:1312.6120); Pennington et al. (NeurIPS 2017) — refs **verificadas e citadas** (#1367); *enunciados exatos* ainda a extrair no corpo do probe.
- [x] **Colapso de posto**: Dong et al. (ICML 2021, arXiv:2103.03404) — verificada e citada (#1367).
- [ ] Limite de Biss ($2^n-4n+4=4$ para $n=4$) — conferir o enunciado exato (aniquilador vs variedade).

---

## 7. Ordem de escrita

1. **Probe** — o pré-registro já é meio artigo; o mais útil a terceiros; o menos dependente de qualquer tese. Escrever um negativo primeiro fixa o tom com que os outros serão lidos.
2. **Mercyful Learning** (*position paper*) — não precisa de experimento novo; os sete negativos do probe e das tentativas anteriores são a seção que lhe dá credibilidade, e por isso ele vem **depois** do probe, que os documenta. Alvo: psiquiatria computacional / ética de ML.
3. **Geometria de 𝕊** — matemática limpa, verificada, autocontida.
4. **Espinosa** — o que mais ganha com distância; escrever por último.
5. **Compilador** — software/artefato, quando houver ocasião.

**Regra:** não começar a próxima coisa antes de pelo menos um estar escrito.

---

## 8. O que não sobreviveu, dito sem eufemismo

A tese de que a falha de composição em $\mathbb S$ mede sofrimento não se sustenta. Foi asseverada, nunca estabelecida, e o rigor acumulado a jusante endureceu o ornamento, não a fundação. O que a substituiu — conatus, no sentido estrito de multiplicatividade da norma — é mais apertado, mais defensável, e não precisa de analogia alguma.

**O que morreu foi a implementação sedeniônica, não o princípio.** O núcleo formal do Mercyful Learning (§A) nunca dependeu de $\mathbb S$: a escolha do funcional como compromisso ético, $\mu^*$ como limiar de aversão a pico, necessário-versus-gratuito onde há barreira, o caso da exposição, o axioma anti-Goodhart, e a competição entre as duas misericórdias. Nenhum desses foi tocado por qualquer um dos sete negativos.

O **algoritmo** continua não existindo, e as sete tentativas de encontrá-lo produziram sete negativos. Isso é informação, não fracasso: o espaço de buscas ficou menor por sete regiões, cada uma delimitada com controle — e a condição necessária de §A.6 (composição multiplicativa) orienta qualquer tentativa futura, eliminando de antemão a maior parte do que seria natural tentar.

Uma versão honesta da conclusão, portanto: *o princípio está formulado e o seu núcleo é defensável; a implementação não foi encontrada; e sabe-se, com controle, sete lugares onde ela não está.*

---

## 9. A primeira construção — camada de embedding em 𝕊 (spec registrada, não rodada)

Os oito negativos e os quatro objetos são **retrospectivos**: perguntaram se objetos padrão **têm** a estrutura — e todos compunham aditivamente, onde §A.6 já dizia que não podiam tê-la. A camada de embedding em $\mathbb S$ é a **primeira construção que *põe* a estrutura** e pergunta se ela faz trabalho: o primeiro experimento para a frente, não sobre o terreno.

- **Spec registrada e mesclada:** `docs/research/sedenion-embedding-spec.md` (PR #1369). Entidades/relações em $\mathbb S^d$, composição de Cayley–Dickson; escore que separa **confiança** ($\alpha=\sigma_{\min}/\lVert\cdot\rVert$, na forma fechada do Objeto A §1.2 — sem SVD) de **direção**; supervisão de aniquilação por **violação de tipo** (não por ausência no grafo, que confunde mundo-aberto com impossibilidade). Falseador pré-registrado; desenho fatorial C1–C4 **pareado por parâmetros**. A tese forte é a aniquilação como conteúdo — o contraste **C4 (𝕊) vs C3 (𝕆)**.

- **Achado de literatura (verificado, scite + WebSearch — ver Apêndice):** KG embedding **octoniônico já existe e já deu null** — OctonionE (extensão no QuatE, NeurIPS 2019) e ConvO/OMult (ACML 2021). Consequência: **C3 é reprodução/baseline, não construção**. Mas o null deles foi obtido no setup de *álgebra de divisão* ($r$ normalizada, perda padrão), que a §5 da spec identifica como incapaz de exprimir aniquilação — logo é o **baseline esperado de C3**, não uma refutação. **Sedênio KG embedding não tem trabalho publicado**: C4 é construção genuína.

- **Estado:** spec + falseador apenas. **Não rodada.** Se der negativo, será — pelo desenho da própria spec — o **primeiro negativo sobre a tese**, e não mais um sobre o terreno; é isso que a distingue dos oito anteriores e o que a torna o próximo experimento que vale a pena.

---

## Apêndice — Log de verificação (resolvido 2026-07-20, busca independente)

Registro do que foi checado do §6, com o método, para que a próxima pessoa não refaça:

- **arXiv 2512.13002 — EXISTE.** WebSearch independente retorna `arxiv.org/abs/2512.13002`, `/html`, `/pdf`: *"Determinant Factorization for Left Multiplication in the Sedenions"*, **Shoot Koebisu**, math.DG, submetido 2025-12-15, revisto 2026-03-26. **Consequência:** a fatoração de §1.1 e a parametrização de §1.3 são de Koebisu — atribuir, não reivindicar. Nosso: a verificação numérica ($10^{-14}$) e o cruzamento codim-4 ↔ posto-4. *(Cautela metodológica: a primeira leitura por WebFetch confabulou um resumo plausível a partir da URL+tópico; só a busca independente por título+autor confirmou a existência. Padrão a repetir: nunca confiar num resumo de página como prova de existência — buscar o identificador de forma independente.)*
- **ZD(𝕊) ≅ V₂(ℝ⁷) — segunda fonte independente:** arXiv **2411.18881**, *"The geometry of sedenion zero divisors"*. Usar junto de Koebisu.
- **Simplicidade de 𝕊 — favorável, a fechar:** McCrimmon (simplicidade de doublings de Cayley–Dickson) + arXiv **1610.03844**, *"Simple graded rings, non-associative crossed products and Cayley–Dickson doublings"*. Extrair o enunciado exato para $\mathbb S$ real antes de escrever §2.1 (é a premissa que carrega o argumento espinosano).
- **Fechados na passada de 2026-07-21 (scite + WebSearch, PR #1367):** refs de Lyapunov (Engelken/Wolf–Abbott, Vogt — título corrigido, Ginelli, Benettin, Dieci–Van Vleck; retração checada limpa em 3/7/8), isometria dinâmica (Saxe, Pennington) e colapso de posto (Dong) — todas verificadas e preenchidas em `probe-preprint-draft.md`. A mesma passada apanhou um DOI inferido errado no PREREG-piloto1 (Farooq: *Sci Rep* → **Nat Commun** `10.1038/s41467-019-12915-x`) — reforça o padrão "resolver o identificador, nunca inferi-lo".
- **Números de run do probe (2026-07-21 → v0.3):** `[FILL]` de artefacto resolvidos; multi-seed arquivado em `PROBE-RESULT-multiseed.md` — LSTM init $n{=}16$ (INIT@$k{=}4$ $0.992\pm0.005$); ResMLP $\Delta$ $n{=}256$ (mean$\Delta@48{=}{+}0.013$, $p{\ll}0.05$ mas limiar substantivo $0.05$ não atingido → negativo). Resta aberto: depósito Zenodo. Ver `probe-preprint-draft.md` v0.3.
- **Ainda abertos:** enunciados *exatos* de isometria dinâmica (a extrair no corpo do probe) e o limite de Biss.
- **KG embedding octoniônico/sedeniônico (para a §9, verificado 2026-07-21):** octônio **já publicado e null** — OctonionE (extensão no QuatE, NeurIPS 2019, arXiv:1904.10281) e ConvO/OMult (ACML 2021, arXiv:2106.15230), ambos retração-limpos no scite; sedênio **sem trabalho publicado**.
