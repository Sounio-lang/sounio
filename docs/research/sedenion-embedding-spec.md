<!-- docs:meta
topic_id: repo.docs.research.sedenion-embedding-spec
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-embedding-spec
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Especificação — Camada de embedding em 𝕊

**Autor:** Demetrios Chiuratto Agourakis
**Data:** 20 de julho de 2026
**Estatuto:** especificação de arquitetura + desenho experimental. Pré-registro do falseador incluído.

> **Nota de verificação de literatura (2026-07-21).** As três checagens que o corpo marcava como
> `[verificar]`/`[FILL]` de literatura foram resolvidas por busca independente (scite MCP + WebSearch),
> com existência/autoria/venue/DOI confirmados — ver §12. Achado que muda o desenho: **KG embedding
> octoniônico já existe e já deu null** (OctonionE, ConvO/OMult), logo **C3 é reprodução/baseline, não
> construção** (§7). Sedênion KG embedding **não** tem trabalho publicado. Os `[FILL]` remanescentes
> (extração de tipos §5.1, tamanhos de $\mathcal N$) são parâmetros de execução a fixar antes de treinar,
> não citações.

---

## 0. O diagnóstico, em uma frase

Espaço de embeddings é espaço vetorial com produto interno. **Não é álgebra.** Divisor de zero exige multiplicação interna. Enquanto o texto entrar em $\mathbb R^d$ com cosseno, não há produto, e sem produto não há aniquilação a encontrar — em nenhuma camada acima.

O embedding deixa de ser pré-processamento e passa a ser **o objeto**.

---

## 1. A lacuna, e por que ela existe pelo motivo errado

A literatura de *knowledge graph embeddings* já embute em álgebras:

| modelo | álgebra | associativa? | divisores de zero? |
|---|---|---|---|
| ComplEx | $\mathbb C$ | sim | não |
| RotatE | $\mathbb C$ (rotações) | sim | não |
| QuatE | $\mathbb H$ | sim | não |
| **OctonionE** (QuatE, §ext.) · **ConvO/OMult** (ConvHyper) | $\mathbb O$ | **não** | não |
| **proposta** | $\mathbb S$ | **não** | **sim** |

Todas as usadas são **álgebras de divisão**, e a escolha é deliberada: divisor de zero quebra a invertibilidade das relações e é tratado como *defeito*.

> **O octônio já foi tentado — e o resultado publicado é null.** OctonionE (extensão do QuatE, Zhang et
> al., NeurIPS 2019) reporta *"performs equally to QuatE… extending to Octonion space does not give
> additional benefits"*, atribuindo isso à perda de associatividade; ConvO/OMult (Demir et al., ACML 2021)
> repetem multiplicação octoniônica em setup convolucional. **Crucialmente, ambos operam no setup de
> álgebra de divisão:** normalizam $r$ (isometria) e usam perda padrão de link prediction — o setup que a
> §5 desta spec identifica como **incapaz de exprimir aniquilação por construção**. O null octoniônico
> deles não testa a tese aqui; ele *é* o baseline de C3 (§7). O que ninguém fez é (a) sedênion e (b)
> aniquilação-como-conteúdo com $r$ não-normalizada e fator de confiança $\alpha$.

**A tese inverte isso.** Aniquilação não é defeito — é conteúdo representacional: a afirmação de que estes dois não compõem. E $\mathbb S$ é o primeiro nível da torre onde ela existe.

---

## 2. A camada

Entidades $h,t\in\mathbb S^{d}$ e relações $r\in\mathbb S^{d}$ — isto é, $d$ cópias de $\mathbb S$, $16d$ parâmetros reais.

Composição por produto de Cayley–Dickson, componente a componente entre as $d$ cópias:

$$h\otimes r \;=\; \big(h^{(1)}r^{(1)},\ \ldots,\ h^{(d)}r^{(d)}\big).$$

### 2.1 A consequência que não existe em $\mathbb H$

Em ComplEx e QuatE, $h\otimes r=0$ **somente** se $h=0$ ou $r=0$. Em $\mathbb S$, pode ser zero com ambos não-nulos.

Leitura semântica: aplicar $r$ a $h$ produz **nada** — a composição aniquila. Isso é estruturalmente diferente de "a consulta tem escore baixo para todas as caudas": o vetor composto é o elemento zero, que está a distância $\lVert t\rVert$ de **toda** cauda não-nula, uniformemente.

> **Aniquilação é a representação natural de "esta consulta não tem completação", e álgebras de divisão não conseguem exprimi-la.** Nelas só se pode obter "o vetor composto calhou de ficar longe de todas as caudas" — afirmação global aprendida, não propriedade da composição.

### 2.2 Relação não pode ser isometria — e isso é o ponto

Em QuatE normaliza-se $r$ para norma unitária, e a relação age como rotação. **Em $\mathbb S$ isso é impossível**: a álgebra não é composicional, $\lVert xy\rVert\neq\lVert x\rVert\lVert y\rVert$, e nenhuma normalização de $r$ a torna preservadora de norma.

Não contorne. A variação de norma sob aplicação da relação **é** o déficit de conatus da §2.2 do registro, e é sinal, não ruído.

---

## 3. Escore: separar magnitude de direção

O escore ingênuo $-\lVert h\otimes r-t\rVert$ confunde as duas coisas que precisam ser supervisionadas separadamente. Use:

$$\text{score}(h,r,t)\;=\;\underbrace{g\big(\alpha(h,r)\big)}_{\text{confiança}}\ \cdot\ \underbrace{\big\langle \widehat{h\otimes r},\ \hat t\big\rangle}_{\text{direção}}$$

com $\widehat{\cdot}$ a normalização unitária, $g$ monótona crescente, e

$$\alpha(h,r)\;=\;\text{grau de não-aniquilação} \quad\text{(definido em §4)}.$$

Assim, aniquilação significa **confiança zero** e não corrompe o sinal direcional. O gradiente da direção continua informativo mesmo onde a magnitude colapsa.

---

## 4. O grau de aniquilação em forma fechada — o retorno do Objeto A

Aqui a geometria de $\mathbb S$ deixa de ser resultado órfão e vira a implementação eficiente.

Para $x\in\mathbb S$ escrito como $x=(x_0+u,\;x_8+w)$ com $u,w\in\operatorname{Im}\mathbb O\cong\mathbb R^7$:

$$A=\lVert u\rVert^2,\quad B=\lVert w\rVert^2,\quad \gamma=\langle u,w\rangle,\quad C=x_0^2+x_8^2,$$
$$D_1=\lVert x\rVert^2,\qquad q=\sqrt{AB-\gamma^2}=\lVert u\wedge w\rVert,$$
$$\boxed{\ \sigma_{\min}(L_x)=\sqrt{D_1-2q\ }\ }$$

**Sem SVD.** Três produtos internos e duas raízes, $O(1)$ por cópia, $O(d)$ por entidade. Uma SVD $16\times16$ por entidade por passo seria proibitiva; esta forma fechada não é.

Defina então

$$\alpha(h,r)\;=\;\frac{\sigma_{\min}\!\big(L_{h\otimes r}\big)}{\lVert h\otimes r\rVert}\ \in[0,1],$$

invariante de escala, zero exatamente sobre a variedade de divisores de zero.

E a variedade tem caracterização exata (registro §1.1): $D_2=0\iff x_0=x_8=0,\ A=B,\ \gamma=0$ — quatro condições, codimensão 4. Isso permite escrever **distância explícita à variedade** como termo de perda, o que nenhum outro modelo de embedding algébrico pode fazer, porque nenhum tem o conjunto singular em forma fechada.

---

## 5. A perda: aniquilação aprendida, não penalizada

**O problema a evitar.** Sob perda padrão de link prediction, $\lVert h\otimes r\rVert\to0$ derruba o escore de *tudo*, inclusive da cauda verdadeira. A descida de gradiente então **foge** da região de aniquilação. O objetivo padrão embute exatamente o pressuposto "aniquilação é defeito".

$$\mathcal L \;=\; \mathcal L_{\text{link}} \;+\; \lambda_{+}\!\!\sum_{(h,r)\in\mathcal P}\!\!\big[\,\alpha(h,r)\ \text{deve ser alto}\,\big] \;+\; \lambda_{-}\!\!\sum_{(h,r)\in\mathcal N}\!\!\big[\,\alpha(h,r)\ \text{deve ser}\ 0\,\big]$$

### 5.1 De onde vem $\mathcal N$ — a supervisão de aniquilação

**Não use "não observado no grafo".** Grafos são incompletos; a hipótese de mundo aberto torna ausência ≠ impossibilidade, e treinar aniquilação sobre ausência ensina o modelo a aniquilar o que apenas falta.

Use **violação de tipo**: pares $(h,r)$ em que o tipo de $h$ é incompatível com o domínio de $r$. Isso é genuinamente impossível, não meramente não-observado. FB15k-237 tem informação de tipo suficiente para construir $\mathcal N$.

[FILL: procedimento exato de extração de tipos e tamanho de $\mathcal N$; declarar antes de treinar]

### 5.2 Termo de aniquilação

$$\big[\,\alpha\to0\,\big]\;=\;\alpha(h,r)^2
\qquad\text{ou}\qquad
\big[\,\alpha\to0\,\big]\;=\;\operatorname{dist}\big(h\otimes r,\ \mathcal{ZD}\big)^2$$

a segunda usando a caracterização de §4. Testar as duas; a segunda é mais forte porque empurra para a variedade e não apenas para norma pequena.

---

## 6. Não-associatividade: o aninhamento como conteúdo

Em $\mathbb C$ e $\mathbb H$ a composição de caminhos é associativa: $(h\otimes r_1)\otimes r_2=h\otimes(r_1\otimes r_2)$. A ordem de agrupamento não é informação.

Em $\mathbb O$ e $\mathbb S$, é. *"Amigo do (pai de X)"* e *"(amigo do pai) de X"* tornam-se objetos distintos **na álgebra**, sem que ninguém precise codificar a distinção.

Isto é a tese — *não-associatividade como regra do conhecimento, não metáfora* — posta onde é mensurável.

**Teste:** consultas multi-hop e *path queries*. Benchmarks padrão da área (verificados, §12):

- **Path queries** — Guu, Miller & Liang, *Traversing Knowledge Graphs in Vector Space* (EMNLP 2015): objetivo composicional sobre caminhos, exatamente onde o agrupamento importa.
- **Complex query answering** — Query2Box (Ren, Hu & Leskovec, ICLR 2020) e BetaE (Ren & Leskovec, NeurIPS 2020), com *splits* de consulta sobre FB15k-237 e NELL995.

Este é o teste **mais forte** da tese, mais que link prediction simples.

---

## 7. Desenho fatorial — isola qual propriedade importa

| condição | álgebra | associativa | div. de zero |
|---|---|:--:|:--:|
| C1 | $\mathbb C$ (ComplEx) | ✓ | ✗ |
| C2 | $\mathbb H$ (QuatE) | ✓ | ✗ |
| **C3** | $\mathbb O$ | ✗ | ✗ |
| **C4** | $\mathbb S$ | ✗ | ✓ |

**C3 é o controle que decide.** Sem ele, um ganho de $\mathbb S$ é atribuível a não-associatividade *ou* a aniquilação, e a tese não se distingue. Com ele:

- C4 > C3 > C1,C2 → ambas as propriedades contribuem;
- C4 ≈ C3 > C1,C2 → é não-associatividade, aniquilação não acrescenta;
- C4 > C3 ≈ C1,C2 → **é a aniquilação**, que é a tese forte;
- todas equivalentes → a álgebra não importa.

**Pareamento obrigatório:** igualar **contagem total de parâmetros**, não $d$. $\mathbb S$ tem 16 reais por unidade, $\mathbb H$ tem 4, $\mathbb C$ tem 2. Comparar $d$ igual é dar $8\times$ mais parâmetros a $\mathbb S$ e o resultado não valeria nada.

> **C3 é reprodução, não construção (verificado 2026-07-21).** KG embedding octoniônico já está publicado —
> **OctonionE** como extensão no paper do QuatE (Zhang et al., NeurIPS 2019) e **ConvO/OMult** (Demir et
> al., ACML 2021). Consequência dupla: (i) C3 deve **reproduzir** OctonionE/ConvO, não ser apresentado como
> construção original; (ii) o resultado publicado deles é **octônion ≈ quatérnion, sem ganho** — mas obtido
> no setup de *álgebra de divisão* ($r$ normalizada, perda padrão), que a §5 mostra não conseguir exprimir
> aniquilação. Portanto o null octoniônico prévio **é o baseline esperado de C3** e transfere o peso do
> desenho para o contraste **C4 vs C3**, que é onde a aniquilação — a razão de $\mathbb S$ e não $\mathbb O$
> — de facto se testa. Isso *afia* a tese em vez de enfraquecê-la, mas obriga a citar o precedente e a não
> reivindicar C3 como novo.

---

## 8. Benchmarks e métricas

- **Link prediction:** FB15k-237, WN18RR. MRR, Hits@1/3/10, filtrado.
- **Multi-hop / path:** path queries de Guu et al. (2015); complex query answering de Query2Box (2020) e BetaE (2020) — ver §6.
- **Diagnóstico próprio da tese:** distribuição de $\alpha(h,r)$ sobre $\mathcal N$ (violações de tipo) versus $\mathcal P$. Separação nítida é o que mostra que a aniquilação foi *aprendida* e não apenas permitida.

---

## 9. Falseador — pré-registrado

**Qualquer um basta:**

1. **C4 ≈ C3 ≈ C2 ≈ C1 a parâmetros pareados.** A álgebra não importa, e a tese cai no seu próprio terreno, com o desenho que ela mesma escolheu.
2. **$\alpha$ não separa $\mathcal N$ de $\mathcal P$** após treino. Aniquilação foi permitida mas não aprendida; a estrutura está disponível e o modelo não a usa.
3. **C4 > C1,C2 mas C3 ≈ C4.** O ganho é não-associatividade; a aniquilação — o argumento pelo qual $\mathbb S$ e não $\mathbb O$ — não acrescenta nada.
4. **O ganho desaparece ao pareá-lo por parâmetros.** Era capacidade.

Reportar o negativo como negativo. São nove; um décimo não muda a natureza do trabalho.

---

## 10. Notas práticas

- **Inicialização.** Longe da variedade $\mathcal{ZD}$ (isto é, $\alpha$ inicial alto), senão o modelo começa aniquilando tudo e o gradiente direcional morre. Verificar a distribuição de $\alpha$ na init.
- **Estabilidade.** $q=\lVert u\wedge w\rVert$ é não-suave em $q=0$ (todos os $\sigma$ coalescem). Mesmo problema do probe, mesma solução: subgradiente declarado ou suavização $q_\delta=\sqrt{q^2+\delta^2}-\delta$, com análise de sensibilidade em $\delta$.
- **Custo.** Produto de Cayley–Dickson $16\times16$ é barato, e o compilador com aritmética exata em tensor cores (Objeto D) já existe. Esta é a primeira aplicação em que ele é **necessário** e não incidental.
- **Não normalizar $r$.** Ver §2.2.

---

## 11. Por que esta é a primeira coisa desta linha que testa a tese

Os nove negativos anteriores perguntaram se objetos padrão **têm** a estrutura. Todos compunham aditivamente, e a condição necessária (registro §A.6) já dizia que não podiam tê-la.

Esta é a primeira construção que **põe** a estrutura e pergunta se ela faz trabalho. Se der negativo, será o primeiro negativo sobre a tese em vez de sobre o terreno.

---

## 12. Referências (verificadas 2026-07-21)

> **Método.** Existência/autoria/venue/DOI confirmados por busca independente. QuatE e ConvO passaram
> pelo scite (checagem de retração — `retraction_notices` ausente, sem retração); os demais são
> WebSearch-grounded com DOI/identificador lido direto do publisher. Formato Vancouver.

**Álgebras hipercomplexas em KG embedding:**

1. Trouillon T, Welbl J, Riedel S, Gaussier É, Bouchard G. Complex embeddings for simple link prediction. In: Proc. 33rd Int. Conf. on Machine Learning (ICML); PMLR 48; 2016. p. 2071–80. arXiv:1606.06357. *(ComplEx, C1)*
2. Sun Z, Deng Z-H, Nie J-Y, Tang J. RotatE: knowledge graph embedding by relational rotation in complex space. In: Int. Conf. on Learning Representations (ICLR); 2019. arXiv:1902.10197.
3. Zhang S, Tay Y, Yao L, Liu Q. Quaternion knowledge graph embeddings. In: Advances in Neural Information Processing Systems 32 (NeurIPS); 2019. arXiv:1904.10281. doi:10.48550/arXiv.1904.10281. *(QuatE = C2; **OctonionE** introduzido como extensão neste paper — o baseline de C3)*
4. Demir C, Moussallem D, Heindorf S, Ngonga Ngomo A-C. Convolutional hypercomplex embeddings for link prediction. In: Proc. 13th Asian Conf. on Machine Learning (ACML); PMLR 157; 2021. p. 656–71. arXiv:2106.15230. doi:10.48550/arXiv.2106.15230. *(QMult/**OMult**/ConvQ/**ConvO** — multiplicação octoniônica; parte do baseline de C3)*

**Consultas compostas / multi-hop (§6):**

5. Guu K, Miller J, Liang P. Traversing knowledge graphs in vector space. In: Proc. 2015 Conf. on Empirical Methods in Natural Language Processing (EMNLP); 2015. p. 318–27. doi:10.18653/v1/D15-1038. arXiv:1506.01094.
6. Ren H, Hu W, Leskovec J. Query2box: reasoning over knowledge graphs in vector space using box embeddings. In: Int. Conf. on Learning Representations (ICLR); 2020. arXiv:2002.05969.
7. Ren H, Leskovec J. Beta embeddings for multi-hop logical reasoning in knowledge graphs. In: Advances in Neural Information Processing Systems 33 (NeurIPS); 2020. arXiv:2010.11465.

---

## Divulgação de uso de IA (GAIDeT / ICMJE 2025)

- **Claude Fable 5 (Claude Code)** — verificação de literatura das §1/§6/§7 (existência de KG embedding
  octoniônico/sedeniônico; benchmarks de path/complex-query) via scite MCP + WebSearch independente;
  resolução dos marcadores `[verificar]`/`[FILL]` de citação; formatação e registro deste documento.

O autor revisou, verificou e assume responsabilidade integral pelo conteúdo, incluindo o desenho experimental, todos os resultados numéricos e sua interpretação.
