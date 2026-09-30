# Dois geradores, a mesma página

Ambos leem os mesmos `artifacts/**/*.json` e o **mesmo CSS** (`web/src/**/*.css`
— o gerador Swift lê os arquivos do projeto React em vez de duplicá-los). A
comparação isola o modelo de geração.

Medido em 4 de setembro de 2026, sobre 523 artefatos.

## O que o navegador baixa para ver `/proof`

| | Swift | React + Vite |
|---|---|---|
| Transferido | **13 KB** gzip | **86 KB** gzip |
| Sem compressão | 183 KB | 368 KB |
| JavaScript | zero | 354 KB |
| Linhas entregues | **as 523** | 86, com "show all" por território |

Seis vezes mais conteúdo por um sexto do peso. Para uma página que é
essencialmente uma tabela longa, não há discussão.

## Build

| | Swift | React + Vite |
|---|---|---|
| Geração completa | 2,6 s | 3,4 s |
| Relink incremental | 1,6 s | — |
| Etapas | evidência → páginas | evidência → portão → tsc → vite |

## As garantias

| | Swift | TypeScript |
|---|---|---|
| Construir um `Claim` fora do módulo | **impossível** — `init` privado, erro de compilação | possível: o módulo usa `as Claim` |
| Prosa com numeral sem lastro | **recusada no construtor**, no ponto de uso | detectada por regex varrendo arquivos, depois |
| Onde o erro aparece | na linha que tentou afirmar | no relatório do portão, com arquivo e linha |

O erro que o Swift dá:

```
error: 'Claim' initializer is inaccessible due to 'private' protection level
```

Não há `as!`, não há literal, não há caminho. Em TypeScript a marca de tipo é
convenção — funciona, e um cast a contorna.

## O que o Swift não faz

- **MDX não tem parser em Swift.** Markdown tem (`swift-markdown`); MDX é
  formato do ecossistema JS. Documentação com componentes embutidos exige JS.
- **Nenhuma ilha interativa.** O filtro de espécie do `/proof`, o alternador de
  tema, os simuladores three.js, o seletor de idioma por página — nada disso
  existe em HTML estático.
- **Sem `zarrita`, sem WebGPU, sem KaTeX.** São bibliotecas JS sem equivalente.

## O que o React não faz

- Não impede fabricar um `Claim`. Impede por convenção, não por compilador.
- Não entrega 523 linhas por 13 KB.
- Não roda sem JavaScript.

## A arquitetura híbrida, medida

Construída: o Swift gera o HTML e as ilhas montam por cima.

| `/proof` | bruto | gzip | linhas entregues |
|---|---|---|---|
| **Híbrido** (HTML + ilha) | 186 KB | **14,8 KB** | **523** |
| SPA React | 368 KB | 86,1 KB | 86 |

O SPA é **5,8× mais pesado** entregando **um sexto** do conteúdo.

### A decisão que barateou tudo

A ilha de filtro **não conhece os artefatos**. O Swift já escreveu as 523
linhas com `data-kind` em cada uma; a ilha só marca `data-filter` no contêiner
e o CSS esconde o resto, com `:has()` para sumir com territórios vazios.

A consequência é que a página funciona sem JavaScript:

| | sem JS | com JS |
|---|---|---|
| Linhas | 523 | 523 |
| Territórios | 19 | 19 |
| Botões de filtro | 0 | 3 |

Sem JavaScript a página fica **completa** — sem filtro, mas inteira. Não é
degradação graciosa: é a página toda, e o JS só acrescenta.

### O erro que cometi no caminho

A primeira versão da ilha era React. Custava **73 KB comprimidos para desenhar
três botões** que só marcam um atributo — mais do que todo o HTML das 523
linhas. Reescrita em TypeScript puro: **0,64 KB**, 114 vezes menor, mesmo
comportamento.

React continua na arquitetura, e será carregado só nas páginas que tiverem
ilhas que realmente precisam dele: simuladores three.js, o seletor de idioma
por página, os painéis de dissertação. Uma ilha que manipula atributos não é
uma dessas.

(Houve um segundo erro antes desse: em modo *lib* o Vite não fixa
`NODE_ENV`, e o primeiro build empacotou o React de desenvolvimento inteiro —
962 KB. Corrigido com `define`.)

## As três páginas, geradas

| | bruto | gzip |
|---|---|---|
| `index.html` | 23 KB | 5,6 KB |
| `honesty.html` | 26 KB | 7,7 KB |
| `proof.html` | 185 KB | 14,2 KB |
| `islands.js` | 1 KB | 0,6 KB |

O visitante que abre só o argumento baixa **7,7 KB**. No SPA ele baixaria os
86 KB do bundle inteiro para ver qualquer página.

As duas versões do `/honesty` renderizam idênticas — 3 atos, 4 evidências,
3561 px de altura em ambas — porque lêem o mesmo CSS.

### Um bug que a paridade revelou

Os numerais saíam como `1.759`: formatação pt-BR numa página cuja língua-fonte
é o inglês. Estava errado nos **dois** geradores, porque portei o mesmo engano.
Corrigido para `en-GB` (`1,759`) em ambos; quando os locales entrarem, a
formatação passa a receber o locale da página.

## Leitura

**Swift gera e garante, JavaScript acrescenta o que exige interação — e o
mínimo possível dele.** As páginas de conteúdo saem estáticas e leves; as
ilhas são a exceção, não a regra, e cada uma paga o seu próprio peso.
