# Site v2 — decisões

Registro do que vale hoje, do que foi derrubado, e por quê.

Atualizado em 10 de setembro de 2026 (segunda revisão).

---

## 0. Como o registro visual mudou, e a lição

As primeiras 24 decisões foram tomadas por escolha múltipla, uma pergunta de
cada vez. Produziram um sistema **coerente** — creme quente, artes em duotone,
Fraunces, verdete, layout de revista — que passava em todos os contrastes e
combinava consigo mesmo. E estava errado.

O veredito do dono do projeto: *"visual ridículo para uma linguagem de
programação epistêmica"*. Estava certo, e o diagnóstico é preciso: creme, foto
em duotone de dois idosos gregos e serifada de alto contraste é linguagem
visual de **azeite artesanal e hotel-boutique**. Um compilador que se recusa a
imprimir `var=0.000` não fala assim.

A causa raiz: a identidade foi construída a partir do **nome** — Cabo Sounion,
mármore, o templo — em vez de a partir da **linguagem**. O nome é a parte
romântica; a linguagem é um instrumento.

**A lição do processo**, que vale mais que qualquer decisão abaixo: escolher
cada peça separadamente, mesmo escolhendo bem cada uma, produz coerência sem
desejo. O resultado parece **montado**, não desenhado. Vinte e quatro perguntas
não substituem uma ideia.

O registro atual saiu de **uma** ideia, não de um questionário.

---

## 1. A ideia

> **Um valor é uma banda, não um ponto.**

É a tese da linguagem, e é desenhável. Todo portão que reporta `passed/total` é
uma proporção estimada de uma amostra finita; o site desenha o intervalo de
Wilson 95%, que é o que a amostra sustenta.

A consequência que dá a manchete: **101 portões relatam que todos os testes
passaram, e nenhum deles alcança a certeza.** `1/1` sustenta [0.207, 1.000] —
quase nada. O melhor evidenciado do corpus, 471/471, para em [0.992, 1.000].

Nenhuma outra linguagem pode usar este desenho, porque nenhuma outra tem isso
no tipo.

---

## 1a. O ângulo, e por que o anterior estava errado

O veredito do dono: *"você vai emocionar quem conhece o website? o cientista, o
pesquisador que vai olhar pensando em adotar?"* Não ia. E o defeito não era
visual.

**O site inteiro estava escrito da perspectiva de quem construiu.** Abrir com
"646.212 linhas do compilador escritas na própria linguagem" é o anúncio de
carro que começa pelo número de soldas no chassi. Responde "isto é sério?" —
que não é motivo para migrar. Ninguém troca de linguagem porque a outra é
grande. E as três páginas falavam do **projeto**: portões, registros, bandas,
territórios. Um pesquisador abria e não se via em lugar nenhum.

A emoção que Sounio tem para vender não é entusiasmo, é **alívio** — e o medo
que vem antes dele. O público é gente com medo silencioso de que um dos seus
números esteja errado.

### As quatro batidas

| | |
|---|---|
| **1. O medo** | uma dose clínica. `70 × 10 = 700.0`, confiante, e jogou fora a incerteza na linha 3 |
| **2. O gesto** | a janela terapêutica é 600–800 mg. Arraste o σ da balança: o ponto não sai de 700, a banda cresce até vazar |
| **3. A credencial** | **656** programas que este compilador recusa. Não avisos: não constrói |
| **4. Não é protótipo** | aí sim o tamanho — mas depois do motivo, não antes |
| **5. A porta** | por que acreditar: o que acontece aqui quando algo é encontrado |

A tese, dita num gesto que a pessoa faz com o dedo: **um ponto não cruza uma
linha; só uma banda cruza.** É interativo em vez de ilustrado porque ninguém
acredita numa animação e todo mundo acredita no próprio dedo.

### O corpus de recusas estava ali e o site o ignorava

`tests/compile-fail/` tem **681** arquivos — **656** vivos, **25** desligados
com `//@ ignore`. Cada um é um programa que outras linguagens compilam caladas.
É o melhor argumento de adoção que este projeto tem e não aparecia em lugar
nenhum do site.

Os desligados são contados **à parte**, e a página diz isso: uma recusa que
está desligada não é uma recusa. Foi assim que se descobriu que
`aleatoric_treated_as_epistemic` — o exemplo cientificamente mais forte de
todos — está `//@ ignore`d e portanto não podia ser citado como vivo.

### A cena é extraída, não escrita

Os parâmetros da abertura saem de `examples/real_world/01_dose_uncertainty.sio`
por `scripts/measure-dose-scene.mjs`, que **falha alto** se não os encontrar. A
cena não pode divergir em silêncio do exemplo que diz ilustrar — e foi
exatamente essa extração que revelou o achado do §6.

Três artefatos novos, todos **registros**: `site/corpus.v1`,
`site/refusals.v1`, `site/dose_scene.v1`.

### O que viaja no artefato, e por quê

Além dos valores da cena, viajam a **escala do campo** (`mark_lines`), o
**alcance do slider** (`sigma_min/max/step`) e a **extensão do eixo**
(`axis_lo/hi`, `tick_step`). Nenhum desses é decoração:

- o máximo do slider é **derivado do ponto de cruzamento** (σ = 3,71 kg) — se
  não passasse dele, o leitor nunca veria a banda deixar a janela e a
  demonstração não existiria;
- a extensão do eixo é derivada da banda mais larga que o leitor consegue
  produzir, senão o desenho estoura no fim do curso;
- e os ticks são valores em **mg impressos na tela**: um tick escolhido à mão
  que estivesse errado enganaria.

Assim o portão de evidência não precisa abrir excepção para nenhum deles, e os
dois geradores leem os mesmos parâmetros em vez de cada um escolher os seus.

### Uma matemática, duas renderizações

`web/src/dose.ts` guarda **só a fórmula** — a propagação GUM pelo método delta.
O componente React e a ilha estática a importam. Desenham coisas diferentes;
não podem calcular coisas diferentes. A ilha passou de 0,64 KB para **1,51 KB**
comprimidos e agora carrega a demonstração interativa inteira.

Sem JavaScript o rig fica parado no σ do exemplo — um valor legítimo da cena,
não um estado de erro.

---

## 1b. A segunda cena: hidrogénio

*"Quem trabalha com combustão não se comove com dose de vancomicina."* A cena
do H2 existe para um público diferente, e por isso tinha de ter uma **tese
diferente** — repetir "um ponto não cruza uma linha" noutro campo seria a mesma
página duas vezes.

> **Uma banda que muda quando você muda o passo de integração está a reportar
> o integrador, não a química.**

E a direção do efeito é o que assusta: **refinar o passo faz a incerteza
encolher.** Parece convergência. Não é.

### O que foi medido, e por três implementações

Sob um fator quatro no passo `dt`:

| implementação | razão da banda |
|---|---|
| réplica Python (quadratura independente por passo) | 1,99994 – 2,00841 |
| cross-check em C++ (mesma fórmula, escrita do protocolo publicado) | 1,99994 – 2,00841 |
| **Sounio nativo (propagação coerente de sensibilidade)** | **0,999999 – 1,000001** |

Duas implementações escritas independentemente concordam — e estão erradas do
mesmo jeito. A quadratura soma uma incerteza **independente** a cada passo,
quando os passos partilham os mesmos parâmetros de velocidade e não são
independentes. Quadratura sobre termos dependentes emite um limite **mais
apertado que a verdade**, que é a única direção de erro que esta linguagem
trata como mentira e não como aproximação.

### A ponte com as recusas

`tests/compile-fail/dsep_fork_unconditioned` recusa exactamente essa
composição, com código E255, e está **vivo**. O comentário do próprio teste diz
a consequência: *"would emit a bound tighter than the truth, which SEMANTICS.md
Invariant 2 calls a lie"*.

**Mas a página diz o que ainda não é verdade:** ligar essa recusa ao caminho de
quadratura do módulo de química é a PR #1758, **aberta e não fundida**. Logo a
correção medida acima é *medida*, não *imposta* — e a nota abaixo da cena diz
qual é qual.

### A interação para onde a medição parou

Três botões, não um slider. A medição tem três valores de `dt`; um contínuo
insinuaria um contínuo que ninguém mediu — e numa página sobre não afirmar além
da amostra, esse seria o erro mais caro possível.

As razões são normalizadas ao valor da própria implementação em `dt = 4e-9`,
porque o documento-fonte não publica âncora absoluta para o lado da quadratura.
Assim os dois lados continuam a ser medições.

### O qualificador que a cena mostra sozinha

`H2` e `O2` não se movem em nenhum dos lados: a banda deles é dominada pela
incerteza semeada na mistura inicial, que não acumula. A secção 5.2 do
`RESULTS.md` diz que enunciar "a banda escala com √dt" sem esse qualificador é
falso para **2 das 8** espécies. A cena mostra isso no próprio gráfico em vez
de o esconder.

### A ilha não calcula

O gerador escreve as três larguras medidas em `data-dt4/dt2/dt1`; a ilha só
troca qual está aplicada. Mesmo princípio do filtro do `/proof`: o cálculo fica
onde há evidência, e o JavaScript move atributos. Total agora: **1,67 KB**
comprimidos para as duas cenas interativas mais o tema e o filtro.

---

## 1c. A escala, e o achado que ela produziu

A primeira versão do site desenhava o resíduo de CI — 523 artefatos, 147 barras
— como se fosse o objeto. O veredito do dono foi: *"é como se você não tivesse
captado o tamanho de Sounio"*. Estava certo. O objeto é uma linguagem cujo
compilador está escrito nela mesma, e isso tem tamanho.

O corpus escrito inteiro passou a ser desenhado, uma marca por fatia igual de
linhas. **Deixou de ser a abertura** na revisão do §1a — o tamanho responde à
segunda pergunta do leitor, não à primeira — e hoje vive no `/proof`, com a
versão achatada na quarta batida da home.

### Por linha, não por arquivo

Por arquivo o corpus de testes tem mais entradas que todo o resto somado —
4 467 contra 591 do compilador — e domina a imagem. Isso desenha quantos
arquivos alguém criou, não o que a linguagem é. Por **linha** a proporção se
inverte e o compilador ocupa a maior área, que é o argumento.

### O que a mudança de unidade expôs

Contar por linha obrigou a olhar o que estava sendo contado. Um balde
`other` de 357 arquivos tinha 1 126 961 linhas — três mil linhas por arquivo, e
foi essa densidade anómala que denunciou. **97,7% era gerado ou arquivado.** Um
único arquivo, `probe_sparse_300000.sio`, tem 303 010 linhas de entrada de
benchmark gerada.

Três árvores ficam de fora de toda contagem de linhas escritas —
`artifacts/`, `archive/` e `bootstrap/` — e o montante excluído é publicado no
`/honesty`, na mesma escala do que remove. Contadas, a manchete teria dito
3 437 936 em vez de 2 100 978.

O erro é exatamente o que o terceiro acto do `/honesty` nomeia: *um valor
plausível no lugar de algo que nunca foi o que dizia ser*. Ele está na página
com o meu nome, porque uma página sobre recusar números sem lastro que
escondesse o seu próprio quase-erro não valeria nada.

### A medição é um artefato

`site-next/scripts/measure-corpus.mjs` não emite HTML nem números: escreve
`artifacts/site/corpus.v1.json`, que o indexador lê como qualquer outro. Se os
números do corpus entrassem no site por um caminho especial, a garantia que o
site anuncia teria um buraco do tamanho da própria manchete. É **registro**, não
portão: medição com procedência, sem veredito a declarar.

A **escala do campo** (`mark_lines`) também viaja dentro do artefato. Duas
razões: ela vira afirmação com procedência como qualquer outra — o portão de
evidência não precisa abrir excepção para ela — e os dois geradores passam a
ler a mesma escala em vez de cada um escolher a sua.

### As três páginas

| Página | O que leva |
|---|---|
| `/` | a barra achatada, na quarta batida: "e não é protótipo" |
| `/proof` | o campo inteiro — é a página do "tudo, sem edição" |
| `/honesty` | o excluído, com o quase-erro narrado |

Aqui a cor codifica **domínio**, não largura de banda. São grandezas
diferentes, e por isso todo gráfico desta família traz a sua própria legenda
colada a ele.

---

## 1d. Contraste: dois tokens reprovavam

Medidos os pares de texto que realmente aparecem, `--faint` dava **2,05** no
escuro e **2,16** no claro — contra 4,5 exigidos. `--ink-3` parava em 4,2–4,5.
Ambos carregam prosa real: notas, legendas do campo, o colofão, a lista do
excluído.

Resolvidos numericamente, mantendo matiz e saturação e movendo só a
luminosidade, contra o **pior** fundo de cada tema (`--raised` no escuro,
`--bg` no claro): `--ink-3` a 5,81 e `--faint` a 4,58. O custo é que a camada
"faint" ficou menos apagada; a alternativa era manter texto ilegível por
estética.

---

## 2. O registro visual

| | |
|---|---|
| Tipografia | **JetBrains Mono em tudo**, inclusive títulos |
| Terreno escuro (padrão) | `#0A0C10` |
| Terreno claro | `#F7F8F7` — o mesmo instrumento sobre papel de laboratório |
| Densidade | alta, de propósito: quem lê diagnóstico confia em densidade |
| Cor | codifica **largura de banda**, não veredito |
| `constrains` | `#43D9AD` escuro · `#0E7C61` claro — a amostra restringe |
| `barely` | `#E8A33D` escuro · `#8F6008` claro — a amostra mal restringe |
| `refused` | `#F0616B` escuro · `#BE2F2C` claro |
| Fotografia | **nenhuma** |

A interpolação entre `constrains` e `barely` é feita em CSS com `color-mix`,
lendo um `--t` no contêiner da banda: acompanha o tema sem recalcular em JS.

---

## 3. Decisões que sobreviveram à mudança de registro

| Decisão | Estado |
|---|---|
| Ambos os temas | ✅ construído — escuro primário, claro é o mesmo instrumento |
| Língua-fonte inglês | ✅ |
| Público: quem precisa confiar num número | ✅ a home abre pelo cálculo clínico |
| Ação principal: ler o argumento | ✅ |
| Voz didática | ✅ cada ato do `/honesty` abre com o que ensina |
| Texto existente como rascunho | ✅ |
| Marcação por afirmação, inline | ✅ `Claim` no meio da prosa |
| Duas espécies: portões e registros | ✅ 270 e 253 |
| `/proof` por território | ✅ 19 territórios |
| Data junto do número | ✅ |
| Playground fora do ar até compilar | ✅ |
| `next.souniolang.org` na Vercel | ✅ workflow pronto |
| Sequência: argumento e `/proof` primeiro | ✅ ambos construídos |
| Coexistir em subdomínio | ✅ |
| Gerador: decidir vendo os dois | ✅ **resolvido — híbrido** |

### O gerador, resolvido

Swift gera o HTML estático e garante as afirmações; JavaScript acrescenta só o
que exige interação, e o mínimo possível dele. `/proof` inteiro sai em
**15,7 KB** comprimidos com as 523 linhas e 147 bandas, contra 88,4 KB do
bundle React. A ilha de filtro tem **0,6 KB** e não conhece os artefatos: o
Swift já escreveu as linhas com `data-kind`, ela só marca um atributo.

Sem JavaScript a página fica completa — sem filtro, mas inteira.

---

## 4. Decisões derrubadas

| Decisão antiga | O que houve |
|---|---|
| Artes protagonistas | Fora. Fotografia não passa credibilidade para quem avalia um sistema de tipos |
| Duotone verdete e creme | Não existe mais |
| Terreno creme `#FBF8F2` | Substituído por papel de laboratório frio |
| Verdete `#2E9C86` como marca | Sobrevive com outro papel: é a cor de "medição que restringe" |
| Ouro só em numerais | Fora — a cor agora significa largura, não categoria |
| Estados frio/âmbar/vermelho | Substituídos pela codificação de largura + `refused` |
| Fraunces + Manrope | Substituídos por monoespaçada única |
| Layout assimétrico de revista | Substituído por densidade |
| Uma abertura de arte por seção | Fora |
| Lockup coluna + palavra | Sobrevive, discreto, 18 px na barra |

---

## 5. O portão de evidência

A regra: um numeral na prosa é uma afirmação factual. Ou vem de
`claim(artefato, métrica)`, ou o build para.

Isentos, e a lista é deliberadamente mínima: versões, anos, datas com nome de
mês, horas de relógio, níveis de portão `E0`–`E5`, código dentro de `<code>` e
crases, e as constantes do **método** (`95%`, `z = 1.96`) — que descrevem como
o intervalo é calculado, não resultados.

A lista cresceu uma vez nesta revisão: versões de **nomes próprios** de padrões
e ferramentas (`GRI-Mech 3.0`, `Cantera 3.2.0`, `NASA-7`). A regra é explícita
por nome, e não genérica do tipo "maiúscula seguida de número" — uma regra
genérica isentaria `Madaros 4.71` e viraria exactamente o contrabando que o
portão existe para impedir. Duas outras tentações foram recusadas em vez de
isentas: os limites do slider e a extensão do eixo viraram métricas do
artefato, e `C++23` perdeu a versão no texto porque ela não sustentava o
argumento.

Os trechos isentos são **mascarados**, não usados para pular a linha inteira.
Pular a linha porque ela contém um ano deixaria passar qualquer numeral
inventado na mesma linha. As duas implementações — `gate-claims.mjs` e o
construtor de `Prose` em Swift — precisam concordar, senão a garantia do site
depende de qual gerador foi usado.

Em Swift o `Claim` tem `init` privado: fabricar um fora do módulo é erro de
compilação, sem cast que escape. Em TypeScript a marca é convenção.

---

## 6. Achados que a construção produziu

- **O site arredondava as próprias afirmações em silêncio.** `Claim` formatava
  com `toLocaleString` sem opções, e o padrão corta em **três casas decimais**.
  A tabela de contraste do H2 publicou `1.99994` como `2` e `0.999999` como
  `1` — apagando exactamente a diferença que a medição existe para mostrar. O
  portão garantia a **procedência** do número e nada garantia o **número**.
  Corrigido nos dois formatadores (TS e Swift). É o defeito mais embaraçoso
  desta construção: um site cuja tese é fidelidade numérica, infiel na última
  polegada, e invisível até um valor com seis casas significativas aparecer.
- **O exemplo carro-chefe de incerteza subestima a própria incerteza.**
  `examples/real_world/01_dose_uncertainty.sio` documenta no cabeçalho
  `σ = 20.6 mg`, IC `[659.6, 740.4]`. A aritmética que o próprio arquivo
  escreve — `Var = 10²·4 + 70²·0,25 = 1625` — dá `σ = 40.3` e IC
  `[621.0, 779.0]`: uma banda de 158 mg onde o cabeçalho promete 80,8.
  **Qual dos dois está errado não está decidido**, e a página diz isso: pode
  ser cabeçalho velho, ou o código pode ter ganho uma incerteza na diretriz de
  dosagem que o cabeçalho nunca pretendeu — e aí a pergunta é se a faixa de uma
  diretriz clínica entra na propagação, o que é decisão de modelagem e não
  erro de digitação. O que está estabelecido é que discordam, e que discordam
  na direção que esta linguagem existe para impedir. Sob investigação em tarefa
  separada.
- **Dois números publicados no site atual não têm artefato nenhum:** o muro de
  reclamation "vermelho de propósito aos 4.194.304 allocations (rc=182)".
  Procurados em todos os 523 artefatos — não existem. Pendente: dar-lhes um
  artefato ou tirá-los do texto.
- **O corpus tem duas espécies**, não uma com metade quebrada. Chamar 253
  registros de "desconhecidos" era erro de categoria do indexador.
- **Os `sprintNN` fragmentavam o corpus em 88 territórios** de um artefato
  cada. Normalizados para um, os 140 viraram 19.
- **`v is Bool` mente no Foundation do Linux** para qualquer `NSNumber` valendo
  0 ou 1. O indexador Swift descartava em silêncio toda métrica igual a zero —
  justamente os `failed: 0` que este site existe para publicar.

---

## 7. Em aberto

- **O gerador Swift continua sem compilar nesta máquina.** Portados agora:
  `Corpus.swift`, `Dose.swift` (cena, rig e recusas) e as três páginas em
  `Pages.swift`. O contrato de marcação foi conferido classe a classe contra o
  CSS, o React e a ilha — e o verificador achou duas divergências reais que
  foram corrigidas: o React não emitia `.rig-var`, que é o nó que a ilha
  atualiza, e `Part.code` emitia `<b>` em vez de `<code>`, o que fazia o
  literal citado divergir do React e escapar da regra de código do portão.
  Mas não há toolchain Swift aqui e o Docker sobe sem conseguir extrair camadas
  (falta privilégio de mount), então **nada disso passou pelo compilador**.
  `swift build` é o primeiro passo antes de confiar no gerador híbrido.
- ~~`npm run measure` ainda não está no CI.~~ **Resolvido**, e de um modo
  diferente do previsto. Os scripts mediam `origin/main`, que se move: os
  artefatos descreviam uma árvore diferente da publicada e ninguém conseguiria
  repetir a medição sem adivinhar o momento. Passaram a medir `HEAD`. A CI
  remede e **compara** — não regenera: os números do site passam por revisão
  como qualquer outra mudança, em vez de a CI os reescrever sem que nenhum
  commit explique por que a manchete mudou.
- **Mais cenas, além da dose e do hidrogénio.** Restam `particle_physics`, `ode`,
  `climate_ensemble`, `sensor_fusion`. É aqui que entra investimento de design,
  não em efeito.
- Os dois números órfãos acima.
- Criar o projeto Vercel e apontar o DNS (passos humanos, em `DEPLOY.md`).
- Navegação lateral da documentação, seletor de idioma por página e marcação de
  traduções divergentes — decididos, não construídos.
- Quando o WASM real do compilador entra, e o que o playground vira até lá.
