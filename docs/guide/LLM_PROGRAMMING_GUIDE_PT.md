<!-- docs:meta
topic_id: repo.docs.guide.llm-programming-guide-pt
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.llm-programming-guide-pt
-->

# Escrever Sounio com um modelo de linguagem

[Ελληνικά](LLM_PROGRAMMING_GUIDE_EL.md) · [English](LLM_PROGRAMMING_GUIDE.md)

**Edição:** outubro de 2026. **Preparação:** 2026-10-16.
**Base de código:** `f5b2c5850be9f3d3995719ffcade44a6d851658a`.

**Status: research language (pre-1.0).** APIs may change; not recommended for production use. Não descreva a Sounio como pronta para uso em produção e não nomeie uma versão do compilador nessa linha.

Este guia ensina a gerar, compreender, corrigir e verificar programas Sounio.
A estratégia é simples: partir de um programa completo, modificar uma coisa
por vez e observar o resultado no compilador identificado. O material serve
como contexto recuperável para modelos pequenos e como referência aprofundada
para agentes maiores. Parecer com Rust não torna Sounio uma variante de Rust.

A [especificação](../spec/LANGUAGE_SPECIFICATION.md) distingue contratos
adotados, comportamento observado e decisões ainda abertas. Este guia ensina
a usar essa referência: não inventa uma segunda linguagem mais fácil para o
modelo, nem transforma limitações da implementação em decisões de projeto.

**Escopo da verificação:** o [recibo de execução](../llm-guide/reviews/example-execution.md)
registra 50 invocações sobre 47 arquivos-fonte distintos, com o Madaros Linux
precompilado identificado: 30 contratos de execução atendidos, quatro casos
somente de checagem atendidos, oito recusas pelo motivo esperado e oito casos
que falharam nas próprias condições esperadas. Estes últimos estão em
quarentena, fora do caminho recomendado. Isso não demonstra rebuild do
compilador, conformidade completa, certificação numérica ou sucesso de todo LLM.

[English](LLM_PROGRAMMING_GUIDE.md) ·
[Receitas](../llm-guide/cookbook.md) ·
[Atlas de programas completos](../llm-guide/examples.md) ·
[Manifesto dos exemplos](../llm-guide/example-manifest.json) ·
[Diagnósticos](../llm-guide/error-catalog.md) ·
[Descoberta de APIs](../llm-guide/stdlib-index.md)

Para começar com um contexto completo do tamanho da tarefa, use o
[currículo trilíngue de trinta roteiros](../llm-guide/task-playbook.md):
`node scripts/docs/sounio_task_context.mjs show first-program --lang pt`.
O comando fornece fontes completas de referência/controle, resultados
históricos do perfil, passos de adaptação e armadilhas sem executar compilador
ou LLM. Os roteiros reutilizam programas existentes; não são trinta novas
execuções. O [guia de tradução e reparo](../llm-guide/translation-and-repair.md)
explica ports que preservam intenção, diagnóstico por etapa e reparos limitados.

<!-- llm:section orientation -->
## 1. O contexto certo para o modelo

Não comece despejando o repositório inteiro no contexto. Use três camadas:

| Camada | Conteúdo | Quando usar |
|---|---|---|
| Inicial | [`llms.txt`](../../llms.txt), tarefa, ambiente e um exemplo completo | Primeiro programa ou alteração pequena. |
| Da tarefa | Seção relevante deste guia, receita, declaração real da API e controle de recusa | Bibliotecas, referências, efeitos e erros. |
| Profunda | Regras da spec, caminhos da implementação e recibos identificados | Divergência semântica, precisão, target ou investigação do compilador. |

O pacote mínimo da tarefa informa entradas, saída exigida, arquivos
permitidos, compilador/artefato, host/target, raiz da biblioteca e critério de
sucesso. “Use um vetor limitado de valores reais” e “implemente um vetor
genérico de alocação dinâmica” são tarefas diferentes. Escolha a representação
a partir da necessidade, não de um nome parecido em outra linguagem.

Para um modelo menor, forneça um programa e uma mudança de cada vez. O exemplo
precisa conter as declarações, importações, efeitos e convenção de retorno.
Uma lista de arquivos, um trecho com funções inexistentes ou um contexto
gigantesco não substitui essa estrutura.

Para tarefas usuais, os quatro programas completos deste guia permitem
começar. Para usar uma biblioteca, acrescente a declaração e o consumidor que
a importa. Para uma hipótese científica, recupere também o contrato relevante:
não peça ao modelo que reduza a hipótese para algo familiar apenas para
conseguir escrever código.

Quando o pedido já é claro, use uma escolha razoável e declare-a; não pare a
cada detalhe editorial. Pergunte somente quando a escolha alterar a semântica,
os dados, o ambiente necessário ou os limites autorizados.

<!-- llm:section workflow -->
## 2. Primeiro resultado correto e ciclo de trabalho

### 2.1 Identifique o executável antes de interpretar resultados

Siga o [preflight inicial](../../llms.txt#first-success). Ele identifica o
perfil solicitado, host compatível, artefato bruto e digest, raiz da biblioteca
e contrato do exemplo **antes** do par de checagem/execução. O launcher público
é `./bin/souc`; o binário selecionado por ele não é esse script de shell.
O hash do wrapper não identifica os bytes do compilador que ele executa.

O perfil documentado dos exemplos usa o precompilado Madaros Linux x86-64.
Um compilador reconstruído ou um artefato nativo Apple Silicon corresponde a
outro perfil. O [guia macOS ARM64](../compiler/MADAROS_MACOS_ARM64.md)
explica a preparação nativa. Não tente executar ELF incompatível, trocar
silenciosamente para o seed ou apagar um override para obter um aparente passe.

Os comandos são usados no ambiente autorizado para a tarefa. Uma receita
pública não anula regras da sessão ou do repositório que exijam execução
remota. Validações completas e de stress deste projeto seguem Foundry/Slurm;
um comando ilustrativo não autoriza essa carga no notebook ou workspace
compartilhado.

### 2.2 Um primeiro programa inteiro

Fonte: [`examples/hello.sio`](../../examples/hello.sio), ID `hello`.
O contrato esperado é saída de processo zero e stdout contendo
`Hello, Sounio`. O launcher pode imprimir mensagens de compilação junto ao
resultado; o manifesto não inventa igualdade exata do fluxo inteiro.

<!-- llm:example hello -->
```sio
// First Sounio program

fn main() -> i32 with IO {
    println("Hello, Sounio")
    return 0
}
```

Depois do preflight, na raiz do repositório:

```sh
./bin/souc check examples/hello.sio
./bin/souc run examples/hello.sio
```

O recibo registra o que aconteceu com esse arquivo. Ao modificar o programa,
o modelo precisa verificar o arquivo modificado: não pode copiar a saudação
esperada para o relatório e chamá-la de saída observada.

### 2.3 Um passo de cada vez

1. Leia o arquivo inteiro e os efeitos declarados.
2. Faça uma pequena mudança completa, no arquivo permitido.
3. Cheque esse arquivo, não o exemplo original por engano.
4. Leia a mensagem completa e corrija a primeira causa relevante.
5. Se a tarefa pede execução, execute e confira o comportamento exigido.
6. Use um controle válido próximo e uma recusa intencional quando eles
   distinguirem a regra sob teste.
7. Registre resultados reais e o que não foi estabelecido.

Uma checagem sustenta a etapa alcançada pelo adaptador. Emitir sustenta a
produção de um artefato. Executar sustenta as observações efetivamente
conferidas. Um AST recuperado, um caminho de teste, a versão impressa ou
um exit isolado não demonstram todas essas etapas.

### 2.4 CLI e MCP não são automaticamente o mesmo ambiente

`sounio_check` é uma ferramenta MCP quando configurada, não um comando de
shell que se deva presumir instalado. Leia o
[procedimento de atribuição MCP](../../llms.txt#mcp-profile-attribution)
antes de combinar seu resultado com comandos no terminal.

A resposta atual de checagem MCP não identifica por si só o digest do binário
bruto e a raiz da biblioteca. A implementação de execução pode fazer o
precheck antes de aplicar overrides de ambiente por chamada. Portanto, uma
chamada de ferramenta não prova um perfil único para check e run.

Obtenha identidade do lado do servidor sob o ambiente efetivo. Se isso não
estiver disponível, trate a atribuição como desconhecida e repita check/run
por uma CLI identificada. Não invente campos de identidade na resposta.

<!-- llm:section syntax -->
## 3. Sintaxe sem contaminação de outra linguagem

| Hábito do modelo | Procedimento Sounio |
|---|---|
| Escrever `let mut valor` | Use `var valor` para binding reatribuível. |
| Escrever `&mut valor` | Use a referência exclusiva `&!valor` da API selecionada. |
| Usar `println!` ou `assert!` | Use chamadas ordinárias `println` e `assert`. |
| Remover todo ponto e vírgula | Não: arrays usam `;`, e posições de statement pertinentes o aceitam. |
| Exigir receptor em todo método | Receptores de instância são explícitos; construtores associados podem não ter `self`. |
| Copiar atributos ou derive de Rust | Procure o mecanismo real de declaração/teste Sounio. |
| Imprimir com argumentos de formatação supostos | Leia a assinatura real; o exemplo inicial passa uma string. |
| Proibir genericidade, traits e closures | Verifique implementação, limites e testemunhos do artefato. |
| Confundir sintaxe aceita com runtime | Separe parsing, checagem, lowering, emissão e execução. |

Bindings podem usar anotação explícita ou tipo inferido do inicializador.
Os parâmetros de função precisam das anotações exigidas pela sintaxe
selecionada. Defina todos os nomes: o compilador não fornece automaticamente
um helper porque seu nome parece natural.

Quebras de linha, pontuação opcional, operandos de return e expressões
multilinha dependem dos predicados do parser. Consulte o
[capítulo de sintaxe](../spec/2026-10/01-lexical-syntax-and-modules.md)
em vez de aplicar uma limpeza mecânica de pontuação. Um workaround não
executado não deve virar uma regra universal no guia.

Strings, caracteres, buffers de bytes e handles estrangeiros são
representações diferentes. Imprimir um valor não garante serialização sem
perda ou uma convenção de indexação Unicode. Não transfira interpolação,
concatenação ou ABI de outra linguagem sem verificar a operação real.

<!-- llm:section types -->
## 4. Tipos e representações

### 4.1 Escolha o domínio antes da estrutura de dados

Grafias inteiras e de ponto flutuante nomeiam tipos; não são uma promessa
automática de overflow, arredondamento e ABI iguais em todos os targets.
Respeite a largura e o formato solicitados. Floats largos não são registros
double-double ou quad-double, e usar um formato menor não preserva a precisão
pedida.

Arrays, tuplas, structs, enums, referências, funções e registros de biblioteca
têm regras diferentes. Aceitar um nome no checker não demonstra todas as
operações ou todos os layouts. Veja os capítulos de
[tipos](../spec/2026-10/02-types-generics-and-ownership.md) e
[numéricos](../spec/2026-10/05-numerics-units-and-algebra.md).

### 4.2 Um programa completo com struct

Fonte: [`examples/native/struct_basic.sio`](../../examples/native/struct_basic.sio),
ID `struct`. A verificação delimitada exige exit zero e stdout contendo
`7`; isso é mais fraco que uma especificação de serialização completa.

<!-- llm:example struct -->
```sio
// Struct literal construction and field access in native-v2 driver.
// Expected: prints "7", exits 0.

struct Point { x: i64, y: i64 }

fn main() -> i64 with IO {
    let p = Point { x: 3, y: 4 }
    let s = p.x + p.y
    print_int(s)
    print_char(10)
    0
}
```

Adapte um campo ou operação, defina a observação da nova tarefa e verifique
novamente. Não substitua um campo científico por um inteiro bruto apenas
porque a assinatura mais rica encontrou uma limitação.

### 4.3 Arrays, tuplas e coleções

Tipo de elemento, capacidade, inicialização, índices e empréstimos precisam
concordar. Repetição e tipos de array contêm ponto e vírgula como sintaxe.
Bounds checks e mutação têm contratos próprios. Não converta todo índice
mecanicamente: use o tipo esperado pela API concreta.

Acesso a tupla e desestruturação de padrão seguem caminhos diferentes.
O teste selecionado de desestruturação aninhada falhou; programas de acesso
e assinatura de tupla passaram. Isso não constitui proibição universal de
tuplas.

Há diversas representações parecidas com vetor. A API pública de funções
livres de `VecF64` no índice não se confunde com os registros próximos
`IntVec`/`FloatVec` não públicos, com um heap genérico, array fixo ou
sequência especializada pelo compilador. Leia capacidade, layout do elemento,
alocação e resultado de falha. Zero retornado pode ser um sentinela, não um
elemento validado.

<!-- llm:section functions -->
## 5. Funções, chamadas genéricas e valores chamáveis

Um exemplo de função precisa mostrar declaração, tipos, efeitos, chamada e
resultado do programa. Um trecho que usa `map`, `sqrt` ou `add_one`
sem defini-los ou importá-los não é um tutorial executável.

Referências a funções nomeadas são um bom início para tarefas de ordem
superior. O exemplo [`cook-function-ref`](../llm-guide/examples.md#cook-function-ref)
tem execução delimitada registrada. Closures também precisam de contrato de
captura, tipo chamável, efeitos, duração e target. O `cook-closure` escolhido
falhou nesse artefato e está em quarentena; isso não prova que todas as
closures sejam inexistentes ou que todas funcionem.

### Uma rota genérica explícita verificada

Fonte: [`tests/run-pass/generics_multi_param.sio`](../../tests/run-pass/generics_multi_param.sio),
ID `generics`. O programa original tem dois controles internos; o catálogo
exige exit zero e o primeiro marcador de sucesso. Isso não demonstra resolução
irrestrita de traits, métodos genéricos ou toda combinação de especialização
e ABI.

<!-- llm:example generics -->
```sio
//@ run-pass
//@ expect-stdout-contains: first_of_two: PASS
//@ expect-stdout-contains: second_of_two: PASS
// Tests two-type-parameter generic functions via turbofish syntax

fn first<T, U>(a: T, b: U) -> T {
    a
}

fn second<T, U>(a: T, b: U) -> U {
    b
}

fn main() -> i64 with IO {
    var exit_fails: i64 = 0
    let x = first::<i64, i64>(42, 99)
    if x == 42 { print("first_of_two: PASS\n") } else { print("first_of_two: FAIL\n"); exit_fails = exit_fails + 1 }

    let y = second::<i64, i64>(10, 77)
    if y == 77 { print("second_of_two: PASS\n") } else { print("second_of_two: FAIL\n"); exit_fails = exit_fails + 1 }
    if exit_fails > 0 {
        return 1
    }
    return 0
}
```

Mantenha a grafia exata de chamada mostrada ao adaptar essa rota. Os outros
dois fixtures escolhidos de função genérica simples e par genérico falharam:
o nome do arquivo não provava suporte no artefato. Não mude a expectativa
para ocultar isso, nem declare a linguagem inteira sem genéricos.

Declaração de trait, bound, resolução de nome, seleção de método e
enforcement de obrigação são operações distintas. Um bound reconhecido pelo
parser não é prova de coerência ou satisfação. Leia o código e um chamador
que realmente interrogue a obrigação solicitada.

<!-- llm:section control-flow -->
## 6. Controle de fluxo e ordem observável

Comece pelos programas completos de
[while](../llm-guide/examples.md#while-loop),
[iteração de array](../llm-guide/examples.md#cook-for-array) e
[intervalos/opções](../llm-guide/examples.md#cook-range-option).
Leia o retorno da entrada e as condições observadas: alguns programas de
teste usam exit não zero como resultado correto.

Expressões if, laços, padrões, return, break, continue e caudas de bloco têm
regras sintáticas e de checagem. Confira todos os caminhos de retorno.
Se um programa só imprime seu marcador quando uma comparação dá certo,
exija esse marcador: sair com zero sozinho pode deixar a falha invisível.

Não use `&&` ou `||` para proteger uma operação perigosa apenas porque
outra linguagem faz curto-circuito. O lowering ordinário e caminhos de target
inspecionados não estabelecem um contrato portátil único para avaliação com
efeitos. Quando segurança ou ordem importarem, use controle de fluxo explícito
e um teste que distinga o comportamento requerido no perfil selecionado.

Isso não proíbe operadores lógicos em expressões booleanas usuais. Significa
que um conjunto de efeitos não é um traço de execução. Observe ordem,
multiplicidade de eventos, mutação e canais científicos afetados por branches.

Recursão e buffers grandes podem atingir capacidades da implementação.
Crash e timeout não provam invalidade da linguagem nem autorizam eliminar
uma proteção. Registre o limite e reduza o reproducer sem trocar o significado
do problema.

<!-- llm:section ownership -->
## 7. Referências, empréstimos e recursos

Referência compartilhada, referência exclusiva, ponteiro bruto, handle linear
e valor possuído não são intercambiáveis só porque alguma ABI os representa
em uma palavra. Preserve contratos em chamadas e retornos.

Leia inteiros os exemplos verificados de
[liberação de empréstimo](../llm-guide/examples.md#cook-borrow-release) e
[referência a array](../llm-guide/examples.md#cook-array-ref).
Inicialização, aquisição, uso, liberação/escopo e mutação observada fazem
parte do caso; não copie só a expressão que aparece no erro.

Diante de conflito, localize as aquisições e a duração dos usos. Encurte o
empréstimo ou reorganize acessos disjuntos somente preservando a tarefa.
Não use ponteiro bruto para apagar o erro, remova a linearidade, copie um
handle consumido ou altere a expectativa do teste.

As análises são delimitadas e dependem do caminho. A existência de um módulo
de lifetimes ou estratégia de drop não prova NLL completo, inserção de
destrutores, validade de ponteiros e todos os movimentos parciais. Ausência de
enforcement também não cancela uma regra de recursos já adotada.

Uma recusa útil acompanha um controle válido próximo. O
[negativo de conflito](../llm-guide/examples.md#negative-borrow-conflict)
foi confrontado com sua mensagem pretendida, não com “qualquer exit não zero”.

<!-- llm:section effects -->
<a id="effects"></a>
## 8. Efeitos sem anotações indiscriminadas

Declaração, inferência, propagação, checagem e execução de efeitos são
mecanismos diferentes. O [capítulo de efeitos](../spec/2026-10/03-effects-handlers-and-concurrency.md)
separa os contratos e os comportamentos observados.

| Operação | O que conferir |
|---|---|
| Imprimir ou acessar I/O | Contrato `IO` da operação chamada. |
| Alterar por referência ou estrutura compartilhada | Contrato de mutação e caminho de `Mut`. |
| Reatribuir variável local | Semântica de mutação local e engine; não uma regra universal de `Mut`. |
| Dividir, obter resto, assert, indexar ou converter | Contrato dos operandos/API e obrigações `Div`/`Panic` pertinentes; não uma tabela universal pela grafia. |
| Métrica científica ou observação | Representação e fronteira `Epistemic`/`Observe`. |
| Chamada por referência de função | Assinatura chamável e efeitos retidos naquela rota. |

Não adicione todos os efeitos em `main` como reparo genérico. Comece pelo
conjunto exigido e propague pela cadeia real de chamadas. Se a tarefa alterar
o contrato, explique. Se o checker não aplicar uma regra, mantenha a semântica
pretendida e registre a lacuna.

### Um exemplo inteiro de subconjunto de efeitos

Fonte: [`tests/run-pass/effect_superset_ok.sio`](../../tests/run-pass/effect_superset_ok.sio),
ID `effects`. O recibo exige `effect superset: PASS` e exit zero.

<!-- llm:example effects -->
```sio
//@ run-pass

// Caller declares more effects than callee needs — should pass
fn pure_add(a: i64, b: i64) -> i64 {
    a + b
}

fn needs_io() with IO {
    print("io\n")
}

fn main() -> i32 with IO, Mut, Div {
    // Calling pure function from effectful context is fine
    let x = pure_add(2, 3)
    // Calling IO function from IO+Mut+Div context is fine
    needs_io()
    if x == 5 {
        print("effect superset: PASS\n")
    }
    return 0
}
```

O [negativo de efeito ausente](../llm-guide/examples.md#negative-effect-missing)
produziu a recusa esperada nesse artefato. Texto histórico dizendo que E035
nunca dispara no default não pode substituir esse resultado.

Nomes de efeito desconhecidos e excesso de capacidade têm diagnósticos
distintos. Um conjunto limitado não é polimorfismo geral de rows; alterar
um conjunto ambiente não implementa operações e continuações de um handler.
Aceitação de sintaxe handler/async não cria o runtime correspondente.

<!-- llm:section scientific -->
## 9. Preserve a informação científica

### 9.1 Identifique primeiro qual valor está sendo manipulado

| Superfície | Informação a preservar |
|---|---|
| `Knowledge<T>` do compilador | Predicados do checker e canais científicos do lowering selecionado. |
| `Epistemic` de biblioteca | Valor, variância, confiança e precondições do registro/API. |
| EISA `EReg` / `EReg2` | Valor, correção aritmética, incerteza e convenções de origem distintas. |
| Anotações de unidade | Dimensão, representação, conversões e fronteiras verificadas. |
| Canais/contagens quânticos | Presença, dependência, hipóteses de amostragem/modelo e proveniência do dispositivo. |

Consulte a [referência científica](../spec/2026-10/04-knowledge-uncertainty-and-provenance.md).
Nomes semelhantes não tornam tipos substituíveis. Os exemplos de
[biblioteca epistêmica](../llm-guide/examples.md#cook-epistemic-library) e
[EISA](../llm-guide/examples.md#cook-eisa) têm contratos próprios. Asserções
herdadas de um fixture não equivalem a uma auditoria numérica independente.

Não desembrulhe silenciosamente um valor científico em `f64`, descarte
variância, substitua incerteza ausente por zero, reduza precisão ou retire
predicado de proveniência para fazer a chamada passar. Se a preservação for
impossível, apresente a mudança semântica necessária antes de executá-la.

Valores centrais iguais não demonstram origem comum, variância igual,
independência ou causalidade. Erro aritmético não é incerteza de medição.
Confiança, ignorância, validade e cobertura estatística precisam de definições
próprias. Canal ausente não é um zero numérico observado.

### 9.2 Unidades, ordem e ferramentas de verificação

Os fixtures escolhidos de soma/escala com unidades falharam no precompilado.
Continuam como diagnósticos, não como tutorial de unidades funcionando.
Use a especificação para identificar o contrato de anotação/operação e exija
recibo de execução no perfil nomeado antes de tratá-lo como rota funcionando.
Não troque de engine sem registrar outro perfil.

Preserve parênteses e ordem quando arredondamento ou álgebra não associativa
forem relevantes. Uma transformação válida em um modelo associativo não
preserva automaticamente o programa e seus canais científicos.

Compare segundo o domínio da tarefa. Uma tolerância de `check_near` herdada
não se transfere automaticamente para outra magnitude, valor excepcional,
incerteza pequena ou hipótese de distribuição. Siga o protocolo ADR009
atual em [`SPEC-011`, fundamentos](../spec/2026-10/00-foundations-and-conventions.md):
não substitua ferramentas numéricas autorizadas por um oracle conveniente.
Prova é Lean 4; concordância numérica e validação empírica têm outros critérios.

Um teste com GUM no nome não demonstra conformidade JCGM completa ou
rastreabilidade metrológica. Relação ontológica não é diagnóstico clínico;
simulação não é experimento em hardware. Preserve a hipótese e explicite
a evidência que ainda falta para testá-la.

<!-- llm:section modules -->
## 10. Importações, visibilidade, pacotes e APIs reais

Use o [índice de descoberta](../llm-guide/stdlib-index.md) como procedimento:

1. Localize módulo e declaração concretos.
2. Leia visibilidade, assinatura completa, efeitos e corpo.
3. Leia um consumidor real que faça a importação.
4. Preserve contexto de arquivos/módulos e raiz da biblioteca.
5. Cheque o consumidor no perfil escolhido.
6. Execute somente se houver entrada e a tarefa pedir comportamento runtime.

O [provedor](../llm-guide/examples.md#cook-import-provider) e o
[consumidor](../llm-guide/examples.md#cook-import-main) são check-only neste
corpus. Não chame isso de runtime aprovado. A fonte do consumidor retorna 42;
uma execução posterior precisa desse contrato, não do zero suposto.

Copiar um consumidor para um diretório temporário arbitrário pode mudar a
resolução de módulos. Preserve os arquivos companheiros e o contexto de busca
ou adapte-os explicitamente. Estar em `stdlib` não torna toda declaração
pública, importável, implementada no target ou numericamente verificada.

A busca atual de pacotes usa caminhos convencionais derivados da importação.
Campo de manifesto não prova resolução guiada pelo manifesto, seleção de
versões, registro público ou operação de publicação. Veja o
[capítulo de pacotes/ferramentas](../spec/2026-10/07-library-packages-and-tools.md).

Não invente helpers por analogia com bibliotecas conhecidas. Um nome
plausível, uma anotação gerada ou uma conclusão de LSP é candidato, não API.
Prefira a declaração real a um índice histórico extenso.

<!-- llm:section targets -->
## 11. Host, FFI, concorrência e dispositivos

Mantenha o target no pacote da tarefa. Launcher público, driver bruto, seed,
wrapper e programa emitido podem ter CLIs diferentes. Não passe verbos do
launcher para um compilador bruto posicional.

| Afirmação | Evidência necessária além do parser |
|---|---|
| Executável nativo | Formato/arquitetura, emissão, execução, exit e saída reais. |
| Chamada estrangeira | Símbolo, ABI, layout, efeitos, resolução e comportamento observado. |
| Async/concorrência | Runtime alcançável, ordem, recursos, falhas e escalonamento. |
| Handler | Despacho de operação, continuação e caminhos de check/lowering/runtime. |
| Resultado GPU | Runtime/dispositivo efetivo e saídas, não apenas PTX/MSL. |
| Hardware quântico | Recibo de job/circuito/device/shots/contagens, não simulador. |
| Resultado formal | Modelo, hipóteses, toolchain, build do checker e relação com produção. |

O [capítulo de execução](../spec/2026-10/06-execution-memory-and-targets.md)
identifica caminhos implementados e recusados. Apple Silicon tem uma rota
ARM64/Metal própria, não um ELF Linux distribuído que por acaso roda no macOS.
Emissão, assinatura, saída de processo e dispositivo real são observações
separadas.

Um nome de target aceito pela CLI ou schema MCP não prova suporte a todas
as operações. Não converta dispositivo ausente, IR não suportada, símbolo
estrangeiro ou runtime inexistente em fallback CPU bem-sucedido sem contrato
e divulgação explícitos.

<!-- llm:section diagnostics -->
## 12. Corrija a causa, não silencie o erro

Use o [guia de diagnósticos](../llm-guide/error-catalog.md). Identifique
engine, adaptador, revisão, etapa, stdout/stderr completos e a primeira
mensagem causal. O número isolado não basta: códigos podem colidir em
namespaces e rotas diferentes.

Um ciclo de reparo útil:

1. Leia o span e a declaração/chamada que o contém.
2. Inspecione a função chamada e a definição importada.
3. Compare com controle válido próximo sob o mesmo artefato e ambiente.
4. Corrija a menor causa preservando a tarefa.
5. Recheque e, se necessário, execute o arquivo alterado.
6. Verifique o negativo pertinente e registre a recusa específica.

Não remova testes, enfraqueça saídas esperadas, aumente tolerâncias sem
justificação do domínio, force casts científicos ou desligue verificações
de efeitos/ownership. Compilador quebrado ou dependência ausente não faz
todos os negativos passarem.

O atlas preserva oito casos que não cumpriram as condições esperadas. Dois
negativos de unidades retornaram não zero, mas sem o diagnóstico exigido:
não demonstram a fronteira pretendida. Os seis positivos falhos também não
entram no caminho de exemplos recomendados.

A maioria das explicações individuais por código não foi revalidada nesta
auditoria e ganhou aviso visível. E035 e E249 receberam correções específicas.
Restabeleça os demais exemplos antigos contra o artefato antes de aplicá-los.

Quando o reparo exigir escolher uma semântica ainda aberta, alterar API
pública ou consertar compilador/biblioteca fora da tarefa, forneça o reproducer
e a decisão necessária. Não troque a pergunta de pesquisa por outra mais
simples e anuncie que a original foi resolvida.

<!-- llm:section recipes -->
## 13. Caminho progressivo pelos programas completos

O [cookbook](../llm-guide/cookbook.md) explica lições e adaptações.
O [atlas](../llm-guide/examples.md) preserva as fontes completas.
O [manifesto](../llm-guide/example-manifest.json) registra verificações por
caso, sem inferir “produção” a partir de um nome de arquivo.

| Objetivo | Exemplo inicial |
|---|---|
| Imprimir e retornar resultado do processo | `hello` |
| Reatribuição e while | `while-loop`, `cook-while` |
| Iteração de array fixo | `cook-for-array` |
| Intervalos e iteração por opção | `cook-range-option` |
| Criação e campos de struct | `struct`, `struct-parameter`, `struct-return` |
| Tuplas e assinaturas de função | `cook-tuple`, `cook-tuple-signature` |
| Referências e mutação observável | `cook-borrow-release`, `cook-array-ref` |
| Match de enum | `cook-enum`, com exit correto 42 |
| Função nomeada como valor | `cook-function-ref` |
| Chamada genérica explícita | `generics` |
| Módulo provedor/consumidor | `cook-import-provider`, `cook-import-main`, somente check |
| APIs de resultado e vetor | `cook-result`, `cook-vec` |
| Epistêmico e EISA distintos | `cook-epistemic-library`, `cook-eisa` |
| Recusa intencional | `negative-*` verificado para aquela regra |

Leia a verificação antes de escolher o ID. Os casos em quarentena de closure,
genérico simples, par genérico, desestruturação aninhada e unidades são
exercícios de diagnóstico, não o primeiro caminho de sucesso.

Não conte IDs repetindo a mesma fonte como programas independentes.
São 50 entradas e 47 arquivos distintos: 30 resultados de execução,
quatro de checagem apenas, oito recusas corretas e oito contratos não
cumpridos. Nenhuma dessas contagens estabelece conformidade completa.

<!-- llm:section prompts -->
## 14. Prompts reutilizáveis

Os textos a seguir são templates, não conversas executadas nem benchmark.
Preencha os campos e forneça o exemplo referido. Resultado imaginado pelo
modelo não substitui chamada de ferramenta e saída real.

### Criar um programa

```text
Escreva um programa Sounio completo para [tarefa].
Entradas: [tipos, valores/fontes, hipóteses permitidas].
Observações exigidas: [stdout, exit, recursos/canais científicos].
Perfil: [engine, binário bruto/hash, host/target, stdlib].
Arquivos permitidos: [caminhos]. Não publique alterações.
Use o exemplo completo fornecido [ID] e declarações reais da API.
Preserve [unidades, incerteza, precisão, ownership, efeitos e ordem].
Não invente helpers. Informe pré-requisito ausente antes de prosseguir.
Cheque o arquivo alterado e execute para conferir o contrato exigido.
Se não houver execução disponível, entregue comandos e marque-os não rodados.
```

### Corrigir uma falha

```text
Diagnostique esta falha Sounio no perfil informado.
Aqui estão comando, exit, stdout, stderr e fonte completos.
Aqui estão declaração importada/chamada e controle válido próximo.
Identifique a primeira causa relevante e corrija o mínimo necessário.
Não apague tipos científicos, amplie efeitos indiscriminadamente, desligue
guards, altere expectativas, troque de engine ou reduza precisão em silêncio.
Recheque a mesma fonte/perfil e execute a observação afetada.
Relate antes/depois e o que continua sem evidência.
```

### Descobrir uma API

```text
Encontre uma API Sounio existente para [operação].
Recupere declaração, visibilidade, representação, assinatura de efeitos,
precondições/falhas do corpo e consumidor real que faça a importação.
Separe presença na fonte, importação aceita e execução.
Escolha exemplo completo verificado; não invente API de outra linguagem.
Preserve os canais solicitados e as restrições de ferramentas numéricas.
```

### Estender um programa científico

```text
Estenda a fonte fornecida sem alterar a pergunta de pesquisa.
Declare domínio do valor, modelo de incerteza/dependência/proveniência,
unidades, precisão e contrato de observação antes de editar.
Separe hipótese do modelo, comportamento do compilador, verificação numérica,
prova formal e evidência empírica. Use somente ferramentas autorizadas.
Se uma transformação perder informação ou exigir decisão semântica,
informe isso; não substitua a tarefa por uma conta mais conveniente.
```

Com modelos menores, uma mudança de cada vez. Com agentes maiores, permita
plano delimitado e recuperação de APIs, mantendo um escritor por arquivo
compartilhado e evidência explícita em cada etapa.

### Cartões de tarefa por domínio

Estes cartões tornam uma solicitação científica concreta sem fingir que um
prompt de domínio é um recibo do compilador. Anexe a fonte completa, a
declaração da biblioteca, a observação esperada e o perfil de execução. O
modelo deve recuperar a API atual do Sounio antes de escrever; os nomes abaixo
são áreas de busca, não autorização para inventar APIs. Mantenha hipóteses do
modelo, dados científicos, evidência do compilador, verificação numérica e
decisões de publicação em camadas separadas.

#### Química, cinética e materiais

```text
Você está auxiliando uma tarefa de química em Sounio.

Tarefa de pesquisa: [reação, equilíbrio, cinética, espectroscopia, materiais
                    ou outra pergunta precisa].
Entradas científicas: [espécies, concentrações, unidades, temperatura,
                       pressão, grade temporal, medidas, incerteza, proveniência].
Observação exigida: [tabela, curva, invariável, status, arquivo ou saída].
Arquivos permitidos: [caminhos exatos da fonte e dos testes]. Não publique.
Perfil: [revisão, compilador bruto/hash, host/target, raiz da stdlib].

Antes de editar:
1. Leia a fonte fornecida e recupere as declarações reais e um consumidor
   real para cada API de química, unidades, arrays, numérica ou resultado.
2. Liste espécies e unidades explicitamente. Diga se cada valor é medido,
   assumido, ajustado, simulado ou derivado, preservando a proveniência.
3. Declare precisão, incerteza/dependência, condições de contorno e falhas.
   Não converta unidades em silêncio, descarte incerteza nem troque o modelo
   de reação por uma conta mais simples.
4. Separe afirmação de check/type-check, observação de execução e verificação
   numérica autorizada. Simulação não é experimento; exemplo aprovado não é
   garantia química ou industrial.

Faça a menor mudança vinculada à fonte. Use apenas APIs confirmadas na fonte
e no perfil. Entregue arquivo completo ou patch exato e relate comando de
check, execução, exit, stdout, stderr, hash da fonte e passos não rodados.
Se faltar declaração ou pré-requisito, pare e identifique a lacuna; não
invente helper.
```

#### IA/ML, tensores e programas diferenciáveis

```text
Você está auxiliando uma tarefa de IA/ML em Sounio.

Classe: [preparação de dados, transformação tensorial, inferência, treino,
        autodiff, otimização, avaliação ou reprodutibilidade].
Entradas: [schema/fonte dos dados, shapes, eixos, dtype, precisão, dispositivo,
          seed, proveniência, restrições de privacidade].
Observação exigida: [shape/valor, loss, métrica, artefato, exit ou recusa].
Arquivos permitidos: [fonte, fixture e testes exatos]. Não publique.
Perfil: [revisão, compilador bruto/hash, host/target, stdlib, dispositivo/runtime].

Antes de editar:
1. Classifique a operação: construção, transformação, modelo, autodiff,
   otimizador, inferência ou avaliação. Recupere declarações reais e consumidor
   completo das fontes atuais de tensor, autodiff, ML ou exemplos; não importe
   uma API de outra linguagem por analogia.
2. Registre shape, rank, ordem dos eixos, dtype, precisão, batching, máscara,
   dispositivo, aleatoriedade e ownership. Preserve proveniência e incerteza;
   ausência de benchmark não pode virar alegação de desempenho.
3. Separe implementação, compilação, execução, benchmark, dados de treino e
   verificação estatística/numérica. Operação tensorial aprovada não é acurácia,
   segurança nem generalização.
4. Não fabrique dataset, checkpoint, acelerador, seed, métrica ou resultado.
   Se a capacidade não existir no perfil, declare a falta e proponha o próximo
   passo mínimo ligado à fonte.

Faça uma mudança mínima por vez. Entregue fonte completa ou patch exato e um
registro reproduzível: comandos, exits, streams brutos, hashes de fonte/
artefato, identidade dos dados e passos não rodados. Limite as conclusões ao
perfil observado.
```

#### Medicina, ontologia e PBPK somente para pesquisa

```text
Você está auxiliando uma tarefa de medicina, ontologia ou PBPK em Sounio
somente para pesquisa. Isto não é aconselhamento clínico, diagnóstico,
prescrição, validação de dispositivo nem recomendação para paciente.

Pergunta: [questão precisa de terminologia, ontologia, PK/PD, PBPK, coorte,
           mecanismo ou reprodutibilidade].
Entradas e governança: [dados desidentificados ou fixture sintético,
                        proveniência, unidades, população, hipóteses, permissão].
Observação exigida: [relação ontológica, registro tipado, simulação,
                     tabela de sensibilidade/resultado, recusa ou status].
Arquivos permitidos: [caminhos exatos da fonte e dos testes]. Não publique.
Perfil: [revisão, compilador bruto/hash, host/target, raiz da stdlib].

Antes de editar:
1. Recupere as declarações reais de ontologia, clínica, PBPK, incerteza e
   bibliotecas, além de um consumidor real. Classifique cada afirmação como
   fato de vocabulário/ontologia, hipótese de modelo, resultado de simulação,
   resultado do compilador, prova formal ou alegação clínica empírica.
2. Preserve proveniência, população/coorte, unidades, precisão, incerteza,
   dependência, ausências e governança. Não extraia conclusão individual de
   modelo populacional nem relação clínica de uma relação terminológica.
3. Mantenha distintas a correspondência ontológica, execução do modelo,
   verificação numérica e validação clínica. Código executado não prova
   diagnóstico, eficácia, segurança, aprovação regulatória ou benefício.
4. Use somente a cadeia numérica/probatória autorizada. Se faltar requisito
   clínico, de governança, de dados ou de execução, marque não verificado e
   pare, em vez de preencher a lacuna.

Faça a menor mudança vinculada à fonte. Entregue fonte ou patch exato e
registro limitado com comandos, exits, saída bruta, identidade de fonte/
artefato/dados, hipóteses e validações não rodadas. Inclua a limitação de
pesquisa em qualquer resumo voltado ao usuário.
```

<!-- llm:section delivery -->
## 15. Entrega verificável e limites honestos

Antes de devolver o programa, confira:

- Arquivo inteiro, declarações, importações, entrada quando pertinente e
  efeitos exigidos pelas operações.
- A fonte checada/executada é a alterada e corresponde ao hash registrado.
- Artefato, engine, adaptador, host/target, ambiente e raiz da biblioteca
  identificados; resultados CLI/MCP não misturados silenciosamente.
- Checagem, emissão, execução e asserções de saída relatadas separadamente.
- Negativos recusados pelo motivo pretendido, não por parse alheio,
  dependência, timeout ou crash.
- Unidades, precisão, variância, proveniência, ownership e ordem preservados.
- Verificação numérica, Lean, corpus completo, GPU, CI remota, publicação e
  aceitação humana não declarados quando não executados.

Use o registro de entrega em [`llms.txt`](../../llms.txt#delivery-record).
O [relatório de preparação e revisão](../llm-guide/reviews/guide-preparation.md)
explica a evidência deste próprio guia. Não se infere taxa de sucesso de
modelos a partir de fixtures que passaram.

Inglês, português, receitas e atlas são mantidos juntos.
`node scripts/docs/validate_llm_guide.mjs` verifica cobertura temática,
identidade de fontes, blocos completos, quarentena e links. Os testes de
mutação verificam contratos estruturais; nem o validador nem seus testes
executam o compilador ou provam a prosa. Recibos de execução e decisões
semânticas mantêm critérios próprios.
