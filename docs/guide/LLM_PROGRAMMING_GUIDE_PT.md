<!-- docs:meta
topic_id: repo.docs.guide.llm-programming-guide-pt
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.llm-programming-guide-pt
-->

# Guia de Programação Sounio para LLMs

Referência definitiva de sintaxe para LLMs que escrevem código em Sounio. Todos os exemplos foram verificados diretamente a partir de `tests/run-pass/` ou de arquivos ativos da biblioteca padrão (`stdlib/`).

**Sounio NÃO é Rust.** Embora a sintaxe pareça semelhante à primeira vista, a semântica e a gramática diferem significativamente. Em caso de dúvida, verifique arquivos reais de extensão `.sio`.

---

## 1. Olá, Mundo (Hello World)

```sio
// Origem: tests/run-pass/hello.sio
fn main() with IO {
    println("Hello, Sounio!")
}
```

---

## 2. Variáveis

```sio
let x = 5                    // Imutável
var y: i32 = 10              // Mutável (pode ser reatribuída)
y = y + 1                    // Correto: var permite reatribuição

// NENHUM ponto e vírgula ao final das linhas!
// let x = 5;   <-- INCORRETO
```

*   **Não use `let mut`** — use `var` para bindings mutáveis.

---

## 3. Tipos de Dados

### Primitivos
`i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `bool`, `char`

### Arrays de Tamanho Fixo [Produção]
```sio
var buffer: [u8; 256] = [0; 256]
let data: [i64; 4] = [1, 2, 3, 4]
let matrix: [f64; 9] = [0.0; 9]
```

### Vetores Dinâmicos (Vec) [Produção]
```sio
// Origem: tests/run-pass/for_in_loops.sio:29
let vec: Vec<i32> = [1, 2, 3, 4]
for x in vec {
    // laço de iteração
}
```
A biblioteca padrão (`stdlib/`) fornece tipos vetoriais monomórficos rápidos como `IntVec` e `FloatVec` através de `stdlib/collections/vec.sio`, incluindo métodos como `push`, `pop` e `len`.

### Tuplas
```sio
let pair = (1, 2)

// Desestruturação funciona diretamente:
// Origem: tests/run-pass/tuple_destructure_let.sio
let (a, b) = (1, 2)
let (x, (y, z)) = (10, (20, 30))
let (first, _) = (5, 10)       // caractere coringa (wildcard)
```

### Estruturas (Structs) [Produção]
```sio
struct Point { x: f64, y: f64 }
let p = Point { x: 1.0, y: 2.0 }

linear struct Handle { fd: i32 }   // Tipos lineares (devem ser consumidos)
```

### Enumerações (Enums) [Beta]
```sio
// Origem: tests/run-pass/native_enum_basic.sio
enum Color { Red, Green, Blue }

let r = Color::Red
let g = Color::Green
```
*Nota*: Definição de enums e acesso a variantes funcionam. Passar valores de enums para funções que esperam inteiros primitivos pode exigir conversão explícita com `as`, pois o verificador de tipos distingue tipos enums de números comuns.

### Tipos de Refinamento (Refinement Types) [Beta]
```sio
type Probability = { p: f64 | p >= 0 }
fn divide(num: i32, denom: { d: i32 | d != 0 }) -> i32 with Panic {
    num / denom
}
```

### Unidades de Medida (Units of Measure) [Produção]
```sio
unit kg;
unit mg = 0.001 * kg;
let dose: mg = 500.0
```

### Tipos Epistêmicos [Produção]
```sio
let m: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
```

### Declarações de Álgebra [Beta]
```sio
algebra Octonion over f64 {
    add: commutative, associative
    mul: alternative, non_commutative
    reassociate: fano_selective
}
```
Propriedades algébricas suportadas:
-   `add`: `commutative`, `associative`
-   `mul`: `commutative`, `associative`, `alternative`, `non_commutative`
-   `reassociate`: `free`, `blocked`, `fano_selective`

A propriedade `mul: alternative` deriva automaticamente requisitos de não-associatividade (`NonAssoc`) para funções que multiplicam tipos estruturados sob este domínio.

### Tipos de Observação [Beta]
```sio
fn sense() -> Unobserved<f64> with Observe {
    37.2
}

fn above_threshold(x: Unobserved<f64>) -> bool with Observe {
    x > 36.0
}
```
O wrapper `Unobserved<T>` transporta um valor antes de sua observação formal. Comparações e outras barreiras de observação requerem obrigatoriamente a declaração e o uso do efeito `with Observe`. Funções puras podem repassar instâncias de `Unobserved<T>` sem restrições.

---

## 4. Funções e Métodos

```sio
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// Funções com efeitos declarados (OBRIGATÓRIO para certas operacoes)
fn divide(a: f64, b: f64) -> f64 with Div, Panic {
    a / b
}

// Retorno antecipado explícito
fn abs(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    x
}
```

### Referências de Função [Produção]
```sio
// Origem: tests/run-pass/closure_fn_ref.sio
fn square(x: i64) -> i64 { x * x }
fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

// Armazenamento em variáveis
let f = square
let r = f(7)          // 49

// Retorno a partir de funções
fn select_op(which: i64) -> fn(i64) -> i64 with Mut, Panic, Div {
    if which == 0 { add_one }
    else { negate }
}
```

### Padrões de Ordem Superior [Produção]
```sio
// Origem: tests/run-pass/closure_higher_order.sio
fn map4(arr: [i64; 4], f: fn(i64) -> i64) -> [i64; 4] with Mut, Panic, Div {
    var out: [i64; 4] = [0; 4]
    var i: i64 = 0
    while i < 4 { out[i] = f(arr[i]); i = i + 1 }
    out
}

fn fold4(arr: [i64; 4], init: i64, f: fn(i64, i64) -> i64) -> i64 with Mut, Panic, Div {
    var acc = init
    var i: i64 = 0
    while i < 4 { acc = f(acc, arr[i]); i = i + 1 }
    acc
}

// Utilizacao:
let doubled = map4(data, dbl)
let sum = fold4(data, 0, add)
let sum_sq = fold4(map4(data, sq), 0, add)   // encadeamento
```

*   **Nota:** Construtores literais de closures anônimas (`|x| x + 1`) são BLOQUEADOS. Apenas referências de funções nomeadas declaradas previamente são válidas.

### Blocos impl [Produção]
```sio
// Origem: stdlib/collections/vec.sio
impl IntVec {
    fn new() -> IntVec {
        IntVec { data: [0; 4096], len: 0 }
    }

    fn push(self: &! IntVec, val: i64) {
        if self.len < VEC_CAP {
            self.data[self.len] = val
            self.len = self.len + 1
        }
    }

    fn len(self: &IntVec) -> i64 {
        self.len
    }
}
```
Os métodos devem declarar o receptor de forma explícita através de `self: &Type` ou `self: &! Type` — não há busca ou vinculação implícita do receptor `self`.

---

## 5. Sistema de Efeitos [Produção]

Os efeitos rastreiam os efeitos colaterais de uma função. A ausência de declaração de efeitos resulta em erro de compilação imediato.

| Efeito | Exigido Quando | Exemplo de Operação |
|--------|----------------|---------------------|
| `IO` | Impressão em console, operações com arquivos e variáveis de ambiente | `println("texto")` |
| `Mut` | Mutação de referências exclusivas `&!` ou escrita em arrays | `arr[i] = 42`, `*x = 10` |
| `Div` | Operações de divisão `/` ou resto de divisão `%` | `a / b` (deve ser declarado junto com `Panic`) |
| `Panic` | Índices de arrays, asserções de validade e conversões de tipo (`as`) | `arr[i]`, `assert(cond)` |
| `Alloc` | Alocação dinâmica de memória no heap | Operação interna rara |
| `Observe` | Conversão/observação de variáveis do tipo `Unobserved<T>` | `if leitura > 36.0 { ... }` |
| `Async` | Execução paralela e operações assíncronas concorrentes | `spawn { ... }` |
| `GPU` | Codegen e kernels de GPU nativos | Operação especializada |
| `Prob` | Execuções probabilísticas sob amostragem | Operação de amostragem |

```sio
fn pure_add(a: i64, b: i64) -> i64 { a + b }                   // Sem efeitos = função puramente matemática
fn mutate(x: &!i32) with Mut { *x = 42 }                        // Mutação local
fn divide(a: f64, b: f64) -> f64 with Div, Panic { a / b }      // Divisão aritmética
fn observe(x: Unobserved<f64>) -> bool with Observe { x > 0.0 } // Barreira de observação
fn process() with IO, Mut, Panic, Div { /* múltiplos */ }       // Propagação múltipla de efeitos
```

Os efeitos colaterais propagam-se de forma ascendente. Uma função chamadora com efeitos declarados mais abrangentes (ex: `with IO, Mut`) pode invocar uma função com efeitos mais restritos (ex: puras ou apenas `with Mut`).

---

## 6. Controle de Fluxo

### Condicionais if/else [Produção]
```sio
if x > 0 {
    println("positivo")
} else if x < 0 {
    println("negativo")
} else {
    println("zero")
}

// Como expressão (retorna valor)
let result = if condition { value1 } else { value2 }
```

### Laço while [Produção]
```sio
var i = 0
while i < 10 {
    process(i)
    i = i + 1
}
```

### Laço for-in [Produção]
```sio
// Origem: tests/run-pass/for_in_loops.sio

// Intervalo numérico semi-aberto (exclusivo)
for i in 0..5 { /* executa para: 0, 1, 2, 3, 4 */ }

// Intervalo numérico fechado (inclusive)
for i in 0..=5 { /* executa para: 0, 1, 2, 3, 4, 5 */ }

// Intervalo com limite dinâmico (variável)
let n = 10
for i in 0..n { /* executa de 0 a 9 */ }

// Iteração direta sobre Arrays
let arr = [10, 20, 30]
for x in arr { sum = sum + x }

// Iteração direta sobre Vetores da stdlib (Vec)
let vec: Vec<i32> = [1, 2, 3, 4]
for x in vec { sum = sum + x }

// Laços aninhados funcionam perfeitamente
for i in 0..3 {
    for j in 0..3 { /* ... */ }
}

// Expressões break e continue
for i in 0..100 {
    if i >= 5 { break }
}
for i in 0..10 {
    if i % 2 == 0 { continue }
    odd_sum = odd_sum + i
}
```

### Correspondência de Padrões (match) [Produção]
```sio
// Origem: tests/run-pass/native_enum_basic.sio
fn color_to_int(c: i64) -> i64 {
    match c {
        Color::Red => 10
        Color::Green => 20
        Color::Blue => 30
        _ => 0
    }
}
```

---

## 7. Referências e Ponteiros

### Referência Compartilhada (Shared Reference) `&T` [Produção]
```sio
fn read_ref(r: &i64) -> i64 { *r }
let val = read_ref(&x)
```

### Referência Exclusiva (Exclusive Reference) `&!T` [Produção]
```sio
// Origem: tests/run-pass/array_mut_ref.sio
fn fill(arr: &![i64; 8]) with Mut, Panic {
    (*arr)[0] = 99       // DESREFERENCIAÇÃO EXPLÍCITA obrigatória para arrays básicas
    (*arr)[1] = 42
}

fn main() -> i64 with IO, Mut, Panic {
    var buf: [i64; 8] = [0, 0, 0, 0, 0, 0, 0, 0]
    fill(&! buf)         // Nota: espaço opcional entre o & e o !
    buf[0]               // retorna 99
}
```

*   **Aviso de Limitação**: Modificações em arrays puras passadas por `&![T; N]` podem se perder no interpretador sob algumas condições. Como contorno seguro, use o padrão de empacotamento em `struct`.

```sio
// Padrão de contorno de segurança — wrap em struct
struct SortBuf { data: [i64; 10000] }
fn sort(b: &! SortBuf) with Mut { b.data[0] = 99 }   // funciona perfeitamente
```

---

## 8. Operadores

### Aritmética
`+`, `-`, `*`, `/` (exige `Div`), `%` (exige `Div`)

### Comparação
`==`, `!=`, `<`, `<=`, `>`, `>=`

### Lógica
`&&`, `||`, `!` (com curto-circuito)

### Manipulação de Bits
`&`, `|`, `^`, `>>`, `<<`

*   **Operandos de deslocamento devem ser explicitamente `u8`:**
```sio
let high = byte >> 4u8
let low = byte & 15u8
```

### Proibição de Menos Unário
```sio
let neg = 0 - 42       // CORRETO
// let neg = -42        // ERRO DE COMPILAÇÃO
```

### Concatenação de Arrays
```sio
let combined = a ++ b   // Concatenação de arrays estáticas
```

### Conversão de Tipos (Type Casting)
```sio
let u: u8 = i as u8
let idx = n as usize    // Obrigatório para indexação de arrays locais
```

---

## 9. Módulos e Importações [Produção]

```sio
// Origem: tests/run-pass/import_basic_main.sio
use import_basic_a::{imported_add}

fn main() -> i64 {
    let result = imported_add(3, 4)
    result
}
```
A visibilidade de elementos entre módulos é controlada usando as palavras-chave `pub fn` e `pub struct`.

```sio
// stdlib/encoding/hex.sio
pub fn hex_encode(data: &[u8; 256], data_len: i32, out: &![u8; 512]) -> i32
    with Mut, Div, Panic { /* ... */ }
```

---

## 10. Manipulação de Cadeias de Texto (Strings)

Literais de string estáticos funcionam diretamente para rotinas de console e depuração:
```sio
println("Hello, World!")
print("value = ")
```

Para strings dinâmicas mutáveis, utilize buffers de tamanho fixo em formato de arrays de bytes:
```sio
var name: [i8; 64] = [0; 64]
name[0] = 72i8    // caractere 'H'
name[1] = 101i8   // caractere 'e'
```

---

## 11. Tratamento de Erros

Não há exceções ou blocos try-catch. Utilize retorno múltiplo por meio de tuplas de status ou enums de opção:

```sio
// Abordagem simples: tupla contendo o código de erro
fn safe_divide(a: f64, b: f64) -> (f64, i32) with Div, Panic {
    if b == 0.0 { (0.0, 1) }    // código de erro 1
    else { (a / b, 0) }          // sucesso (código 0)
}

// A biblioteca padrão fornece os tipos monomórficos mais comuns:
// stdlib/core/result.sio — IntResult, FloatResult
// stdlib/core/option.sio — IntOption, FloatOption
```

---

## 12. Depuração e Asserções

```sio
// Asserções em linha (inline assertions)
fn test_addition() {
    let result = add(2, 3)
    assert(result == 5)
}

// Funções de verificação aproximada em stdlib/test/helpers.sio
pub fn check_near(a: f64, b: f64, tol: f64) -> bool {
    let d = a - b
    let ad = if d < 0.0 { 0.0 - d } else { d }
    ad < tol
}
```

Anotações de cabeçalho comuns suportadas no executor de testes do repositório:
-   `//@ run-pass` — O teste deve compilar e rodar com saída bem-sucedida.
-   `//@ compile-fail` — O teste deve falhar durante a fase de análise ou type-checking.
-   `//@ error-pattern: <texto>` — Padrão de erro de diagnóstico esperado no terminal.
-   `//@ ignore` — Ignora este teste durante a execução da suite.

---

## 13. Interface de Função Estrangeira (FFI) [Produção, Limitado]

```sio
extern "C" {
    fn sqrt(x: f64) -> f64
    fn pow(x: f64, y: f64) -> f64
}
```

**Funções FFI Matematica Suportadas nativamente (JIT):**
-   Assinatura `f64 -> f64`: `sqrt`, `sin`, `cos`, `tan`, `exp`, `log`, `floor`, `ceil`, `atan`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `cbrt`, `round`, `log2`, `log10`.
-   Assinatura `(f64, f64) -> f64`: `pow`, `atan2`.
-   *Aviso*: Chamadas diretas de FFI que lidam com ponteiros e alocação dinâmica de inteiros (`malloc`, etc.) podem causar falha silenciosa de interrupção em JIT — evite usá-las, preferindo as stubs dedicadas integradas ao compilador native-v2.

---

## 14. Programação Assíncrona [Produção — Apenas Execução Native]

Todas as primitivas assíncronas do Sounio utilizam o modelo de isolamento fork do sistema operacional (Copy-on-Write), garantindo alta velocidade e segurança contra vazamento de memória. Requer a declaração do efeito `with Async`.

```sio
// Origem: tests/run-pass/async_spawn.sio
fn main() with IO, Async {
    let h1 = spawn { 10 + 5 }     // Processamento paralelo via fork
    let h2 = spawn { 20 + 1 }
    let r1 = h1.await              // Sincroniza e consome o resultado do mmap
    let r2 = h2.await
    print("r1="); print_i64(r1)   // Exibe 15
    print(" r2="); print_i64(r2)  // Exibe 21
}
```

### Canais de Comunicação (Pipe-Backed)

```sio
// Origem: tests/run-pass/async_channels.sio
fn main() with IO, Async {
    let (tx, rx) = channel::<i64>()
    let h = spawn { tx.send(42).await }
    let v = rx.recv().await
    h.await
    print_i64(v)   // Retorna 42
}
```

### Temporizadores `sleep(ms).await`

```sio
// Origem: tests/run-pass/async_sleep.sio
fn main() with IO, Async {
    sleep(10).await               // Nano-suspensão de 10 milissegundos
    let t1 = spawn { sleep(5).await; 1 }
    let t2 = spawn { sleep(5).await; 2 }
    let r1 = t1.await
    let r2 = t2.await             // Ambos executados paralelamente no SO
}
```

### Operação de Junção `join(h1, h2)`

```sio
// Origem: tests/run-pass/async_join.sio
fn main() with IO, Async {
    let h1 = spawn { 10 }
    let h2 = spawn { 20 }
    let (r1, r2) = join(h1, h2)  // Retorna uma tupla de valores (i64, i64)
}
```

**Regras estritas da concorrência assíncrona:**
-   `spawn { bloco }` requer obrigatoriamente a declaração `with Async` no escopo da função chamadora.
-   O processo filho em execução concorrente opera em isolamento total, não sendo capaz de alterar o valor de variáveis locais do processo pai (comportamento nativo de isolamento de memória virtual fork).
-   `join` suporta exatamente 2 manipuladores de thread. Para junções mais complexas, execute aguardas consecutivas individuais por meio do método `.await`.

---

## 15. Declaração de Ontologias [Produção]

```sio
// Origem: tests/run-pass/ontology_roles_basic.sio
ontology Pharma {
    class Drug
    class Disease
    class Rapamycin subclass_of Drug
    role treats domain Drug range Disease
    role treated_by inverse_of treats
    role has_part transitive
    disjoint Drug, Disease

    class StrongDrug subclass_of Drug {
        property potency: f64 where potency >= 10.0
    }
}
```

As classes declaradas na ontologia tornam-se tipos válidos que podem ser passados em assinaturas de funções e verificados em tempo de compilação. Disjunção, herança e subsumpção semântica são plenamente verificadas e validadas de forma estática pelo compilador.

---

## 16. Blocos de Estudo Clínico (Clinical Study) [Beta]

```sio
// Origem: tests/run-pass/study_block_basic.sio
study MyTrial {
    title: "Rapamycin Dosing Study"
    design: parallel_rct
    participants { sample_size: 120, power: 0.80 }
    outcomes { primary: blood_concentration }
    analysis {
        hypothesis H1 { outcome: blood_concentration, direction: greater, effect_size: 0.5 }
        alpha: 0.05
        correction: bonferroni
    }
}
```

Blocos declarativos consistentes com as especificações internacionais de saúde CONSORT, rastreando hipóteses pré-registradas, correções de testes múltiplos e proveniência para auditoria rigorosa de ensaios clínicos.

---

## 17. O Que NÃO Funciona no Compilador (Verificado)

| Recurso | Status de Restrição | Solução Alternativa |
|---------|---------------------|---------------------|
| Ponto e vírgula `;` | Bloqueado | Remova o caractere `;` das extremidades |
| Referência mutável `&mut` | Proibido | Substitua pela sintaxe `&!` |
| Binding de mutação `let mut` | Proibido | Substitua pela palavra-chave `var` |
| Macros de Rust (`assert!`, etc.) | Proibido | Use chamadas comuns: `assert()`, `println()` |
| Literais de closure (`|x| x+1`) | Bloqueado | Use referências a funções nomeadas normais |
| Atributos de cabeçalho (`#[test]`) | Bloqueado | Defina funções locais de teste comuns |
| Menos unário (`-42`) | Bloqueado | Substitua pelo termo aritmético `0 - 42` |
| FFI para inteiros complexos | JIT instável | Use os stubs nativos pré-compilados |

---

## 18. Códigos de Exemplo para Estudo

| Arquivo de Exemplo | O Que Demonstra Praticamente |
|--------------------|-----------------------------|
| `tests/run-pass/hello.sio` | Primeiros passos, `println` e declaração básica do efeito `IO` |
| `tests/run-pass/for_in_loops.sio` | Variações do laço for-in, inclusive, exclusivo e desvios de controle |
| `tests/run-pass/closure_fn_ref.sio` | Referências de funções e passagem como parâmetro de ordem superior |
| `tests/run-pass/closure_higher_order.sio` | Implementações reais dos laços map e fold sobre dados primitivos |
| `tests/run-pass/native_enum_basic.sio` | Definições de enums, correspondência estática e análise de padrões `match` |
| `tests/run-pass/array_mut_ref.sio` | Mutação de arrays estáticos via referências exclusivas `&!` com deref |
