<!-- docs:meta
topic_id: repo.docs.guide.sounio-gotchas-pt
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.guide.sounio-gotchas-pt
-->

# Armadilhas Comuns e Erros Frequentes em Sounio (Gotchas)

Este guia cobre os erros mais comuns cometidos por desenvolvedores e LLMs ao programar em Sounio. Estude-os para evitar falhas de compilação e comportamento inesperado.

**Referência completa de sintaxe**: [docs/guide/LLM_PROGRAMMING_GUIDE_PT.md](LLM_PROGRAMMING_GUIDE_PT.md)

## 1. PONTOS E VÍRGULAS - O Erro Número 1

### Exemplo de Erro
```sio
// [INCORRETO]
let x = 5;
let y = 10;
fn foo() -> i32 {
    let result = x + y;
    result;
}
```

### Por que está Errado
- As expressões e declarações em Sounio NÃO terminam com ponto e vírgula.
- O parser trata o caractere `;` como um separador de instruções (para colocar múltiplos comandos na mesma linha), e não como um terminador.
- Adicionar um `;` ao final de uma expressão altera o seu tipo de retorno para `()`, quebrando a compilação.

### Correção
```sio
// [CORRETO]
let x = 5
let y = 10
fn foo() -> i32 {
    let result = x + y
    result
}
```

---

## 2. `&mut` vs `&!`

### Exemplo de Erro
```sio
// [INCORRETO] - Sintaxe herdada do Rust
fn increment(x: &mut i32) with Mut {
    *x = *x + 1
}
var counter: i32 = 0
increment(&mut counter)
```

### Correção
```sio
// [CORRETO] - Sounio utiliza &! (dois tokens: & seguido de !)
fn increment(x: &!i32) with Mut {
    *x = *x + 1
}
var counter: i32 = 0
increment(&!counter)
```

---

## 3. EFEITOS AUSENTES (MISSING EFFECTS)

### Exemplo de Erro
```sio
// [INCORRETO] - Falta declarar o efeito Mut
fn set_value(x: &!i32) {
    *x = 42  // ERRO: mutação requer 'with Mut'
}

// [INCORRETO] - Falta declarar os efeitos Div e Panic
fn divide(a: f64, b: f64) -> f64 {
    a / b  // ERRO: divisão requer 'with Div, Panic'
}

// [INCORRETO] - Falta declarar o efeito IO
fn say_hello() {
    println("Hello")  // ERRO: I/O requer 'with IO'
}
```

### Correção
```sio
// [CORRETO]
fn set_value(x: &!i32) with Mut { *x = 42 }
fn divide(a: f64, b: f64) -> f64 with Div, Panic { a / b }
fn say_hello() with IO { println("Hello") }
```

### Referência de Efeitos
| Efeito | Quando é Necessário |
|--------|---------------------|
| `Mut` | Mutação através de referência exclusiva `&!`, ou atribuição em arrays |
| `Div, Panic` | Divisão `/` ou módulo `%` |
| `Panic` | Acesso a índices de arrays, asserções com `assert()` e conversões com `as` |
| `IO` | Uso de `print()`, `println()` e operações com arquivos |

---

## 4. MACROS DE RUST NÃO EXISTEM

### Exemplo de Erro
```sio
// [INCORRETO] - Macros com ponto de exclamação !
assert!(x == 5)
println!("hello {}", name)
vec![1, 2, 3]
```

### Correção
```sio
// [CORRETO] - Funções normais em Sounio (sem exclamação)
assert(x == 5)
println("hello")
print(name)
```

---

## 5. NÚMEROS NEGATIVOS E MENOS UNÁRIO

### Exemplo de Erro
```sio
// [INCORRETO] - O operador menos unário não existe na gramática de Sounio
let neg = -42
let result = -x
```

### Correção
```sio
// [CORRETO]
let neg = 0 - 42
let result = 0 - x
let value = a - (0 - b)  // Equivalente a a + b
```

---

## 6. DESLOCAMENTO DE BITS REQUER OPERANDO u8

### Exemplo de Erro
```sio
// [INCORRETO] - O operando de deslocamento (shift amount) precisa ser estritamente u8
let shifted = byte >> 4       // ERRO: 4 é inferido como i32!
```

### Correção
```sio
// [CORRETO]
let shifted = byte >> 4u8
let masked = byte & 15u8
let high = (byte >> 4u8) & 15u8
```

---

## 7. INCOMPATIBILIDADE DE TAMANHO DE ARRAYS

### Exemplo de Erro
```sio
// [INCORRETO] - O tamanho de inicialização deve coincidir exatamente com o tamanho do tipo
var small_buffer: [u8; 10] = [0; 256]  // ERRO: 256 != 10!
```

### Correção
```sio
// [CORRETO]
var buffer: [u8; 256] = [0; 256]
```

---

## 8. LITERAIS DE CLOSURE vs REFERÊNCIAS DE FUNÇÃO

### Exemplo de Erro
```sio
// [INCORRETO] - Closures anônimas ou literais de lambda são BLOQUEADOS
let doubled = numbers.iter().map(|x| x * 2).collect()
let callback = |x| { x + 1 }
```

### Por que está Errado
- Sounio NÃO possui suporte a closures literais (`|x| expr`).
- Contudo, **referências de funções nomeadas funcionam perfeitamente** como valores de primeira classe.

### Correção
```sio
// [CORRETO] - Referências de função nomeadas
fn double(x: i64) -> i64 { x * 2 }
fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

let f = double            // Armazena a referência da função em uma variável
let r = f(7)              // Chamada através da variável: retorna 14
let r2 = apply(double, 5) // Passagem como argumento: retorna 10

// Padrões de ordem superior funcionam (exemplo de stdlib)
fn map4(arr: [i64; 4], f: fn(i64) -> i64) -> [i64; 4] with Mut, Panic, Div {
    var out: [i64; 4] = [0; 4]
    var i: i64 = 0
    while i < 4 { out[i] = f(arr[i]); i = i + 1 }
    out
}
let doubled = map4(data, double)
```

---

## 9. MÉTODOS EM TIPOS PRIMITIVOS DO CORE

### Exemplo de Erro
```sio
// [INCORRETO] - Tipos nativos do core não têm métodos associados diretamente
let text = "hello"
let len = text.len()
let upper = text.to_uppercase()
let first = arr.first()
```

### Por que está Errado
- Os tipos primitivos e fundamentais (`i32`, `[T;N]`, literais de string) não possuem métodos orientados a objeto.
- Contudo, os **tipos declarados na stdlib possuem métodos** através de blocos `impl`.

### Correção
```sio
// Para tipos primitivos do core: use laços manuais e funções auxiliares
var i = 0
while i < len {
    process(array[i as usize])
    i = i + 1
}

// Para tipos estruturados e coleções da stdlib: métodos impl funcionam
impl IntVec {
    fn len(self: &IntVec) -> i64 { self.len }
    fn push(self: &! IntVec, val: i64) { /* ... */ }
}
```

---

## 10. CONVERSÕES EXPLÍCITAS DE TIPO COM `as`

### Exemplo de Erro
```sio
// [INCORRETO] - Conversões implícitas não são suportadas em indexação
let i: i32 = 5
let arr: [u8; 256] = [0; 256]
let val = arr[i]     // ERRO: requer [usize], não [i32]
```

### Correção
```sio
// [CORRETO]
let val = arr[i as usize]
let u: u8 = i as u8
```

---

## 11. TRATAMENTO DE EXCEÇÕES NÃO EXISTE

### Exemplo de Erro
```sio
// [INCORRETO]
try {
    let x = risky_operation()
} catch (error) {
    print("Error!")
}
```

### Correção
```sio
// [CORRETO] - Retorne tuplas de status ou códigos de erro
fn divide(a: f64, b: f64) -> (f64, i32) with Div, Panic {
    if b == 0.0 { (0.0, 1) }
    else { (a / b, 0) }
}
let (result, err) = divide(10.0, 0.0)
if err != 0 { println("erro de divisao") }
```

---

## 12. ERRO DE CORRUPÇÃO EM MUTAÇÃO DE ARRAY BRUTO com `&!`

### Exemplo de Erro
```sio
// [COMPORTAMENTO INDESEJADO] - O interpretador perde mutações locais em arrays brutas
fn sort_broken(arr: &![i64; 10000]) with Mut {
    arr[0] = 99  // Mutação invisível ao chamador em alguns caminhos JIT antigos!
}
```

### Por que Ocorre
- Algumas revisões do interpretador possuem limitações no rastreamento de mutações diretas em `&![T; N]`.
- O padrão de empacotamento em `struct` ou de desreferenciação explícita propaga as alterações corretamente.

### Correção
```sio
// [CORRETO] - Empacotar em uma Struct
struct SortBuf { data: [i64; 10000] }
fn sort(b: &! SortBuf) with Mut {
    b.data[0] = 99  // Funciona perfeitamente
}

// [CORRETO] - Desreferenciação Explícita (Funciona perfeitamente no JIT e Compilador Native)
fn fill(arr: &![i64; 8]) with Mut, Panic {
    (*arr)[0] = 99   // Desreferenciação explícita para forçar a mutação através do ponteiro
}
```

---

## 13. MANIPULAÇÃO DE STRINGS

### Exemplo de Erro
```sio
// [INCORRETO] - Strings não são objetos de alocação dinâmica extensíveis por padrão
let greeting = "Hello"
greeting.push_str(" World")
let upper = greeting.to_uppercase()
```

### Correção
```sio
// [CORRETO] - Use strings literais estáticas para formatação e saída
println("Hello, World!")

// Para strings mutáveis em buffers: use arrays fixas de bytes
var greeting: [i8; 64] = [0; 64]
greeting[0] = 72i8   // 'H'
greeting[1] = 101i8  // 'e'
```

---

## 14. INTERFACE DE FUNÇÃO ESTRANGEIRA (FFI) RESTRITA A MATEMÁTICA

### Exemplo de Erro
```sio
// [INCORRETO] - Chamadas de FFI que retornam inteiros complexos podem falhar silenciosamente no JIT
extern "C" { fn malloc(size: i64) -> i64 }
extern "C" { fn getpid() -> i32 }
```

### Por que está Errado
- Atualmente, no backend JIT, apenas funções matemáticas de assinatura `f64 -> f64` e `(f64, f64) -> f64` são garantidas para execução direta nativa.
- Funções FFI complexas não matemáticas devem usar stubs gerados no compilador native-v2.

### Correção
```sio
// [CORRETO] - Funções nativas da biblioteca matemática padrão
extern "C" {
    fn sqrt(x: f64) -> f64
    fn sin(x: f64) -> f64
    fn pow(x: f64, y: f64) -> f64
}
// Lista completa suportada: sqrt, sin, cos, tan, exp, log, floor, ceil, atan, sinh,
//                          cosh, tanh, asin, acos, cbrt, round, log2, log10, pow, atan2.
```

---

## Lista de Verificação (Checklist) Antes de Salvar Código Sounio

- [ ] Nenhum ponto e vírgula `;` foi colocado no final de instruções ou funções.
- [ ] Usou-se `&!` para referências mutáveis em vez de `&mut`.
- [ ] Usou-se `var` para variáveis reatribuíveis em vez de `let mut`.
- [ ] Todos os efeitos colaterais foram declarados (ex: `with Mut, Div, Panic, IO`).
- [ ] Nenhuma macro de Rust foi utilizada (`assert` em vez de `assert!`, `println` em vez de `println!`).
- [ ] Menos unário foi substituído por `0 - x` (ex: `0 - 42` em vez de `-42`).
- [ ] Deslocamento de bits usa operando `u8` (ex: `x >> 4u8`).
- [ ] Tamanhos das arrays coincidem perfeitamente na inicialização (ex: `[u8; 256] = [0; 256]`).
- [ ] Conversões de tipo na indexação de arrays são explícitas (ex: `arr[i as usize]`).
- [ ] Referências de função nomeadas foram usadas no lugar de lambdas/closures literais.
- [ ] Erros são retornados via tuplas ou códigos de status, sem cláusulas `try/catch`.
- [ ] Alterações em `&![T; N]` usam `(*arr)[i]` ou estrutura wrapper.
- [ ] Funções FFI declaradas limitam-se ao escopo matemático padrão (`sqrt`, `sin`, etc.).

---

**Resumo**: Sounio NÃO é Rust. Estude a pasta `tests/run-pass/` para exemplos plenamente validados pelo compilador e consulte [docs/guide/LLM_PROGRAMMING_GUIDE_PT.md](LLM_PROGRAMMING_GUIDE_PT.md).
