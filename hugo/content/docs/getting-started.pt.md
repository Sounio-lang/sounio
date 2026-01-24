---
title: "Getting Started with Sounio"
description: "Installation and first steps with the Sounio compiler"
layout: "docs"
---

Bem-vindo ao **Sounio**, uma linguagem de programação de sistemas para epistemic computing — onde todo valor pode carregar sua incerteza.

## Instalação

### A Partir de Binário (Recomendado)

Baixe a versão mais recente para sua plataforma:

```bash
# Linux/macOS
curl -sSf https://souniolang.org/install.sh | sh

# Ou baixe diretamente
wget https://github.com/sounio-lang/sounio/releases/latest/download/souc-linux-x64.tar.gz
tar xzf souc-linux-x64.tar.gz
sudo mv souc /usr/local/bin/
```

### A Partir do Código-Fonte

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio/compiler
cargo build --release
sudo cp target/release/souc /usr/local/bin/
```

### Verificar a Instalação

```bash
souc --version
# souc 0.93.0
```

## Seu Primeiro Programa

Crie um arquivo `hello.sio`:

```sounio
fn main() -> i32 {
    print("Hello, Sounio!")
    println()
    0
}
```

Compile e execute:

```bash
souc run hello.sio
# Output: Hello, Sounio!
```

Ou apenas verifique os tipos:

```bash
souc check hello.sio
```

## Conceitos Principais

### 1. Tipos Epistêmicos

O recurso assinatura do Sounio é o tipo `Knowledge<T>` — valores que carregam sua incerteza:

```sounio
import sounio::epistemic::*

fn main() -> i32 {
    // Value with uncertainty
    let measurement = Knowledge::new(
        value: 42.0,
        uncertainty: 0.5,
        confidence: 0.95
    )

    // Uncertainty propagates through operations
    let doubled = measurement.mul(Knowledge::exact(2.0))

    print(doubled.to_string())
    // Output: 84.0000 +/- 1.9600 (95% CI)

    0
}
```

### 2. Variáveis

```sounio
let x = 5              // immutable
var y = 10             // mutable

y = y + 1              // OK: y is mutable
// x = 6               // Error: x is immutable
```

### 3. Referências

O Sounio usa `&!` para referências mutáveis (não `&mut` como no Rust):

```sounio
fn increment(x: &!i32) {
    *x = *x + 1
}

fn main() -> i32 {
    var value = 10
    increment(&!value)
    print(value)  // 11
    0
}
```

### 4. Unidades Físicas

Análise dimensional com segurança de tipos:

```sounio
let distance: f64<m> = 100.0 m
let time: f64<s> = 9.58 s
let speed = distance / time  // Type: f64<m/s>

// Compile error: can't add meters and seconds
// let invalid = distance + time
```

### 5. Efeitos

Funções declaram seus efeitos colaterais:

```sounio
fn read_file(path: &str) -> String with IO {
    // Can perform I/O
}

fn pure_function(x: i32) -> i32 {
    // No effects allowed
    x * 2
}
```

### 6. DSL MedLang

Sintaxe específica de domínio para farmacometria:

```sounio
import sounio::medlang::*

model OneCompartment {
    param CL: Knowledge<f64> = Knowledge::new(
        value: 10.0,
        uncertainty: 3.0,
        confidence: 0.95
    )
    param V: Knowledge<f64> = Knowledge::new(
        value: 50.0,
        uncertainty: 12.5,
        confidence: 0.95
    )

    compartment Central { volume: V }
    flow Central -> Elimination: CL

    observe Cp = Central.concentration
}
```

## Estrutura do Projeto

Um projeto típico do Sounio:

```
my_project/
├── src/
│   ├── main.sio
│   └── lib.sio
├── tests/
│   └── test_main.sio
├── examples/
│   └── demo.sio
└── sounio.toml
```

## Referência de Comandos

```bash
# Type-check a file
souc check file.sio

# Run a file (JIT compilation)
souc run file.sio

# Compile to executable
souc build file.sio -o output

# Show AST
souc check file.sio --show-ast

# Show types
souc check file.sio --show-types

# Watch mode (recompile on changes)
souc watch file.sio

# Get help
souc --help
```

## Exemplos

O diretório `examples/` contém muitos exemplos funcionais:

| File | Description |
|------|-------------|
| `hello.sio` | Olá Mundo |
| `fibonacci.sio` | Fibonacci recursivo e iterativo |
| `uncertainty.sio` | Propagação de incerteza em Knowledge<T> |
| `pkpd.sio` | Modelo PK de dois compartimentos |
| `effects.sio` | Demonstração de efeitos algébricos |
| `gpu.sio` | Exemplo de kernel GPU |
| `ode_demo.sio` | Resolução de EDO |
| `autodiff.sio` | Diferenciação automática |

Execute qualquer exemplo:

```bash
cd examples
souc run hello.sio
souc run fibonacci.sio
souc run uncertainty.sio
```

## Próximos Passos

- [Referência da Linguagem](./LLM_PROGRAMMING_GUIDE.md) — Guia completo de sintaxe
- [Biblioteca Padrão](../stdlib/) — Navegue pela stdlib
- [Exemplos](../examples/) — Exemplos de código funcionais
- [CHANGELOG](../CHANGELOG.md) — Histórico de versões

## Obtendo Ajuda

- **GitHub Issues**: [sounio-lang/sounio](https://github.com/sounio-lang/sounio/issues)
- **Discussões**: [GitHub Discussions](https://github.com/sounio-lang/sounio/discussions)
- **Site**: [souniolang.org](https://souniolang.org)

---

🏛️ **Sounio** — Computação no Horizonte da Certeza
