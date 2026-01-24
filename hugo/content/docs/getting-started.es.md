---
title: "Getting Started with Sounio"
description: "Installation and first steps with the Sounio compiler"
layout: "docs"
---

Bienvenido a **Sounio**, un lenguaje de programación de sistemas para epistemic computing — donde cada valor puede llevar su incertidumbre.

## Instalación

### Desde Binario (Recomendado)

Descarga la última versión para tu plataforma:

```bash
# Linux/macOS
curl -sSf https://souniolang.org/install.sh | sh

# Or download directly
wget https://github.com/sounio-lang/sounio/releases/latest/download/souc-linux-x64.tar.gz
tar xzf souc-linux-x64.tar.gz
sudo mv souc /usr/local/bin/
```

### Desde el Código Fuente

```bash
git clone https://github.com/sounio-lang/sounio.git
cd sounio/compiler
cargo build --release
sudo cp target/release/souc /usr/local/bin/
```

### Verificar la Instalación

```bash
souc --version
# souc 0.93.0
```

## Tu Primer Programa

Crea un archivo `hello.sio`:

```sounio
fn main() -> i32 {
    print("Hello, Sounio!")
    println()
    0
}
```

Compila y ejecuta:

```bash
souc run hello.sio
# Output: Hello, Sounio!
```

O solo verifica los tipos:

```bash
souc check hello.sio
```

## Conceptos Clave

### 1. Tipos Epistémicos

La característica principal de Sounio es el tipo `Knowledge<T>` — valores que llevan su incertidumbre:

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

### 2. Variables

```sounio
let x = 5              // immutable
var y = 10             // mutable

y = y + 1              // OK: y is mutable
// x = 6               // Error: x is immutable
```

### 3. Referencias

Sounio usa `&!` para referencias mutables (no `&mut` como en Rust):

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

Análisis dimensional con tipos seguros:

```sounio
let distance: f64<m> = 100.0 m
let time: f64<s> = 9.58 s
let speed = distance / time  // Type: f64<m/s>

// Compile error: can't add meters and seconds
// let invalid = distance + time
```

### 5. Efectos

Las funciones declaran sus efectos secundarios:

```sounio
fn read_file(path: &str) -> String with IO {
    // Can perform I/O
}

fn pure_function(x: i32) -> i32 {
    // No effects allowed
    x * 2
}
```

### 6. DSL de MedLang

Sintaxis específica del dominio para farmacometría:

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

## Estructura del Proyecto

Un proyecto típico de Sounio:

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

## Referencia de Comandos

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

## Ejemplos

El directorio `examples/` contiene muchos ejemplos funcionales:

| File | Description |
|------|-------------|
| `hello.sio` | Hola Mundo |
| `fibonacci.sio` | Fibonacci recursivo e iterativo |
| `uncertainty.sio` | Propagación de incertidumbre en Knowledge<T> |
| `pkpd.sio` | Modelo PK de dos compartimentos |
| `effects.sio` | Demostración de efectos algebraicos |
| `gpu.sio` | Ejemplo de kernel GPU |
| `ode_demo.sio` | Resolución de EDO |
| `autodiff.sio` | Diferenciación automática |

Ejecuta cualquier ejemplo:

```bash
cd examples
souc run hello.sio
souc run fibonacci.sio
souc run uncertainty.sio
```

## Próximos Pasos

- [Language Reference](./LLM_PROGRAMMING_GUIDE.md) — Guía completa de sintaxis
- [Standard Library](../stdlib/) — Explora la stdlib
- [Examples](../examples/) — Ejemplos de código funcionales
- [CHANGELOG](../CHANGELOG.md) — Historial de versiones

## Obtener Ayuda

- **GitHub Issues**: [sounio-lang/sounio](https://github.com/sounio-lang/sounio/issues)
- **Discusiones**: [GitHub Discussions](https://github.com/sounio-lang/sounio/discussions)
- **Sitio web**: [souniolang.org](https://souniolang.org)

---

🏛️ **Sounio** — Computa en el Horizonte de la Certeza
