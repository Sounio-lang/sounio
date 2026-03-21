---
title: "Contributing to Sounio"
description: "How to contribute code, documentation, and ideas to the Sounio project"
layout: "contributing"
---

¡Gracias por su interés en contribuir a Sounio! Este documento proporciona pautas e instrucciones para contribuir.

## Código de Conducta

Sé respetuoso. Sé constructivo. Sé paciente. Estamos construyendo algo que importa.

## Primeros Pasos

### Prerrequisitos

- **Rust 1.70+** — El compilador está escrito en Rust
- **Git** — Control de versiones
- **LLVM 15+** (opcional) — Para el backend de LLVM

### Compilación desde el Código Fuente

```bash
# Clone the repository
git clone https://github.com/sounio-lang/sounio.git
cd sounio

# Build the compiler
cd compiler
cargo build --release

# Run tests
cargo test

# Run the compiler
./target/release/souc run examples/hello.sio
```

## Flujo de Trabajo de Desarrollo

### 1. Fork y Clonar

```bash
git clone https://github.com/YOUR_USERNAME/sounio.git
cd sounio
git remote add upstream https://github.com/sounio-lang/sounio.git
```

### 2. Crear una Rama

```bash
git checkout -b feature/your-feature-name
```

Convenciones de nomenclatura de ramas:
- `feature/` — Nuevas funcionalidades
- `fix/` — Correcciones de errores
- `docs/` — Documentación
- `refactor/` — Refactorización de código
- `test/` — Adiciones de pruebas

### 3. Realizar Cambios

- Sigue las pautas de estilo de código a continuación
- Agrega pruebas para nueva funcionalidad
- Actualiza la documentación según sea necesario

### 4. Probar Tus Cambios

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Check formatting
cargo fmt --check

# Run clippy
cargo clippy
```

### 5. Confirmar

Sigue el formato del mensaje de commit:

```
[component] Brief description

Components: lexer, parser, ast, check, types, effects, hir, hlir,
           codegen, cli, docs, stdlib, tests, epistemic
```

Ejemplos:
```
[parser] Add support for Knowledge<T> generic syntax
[stdlib] Implement bootstrap_correlation in connectivity module
[docs] Update README with new examples
```

### 6. Empujar y Crear PR

```bash
git push origin feature/your-feature-name
```

Luego crea una Solicitud de Extracción en GitHub.

## Pautas de Estilo de Código

### Rust (Compilador)

- Usa `rustfmt` para el formateo
- Ejecuta `clippy` antes de confirmar
- No uses `unwrap()` en código de biblioteca — usa `?` o manejo de errores adecuado
- Usa `thiserror` para tipos de error
- Usa `miette` para diagnósticos con spans de fuente
- Todos los elementos públicos necesitan comentarios de documentación

### Sounio (stdlib)

```sio
// Use descriptive names
fn compute_bootstrap_confidence_interval(data: &[f64], n_boot: i32) -> ConfidenceInterval

// Document functions
/// Computes the modularity of a network using the Louvain algorithm.
///
/// # Arguments
/// * `weights` - Adjacency matrix (N x N)
/// * `resolution` - Resolution parameter (default: 1.0)
///
/// # Returns
/// Modularity value in range [-0.5, 1.0]
fn louvain_modularity(weights: &[[f64]], resolution: f64) -> f64

// Use Knowledge<T> for uncertain values
let result = Knowledge::new(
    value: computed_value,
    uncertainty: computed_uncertainty,
    source: "bootstrap"
)
```

## Qué Contribuir

### Alta Prioridad

- [ ] Implementación del Protocolo de Servidor de Lenguaje (LSP)
- [ ] Optimizaciones del backend de LLVM
- [ ] Administrador de paquetes (`siopkg`)
- [ ] REPL interactivo
- [ ] Más módulos de stdlib

### Prioridad Media

- [ ] Mejoras en la documentación
- [ ] Programas de ejemplo
- [ ] Benchmarks de rendimiento
- [ ] Integraciones con editores

### Siempre Bienvenido

- Correcciones de errores
- Mejoras en la cobertura de pruebas
- Aclaraciones en la documentación
- Correcciones de errores tipográficos

## Contribuciones a stdlib

La biblioteca estándar (`stdlib/`) contiene módulos específicos del dominio:

| Módulo | Descripción |
|--------|-------------|
| `epistemic/` | Tipos de incertidumbre centrales |
| `medlang/` | DSL para modelado PK/PD |
| `fmri/` | Pipeline de neuroimagen |
| `causal/` | Inferencia causal |
| `connectivity/` | Análisis de redes |
| `gpu/` | Aceleración GPU |
| `optimize/` | Optimización |
| `signal/` | Procesamiento de señales |
| `data/` | DataFrames |
| `mcmc/` | Muestreo MCMC |
| `random/` | RNG |
| `quantum/` | Computación cuántica |
| `linalg/` | Álgebra lineal |
| `ode/` | Solvers de EDO |
| `bayes/` | Inferencia bayesiana |

Al agregar a stdlib:
1. Sigue los patrones existentes en el módulo
2. Incluye propagación de incertidumbre donde sea apropiado
3. Agrega comentarios de documentación completos
4. Escribe pruebas

## ¿Preguntas?

- Abre un issue para errores o solicitudes de funcionalidades
- Usa discusiones para preguntas
- Verifica issues existentes antes de crear nuevos

## Licencia

Al contribuir, aceptas que tus contribuciones se licenciarán bajo la Licencia MIT.

---

*¡Gracias por ayudar a construir el futuro de la computación epistémica!* 🏛️
