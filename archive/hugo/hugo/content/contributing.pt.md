---
title: "Contribuindo para Sounio"
description: "Como contribuir código, documentação e ideias para o projeto Sounio"
layout: "contributing"
---

Obrigado pelo seu interesse em contribuir para Sounio! Este documento fornece diretrizes e instruções para contribuir.

## Código de Conduta

Seja respeitoso. Seja construtivo. Seja paciente. Estamos construindo algo que importa.

## Começando

### Pré-requisitos

- **Rust 1.70+** — O compilador é escrito em Rust
- **Git** — Controle de versão
- **LLVM 15+** (opcional) — Para o backend LLVM

### Compilando do Código-Fonte

```bash
# Clonar o repositório
git clone https://github.com/sounio-lang/sounio.git
cd sounio

# Compilar o compilador
cd compiler
cargo build --release

# Executar testes
cargo test

# Executar o compilador
./target/release/souc run examples/hello.sio
```

## Fluxo de Trabalho de Desenvolvimento

### 1. Fazer Fork e Clonar

```bash
git clone https://github.com/YOUR_USERNAME/sounio.git
cd sounio
git remote add upstream https://github.com/sounio-lang/sounio.git
```

### 2. Criar uma Branch

```bash
git checkout -b feature/your-feature-name
```

Convenções de nomenclatura de branch:
- `feature/` — Novas funcionalidades
- `fix/` — Correções de bugs
- `docs/` — Documentação
- `refactor/` — Refatoração de código
- `test/` — Adições de testes

### 3. Fazer Alterações

- Siga as diretrizes de estilo de código abaixo
- Adicione testes para novas funcionalidades
- Atualize a documentação conforme necessário

### 4. Testar Suas Alterações

```bash
# Executar todos os testes
cargo test

# Executar teste específico
cargo test test_name

# Verificar formatação
cargo fmt --check

# Executar clippy
cargo clippy
```

### 5. Fazer Commit

Siga o formato da mensagem de commit:

```
[component] Brief description

Components: lexer, parser, ast, check, types, effects, hir, hlir,
           codegen, cli, docs, stdlib, tests, epistemic
```

Exemplos:
```
[parser] Add support for Knowledge<T> generic syntax
[stdlib] Implement bootstrap_correlation in connectivity module
[docs] Update README with new examples
```

### 6. Enviar e Criar PR

```bash
git push origin feature/your-feature-name
```

Em seguida, crie um Pull Request no GitHub.

## Diretrizes de Estilo de Código

### Rust (Compilador)

- Use `rustfmt` para formatação
- Execute `clippy` antes de fazer commit
- Nenhum `unwrap()` em código de biblioteca — use `?` ou tratamento de erro apropriado
- Use `thiserror` para tipos de erro
- Use `miette` para diagnósticos com spans de código-fonte
- Todos os itens públicos precisam de comentários de documentação

### Sounio (stdlib)

```sio
// Use nomes descritivos
fn compute_bootstrap_confidence_interval(data: &[f64], n_boot: i32) -> ConfidenceInterval

// Documente funções
/// Computes the modularity of a network using the Louvain algorithm.
///
/// # Arguments
/// * `weights` - Adjacency matrix (N x N)
/// * `resolution` - Resolution parameter (default: 1.0)
///
/// # Returns
/// Modularity value in range [-0.5, 1.0]
fn louvain_modularity(weights: &[[f64]], resolution: f64) -> f64

// Use Knowledge<T> para valores incertos
let result = Knowledge::new(
    value: computed_value,
    uncertainty: computed_uncertainty,
    source: "bootstrap"
)
```

## O que Contribuir

### Alta Prioridade

- [ ] Implementação do Language Server Protocol (LSP)
- [ ] Otimizações do backend LLVM
- [ ] Gerenciador de pacotes (`siopkg`)
- [ ] REPL interativo
- [ ] Mais módulos stdlib

### Prioridade Média

- [ ] Melhorias na documentação
- [ ] Programas de exemplo
- [ ] Benchmarks de desempenho
- [ ] Integrações com editores

### Sempre Bem-vindo

- Correções de bugs
- Melhorias na cobertura de testes
- Clarificações na documentação
- Correções de digitação

## Contribuições da stdlib

A biblioteca padrão (`stdlib/`) contém módulos específicos do domínio:

| Módulo | Descrição |
|--------|-----------|
| `epistemic/` | Tipos de incerteza central |
| `medlang/` | DSL de modelagem PK/PD |
| `fmri/` | Pipeline de neuroimagem |
| `causal/` | Inferência causal |
| `connectivity/` | Análise de rede |
| `gpu/` | Aceleração de GPU |
| `optimize/` | Otimização |
| `signal/` | Processamento de sinal |
| `data/` | DataFrames |
| `mcmc/` | Amostragem MCMC |
| `random/` | RNG |
| `quantum/` | Computação quântica |
| `linalg/` | Álgebra linear |
| `ode/` | Solucionadores de ODE |
| `bayes/` | Inferência Bayesiana |

Ao adicionar à stdlib:
1. Siga os padrões existentes no módulo
2. Inclua propagação de incerteza quando apropriado
3. Adicione comentários de documentação abrangentes
4. Escreva testes

## Dúvidas?

- Abra uma issue para bugs ou solicitações de recursos
- Use discussões para perguntas
- Verifique issues existentes antes de criar novas

## Licença

Ao contribuir, você concorda que suas contribuições serão licenciadas sob a Licença MIT.

---

*Obrigado por ajudar a construir o futuro da computação epistêmica!* 🏛️
