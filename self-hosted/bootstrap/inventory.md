# Inventário das Construções Sintáticas do Self-Hosted

Data: 2026-03-14 02:42:55 UTC
Diretório analisado: `self-hosted/`
Extensão de arquivo: `.sio`

## Método de Coleta

- Contagem de ocorrências de palavras-chave via `grep -r` (case-sensitive, palavra inteira)
- Exclui comentários e strings? Não, contagem bruta.
- Agrupamento por módulo (subdiretório) usando `grep` com saída de caminhos.

## Estatísticas Gerais

- Total de arquivos `.sio`: 397
- Total de linhas (aproximado): (não coletado)
- Construção mais frequente: `let` (39 257 ocorrências)
- Construções menos frequentes: `throw` (0), `catch` (4), `try` (11)
- Módulos com maior densidade de código: `ir/`, `check/`, `native/`

## Ocorrências por Construção

| Construção | Ocorrências | Exemplo |
|------------|-------------|---------|
| `match` | 1809 | `match x { ... }` |
| `use` | 1556 | `use lexer::{Token}` |
| `enum` | 221 | `enum ExprKind { ... }` |
| `struct` | 1683 | `struct Parser { ... }` |
| `fn` | 22943 | `fn parse_expr() { ... }` |
| `let` | 39257 | `let x = 5;` |
| `var` | 18819 | `var x = 5;` |
| `if` | 32672 | `if cond { ... }` |
| `while` | 4452 | `while cond { ... }` |
| `for` | 2920 | `for i in 0..n { ... }` |
| `return` | 15060 | `return result;` |
| `break` | 329 | `break;` |
| `continue` | 109 | `continue;` |
| `type` | 1193 | `type Alias = T;` |
| `impl` | 131 | `impl Parser { ... }` |
| `trait` | 65 | `trait Serializable { ... }` |
| `mod` | 171 | `mod lexer;` |
| `import` | 73 | `import std.io;` |
| `extern` | 69 | `extern "C" { ... }` |
| `knowledge` | 44 | `knowledge<T>` |
| `units` | 57 | `units m, s;` |
| `effect` | 307 | `effect CanFail { ... }` |
| `handle` | 275 | `handle e { ... }` |
| `spawn` | 16 | `spawn task;` |
| `async` | 55 | `async fn f() { ... }` |
| `await` | 37 | `await future;` |
| `try` | 11 | `try { ... }` |
| `catch` | 4 | `catch e { ... }` |
| `throw` | 0 | `throw error;` |

## Distribuição por Módulo

Os seguintes subdiretórios possuem a maior densidade de construções (total de ocorrências de todas as palavras‑chave):

| Módulo | Total de Ocorrências | Top 3 Construções |
|--------|----------------------|-------------------|
| `ir/` | 25 292 | `let` (7 384), `if` (6 323), `fn` (3 202) |
| `check/` | 22 915 | `let` (5 817), `if` (5 059), `fn` (3 515) |
| `native/` | 18 509 | `if` (4 861), `let` (4 020), `fn` (3 068) |
| `compiler/` | 16 210 | `if` (4 426), `let` (3 954), `var` (2 157) |
| `gpu/` | 7 325 | `let` (1 883), `fn` (1 432), `if` (1 287) |
| `.` (raiz) | 7 166 | `let` (2 621), `if` (1 533), `fn` (1 434) |
| `lsp/` | 5 864 | `let` (1 900), `fn` (1 228), `if` (996) |
| `vm/` | 4 491 | `let` (1 592), `if` (922), `fn` (740) |
| `io/` | 3 074 | `if` (1 056), `let` (675), `return` (522) |
| `wasm/` | 2 799 | `let` (793), `var` (559), `fn` (484) |

> Nota: Os valores são aproximados, baseados na contagem bruta de palavras‑chave; incluem ocorrências em comentários e strings.

## Observações

- Lista de construções não suportadas pelo compilador nativo (AST‑direct) atual.
- Sugestões para a Fase 1 (extensão do compilador nativo).

## Arquivos Analisados

Total de arquivos `.sio`: 397

Lista completa disponível em `self-hosted/`.