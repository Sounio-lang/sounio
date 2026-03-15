# Inventário das Construções Sintáticas do Self-Hosted

Data: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Diretório analisado: `self-hosted/`
Extensão de arquivo: `.sio`

## Método de Coleta

- Contagem de ocorrências de palavras-chave via `grep -r` (case-sensitive, palavra inteira)
- Exclui comentários e strings? Não, contagem bruta.
- Agrupamento por módulo (subdiretório) usando `grep` com saída de caminhos.

## Estatísticas Gerais

(Números a serem preenchidos)

## Ocorrências por Construção

| Construção | Ocorrências | Exemplo |
|------------|-------------|---------|
| `match` | | |
| `use` | | |
| `enum` | | |
| `struct` | | |
| `fn` | | |
| `let` | | |
| `var` | | |
| `if` | | |
| `while` | | |
| `for` | | |
| `return` | | |
| `break` | | |
| `continue` | | |
| `type` | | |
| `impl` | | |
| `trait` | | |
| `mod` | | |
| `import` | | |
| `extern` | | |
| `knowledge` | | |
| `units` | | |
| `effect` | | |
| `handle` | | |
| `spawn` | | |
| `async` | | |
| `await` | | |
| `try` | | |
| `catch` | | |
| `throw` | | |

## Distribuição por Módulo

(Subdireitórios com mais ocorrências)

## Observações

- Lista de construções não suportadas pelo compilador nativo (AST‑direct) atual.
- Sugestões para a Fase 1 (extensão do compilador nativo).

## Arquivos Analisados

Total de arquivos `.sio`: (a ser preenchido)

