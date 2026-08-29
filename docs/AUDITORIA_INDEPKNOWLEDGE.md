# Auditoria — a faixa CI (`IndepKnowledge<T, A, B | Z>`) não existe

Data: 28-ago-2026. Motivo: pedido de "ligar o IndepKnowledge no checador", como segunda metade da
Fase 1 (a primeira, a lei de composição, está em `24aba6b9c`).

**Veredito: não há o que ligar.** O tipo não é construível, não é parseável, e o validador não é
chamado. O que existe é a moldura sem a peça.

## O que foi verificado, arquivo por arquivo

| peça | estado |
|---|---|
| `check/types.sio:77` — variante `TyCondIndep` no enum | existe |
| `check/types.sio:1770` — construtor `ty_cond_indep(...)` | existe, **ZERO chamadores** |
| `check/epistemic.sio:1099` — `check_cond_indep_type` | existe, chamado **só** por `is_valid_cond_indep` |
| `check/epistemic.sio:1119` — `is_valid_cond_indep` | existe, **ZERO chamadores** |
| `check/epistemic.sio:1125` — `cond_indep_structurally_holds` | existe, **ZERO chamadores** |
| `check/check.sio:1845` — impressão de `IndepKnowledge<` | existe (só imprime) |
| `check/compat.sio:446` — compatibilidade entre dois `TyCondIndep` | existe |
| `parser/ast.sio` — variante `TypeCondIndep` em `TypeExprKind` | **NÃO EXISTE** |
| `parser/types.sio` — função de parsing | **NÃO EXISTE** |
| lexer — token para `IndepKnowledge` | **NÃO EXISTE** |
| `check/check.sio:6449` — braço no despacho de lowering | **NÃO EXISTE** |

Como não há variante no `TypeExprKind` nem token no lexer, **não existe sintaxe** que produza um
`TyCondIndep`. O construtor nunca é chamado. O validador valida um valor que nada cria.

## O agravante: o comentário afirma o que não faz

`cond_indep_structurally_holds` diz, no próprio corpo:

> "Full SMT-backed graph check is wired through causal.sio; this is the lightweight fallback"

Não está ligado a `causal.sio`. A função retorna `true` em todos os caminhos não-degenerados.

## Os testes não exercitam o recurso

- `tests/frontend/cond_indep_basic.sio` — marcado `run-pass`, **passa**, e não contém uma única
  anotação `IndepKnowledge`. É aritmética `f64` com comentários afirmando independência
  condicional. O cabeçalho diz "The compiler verifies that the claimed independence is
  structurally valid" — o programa nunca pede verificação nenhuma.
- `tests/compile-fail/cond_indep_violation.sio` — a linha com o tipo está **comentada**
  (`//   let ci: IndepKnowledge<f64, x, y | {}> = ...`).
- `tests/stdlib/autodiff/test_epistemic_bridge.sio` — marcado `//@ ignore`, nunca roda.

Ou seja: a alegação de "primeiro sistema de tipos a codificar independência condicional em tempo
de compilação" (comentário do próprio teste, Sprint 20 Track CI) está sustentada por testes que
não usam o recurso, porque o recurso não tem sintaxe.

## Por que isto importa além do CI

É o mesmo desconto que a Fase 3 do plano identifica na fachada Φ: um cético que encontre isto
reprecifica tudo o que está ao lado. E aqui é pior que no serviço de inferência, porque a
independência condicional é **a peça de que a Fase 1 depende** — a lei de composição consertada
em `24aba6b9c` degrada para o limite conservador justamente por não ter testemunha verificável.

## O que custaria ligar de verdade

1. token `IndepKnowledge` no lexer;
2. variante `TypeCondIndep` em `parser/ast.sio::TypeExprKind`;
3. função de parsing em `parser/types.sio` — inclui aceitar `|` dentro de argumentos de tipo,
   sintaxe que **não existe** hoje em nenhum tipo;
4. braço no despacho de lowering (`check/check.sio:6449`), no molde de
   `TypeKnowledge => c.lower_knowledge_type(...)`;
5. `lower_cond_indep_type`: constrói via `ty_cond_indep`, chama `check_cond_indep_type`, reporta
   E072;
6. resolução dos ids de variável contra um grafo causal DECLARADO, e a chamada real a
   `cat_d_separated` — é aqui que mora o conteúdo, e é o passo que hoje não tem nem desenho;
7. a regra de composição: exigir um valor desse tipo para liberar a quadratura;
8. o espelho em `bootstrap/bootstrap_stage1.sio` (~30 mil linhas), mantendo o bootstrap no ponto
   fixo byte-idêntico.

Os passos 1–5 são mecânicos. O 6 é a contribuição. O 8 é o risco.

## Recomendação

Não fazer às pressas. Um `IndepKnowledge` que parseia e não checa d-separação seria a mesma
fachada, uma camada acima — e desta vez com a minha assinatura no commit.

Enquanto não existir, a Fase 1 sustenta o rigor por outra via, declarada em `24aba6b9c`: o padrão
é o limite conservador, e toda alegação de independência não provada é **grepável**
(`indep_declared`, `_independent`). É mais fraco que tipo, e é honesto sobre ser.
