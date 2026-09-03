<!-- docs:meta
topic_id: repo.docs.auditoria-indepknowledge
authority: repo_only
audience: users
last_validated: 2026-08-29
validated_by: claude-2 (rebase onto integration/sounio-dev-ready-base @ 1c1b6549ad)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.auditoria-indepknowledge
-->

# Auditoria — a faixa CI (`IndepKnowledge<T, A, B | Z>`) não existe

Data: 28-ago-2026. Motivo: pedido de "ligar o IndepKnowledge no checador", como segunda metade da
Fase 1 (a primeira, a lei de composição, está em `24aba6b9c`).

**Veredito: não há o que ligar.** O tipo não é construível, não é parseável, e o validador não é
chamado. O que existe é a moldura sem a peça.

> **Nota de rebase, 29-ago-2026.** As linhas citadas abaixo foram re-medidas contra
> `integration/sounio-dev-ready-base` @ `1c1b6549ad` e **todas tinham deslocado** — a tabela já
> traz os números corrigidos. Os *veredictos* (existe / zero chamadores / não existe) foram
> re-verificados um a um e **continuam válidos**: `ty_cond_indep` e `is_valid_cond_indep` têm
> exatamente uma ocorrência na árvore (a própria definição), `cond_indep_structurally_holds` tem
> duas (definição + comentário), e `TypeCondIndep`, a função de parsing e o token do lexer
> continuam com zero ocorrências. Duas correções de substância, e não só de linha:
>
> - **Uma afirmação do corpo do PR era falsa desde que foi escrita**: "`E072` existe no fonte
>   modular e **0×** no binário". `E072` aparece **6×** em `lean_single.sio` — e já aparecia em
>   `8d203709e1`. O sentido lá é outro ("kernel function must return unit type", linhas 4356 e
>   29880), o que **reforça** a tese das duas cópias mas por uma via diferente: não é ausência,
>   é o mesmo número com dois significados vivos. `TypeExprKind`, `TypeKnowledge` e `TyCondIndep`
>   continuam, esses sim, com **0** ocorrências em `lean_single.sio`.
> - **`tests/frontend/cond_indep_basic.sio`**: o texto diz "não contém uma única anotação
>   `IndepKnowledge`". Continua verdadeiro como anotação; para quem for conferir com `grep`, o
>   nome aparece uma vez — dentro de um comentário.
>
> Nota de escopo: existe um PR irmão, **#1758**, saído do **mesmo** merge-base `8d203709e1`, que
> implementa os passos 1–7 desta recomendação (token, gramática, grafo causal declarável,
> Bayes-Ball) dentro de `lean_single.sio`. Ele **não** é sucessor nem predecessor deste PR: são
> dois ramos irmãos em camadas diferentes. Se #1758 entrar, o veredicto "não há o que ligar"
> passa a descrever apenas a base, não a árvore.

## O que foi verificado, arquivo por arquivo

| peça | estado |
|---|---|
| `check/types.sio:78` — variante `TyCondIndep` no enum | existe |
| `check/types.sio:2086` — construtor `ty_cond_indep(...)` | existe, **ZERO chamadores** |
| `check/epistemic.sio:1110` — `check_cond_indep_type` | existe, chamado **só** por `is_valid_cond_indep` |
| `check/epistemic.sio:1129` — `is_valid_cond_indep` | existe, **ZERO chamadores** |
| `check/epistemic.sio:1138` — `cond_indep_structurally_holds` | existe, **ZERO chamadores** |
| `check/check.sio:10939` — impressão de `IndepKnowledge<` | existe (só imprime) |
| `check/compat.sio:580` — compatibilidade entre dois `TyCondIndep` | existe |
| `parser/ast.sio` — variante `TypeCondIndep` em `TypeExprKind` | **NÃO EXISTE** |
| `parser/types.sio` — função de parsing | **NÃO EXISTE** |
| lexer — token para `IndepKnowledge` | **NÃO EXISTE** |
| `check/check.sio:16602` — braço no despacho de lowering | **NÃO EXISTE** |

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
- `tests/stdlib/autodiff/test_epistemic_bridge.sio` — **este item caducou.** Em `8d203709e1` estava `//@ ignore`; na base de hoje é um `run-pass` de três linhas que imprime `AUTODIFF_BRIDGE_OK` e não exercita nada. O recurso continua sem teste — mudou só a forma de não testá-lo.

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
4. braço no despacho de lowering (`check/check.sio:16602`), no molde de
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

---

# ADENDO — por que os passos 1–5 NÃO foram feitos

Ao começar os passos 1–5 (lexer, AST, parser, despacho, lowering), a primeira coisa a verificar
era se editar `self-hosted/parser` e `self-hosted/check` muda o compilador que roda. **Não muda.**

## O compilador que roda não é o que foi auditado

`bin/souc` é um script que executa `bin/souc-linux-x86_64`. Esse binário é produzido por
`self-hosted/compiler/lean_single.sio` — **30.049 linhas, um arquivo só**, cujo próprio cabeçalho
diz "keep lean_single aligned with the 1M-token modular parser/lexer path".

Contagem em `lean_single.sio`:

| símbolo | ocorrências |
|---|---|
| `TypeExprKind` | **0** |
| `TypeKnowledge` | **0** |
| `TyCondIndep` | **0** |
| `IndepKnowledge` | **0** |
| `ty_cond_indep` | **0** |
| `check_cond_indep_type` | **0** |

E a prova direta: `E072` aparece 2× em `self-hosted/check/epistemic.sio` e **0×** dentro de
`bin/souc-linux-x86_64`. O checador epistêmico modular **não está no compilador que roda**.

Logo, implementar os passos 1–5 no caminho modular produziria código que parece implementar o
recurso, num caminho que não é o compilador. Fachada, com a minha assinatura.

## E há um buraco anterior que torna qualquer garantia de tipo inexequível

Testado contra o binário real:

```
fn f(x: TipoQueNaoExiste<f64>) -> f64 { 1.0 }     → aceito, sem erro nem warning
fn f(x: IndepKnowledge<f64, a, b | z>) -> f64 {…} → aceito, rc=0
fn main() -> i64 { let x: i64 = funcao_inexistente(42); x }
                                                   → COMPILA E RODA, rc=0
```

Um programa que chama função inexistente compila e executa em silêncio. Enquanto isso for
verdade, **nenhuma garantia de tipo é executável**: um erro de digitação no nome do tipo-testemunha
passa direto, e a testemunha "existe" sem existir.

Isto estende o que a memória do projeto já registrava ("nome inexistente = warning +
`xor eax,eax`"): hoje não é nem warning.

## Ordem honesta do trabalho

1. **Resolução de nomes**: nome desconhecido — de tipo ou de função — passa a ser erro em
   `lean_single.sio`. É a fundação; sem ela o resto é decorativo. Blast radius desconhecido: parte
   dos 639 testes que passam pode depender da permissividade, e medir isso É o resultado.
2. **`IndepKnowledge` em `lean_single.sio`**, que é onde tem efeito — não no caminho modular.
3. **Reconciliar** o caminho modular com o `lean_single`, ou aposentar um dos dois. Manter duas
   cópias divergentes do compilador é a causa raiz desta auditoria inteira.

## O que joga a favor

A cadeia de bootstrap **fecha e é reprodutível**, verificado agora:
`gen1` é byte-idêntico a `bin/souc-linux-x86_64`; `gen1 == gen2`; a reconstrução leva **0,95 s**.
Qualquer mudança no compilador é verificável em um segundo, com ponto fixo checável por `cmp`.
