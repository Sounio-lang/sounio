# Tier 0 — Matriz de testemunhas

**Medido em:** 2026-08-14  
**Árvore:** `origin/main` @ `d8dbd487a0`, worktree `witness-matrix-20260814`  
**Compilador:** `bin/souc` -> Madaros v0.80.0 (o binário *enviado*, não um `mc` recompilado)  
**Superfície:** `souc compile` + executar o ELF; `souc check` para os casos de rejeição.

Nota: `--native-v2-compile` (usado pelas rodadas originais) **não existe** no wrapper público.
`souc compile` é a superfície que um usuário real encontra, então é ela que foi medida.

## Pergunta

Os defeitos fechados em `integration/native-v2-honest` (tip `60686c617b`) **nunca foram merjados**
na `main`. A `main` andou 2.529 commits por conta própria. Isto ainda ocorre na árvore enviada?

## Resultado: 8 de 10 fechados por rota independente, **2 miscompiles silenciosos vivos**

| ID | Defeito | Origem (branch órfã) | Obtido | Esperado | Veredito |
|---|---|---|---|---|---|
| w1 | coerção de literal `fn main()->i32{0}` | `4acea3a59e` | 0 | 0 | **FECHADO** (parcial) |
| w2 | float `a+b` (dois params em memória) | `2286fb6d5d` | 7 | 7 | **FECHADO** |
| w3 | float `p.x+p.y` (dois campos) | `c2a783f270` | 7 | 7 | **FECHADO** |
| w3c | controle mem+literal `p.x+1.0` | — | 4 | 4 | controle OK |
| **w4** | **cast/comparação f32** | `a436a68712` | **0** | **12** | **ABERTO — miscompile** |
| w4c | controle f64 mesma forma | — | 12 | 12 | controle OK |
| w5 | closure sem captura `f(41)` | `320f4d2352` | 42 | 42 | **FECHADO** |
| w6 | guarda de `match` nunca lowered | `5ca40eee31` | 7 | 7 | **FECHADO** |
| **w7** | **discriminante de enum cego ao enum** | `e71eac8d99` | **20** | **30** | **ABERTO — miscompile** |
| w8 | índice de campo type-blind `ax/ay` | `6534d904e5` | 42 | 42 | **FECHADO** |
| w8c | controle sem colisão `bx/ay` | — | 42 | 42 | controle OK |
| w9 | release wall multi-módulo `util_add(40,2)` | `8765ca1dc4` | 42 | 42 | **FECHADO** |
| w10 | bypass do typecheck em `use` | `e1ac6f7c87` | rejeita (exit 1) | REJECT | **FECHADO** |

---

## Os dois defeitos abertos, caracterizados

### w7 — discriminante de enum resolvido por nome, cruzando enums

Confirmado e isolado por diferencial:

- `enum Color{Red,Green,Mark}` **sozinho** -> `Color::Mark` roda **30 (correto)**.
- Com `enum Shape{Quad{w:i64},Mark}` **também declarado** -> `Color::Mark` roda **20 (errado)**.

A presença de um `Mark` em outro enum sequestra o discriminante. É exatamente o BUG B da
soundness-round-3: a busca resolve por *nome de variante* e devolve o primeiro match entre
**todos** os enums, e os discriminantes reiniciam em 0 por enum. Programa bem-tipado,
`check: OK`, valor errado em runtime, sem aviso.

### w4 — não é a comparação: **`as f32` trunca a parte fracionária**

A testemunha original atribuía isto à comparação f32. **A atribuição está errada.** Medido:

| Sonda | Obtido | Correto |
|---|---|---|
| `(1.5 as f32) * (100.0 as f32)` | **100** | 150 |
| `(2.5 as f32) * (100.0 as f32)` | **200** | 250 |
| `(1.5 as f64) * 100.0` | 150 | 150 — controle OK |
| f32 `y < x` (verdadeiro) | 77 | 77 — comparação OK |
| f32 `x > y` com 2.5/1.0 | 66 | 66 — comparação OK |
| f32 `x > y` com 1.5/1.0 | **0** | 12 |

A comparação f32 funciona nos dois sentidos. O que quebra é o **round-trip f32**: `1.5 as f32` vira `1.0`
e `2.5 as f32` vira `2.0` — truncamento em direção a zero, como se fosse conversão inteira.
A testemunha `1.5 > 1.0` falha porque vira `1.0 > 1.0`; `2.5 > 1.0` passa por acidente
(`2.0 > 1.0`). **f64 puro preserva.** Locus: o caminho de conversão f32, não o de comparação (ver sonda discriminante abaixo).

Para uma linguagem cuja tese é impedir que um backend *silently lower away scientific meaning*,
este é o pior caso possível: uma conversão de precisão que destrói o valor sem diagnóstico.

### Achado colateral: a coerção de literal fechou só para inteiros

`fn main() -> i32 { 0 }` é **aceito** (w1 fechado). Mas `let x: f32 = 1.5` é **rejeitado**
(`error[E001] expected f32, found f64`) — literais float não adotam `f32` do contexto. Isso
força o `as f32` e, por consequência, expõe todo mundo ao truncamento acima. A rodada
literal-coercion cobria `f32/f64/i8`; na `main` só a metade inteira está fechada.

---

## Como reproduzir

    bash tests/witness_matrix/run.sh                # usa ./bin/souc
    bash tests/witness_matrix/run.sh /caminho/souc

Os dois casos multi-módulo rodam a partir de `tests/witness_matrix/cases/mm/` (precisam de
cwd no diretório do projeto para a resolução de imports).

## Consequência para o plano

O Tier 1 é **muito menor do que se temia**: não é portar 144 commits de uma branch com dois
meses de defasagem. São **dois defeitos**, ambos com testemunha mínima e locus indicado. O
resto do lane órfão foi fechado pela `main` por rota própria — o que sobra dele de valor
durável são os *gates*, não os patches.

## Nota de harness (auto-captura)

A primeira versão do `run.sh` reportou `w9 COMPILE-FAIL` enquanto minha compilação manual
do mesmo programa rodava 42. Divergência entre gate e medição independente é **bug de
harness, não do compilador** — causa: `./bin/souc` é relativo e quebrava após o `cd` para
o diretório do caso multi-módulo. Corrigido absolutizando `SOUC` (linha 13). Registrado
aqui em vez de silenciosamente corrigido, porque um gate que erra para o lado do
FAIL treina o leitor a ignorá-lo.

Contagem final do gate: **11/13 corretas, 2 abertas**, exit 1.

### w4 — sonda discriminante (correção de atribuição)

Objeção correta a uma primeira versão desta análise: **todas** as sondas iniciais multiplicavam
dois valores f32, e as FACTS da rodada dizem que *f32 é representado como f64 no backend*. Logo
elas não separavam "o cast trunca" de "a multiplicação f32 está quebrada".

Sonda com um único f32 no caminho aritmético:

    let x: f32 = 1.5 as f32
    let d: f64 = x as f64
    let t: f64 = d * 100.0     // multiplicação puramente f64
    t as i64                    // obtido: 100 | correto: 150

Como a multiplicação aqui é f64 pura (e o controle f64 puro dá 150), a multiplicação f32 está
**descartada** como causa. O valor já chega truncado.

Afirmação que a evidência sustenta: **o round-trip por f32 destrói a parte fracionária**
(`(1.5 as f32) as f64` = 1.0, enquanto `1.5 as f64` = 1.5). As sondas **não** distinguem se a
perda ocorre no store (`as f32`) ou no load (`as f64` de volta) — não superespecificar sem uma
sonda que separe os dois. O que está descartado: a comparação f32 (funciona nos dois sentidos)
e a aritmética f64.

---

## Verificação pós-fix (2026-08-14, controlada)

Três compiladores, mesma matriz:

| Compilador | Resultado | w4 | w4b | w7 | w5 |
|---|---|---|---|---|---|
| `bin/souc` enviado (prebuilt 25/jul) | 11/13 | 0 | 100 | 20 | **42 OK** |
| fonte da `main`, sem meus commits | **11/15** | 0 | 100 | 20 | **COMPILE-FAIL** |
| fonte da `main` + meus commits | **14/15** | **12** | **150** | **30** | COMPILE-FAIL |

Ou seja: os fixes fecham w4, w4b e w7 e **regridem zero** casos. O w5 falha de forma
idêntica no controle sem os meus commits, logo **não é regressão minha**.

### Achado independente: o binário enviado e a fonte da main divergem

w5 (closure sem captura) **compila e roda 42** sob o `bin/souc` enviado, mas o compilador
construído da fonte atual da `main` recusa:

    error: IR instruction arena contract violated (invalid handle) on region slot -1 generation -1
    Error: refusing to write a binary built on a violated IR arena contract

A recusa em si é o comportamento certo (fail-closed, sem emitir binário). O problema é a
divergência: `bin/madaros-linux-x86_64` é de 25 de julho e antecede a landing da arena de IR
(PR #1695/#1651). Quem usa o compilador enviado vê closures funcionando; quem constrói da
fonte, não. O gate que pegaria isso -- `madaros_full_gate` / `madaros_source_to_elf_gate` --
roda **apenas** no workflow agendado `madaros-prebuilt-refresh.yml`, onde falha e um
`::warning::` que pula o refresh e nunca deixa a main vermelha.

Esta e exatamente a patologia que o Tier 1 item 4 do plano descreve: gate que existe e nunca
bloqueia. Nao investiguei a causa da violacao da arena -- esta fora do escopo despachado.
