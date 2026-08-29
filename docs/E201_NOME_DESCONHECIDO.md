# E201 — nome desconhecido em posição de chamada passa a ser fatal

28-ago-2026. Pré-requisito da Fase 1, e do `IndepKnowledge`: enquanto um nome inexistente passar
em silêncio, **nenhuma garantia de tipo é executável** — um erro de digitação no nome de um
tipo-testemunha ou de uma função de prova passa direto, e a garantia "existe" sem existir.

## O defeito

O compilador tinha **dois caminhos divergentes** para nome desconhecido, em
`self-hosted/compiler/lean_single.sio`:

| caso | comportamento antes |
|---|---|
| variável indefinida | `E200` + `tc_mark_failed()` — **fatal, correto** |
| **chamada** de função indefinida | `tc_undefined_var` → `warning:` + **sem** `tc_mark_failed()`, emite `xor eax,eax` |

Um programa que chamasse uma função inexistente **compilava, rodava, devolvia zero** e saía com
`rc=0`. O valor da chamada virava zero em silêncio.

Conserto: `tc_undefined_var` passa a imprimir `E201` e chamar `tc_mark_failed()`, como o E200 já
fazia. Cinco linhas.

## O resultado — 13 programas passam a ser rejeitados

Suíte de 864 testes, mesmo comando, só o compilador mudando:

| | antes | depois |
|---|---|---|
| passam | 639 | 626 |
| falham | 3 | 16 |

As 3 anteriores (`g2_abide_sounio`, `test_integral_eq`, `test_heap_vec_generic`) são
pré-existentes. **As 13 novas são programas que chamavam funções que não existem:**

`gum_h1_native` · `gum_iso_budget` · `gum_iso_budget_ode` · `knowledge_octonion_structure` ·
`ontology_roles_hierarchy` · `ontology_transitive` · `ontology_type_bridge` ·
`rapamycin_epistemic_adaptive` · `rapamycin_iso_budget` · `rapamycin_rk4_budget` ·
`test_dissertation_e2e` · `test_pipeline_real_e2e` · `test_types`

São os programas-vitrine científicos: GUM, octoniões, PBPK/rapamicina, ontologia, e os dois
end-to-end de dissertação e pipeline.

## Três exemplos, e o terceiro é o pior

| arquivo | linha | chamada | existe no repo? |
|---|---|---|---|
| `tests/run-pass/gum_h1_native.sio` | 97 | `print_float(add_v)` | **não** |
| `tests/run-pass/knowledge_octonion_structure.sio` | 112 | `budget_of(c0)` | **não** |
| `tests/stdlib/geometry/test_types.sio` | 20 | `assert_eq(sum_p.x, expected_sum_x)` | **não** |

Nenhuma das três está definida em `stdlib/` nem em `self-hosted/` — não é import faltando, as
funções não existem.

O caso de `test_types.sio` é o que mais importa: **é um teste cujas asserções não existem.** Ele
passava porque `assert_eq` virava `xor eax,eax`. O comentário na própria linha diz
"// assume assert_eq and == for Knowledge". A suposição nunca foi verdadeira.

E `gum_h1_native.sio` — nomeado no plano como suspeito da Fase 1 — chamava `print_float`, que não
existe: o teste que "verifica" propagação GUM nunca imprimiu nada.

## Verificação

- ponto fixo do bootstrap **mantido**: o binário reconstruído é byte-idêntico ao `gen1`, e
  `gen1 == gen2`. Reconstrução em 0,98 s.
- programa válido de controle continua compilando com `rc=0`.
- `bin/souc-linux-x86_64` regenerado a partir do fonte e commitado junto — fonte e binário não
  podem divergir, que é a causa raiz da auditoria anterior.

## O que fica pendente

Os 13 programas precisam de triagem individual: para cada chamada, implementar a função que
falta ou remover a chamada. **Não afrouxar o E201 para fazer a suíte ficar verde** — o verde
anterior era o problema.
