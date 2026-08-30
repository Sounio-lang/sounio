<!-- docs:meta
topic_id: repo.docs.e201-nome-desconhecido
authority: repo_only
audience: users
last_validated: 2026-08-29
validated_by: claude-2 (rebase onto integration/sounio-dev-ready-base @ 1c1b6549ad)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.e201-nome-desconhecido
-->

# E201 — nome desconhecido em posição de chamada passa a ser fatal

> **Nota de rebase, 29-ago-2026 — leia antes do resto.** Este documento foi escrito contra
> `8d203709e1`. Ao integrar `integration/sounio-dev-ready-base` (205 commits à frente), quase
> tudo que ele descreve **já tinha sido feito na base, de forma independente e melhor**, e o
> PR deixou de carregá-lo. Medido, arquivo por arquivo:
>
> | o que este doc descreve | estado na base, medido 29-ago-2026 |
> |---|---|
> | o conserto de `tc_undefined_var` (imprimir `E201` + `tc_mark_failed()`) | **já feito na base** (`4ff763a9e9`, localização em `bb3a2b2da2`): imprime `error: unknown identifier \`nome\` at <arquivo:linha>` e chama `tc_mark_failed()`. Nomeia o identificador e a origem, coisa que a versão deste PR não fazia. **O PR agora toma a versão da base.** |
> | o número `E201` | **colidia**: `error[E201]` já existe no mesmo `lean_single.sio` com o sentido "parameter uses `ExactlyPrivate<T>` without `with ZD` effect" (linha 29658 na base). A escolha da base — prefixo `error:` sem número — evita a colisão. |
> | 8 dos 13 programas triados (`gum_iso_budget`, `gum_iso_budget_ode`, `knowledge_octonion_structure`, `rapamycin_*` ×3, `test_dissertation_e2e`, `test_pipeline_real_e2e`) | **já consertados na base**, por outro caminho: `epistemic::budget64` (uma API de orçamento de verdade, em vez das oito chamadas a `sensitivity_of` que saíam zero) e um `run_fmri_pipeline_gate` próprio. **O PR toma a base nesses oito.** |
> | `test_types` | a base substituiu o arquivo por um *stub* de três linhas que imprime `GEOMETRY_OK`. A reescrita deste PR **não sobrevive**: ela lê `soma.x` de fora do módulo e a base transformou isso em erro duro (`private struct field access`, 37 sítios em `lean_single.sio`, **0** em `8d203709e1`), e `stdlib/geometry` não declara nada `pub`. Marcar os campos `pub` conserta `Point2D` mas **não** `Point3D`: existe um segundo `struct Point3D` em `stdlib/geo/pure/types.sio` e a tabela de símbolos plana resolve o nome para aquele. **O PR toma o stub da base** e registra a dívida aqui. |
> | ponto fixo do bootstrap "byte-idêntico ao `gen1`, e `gen1 == gen2`" | **VAZIO após o rebase.** O `bin/souc-linux-x86_64` deste PR é agora, byte a byte, o da base (`150e57d2a6`), e não foi reconstruído a partir do fonte integrado. Nenhuma afirmação de ponto fixo pode ser lida deste PR. |
> | `gum_h1_native`, os três `ontology_*` | **sobrevivem** — a base não os tocou. Re-medidos 29-ago-2026 sob `lean_single`: os quatro rodam `rc=0`, e `gum_h1_native` imprime `Sounio U: 66.361342` contra 67 nm publicado (0,95 % de erro relativo). |
> | contagens de suíte (864 testes; 639→626; 638/3/223) | **não re-derivadas.** Um número de suíte medido contra `8d203709e1` não diz nada sobre uma árvore 205 commits à frente, e re-derivá-lo exigiria uma execução completa em cima de um compilador reconstruído. Trate-os como históricos. |
>
> O que resta deste documento é o valor forense: o defeito era real, o mecanismo está descrito
> corretamente, e os dois achados laterais no fim continuam de pé.

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

---

# Triagem dos 13 — resultado

Suíte final: **638 passam, 3 falham, 223 pulados**. As 3 são as pré-existentes
(`g2_abide_sounio`, `test_integral_eq`, `test_heap_vec_generic`), verificadas uma a uma como
falhando também com o stdlib revertido.

| programa | causa | conserto |
|---|---|---|
| `gum_h1_native` | `print_float` não existe | helper local sem divisão (para não forçar `Div` nas assinaturas). **O teste agora imprime: Sounio U = 66,361 contra 67 nm publicado no GUM — dentro dos ±5%.** Estava certo e nunca mostrou. |
| `gum_iso_budget`, `gum_iso_budget_ode`, `knowledge_octonion_structure`, `rapamycin_epistemic_adaptive`, `rapamycin_iso_budget`, `rapamycin_rk4_budget` | `budget_of` é intrínseco **nunca implementado** | orçamento computado com o que EXISTE: `variance_of` (sombra β⁴) + `sensitivity_of(x,k)` (sombra β⁶), oito chamadas literais. ⚠️ **Achado ao ligar: as sensibilidades saem todas ZERO** — o orçamento por canal não é rastreado, só a variância total, que confere (5,25 = 4,0+1,0+0,25). Fica visível. |
| `ontology_transitive`, `ontology_roles_hierarchy`, `ontology_type_bridge` | `make_mammal` / `make_rapamycin` não existiam | construtores no idioma já usado por `make_dog` em `ontology_complex_hierarchy.sio` |
| `test_dissertation_e2e` | `print_i64` não existe | `print_int`, o builtin do compilador |
| `test_types` | **não era Sounio** | reescrito. O anterior usava `import`, `-> void`, `std::print("{}", …)`, `Point2D::new`, `.value()` em campos `f64`, e `Vec2`/`Vec3`, que não existem. Comentários próprios: "assume assert_eq". Agora testa a API real (`point2d_*`, `point3d_cross`, `matrix3_*`) com asserções que valem. |
| `test_pipeline_real_e2e` | **bloqueado por defeito de compilador** | ver abaixo |

## O caso fMRI — três camadas, duas consertadas

O teste passava imprimindo `SCIENCE_FMRI_OK` **sem jamais abrir um arquivo**:
`run_fmri_pipeline_gate` não existia e o compilador devolvia 0. Escrito o gate real
(`stdlib/fmri/pipeline_real.sio`, carrega os dois volumes e confere a grade), a cadeia apareceu:

1. ✅ `nifti_read_file_buffer_into` rejeitava **todo binário** como truncado: checava
   `read_file(path).len()`, que é `strlen` e para no primeiro NUL — offset **2** nesta fixture.
   Limite passa a vir de `file_size` (stat).
2. ✅ `nifti_str_byte_u8` limitava o índice pelo mesmo `strlen`. Novo `nifti_byte_at(src, idx,
   limit)` com limite explícito.
3. 🔴 **Não consertado.** `read_file` devolve `string`; os acessores declaram `[i8; 0]`. A chamada
   é **erro de tipo (E001)** e o que o compilador emite lê ZEROS. *A pista NIfTI nativa nunca foi
   bem-tipada.* Consertar exige decisão de linguagem: uma primitiva de indexação de bytes sobre
   `string`, ou `read_file` devolvendo um tipo de bytes.

A fixture está **perfeita**, verificada byte a byte: NIfTI-1, magic `n+1`, dim 2×2×2×4, uint8,
`vox_offset` 352, 384 bytes exatos — precisa de 384 e tem 384. O defeito nunca foi dela.

O teste ficou `//@ ignore` **com o motivo nomeado no cabeçalho**, não para ficar verde.

## Dois achados laterais, para não se perderem

- `sensitivity_of` existe, compila e **devolve zero** — o orçamento por canal do GUM não está
  rastreado.
- `arity mismatch` e `assignment type mismatch` ainda são **warnings** (`tc_error`), não erros.
  São o próximo E201: mesma classe de buraco, mesma função de reporte.
