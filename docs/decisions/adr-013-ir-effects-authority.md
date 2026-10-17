# ADR-013 — Efeitos de memória são declarados por opcode, numa única autoridade; passes consultam, nunca enumeram

**Status:** proposto (2026-09-25)
**Contexto de origem:** lição do PR triton-lang/triton#11964 (fechado por T. Raoux) e dos achados do Copilot naquele PR.

## Contexto

Em 25/09/2026 o `self-hosted/ir/` tem **cinco** tabelas privadas de "esta instrução tem efeito colateral?", escritas à mão e divergentes entre si:

| ficheiro | predicado | falta / diverge |
|---|---|---|
| `opt_cleanup.sio:1573` | `ocp_has_side_effect` (pareado com a whitelist `ocp_has_dst`) | `IrStorePtr`, `IrStoreGlobal`, `IrCallExtern`, `IrCallIndirect` ficam seguros **só porque não estão em `ocp_has_dst`** — segurança por omissão, não por declaração |
| `optimize.sio:346` | `instr_has_side_effects` | não lista `IrCallExtern`/`IrCallIndirect`/`IrStoreGlobal`/`IrStorePtr`; a DCE desse ficheiro apagaria uma chamada externa com resultado não usado |
| `const_prop.sio:540` | `cp_has_side_effect` (códigos `CP_OP_*` próprios) | trata `ALLOC` como efeito; `opt_cleanup` não — os dois discordam sobre o mesmo opcode no mesmo pipeline vivo (`cp_optimize_function` é chamado de dentro de `opt_cleanup`) |
| `dce.sio:236` | `dce_has_side_effect` (códigos `DCE_OP_*`) | idem `const_prop` |
| `auto_vectorize.sio:364` | `avec_has_side_effect` | só CALL, RETURN, INDEX_SET |

Todas terminam em `_ => false`: **o opcode que ninguém lembrou de listar é tratado como puro.** Esse é exatamente o defeito de forma do primeiro achado do Copilot em #11964 (load volátil modelado como `Write` na própria op, mas o passe consultava um predicado manual que nunca perguntava à op).

Há um segundo buraco, ortogonal: `IrInstr` tem dois campos de fonte (`src1`, `src2`), mas

- `IrIndexSet` e as sete ops epistémicas de três registos guardam o terceiro registo em `imm_i64` (#1682 já foi um miscompile desta classe);
- `IrCallSret` guarda um intervalo contíguo em `field_idx`;
- `IrCall*` guardam os argumentos na pool plana, que `ir_arena_load` deliberadamente não reconstrói;
- `IrVecFmaF64` baixa para `VFMADD231PD dst, src1, src2` — **lê `dst`**.

Um scan `x.src1 == r || x.src2 == r` é incompleto por construção. `ocp_sink_loads` (Sprint 105) usa precisamente esse scan; `effects_self_test_runner.sio` T09 reproduz o defeito (`r5 = imm 7; a[i] = r5` é trocado). É **latente**, não vivo: o caminho `-O` executa `opt_cleanup_function_mfi`, que não chama `ocp_sink_loads`; só o `opt_cleanup_function` por-valor (probes) o alcança.

## Decisão

1. **Uma autoridade:** `self-hosted/ir/effects.sio` (`ir::effects`) é o único lugar que diz o que um `IrOpcode` faz à memória e ao controlo. Bits: `PURE, READ_MEM, WRITE_MEM, CONTROL, CALL, ALLOC, OBSERVABLE, UNKNOWN, READS_DST`.
2. **Default conservador:** `ir_effects_of(op)` termina em `_ => IR_EFF_UNKNOWN`, e `UNKNOWN` é barreira total (pode ler/escrever tudo, nunca removível). Um opcode só sai de `UNKNOWN` com **testemunha de lowering** (linha de `native/lower_ir.sio` ou `codegen_x86_linux.sio`) citada no comentário do arm.
3. **Passes consultam, não enumeram:** nenhum passe novo escreve `match op { ... _ => false }` para efeitos. Usa `ir_effect_may_read_mem`, `ir_effect_may_write_mem`, `ir_effect_is_barrier`, `ir_effect_removable_if_dead`, `ir_effect_commute`.
4. **Leitor de usos completo:** `ir_effect_reads_reg_at(slot, r)` é o único leitor de operandos-fonte aceite em passes que movem ou apagam instruções. Cobre `src1/src2`, `imm_i64` (terceiro registo), pool de argumentos, intervalo sret e acumulador FMA.
5. **Um passe, um propósito, um teste:** cada reescrita tem um runner `*_self_test_runner.sio` que só passa se ela fizer exatamente o que promete, e um gate em `scripts/ci/`.
6. **Ordem de programa é o schedule** para loads fora de loops, até que exista um modelo de pressão com evidência de *runtime* (não de contagem de spills). Registado explicitamente para não ser reaberto por heurística.

## Consequências

- Perda de poder de otimização: nenhuma, na prática — `ocp_has_dst` já limita a DCE a 16 opcodes; `ir_effect_removable_if_dead` cobre os mesmos (com `IrAlloc` do lado conservador do `const_prop`).
- Ganho: um `IrLoadVolatile` futuro recebe `WRITE_MEM|OBSERVABLE` numa linha e todos os passes herdam a resposta certa sem serem editados.
- Custo: ~100 opcodes vetoriais/hipercomplexos ficam `UNKNOWN` até auditoria do lowering; enquanto isso são barreiras. É o comportamento correto para um compilador que emite ISA direto e não tem um `ptxas` atrás para corrigir a ordem.

## Migração (ordem sugerida)

1. `ocp_sink_loads` → `ir_effect_swap_ok` (patch anexo; T09 vira de "reproduzido" para "não reproduzido").
2. `ocp_mfi_dce_once` e `ocp_dce_once` → `ir_effect_removable_if_dead` + `ir_effect_reads_reg_at` (fecha a classe #1682 por construção, não por caso).
3. `optimize.sio::instr_has_side_effects`, `const_prop.sio::cp_has_side_effect`, `dce.sio::dce_has_side_effect`, `auto_vectorize.sio::avec_has_side_effect` → delegar; apagar as tabelas.
4. Gate CI: `scripts/ci/ir_effects_authority_gate.sh` que falha se `grep -n "_ => false" self-hosted/ir/*.sio` aparecer dentro de uma função cujo nome contém `side_effect`.
5. Refinar recursos (heap/global/ptr) **só** depois de `alias_analysis.sio` ter um importador e um teste.

## Achado colateral (não bloqueia)

O prebuilt `bin/madaros-linux-x86_64` (v0.80.0) segfaulta ao executar `ir_arena_store` de uma `IrCallExtern` cujos argumentos vêm numa `Option<Box<IrRegList>>` construída num programa standalone (`effects_t03_probe.sio` reproduz; crash entre "fn built" e "stored"). O compilador em si faz o mesmo em `lower.sio` sem crash, portanto é provável que dependa de estado do arena/pool ou do bug conhecido de escrita global em função unit (v0.80.0). Registado para auditoria; T03 contorna preenchendo a pool diretamente.
