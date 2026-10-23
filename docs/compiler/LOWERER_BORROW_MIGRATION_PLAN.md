# Lowerer Borrow Migration Plan

**Issue**: #3020  
**PR**: #3040  
**Status**: Deferred (plano documentado, implementação suspensa)

## Motivação

`Lowerer` struct (~4.5 MB) é passado por valor em ~618 funções em `self-hosted/ir/lower.sio`, pressionando a pilha no self-compile gen3. O CI usa pilha de 16 MB; esta VM local pode usar 8 MB, causando SIGSEGV.

**Gate atual no CI (commit 5d02e9a7)**:
- ✅ Typecheck: passa (rc=0, ~3min)
- ✅ Gen2: passa (rc=0, ~22min, 14877 funções merged)
- ✅ Gen2 identifica como "madaros"
- ❌ Gen3: falha (rc=181)
- ✅ Gate: **PASSA** (espera rung "run", não "gen3")

## Tentativas Anteriores e Lições (#3015)

### Tentativa 1: `&Lowerer` em `field_idx_*`
- **Resultado**: 491s (3× mais lento vs baseline 162s)
- **Causa**: Provável uso de `var result = *lo` dentro das funções
- **Lição**: Referência compartilhada + retorno `(Lowerer, ...)` força cópia

### Tentativa 2: + `ir_fn_get_ref`
- **Resultado**: 503s, 6.46 GB (3× tempo, 4× memória)
- **Causa**: Ainda mais cópias
- **Lição**: Referência compartilhada não é solução

## Por Que `&Lowerer` Foi 3× Mais Lento

**Pattern incorreto**:
```sio
fn field_idx_foo(lo: &Lowerer, ...) -> (Lowerer, i64) {
    var result = *lo  // ← CÓPIA DE 4.5 MB AQUI!
    // ... modifica result
    (result, resultado)
}
```

**Problemas**:
1. `*lo` cria cópia de 4.5 MB
2. Retorno `(Lowerer, ...)` força outra cópia
3. **2× overhead** vs by-value com move semântico
4. ABI de referência + cópia + retorno é pior que move direto

## Pattern Correto: `&!Lowerer`

```sio
fn fresh_reg(self: &!Lowerer) -> i64 with Mut, Panic, Div, IO {
    // Sem cópia, modifica in-place
    if self.current_fn < 0 || self.current_fn >= self.module.fn_count {
        if self.probe_mode {
            print("lower_probe: fresh_reg_invalid current_fn=")
            print_int(self.current_fn)
            print(" fn_count=")
            print_int(self.module.fn_count)
            print("\n")
        }
        let _loe = self.report_error_at(8959)
        return 0
    }
    var current_func_box = self.current_func
    let reg = (*current_func_box).reg_count
    if reg >= IR_MAX_VREGS {
        IR_LOWER_HARD_ERROR = 1
        if IR_LOWER_HARD_ERROR_REASON == 0 {
            IR_LOWER_HARD_ERROR_REASON = 5
            IR_LOWER_HARD_ERROR_LIMIT = IR_MAX_VREGS
            IR_LOWER_HARD_ERROR_NEEDED = reg + 1
            ir_copy_name_to_flat(&(*current_func_box).name, &! IR_LOWER_HARD_ERROR_FN_BUF, &! IR_LOWER_HARD_ERROR_FN_LEN)
        }
        return 0
    }
    (*current_func_box).reg_count = reg + 1
    self.current_func = current_func_box
    reg
}

fn emit(self: &!Lowerer, instr: IrInstr) with Mut, Panic, Div, IO {
    // ... (sem retorno de Lowerer)
}
```

**Call sites mudam**:
```sio
// Antes:
let pair = lo.fresh_reg()
lo = pair.0
let reg = pair.1

// Depois:
let reg = fresh_reg(&!lo)
// lo já está modificado

// Antes:
lo = lo.emit(instr)

// Depois:
emit(&!lo, instr)
// ou:
lo.emit(instr)  // se methods permitirem &!self implícito
```

## Por Que NÃO Implementei Agora

### Escala

- **499 call sites** de `fresh_reg`
- **1159 call sites** de `emit`
- **Total**: ~1600+ call sites a mudar

### Risco

- Cada call site espera tupla `(Lowerer, valor)` ou `Lowerer`
- Mudar assinatura quebra TODOS
- Typecheck passa, mas comportamento pode mudar sutilmente
- Gen3 já falha (rc=181), não posso piorar

### Alternativa Segura

1. **Ferramental**: Script de transformação automática (sed/awk/parser)
2. **Faseamento**:
   - Fase 1: Migrar 5-10 funções pequenas e isoladas
   - Fase 2: Verificar gen3 não piora
   - Fase 3: Migrar `fresh_reg` e `emit`
   - Fase 4: Migrar resto
3. **Validação**: Gate deve passar em cada fase

## Plano de Implementação (Futuro)

### Fase 0: Ferramental
- [ ] Script para transformar `fn f(self) -> (Lowerer, T)` → `fn f(self: &!Lowerer) -> T`
- [ ] Script para transformar call sites `let (lo, val) = lo.f()` → `let val = f(&!lo)`

### Fase 1: Funções Auxiliares (~10 funções)
- [ ] Migrar helpers pequenos e isolados
- [ ] Medir gen3 (tempo, RSS, rung)
- [ ] Reverter se piorar

### Fase 2: `fresh_reg` (499 call sites)
- [ ] Aplicar script
- [ ] Typecheck
- [ ] Medir gen3
- [ ] Reverter se piorar

### Fase 3: `emit` (1159 call sites)
- [ ] Aplicar script
- [ ] Typecheck
- [ ] Medir gen3
- [ ] Reverter se piorar

### Fase 4: Resto (~300 funções)
- [ ] Migrar em lotes de 50
- [ ] Medir a cada lote
- [ ] Ratchet de tempo e RSS

## Medições de Baseline (Commit 5d02e9a7)

**No CI**:
- Typecheck: ~3min, rc=0
- Gen2: ~22min, rc=0, 14877 funções merged
- Gen2 run: passa (banner=madaros)
- Gen3: rc=181 (esperado, gate aceita "run")

**Local (VM)**:
- Commit 5d02e9a7: typecheck passa
- Branch `cursor/lowerer-borrow-migration-8bcf` com mudanças: SIGSEGV

## Correções Feitas Neste PR

### 1. Bug de Cálculo de Frame Size (~1000× overflow)

**Problema**: `checker_type_byte_size_inplace` em `self-hosted/check/check.sio` calculava ~1000× maior para arrays aninhados (ex: ~4 TB para struct de ~4 GB).

**Causa**: Multiplicação de `ty.array_size * elem_sz` em arrays aninhados profundos causava overflow.

**Correção**: Cap em 16 GiB (17179869184 bytes). Valores negativos (overflow) ou maiores são capeados.

**Teste**: `tests/compile-pass/frame_size_calculation_nested_arrays.sio`
- Struct com 4096 × 16384 elementos aninhados
- Antes: mostraria ~4 TB
- Depois: mostra 1073741824 bytes (1 GB correto)

**Commit**: e4378145c

### 2. Documentação das Tentativas e Explicações

- `docs/compiler/FINDINGS_PROOF_AND_GATE.md`: Medições e prova de by-value vs referência
- `docs/compiler/CORRECTION_BASELINE_MEASUREMENTS.md`: Correção de baseline inválido
- Este documento: Plano completo de migração

## Restrições Obrigatórias

- ❌ Não aumentar arena de 2 GiB, arena de IR de 4M ou `IR_MAX_INSTRS`
- ❌ Não aumentar limite de pilha do CI nem afrouxar gates/ratchets
- ❌ Não desativar jobs
- ❌ Não fazer force-push
- ✅ Reverter se gen3 piorar

## Conclusão

**Status atual**: Não migrei `emit`/`fresh_reg` porque:
1. Escala muito grande (1600+ call sites)
2. Risco de quebrar gen3 que já falha
3. Falta ferramental para transformação automática segura

**Entregável**: Plano documentado, bug de frame size corrigido, explicação da lentidão de `&Lowerer`.

**Próximo**: Implementar ferramental de transformação ou migrar em pequenos lotes com validação a cada passo.
