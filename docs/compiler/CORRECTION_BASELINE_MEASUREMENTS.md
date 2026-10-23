# Correção: Medições Baseline Incorretas

**Data**: 2026-10-23  
**Issue**: #3020  
**PR**: #3040

## Erro Identificado

As medições baseline iniciais reportadas no PR #3040 estavam **completamente incorretas**:

### Medições Iniciais (ERRADAS)
- Tempo gen3: 8.3s
- RSS máximo: 1.9 MB  
- Comando: `gen2.elf self-hosted/compiler/lean_single.sio gen3.elf`

**Problemas**:
1. **Compilador errado**: `gen2.elf` era o `lean_single`, não o Madaros
2. **Fonte errado**: `lean_single.sio` é o compilador single-file antigo (~39K linhas), não o compilador modular completo
3. **RSS não capturado**: O monitor Python só via o wrapper, não o processo real
4. **Procedimento errado**: Não seguiu `scripts/ci/madaros_fixed_point_gate.sh`

## Medições Parciais Corretas

### Setup Correto
- **Gen1**: `artifacts/self-hosted/madaros` (Madaros v0.80.0, 124 MB)
- **Fonte**: `self-hosted/compiler/main.sio` (compilador modular completo)
- **Comando**: `madaros build self-hosted/compiler/main.sio gen2.elf`
- **Limites**: stack 1.5 GB, virt mem unlimited (conforme `scripts/lib/souc_invoke.sh`)

### Gen2 em Progresso (Parcial)
- **RSS observado**: ~14.04 GB (medido via `ps`)
- **Tempo**: >18 minutos (processo interrompido, ainda no lowering)
- **Warnings emitidos**: Centenas de "stack frame too large"
  - Máximo observado: **4054097375883 bytes** (~4 TB) - claramente um bug no cálculo
  - Comum: 11-27 MB para structs de IR/compilador
  - `Lowerer` não apareceu explicitamente, mas outros structs grandes sim

### Warnings de Stack Frame

Exemplos reais extraídos do log:

```
warning: stack frame too large (11109881 bytes) — return of fn compiler_ir_load_result_error
warning: stack frame too large (27406360 bytes) — by-value parameter of fn cgplan_update_spills
warning: stack frame too large (4054097375883 bytes) — by-value parameter of fn hlir_opt_is_param_value
warning: stack frame too large (12298248 bytes) — return of fn hlir_gpu_apply_fusion
```

## Impacto no Diagnóstico Original

### O que permanece válido:
1. ✅ Problema de passagem por valor existe (confirmado pelos warnings do compilador)
2. ✅ Structs grandes são passados por valor em múltiplas funções
3. ✅ Stack frames estão consumindo muita memória
4. ✅ ~14 GB de RSS é compatível com arenas grandes + estruturas passadas por valor

### O que estava errado:
1. ❌ Tempo baseline de ~8s (era outro alvo)
2. ❌ RSS de 1.9 MB (não capturou o processo real)
3. ❌ "Compilação gen3 passa facilmente com 8 MB stack" (não testado corretamente)

### O que não foi concluído:
1. ⏸️ Tempo real do gen2 e gen3 completos
2. ⏸️ RSS máximo via `/usr/bin/time -v`
3. ⏸️ Verificação de fixed point

## Análise Atualizada

### Causa Raiz Confirmada

Os warnings do compilador **confirmam** que:
- Passagem por valor de structs grandes está ocorrendo extensivamente
- Stack frames individuais chegam a **27 MB** (reais, não bugs de cálculo)
- Alguns cálculos de tamanho de frame estão com bug (valores de TB)

### Estimativa de Lowerer

Se `Lowerer` tem ~4.5 MB e é passado por valor em 501 chamadas a `fresh_reg`:
- 501 frames × 4.5 MB ≈ **2.2 GB** só de frames de stack do Lowerer
- Com recursão profunda + outros structs grandes: facilmente >10 GB de pressão de memória

### Por que 14 GB de RSS?

```
Arena BSS:     ~3.8 GB  (observado no build do Madaros)
Stack frames:  ~2-4 GB  (Lowerer + outros structs)
Heap/outros:   ~6-8 GB  (IR, símbolos, buffers)
────────────────────────
Total:         ~14 GB   ✓ compatível
```

## Recomendação Atualizada

A recomendação original **permanece válida**:

**NÃO migrar incrementalmente** sem antes:

1. **Completar medições corretas**:
   - Executar o fixed-point gate completo
   - Capturar RSS com `/usr/bin/time -v`
   - Medir tempo real de gen2 e gen3

2. **Diagnosticar codegen de referências**:
   - Criar teste isolado de assembly
   - Comparar valor vs `&T` vs `&!T`
   - Provar ou refutar "cada `*lo` copia 4.5 MB"

3. **Se codegen estiver correto**: Migração viável
4. **Se codegen estiver bugado**: Consertar primeiro

## Próximos Passos

1. Executar fixed-point gate em ambiente com recursos adequados
2. Criar teste assembly isolado para provar comportamento de cópia
3. Atualizar PR #3040 com números reais
4. Revisar plano de migração com baseline correto

## Lições Aprendidas

1. ✅ Sempre seguir o procedimento do CI (`scripts/ci/madaros_fixed_point_gate.sh`)
2. ✅ Verificar qual compilador está sendo usado (`souc_banner`)
3. ✅ Confirmar qual fonte está sendo compilado
4. ✅ Usar `/usr/bin/time -v` para RSS confiável
5. ✅ Monitorar processos reais, não wrappers
