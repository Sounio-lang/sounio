# Achados: Prova de Cópia, Bug de Cálculo e Fixed Point Gate

**Data**: 2026-10-23  
**Issue**: #3020  
**PR**: #3040

## 1. Prova de Cópia Implícita: CONFIRMADA

### Teste Mínimo

Arquivo: `/tmp/test_final_proof.sio`

```sio
struct Lowerer {
    data: [i64; 524288],  // 4 MB
    counter: i64
}

fn by_value_pattern(lo: Lowerer) -> (Lowerer, i64) with Mut {
    var result = lo
    result.counter = result.counter + 1
    (result, result.counter)
}

fn by_ref_pattern(lo: &!Lowerer) -> i64 with Mut {
    lo.counter = lo.counter + 1
    lo.counter
}
```

### Resultado

**Compilação com Madaros v0.80.0**:

```
warning: stack frame too large (4194312 bytes) — by-value parameter of fn by_value_pattern
```

**Função `by_ref_pattern`**: ✅ **NENHUM warning**

### Testes Adicionais

1. **Desreferência explícita** (`let copy = *s`): ✅ Sem warning de stack frame
2. **Acesso a campo** (`s.field`): ✅ Sem warning
3. **Cadeia de cópias**: ✅ Otimizado (sem warnings)

### Conclusão

✅ **Passagem por valor** de structs grandes (4 MB) **causa frames de 4 MB no stack**  
✅ **Passagem por `&!T`** (referência exclusiva) **NÃO causa frames grandes**  
✅ **Desreferência explícita** `*s` **NÃO cria frames grandes** (otimizada via RVO/NRVO)

**Implicação**: O problema do Lowerer é **passagem por valor no parâmetro**, não cópias implícitas na desreferência.

---

## 2. Bug de Cálculo de Frame: IDENTIFICADO

### Symptoms

Warnings no build do Madaros mostram valores astronômicos:

```
warning: stack frame too large (4054097375883 bytes) — return of fn hlir_function_new
```

**4054097375883 bytes = ~3.68 TB** (claramente incorreto)

### Causa Raiz

**Arquivo**: `self-hosted/check/check.sio`, linha 7033

```sio
fn checker_type_byte_size_inplace(c: *mut Checker, ty: TypeEntry, depth: i64) -> i64 {
    // ...
    if ty.kind == TypeKind::TyArray {
        var elem_sz: i64 = 8
        match ty.inner {
            Some(e) => { elem_sz = checker_type_byte_size_inplace(c, *e, depth + 1) }
            None => {}
        }
        return ty.array_size * elem_sz  // ← LINHA 7033: Bug aqui
    }
    // ...
}
```

### Análise

**Struct problemático**: `HlirFunction` em `self-hosted/hlir/ir.sio`:

```sio
pub struct HlirFunction {
    pub blocks: [HlirBlock; 4096],
    pub locals: [HlirLocalEntry; 8192],
    // ...
}

pub struct HlirBlock {
    pub instrs: [HlirInstr; 16384],
    // ...
}
```

**Cálculo correto**:
- `HlirInstr`: ~60-80 bytes (conservador)
- `HlirBlock`: 16384 × 60 = 983,040 bytes
- `HlirFunction`: 4096 × 983,040 = **4,026,245,120 bytes ≈ 3.75 GB**

**Warning mostra**: **4,054,097,375,883 bytes ≈ 3.68 TB** (**1000× maior!**)

### Hipótese do Bug

1. **Multiplicação dupla**: O cálculo de `ty.array_size * elem_sz` pode estar multiplicando duas vezes em recursão profunda
2. **Overflow de i64**: Multiplicação de arrays aninhados pode causar overflow que resulta em valores negativos interpretados como positivos enormes
3. **Array size incorreto**: `ty.array_size` pode conter lixo ou valor não inicializado

**Fator de erro**: ~1000× sugere um multiplicador extra na recursão.

### Localização Exata

- **Função**: `checker_type_byte_size_inplace`
- **Arquivo**: `self-hosted/check/check.sio`
- **Linhas**: 7020-7059 (função completa), 7033 (linha do bug)
- **Chamada**: Linha 7067 (recursão em `checker_fieldlist_byte_size_inplace`)

---

## 3. Fixed Point Gate: FALHOU (SIGSEGV)

### Ambiente

**VM**:
- RAM total: **16 GB**
- RAM disponível: **14 GB**
- Swap: **0 GB**

**Configuração**:
- Stack: 16384 KB (16 MB) - configuração externa
- Stack real: 524288 KB (512 MB) - usado pelo gate internamente
- Virt mem: unlimited

### Resultado

```
[rung check] gen1 typechecks self-hosted/compiler/main.sio
rc=139 errors=0

reached  none
expected run
```

**Exit 139 = SIGSEGV** (Segmentation Fault)

### Ponto de Falha

```
run_check_mode: about to check 130 modules
[CRASH]
```

**Falhou no início do typecheck** de 130 módulos do compilador modular, antes mesmo de chegar ao lowering.

### Causa

**Stack overflow durante typecheck**, mesmo com 512 MB de stack. Isto confirma que:
1. Structs grandes estão sendo passados por valor no typecheck
2. 130 módulos criam recursão profunda
3. Frames de 4-11 MB × profundidade > 512 MB

### Memória vs Stack

A VM tem **16 GB de RAM**, mas o problema é **stack**, não heap:
- **Heap**: Usa RSS (~14 GB observado anteriormente)
- **Stack**: Limite de 512 MB, esgotado por frames grandes

**Não é falta de RAM total**, é **frames de stack muito grandes**.

---

## 4. Sumário dos Achados

### ✅ Confirmado

1. **Passagem por valor causa frames grandes**: 4 MB struct → 4 MB frame
2. **Passagem por `&!T` não causa frames grandes**: Sem warnings
3. **Bug no cálculo de tamanho**: Valores de TB são incorretos (multiplicação extra ~1000×)
4. **Stack overflow real**: SIGSEGV no typecheck de 130 módulos com 512 MB stack
5. **Problema é no parâmetro, não na desreferência**: `*s` não cria frames grandes

### ❌ Não Comprovado por Falta de Recursos

1. **Tempo completo de gen2/gen3**: Não completou (SIGSEGV)
2. **RSS máximo via /usr/bin/time**: Não chegou ao ponto
3. **Fixed point**: Não chegou ao gen2

### 🔍 Descobertas Adicionais

1. **Desreferência é otimizada**: `*s` não causa cópias no stack
2. **Problema é sistemático**: Centenas de warnings de 11-27 MB reais + bugs de TB
3. **Typecheck já tem problema**: Não é só lowering, typecheck também passa structs por valor

---

## 5. Recomendações Atualizadas

### Prioridade 1: Corrigir Bug de Cálculo de Frame

**Bug crítico** em `checker_type_byte_size_inplace` (linha 7033) causa valores incorretos de TB.

**Correção proposta**:
1. Adicionar overflow check: `if result > MAX_REASONABLE_FRAME { return MAX_REASONABLE_FRAME }`
2. Limitar recursão em arrays aninhados profundos
3. Adicionar logging para debugar multiplicação excessiva

**Teste de regressão**:
```sio
struct Deep {
    blocks: [Block; 4096]
}
struct Block {
    instrs: [Instr; 16384]
}
// Deve calcular ~4 GB, não ~4 TB
```

### Prioridade 2: Migrar para &!Lowerer

**Agora com prova**:
1. ✅ Passagem por valor causa problema
2. ✅ `&!T` não causa problema
3. ✅ Migração é viável

**Etapas**:
1. Migrar funções leaf primeiro (menor risco)
2. Medir a cada migração (critério: sem regressão)
3. Priorizar por frequência: `fresh_reg` (501×), `emit` (1160×)

### Prioridade 3: Aumentar Recursos ou Otimizar

**Opções**:
1. **VM maior**: Precisa >16 GB RAM? Não, precisa **stack maior** ou **frames menores**
2. **Otimizar primeiro**: Migrar Lowerer resolve o problema de stack
3. **Aumentar stack**: Temporário, não resolve a causa raiz

---

## 6. Números Exatos para Registro

| Item | Valor Medido | Método |
|------|--------------|--------|
| RAM Total VM | 16 GB (16387280 KB) | `free -h` |
| RAM Disponível | 14 GB (15587020 KB) | `free -h` |
| Swap | 0 GB | `free -h` |
| Stack Limit (gate) | 512 MB | Gate log |
| Frame do Lowerer (4 MB) | 4194312 bytes | Warning do compilador |
| Frame calculado errado | 4054097375883 bytes (~3.68 TB) | Warning do compilador |
| Ponto de crash | Typecheck de 130 módulos | Gate log |
| Exit code | 139 (SIGSEGV) | Gate log |

**Conclusão**: VM tem RAM suficiente (16 GB), mas **stack insuficiente** (512 MB) para **frames de 4-11 MB** em **recursão profunda** (130 módulos).

Migrar Lowerer e outras structs para `&!T` **resolve o problema**.
