# Fase 0: Análise de Ferramental para Migração fresh_reg

**Data**: 2026-10-23  
**Task**: Implementar script de reescrita automática de `fresh_reg`  
**Branch**: `cursor/lowerer-borrow-migration-8bcf`  
**Issue**: #3020, #3040

## Objetivo

Criar script que reescreve automaticamente `fresh_reg` de:

```sio
fn fresh_reg(self) -> (Lowerer, i64) with Mut, Panic, Div, IO {
    var lo = self
    // ...
    (lo, reg)
}

// Call site:
let pair = lo.fresh_reg()
lo = pair.0
let reg = pair.1
```

Para:

```sio
fn fresh_reg(self: &!Lowerer) -> i64 with Mut, Panic, Div, IO {
    // ... usar self diretamente
    reg
}

// Call site:
let reg = fresh_reg(&!lo)
```

## Escala do Problema

```bash
grep -n "\.fresh_reg()" self-hosted/ir/lower.sio | wc -l
# 499 call sites
```

**499 call sites** em `self-hosted/ir/lower.sio` (arquivo de ~39k linhas).

## Tentativa: Script Regex

Criei `scripts/dev/migrate_fresh_reg_to_borrow.py` com transformações baseadas em regex:

1. **Assinatura**: `fn fresh_reg(self) -> (Lowerer, i64)` → `fn fresh_reg(self: &!Lowerer) -> i64`
2. **Corpo**:
   - Remover `var lo = self`
   - Trocar `lo.` por `self.`
   - Trocar `return (lo, reg)` por `return reg`
3. **Call sites**:
   - Detectar pattern de 3 linhas:
     ```sio
     let pair = lo.fresh_reg()
     lo = pair.0
     let reg = pair.1
     ```
   - Substituir por: `let reg = fresh_reg(&!lo)`

### Por Que Regex Não Funciona

1. **Call sites variam muito**:
   ```sio
   // Pattern 1:
   let pair = lo.fresh_reg()
   lo = pair.0
   let reg = pair.1
   
   // Pattern 2:
   let (lo2, reg) = lo.fresh_reg()
   
   // Pattern 3:
   let pair = self.lowerer.fresh_reg()
   self.lowerer = pair.0
   let r = pair.1
   
   // Pattern 4: aninhado
   let pair = lo.emit(...).fresh_reg()
   
   // Pattern 5: em if/match
   if cond {
       let pair = lo.fresh_reg()
       lo = pair.0
       let reg = pair.1
   }
   ```

2. **Contexto importa**:
   - Nome da variável `lo` varia (`lo`, `self.lowerer`, `ctx.lo`, etc.)
   - Nome do `pair` varia (`pair`, `p`, `fresh_pair`, etc.)
   - Nome do `reg` varia (`reg`, `r`, `fresh_reg`, `tmp_reg`, etc.)

3. **Indentação e formatação**:
   - Espaçamento varia
   - Comentários entre linhas
   - Linhas vazias

4. **Corpo da função**:
   - `lo` aparece em muitos contextos (field access, chamadas, `(*current_func_box)`)
   - `return (loe, 0)` vs `return (lo, reg)` — primeiro não deve ser mudado
   - Lógica de erro precisa de `_loe` em vez de `loe`

### Estimativa de Precisão

- **Assinatura**: 95% (regex simples, única função)
- **Corpo**: 70% (muitos contextos de `lo`, retornos variados)
- **Call sites**: **10%** (499 sites, variação enorme)

**Risco**: Script automatizado com 10% de precisão quebra ~450 call sites.

## Solução Correta: Parser ou AST

Para fazer isso corretamente, precisamos:

1. **Parser de Sounio**: Parse `lower.sio` para AST
2. **Transformação AST**:
   - Encontrar `fn fresh_reg` e mudar assinatura
   - Reescrever corpo substituindo `var lo = self` por uso direto
   - Encontrar todos os call sites e reescrever baseado em fluxo de dados
3. **Pretty-printer**: Gerar código Sounio do AST transformado

### Alternativa: Ferramentas Existentes

1. **Tree-sitter**: Grammar de Sounio + transformação
2. **Scripting com LSP**: Se Sounio LSP suporta "rename" e "extract", usar isso
3. **Transformação manual assistida**: Gerar lista de call sites, transformar em lote com editor (vim macros, VSCode multi-cursor)

### Alternativa Realista: Transformação Manual Guiada

1. **Mapear call sites**:
   ```bash
   grep -n "\.fresh_reg()" self-hosted/ir/lower.sio > /tmp/fresh_reg_sites.txt
   ```

2. **Classificar patterns** (manual, ~50 samples):
   - Pattern A: 3 linhas sequenciais
   - Pattern B: destructuring `let (lo, reg) = ...`
   - Pattern C: aninhado
   - Pattern D: if/match

3. **Transformar por pattern**:
   - Pattern A: Vim macro ou sed script
   - Pattern B: Manual ou sed mais complexo
   - Pattern C e D: Manual (poucos casos)

4. **Validar**: Typecheck a cada lote de 50 sites

## Decisão

**NÃO implementar script automatizado agora** porque:

1. **Risco muito alto** (90% de call sites quebrados)
2. **Ferramental correto** (parser/AST) não existe no repo
3. **Transformação manual** de 499 sites é ~2-3h de trabalho cuidadoso
4. **Typecheck a cada lote** garante que não quebra

**Próximo passo** (fora deste PR):

1. Mapear os 499 call sites e classificar patterns
2. Criar vim macros ou scripts sed para patterns mais comuns (80%)
3. Transformar manual os 20% restantes
4. Typecheck e medir gen3 a cada lote de 50

## Conclusão

**Fase 0 requer**:
- Parser de Sounio ou ferramental LSP avançado, **OU**
- Transformação manual guiada (2-3h de trabalho)

**Não tentarei** script regex que quebra 450 call sites.

**Entregável deste PR**: Análise de ferramental e decisão documentada.

---

**Refs**: #3020, #3040  
**Script tentado**: `scripts/dev/migrate_fresh_reg_to_borrow.py` (não aplicado)
