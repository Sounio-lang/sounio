# Stack Overflow Investigation Results

## Contradição Resolvida com Evidências

### Descoberta: GitHub Actions usa 16MB de stack, não 8MB

**Evidência do CI (PR #3000, run 39763582387)**:
```
Madaros Current-Source f64 Lowering: [madaros-call-arity-13] stack_kb before=16384 after=524288
Madaros Current-Source f64 Lowering: SCO_STACK_LIMIT soft_before_kb=16384 hard_before_kb=unlimited
```

**Fonte**: Linha 9397 e 9410 do log completo do run 39763582387

**VM local (Cursor Cloud Agent)**:
```bash
$ ulimit -s
8192
```

**Conclusão**: GitHub Actions Ubuntu runners têm **16 MB de stack padrão**, não 8 MB. 
PR #3000 passou porque o CI tinha 2× mais stack do que assumimos.

### Implicação

O gate `madaros_global_capacity_gate` com 8191 globals:
- **Passa com 16 MB**: Confirmado no CI do PR #3000
- **Falha com 8 MB**: Confirmado localmente (exit 139, SIGSEGV)

## Análise de Recursão (Opção 2B)

### Top Funções Recursivas Identificadas

Análise estática de `self-hosted/ir/lower.sio` (618 funções que recebem `self`):

**Mais chamadas (hotspots de recursão)**:
1. `fresh_reg` - chamada 55× por outras funções
2. `lower_opt_expr_ref` - chamada 29×
3. `expr_receiver_struct_type_ref` - chamada 29×
4. `expr_to_callee_name_ref` - chamada 24×
5. `lower_expr_ref` - chamada 20×

**Mais complexas (muitas chamadas internas)**:
1. `lower_expr_non_binary_ref` - chama 50 outras funções
2. `lower_call_expr_special_b_ref` - chama 40 outras funções
3. `fo_ctor_register_body` - chama 33 outras funções
4. `lower_expr_variance_non_call_ref` - chama 31 outras funções
5. `expr_result_is_float_ref` - chama 28 outras funções

### Profundidade de Pilha Estimada

Com **Lowerer ~4.5 MB** por valor e recursão profunda:
- 2000 globals → profundidade ~20-30 frames
- Cada frame com múltiplos Lowerers
- Total: **8+ MB fácilmente excedido**

### Desafio da Conversão para &!Lowerer

**Problema**: 618 funções usam o padrão:
```sio
fn foo(self, ...) -> Lowerer {
    let lo = self.bar()
    lo.baz()
}
```

**Solução necessária**: Converter para:
```sio
fn foo(self: &!Lowerer, ...) {
    self.bar()
    self.baz()
}
```

**Escopo**:
- 618 assinaturas de função
- ~10,000+ call sites
- Muitas retornam tuplas `(Lowerer, i64)` → precisa empacotar de outra forma
- Compile time: 3-5 min por tentativa de validação

**Estimativa realista**: Semanas de trabalho cuidadoso, não horas

## Solução de Curto Prazo

Dado que:
1. CI real usa 16 MB (não podemos reduzir isso - seria **loosening the gate**)
2. A refatoração completa &!Lowerer é trabalho de semanas
3. O stack overflow só ocorre em testes de boundary extremos (8191 globals)

**Recomendação técnica**:
- Manter status quo: CI com 16 MB funciona
- Documentar que Madaros requer > 8 MB para programas muito grandes
- Abrir issue para refatoração sistemática de Lowerer

## Medições Finais

| Configuração | Stack | Exit | Comportamento |
|--------------|-------|------|---------------|
| CI (GitHub Actions) | 16 MB | 0 | ✅ PASS (PR #3000) |
| Local (8192 KB) | 8 MB | 139 | ❌ SIGSEGV em 8191-global test |
| Local com override | 8 MB + `MADAROS_STACK_KB=8192` | 0 | ✅ PASS (skip check) |
| Local (unlimited) | unlimited | 0 | ✅ PASS |

## Evidência de Log CI

Trechos relevantes do run 39763582387:

```
[madaros-global-capacity] PASS: 8191 globals execute; 8192 rejects
[madaros-call-arity-13] stack_kb before=16384 after=524288 requested=524288
SCO_STACK_LIMIT soft_before_kb=16384 hard_before_kb=unlimited soft_after_kb=65536
```

Múltiplos pontos no log confirmam: CI começou com 16384 KB (16 MB) de stack.
