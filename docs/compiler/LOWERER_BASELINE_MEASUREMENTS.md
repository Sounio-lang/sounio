# Medições Baseline - Commit 8c4717178

## Configuração do ambiente
- VM: Cloud Agent
- Branch: cursor/lowerer-borrow-migration-8bcf
- Commit base: 8c4717178

## Resultados gen3 (gen2.elf compila lean_single.sio → gen3.elf)

| Stack Limite | Exit Code | Tempo (s) | RSS Máx (MB) | Status |
|--------------|-----------|-----------|--------------|---------|
| unlimited    | 0         | 8.4       | 1.9          | ✅ PASS |
| 16 MB (CI)   | 0         | 8.2       | 1.9          | ✅ PASS |
| 8 MB (VM)    | 0         | 8.3       | 1.9          | ✅ PASS |

## Observações

1. **Tempo muito menor que o relatado**: O PR #3015 menciona 162s com pilha ilimitada, mas aqui obtivemos ~8s. Isso porque o PR media o `make build` completo (JIT → gen1 → gen2 → gen3), enquanto aqui medimos apenas a etapa gen3.

2. **Compilação gen3 passa com 8 MB**: O problema de stack overflow mencionado no #2966 ocorre apenas em testes de boundary extremos (>8000 globais), não na compilação normal do compilador.

3. **RSS extremamente baixo**: O RSS máximo observado (1.9 MB) não captura o uso real de memória do compilador. O valor de 1.58 GB mencionado no #3015 provavelmente vem da arena BSS (696 MB conforme output do compilador).

## Próximos passos

1. Criar teste que reproduza o stack overflow
2. Migrar funções hotspot para `&!Lowerer`
3. Medir impacto de cada mudança
