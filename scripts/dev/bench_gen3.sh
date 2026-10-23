#!/bin/bash
# Benchmark gen3 compilation com diferentes limites de pilha
# Uso: scripts/dev/bench_gen3.sh [stack_kb|unlimited]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Verificar se gen2.elf existe
if [ ! -f gen2.elf ]; then
    echo "❌ gen2.elf não encontrado. Execute 'make build' primeiro."
    exit 1
fi

# Determinar limite de pilha
if [ -n "$1" ]; then
    STACK_LIMIT="$1"
else
    STACK_LIMIT="unlimited"
fi

# Configurar pilha
if [ "$STACK_LIMIT" = "unlimited" ]; then
    ulimit -s unlimited
    STACK_LABEL="unlimited"
else
    ulimit -s "$STACK_LIMIT"
    STACK_LABEL="${STACK_LIMIT}KB"
fi

echo "======================================================================="
echo "BENCHMARK GEN3 COMPILATION"
echo "======================================================================="
echo "Stack limit: $STACK_LABEL ($(ulimit -s))"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Branch: $(git branch --show-current)"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================="
echo ""

# Arquivo de output temporário
OUTPUT_FILE="/tmp/gen3_bench_$$.elf"

# Limpar output anterior
rm -f "$OUTPUT_FILE"

# Medir tempo e memória com /usr/bin/time se disponível
if command -v /usr/bin/time > /dev/null 2>&1; then
    echo "Compilando gen3 (com /usr/bin/time)..."
    /usr/bin/time -v ./gen2.elf self-hosted/compiler/lean_single.sio "$OUTPUT_FILE" 2>&1 | \
        tee /tmp/gen3_bench_output.log
    EXIT_CODE=${PIPESTATUS[0]}
else
    # Fallback para time builtin
    echo "Compilando gen3 (time builtin, sem RSS)..."
    (time ./gen2.elf self-hosted/compiler/lean_single.sio "$OUTPUT_FILE") 2>&1 | \
        tee /tmp/gen3_bench_output.log
    EXIT_CODE=${PIPESTATUS[0]}
fi

echo ""
echo "======================================================================="
echo "RESULTADO"
echo "======================================================================="
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: ✅ SUCESSO"
    echo ""
    echo "Output size: $(ls -lh "$OUTPUT_FILE" | awk '{print $5}')"
    
    # Tentar extrair estatísticas do log
    echo ""
    echo "Estatísticas do compilador:"
    grep -E "(compile:|elf:)" /tmp/gen3_bench_output.log | tail -2
    
    # Extrair RSS máximo se disponível
    if grep -q "Maximum resident set size" /tmp/gen3_bench_output.log; then
        echo ""
        echo "Memória:"
        grep "Maximum resident set size" /tmp/gen3_bench_output.log | tail -1
    fi
    
    # Limpar
    rm -f "$OUTPUT_FILE"
else
    echo "Status: ❌ FALHA"
    if [ $EXIT_CODE -eq 139 ]; then
        echo "        (exit 139 = SIGSEGV, provável stack overflow)"
    fi
fi

echo "======================================================================="
