#!/usr/bin/env python3
"""
Análise de hotspots do Lowerer em self-hosted/ir/lower.sio

Identifica:
- Funções que recebem self e retornam Lowerer
- Frequência de chamadas
- Profundidade de chamadas
- Funções recursivas

Uso: python3 scripts/dev/analyze_lowerer_hotspots.py
"""

import re
from collections import defaultdict
from pathlib import Path

def analyze_lowerer_functions():
    """Analisa funções do Lowerer em lower.sio"""
    
    lower_sio = Path('self-hosted/ir/lower.sio')
    if not lower_sio.exists():
        print(f"Erro: {lower_sio} não encontrado")
        return
    
    content = lower_sio.read_text()
    
    # Padrão para funções que recebem self e retornam Lowerer
    fn_pattern = re.compile(
        r'fn\s+(\w+)\s*\([^)]*self[^)]*\)\s*->\s*(?:\(Lowerer[^)]*\)|Lowerer)',
        re.MULTILINE
    )
    
    functions = [m.group(1) for m in fn_pattern.finditer(content)]
    print(f"Total de funções que recebem self e retornam Lowerer: {len(functions)}")
    
    # Contar chamadas a essas funções
    call_counts = defaultdict(int)
    for fn in functions:
        pattern = rf'\b{fn}\s*\('
        call_counts[fn] = len(re.findall(pattern, content))
    
    # Top 30
    print("\n" + "="*70)
    print("TOP 30 FUNÇÕES MAIS CHAMADAS")
    print("="*70)
    print(f"{'Função':<40} {'Chamadas':>10}")
    print("-"*70)
    
    for fn, count in sorted(call_counts.items(), key=lambda x: x[1], reverse=True)[:30]:
        print(f"{fn:<40} {count:>10}×")
    
    # Funções _ref (comuns em recursão)
    ref_funcs = [fn for fn in functions if fn.endswith('_ref')]
    print(f"\n" + "="*70)
    print(f"FUNÇÕES TERMINADAS EM _ref: {len(ref_funcs)}")
    print("="*70)
    print("Top 15 _ref mais chamadas:")
    print("-"*70)
    
    ref_calls = {fn: call_counts[fn] for fn in ref_funcs}
    for fn, count in sorted(ref_calls.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"{fn:<40} {count:>10}×")
    
    # Estatísticas gerais
    total_calls = sum(call_counts.values())
    avg_calls = total_calls / len(functions) if functions else 0
    
    print(f"\n" + "="*70)
    print("ESTATÍSTICAS GERAIS")
    print("="*70)
    print(f"Total de funções:        {len(functions)}")
    print(f"Total de chamadas:       {total_calls}")
    print(f"Média de chamadas:       {avg_calls:.1f}")
    print(f"Funções _ref:            {len(ref_funcs)}")
    print(f"Funções mais chamadas:   {max(call_counts.values())} ({max(call_counts, key=call_counts.get)})")
    
    # Hotspots do PR #3015
    hotspots_3015 = [
        'fresh_reg',
        'lower_opt_expr_ref',
        'expr_receiver_struct_type_ref',
        'lower_expr_non_binary_ref',
        'lower_call_expr_special_b_ref',
        'fo_ctor_register_body'
    ]
    
    print(f"\n" + "="*70)
    print("HOTSPOTS MENCIONADOS NO PR #3015")
    print("="*70)
    print(f"{'Função':<40} {'Chamadas':>10} {'Status':>10}")
    print("-"*70)
    
    for fn in hotspots_3015:
        count = call_counts.get(fn, 0)
        status = "✓" if fn in functions else "✗ não encontrada"
        print(f"{fn:<40} {count:>10}× {status:>10}")

if __name__ == "__main__":
    analyze_lowerer_functions()
