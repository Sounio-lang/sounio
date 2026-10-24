#!/usr/bin/env python3
"""
Script para migrar fresh_reg de by-value para &!Lowerer.

Transformações:
1. Assinatura: fn fresh_reg(self) -> (Lowerer, i64) → fn fresh_reg(self: &!Lowerer) -> i64
2. Corpo: var lo = self → remover, usar self diretamente
3. Retorno: (lo, reg) → reg
4. Call sites: let pair = lo.fresh_reg(); lo = pair.0; let reg = pair.1 → let reg = fresh_reg(&!lo)
"""

import re
import sys

def migrate_fresh_reg_signature(content: str) -> str:
    """Migrar assinatura de fresh_reg."""
    # Pattern: fn fresh_reg(self) -> (Lowerer, i64)
    pattern = r'fn fresh_reg\(self\) -> \(Lowerer, i64\)'
    replacement = r'fn fresh_reg(self: &!Lowerer) -> i64'
    return re.sub(pattern, replacement, content)

def migrate_fresh_reg_body(content: str) -> str:
    """Migrar corpo de fresh_reg."""
    # Encontrar o corpo da função
    fn_start = content.find('fn fresh_reg(self')
    if fn_start == -1:
        return content
    
    # Encontrar abertura e fechamento de chaves
    brace_start = content.find('{', fn_start)
    if brace_start == -1:
        return content
    
    brace_count = 1
    pos = brace_start + 1
    while pos < len(content) and brace_count > 0:
        if content[pos] == '{':
            brace_count += 1
        elif content[pos] == '}':
            brace_count -= 1
        pos += 1
    
    brace_end = pos - 1
    
    # Extrair corpo
    body = content[brace_start+1:brace_end]
    
    # Remover "var lo = self" no início
    body = re.sub(r'^\s*var lo = self\s*\n', '', body, flags=re.MULTILINE)
    
    # Trocar "lo." por "self."
    body = re.sub(r'\blo\.', 'self.', body)
    
    # Trocar "(*lo)" por "(*self)"
    body = re.sub(r'\(\*lo\)', '(*self)', body)
    
    # Trocar retornos (loe, 0) por 0, (lo, reg) por reg, etc
    body = re.sub(r'return \((\w+), (\w+)\)', r'return \2', body)
    
    # Trocar "lo.report_error_at" no let
    body = re.sub(r'let loe = lo\.report_error_at', 'let _loe = self.report_error_at', body)
    
    # Trocar final "(lo, reg)" por "reg"
    body = re.sub(r'\(lo, (\w+)\)$', r'\1', body, flags=re.MULTILINE)
    
    # Reconstruir
    new_content = content[:brace_start+1] + body + content[brace_end:]
    
    return new_content

def migrate_fresh_reg_call_sites(content: str) -> str:
    """Migrar call sites de fresh_reg."""
    # Pattern 1: let pair = lo.fresh_reg() seguido de lo = pair.0 e let reg = pair.1
    # Muito complexo para regex simples. Vou fazer pattern mais direto:
    
    # Pattern comum: let pair = <expr>.fresh_reg(); <var> = pair.0; let <reg> = pair.1
    # Substituir por: let <reg> = fresh_reg(&!<var>)
    
    # Essa transformação é muito sensível ao contexto. Vou fazer apenas o básico.
    # Pattern: let <pair_var> = <lo_expr>.fresh_reg()
    
    lines = content.split('\n')
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Detectar: let pair = lo.fresh_reg()
        match = re.match(r'(\s*)let (\w+) = (\w+)\.fresh_reg\(\)', line)
        if match:
            indent = match.group(1)
            pair_var = match.group(2)
            lo_var = match.group(3)
            
            # Próxima linha deve ser: lo = pair.0
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                assign_match = re.match(rf'(\s*){lo_var} = {pair_var}\.0', next_line)
                
                if assign_match and i + 2 < len(lines):
                    # Linha seguinte deve ser: let reg = pair.1
                    third_line = lines[i + 2]
                    reg_match = re.match(rf'(\s*)let (\w+) = {pair_var}\.1', third_line)
                    
                    if reg_match:
                        reg_var = reg_match.group(2)
                        # Transformar em: let reg = fresh_reg(&!lo)
                        new_lines.append(f'{indent}let {reg_var} = fresh_reg(&!{lo_var})')
                        i += 3  # pular as 3 linhas
                        continue
        
        new_lines.append(line)
        i += 1
    
    return '\n'.join(new_lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 migrate_fresh_reg_to_borrow.py <file.sio>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Aplicar transformações
    content = migrate_fresh_reg_signature(content)
    content = migrate_fresh_reg_body(content)
    content = migrate_fresh_reg_call_sites(content)
    
    # Escrever de volta
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Migrated fresh_reg in {filepath}")

if __name__ == '__main__':
    main()
