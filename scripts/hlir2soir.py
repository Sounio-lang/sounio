#!/usr/bin/env python3
"""
hlir2soir.py — Convert Sounio HLIR text → Simple Object IR (SOIR) line format.

SOIR line types:
  F <name> <param_count> [<param_vid>...]
  B <block_id> <label>
  Ia <result_vid>
  Ic <result_vid> <const_val>
  Il <result_vid> <ptr_vid>
  Is <ptr_vid> <val_vid>
  Ib <result_vid> <binop> <l_vid> <r_vid>
  Ik <result_vid> <callee_name> <argc> [<arg_vid>...]
  Tr <ret_vid>  (-1 = void)
  Tb <target_block_id>
  Tq <cond_vid> <then_id> <else_id>
"""
import sys, re

VID = r'ValueId\(\s*(\d+)\s*,?\s*\)'
BID = r'BlockId\(\s*(\d+)\s*,?\s*\)'

OP_MAP = {'add':'add','sub':'sub','mul':'mul',
          'slt':'lt','sgt':'gt','sle':'le','sge':'ge',
          'eq':'eq','ne':'ne'}

def find_vids(text):
    return re.findall(VID, text, re.DOTALL)

def find_bid(text):
    m = re.search(BID, text, re.DOTALL)
    return m.group(1) if m else '-1'

def parse_instr(itext):
    """Return a SOIR line for one HlirInstr text block."""
    # Result
    m_res = re.search(r'result:\s*Some\(\s*' + VID, itext, re.DOTALL)
    result_vid = m_res.group(1) if m_res else '-1'

    # Detect op type by looking for distinctive keywords
    if re.search(r'\bop:\s*Alloca\b', itext):
        return f'Ia {result_vid}'

    elif re.search(r'\bop:\s*Const\(\s*Int\(', itext):
        m = re.search(r'Const\(\s*Int\(\s*(-?\d+)', itext)
        val = m.group(1) if m else '0'
        return f'Ic {result_vid} {val}'

    elif re.search(r'\bop:\s*Load\s*\{', itext):
        # ptr is the only ValueId in the Load block
        m_ptr = re.search(r'ptr:\s*' + VID, itext, re.DOTALL)
        ptr = m_ptr.group(1) if m_ptr else '-1'
        return f'Il {result_vid} {ptr}'

    elif re.search(r'\bop:\s*Store\s*\{', itext):
        m_ptr = re.search(r'ptr:\s*' + VID, itext, re.DOTALL)
        m_val = re.search(r'value:\s*' + VID, itext, re.DOTALL)
        ptr = m_ptr.group(1) if m_ptr else '-1'
        val = m_val.group(1) if m_val else '-1'
        return f'Is {ptr} {val}'

    elif re.search(r'\bop:\s*Binary\s*\{', itext):
        # Find the Binary { ... } block to extract op, left, right
        m_bin = re.search(r'op:\s*Binary\s*\{(.*?)\}', itext, re.DOTALL)
        bin_text = m_bin.group(1) if m_bin else itext
        # op name (the 'op: Add' inside Binary)
        m_op = re.search(r'\bop:\s*(\w+)', bin_text)
        binop_raw = m_op.group(1).lower() if m_op else 'add'
        binop = OP_MAP.get(binop_raw, 'add')
        # left and right ValueIds
        m_left = re.search(r'left:\s*' + VID, bin_text, re.DOTALL)
        m_right = re.search(r'right:\s*' + VID, bin_text, re.DOTALL)
        l = m_left.group(1) if m_left else '-1'
        r = m_right.group(1) if m_right else '-1'
        return f'Ib {result_vid} {binop} {l} {r}'

    elif re.search(r'\bop:\s*CallDirect\s*\{', itext):
        m_call = re.search(r'op:\s*CallDirect\s*\{(.*?)\}', itext, re.DOTALL)
        call_text = m_call.group(1) if m_call else itext
        m_name = re.search(r'name:\s*"([^"]+)"', call_text)
        callee = m_name.group(1) if m_name else 'unknown'
        # args: [ValueId(...), ...] — find the args list
        m_args = re.search(r'args:\s*\[(.*?)\]', call_text, re.DOTALL)
        if m_args:
            arg_vids = find_vids(m_args.group(1))
        else:
            arg_vids = []
        argc = len(arg_vids)
        args_str = ' '.join(arg_vids)
        return f'Ik {result_vid} {callee} {argc} {args_str}'.strip()

    return None  # unknown op, skip

def hlir_to_soir(text):
    lines = []

    func_starts = [m.start() for m in re.finditer(r'HlirFunction \{', text)]
    if not func_starts:
        return ''

    for idx, fs in enumerate(func_starts):
        next_fs = func_starts[idx + 1] if idx + 1 < len(func_starts) else len(text)
        func_text = text[fs:next_fs]

        # Function name (after id: FunctionId(...))
        m_id = re.search(r'id:\s*FunctionId\([^)]*\)', func_text, re.DOTALL)
        sf = m_id.end() if m_id else 0
        m_name = re.search(r'name:\s*"([^"]+)"', func_text[sf:])
        if not m_name:
            continue
        func_name = m_name.group(1)

        # Params
        param_vids = re.findall(r'HlirParam\s*\{.*?' + VID, func_text, re.DOTALL)
        pstr = ' '.join(param_vids)
        lines.append(f'F {func_name} {len(param_vids)} {pstr}'.strip())

        # Blocks
        block_starts = [m.start() for m in re.finditer(r'HlirBlock \{', func_text)]
        is_kernel_pos = (func_text.find('is_kernel:') + 1) or len(func_text)

        for bi, bs in enumerate(block_starts):
            if bs >= is_kernel_pos:
                break
            next_bs = block_starts[bi + 1] if bi + 1 < len(block_starts) else is_kernel_pos
            block_text = func_text[bs:next_bs]

            m_bid = re.search(BID, block_text, re.DOTALL)
            m_label = re.search(r'label:\s*"([^"]+)"', block_text)
            block_id = m_bid.group(1) if m_bid else '0'
            label = m_label.group(1) if m_label else 'entry'
            lines.append(f'B {block_id} {label}')

            # Split at terminator
            m_term = re.search(r'\bterminator:', block_text)
            if not m_term:
                continue
            term_start = m_term.start()
            instrs_text = block_text[:term_start]
            term_text = block_text[term_start + len('terminator:'):].strip()

            # Instructions: split by HlirInstr {
            instr_starts = [m.start() for m in re.finditer(r'HlirInstr \{', instrs_text)]
            for ii, ist in enumerate(instr_starts):
                next_ist = instr_starts[ii + 1] if ii + 1 < len(instr_starts) else len(instrs_text)
                itext = instrs_text[ist:next_ist]
                line = parse_instr(itext)
                if line:
                    lines.append(line)

            # Terminator
            if re.match(r'Return\(None\)', term_text):
                lines.append('Tr -1')
            elif re.match(r'Return\(', term_text):
                m = re.search(VID, term_text, re.DOTALL)
                lines.append(f'Tr {m.group(1) if m else "-1"}')
            elif re.match(r'Branch\(', term_text):
                m = re.search(BID, term_text, re.DOTALL)
                lines.append(f'Tb {m.group(1) if m else "0"}')
            elif re.match(r'CondBranch\s*\{', term_text):
                m_c = re.search(r'condition:\s*' + VID, term_text, re.DOTALL)
                m_t = re.search(r'then_block:\s*' + BID, term_text, re.DOTALL)
                m_e = re.search(r'else_block:\s*' + BID, term_text, re.DOTALL)
                c = m_c.group(1) if m_c else '-1'
                t = m_t.group(1) if m_t else '-1'
                e = m_e.group(1) if m_e else '-1'
                lines.append(f'Tq {c} {t} {e}')

    return '\n'.join(lines) + '\n'

if __name__ == '__main__':
    hlir = sys.stdin.read()
    sys.stdout.write(hlir_to_soir(hlir))
