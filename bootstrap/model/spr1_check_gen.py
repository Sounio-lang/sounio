#!/usr/bin/env python3
"""Generator of the spr1_check listing (format A) and of its Lean data.

spr1_check reads a Sporos bytecode file (formal/Sporos/Bytecode.lean) on stdin
and exits 0 iff it decodes to a program that passes Sporos.checkProg
(formal/Sporos/Check.lean); 5 on a malformed or ill-typed file, 6 when memory
is refused or a stack height reaches 2^64, 3 when the three passes disagree.

The listing bytes/spr1_check.hex is the source of record, assembled by
bootstrap/roots/a/seed-a. This script writes that listing; it is a convenience
for the author, not a step of the bootstrap path, and every instruction it
emits is re-decoded here by a transcription of Bootstrap/X86Dec.decode and
compared with the instruction that was meant.

  python3 spr1_check_gen.py OUTDIR      writes spr1_check.hex, spr1_check,
                                        SporosCheckBinData.lean.txt, symbols.txt
"""
import sys, os

W = 1 << 64
ACCEPT = 0x5acce970a55a3c96
REJECT = 0xdeadbeef11110001
OVERFLOW = 0x0bad0f1e33c3c3a5
HMAX_SHIFT = 48
PAD0 = 480
PAD1 = 5300
NEG_ACC = (W - ACCEPT) % W
NEG_OVF = (W - OVERFLOW) % W

BASE = 0x400000
BSS = 0x600000
DESC = [BSS, BSS + 0x1000, BSS + 0x2000]
BSS_SIZE = 0x3000
RES = [0, 3, 1]            # pass k writes rax, rbx, rcx
CAP0 = 0x10000

RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI = range(8)
R8, R9, R10, R11, R12, R13, R14, R15 = range(8, 16)
RN = ['rax', 'rcx', 'rdx', 'rbx', 'rsp', 'rbp', 'rsi', 'rdi',
      'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']
CC = {'o': 0, 'no': 1, 'b': 2, 'ae': 3, 'e': 4, 'ne': 5, 'be': 6, 'a': 7,
      's': 8, 'ns': 9, 'l': 12, 'ge': 13, 'le': 14, 'g': 15}
CCN = {v: k for k, v in CC.items()}
ALU_RR = {'add': 0x01, 'sub': 0x29, 'and': 0x21, 'or': 0x09, 'xor': 0x31}
ALU_EXT = {'add': 0, 'or': 1, 'and': 4, 'sub': 5, 'xor': 6}


def le(v, n):
    return [(v >> (8 * i)) & 0xff for i in range(n)]


def rex(w, r, x, b):
    return 0x40 | (w << 3) | (r << 2) | (x << 1) | b


def memop(reg, base, disp):
    """ModRM (+SIB) (+disp) for [base + disp], disp a signed int."""
    rm = base & 7
    if disp == 0 and rm != 5:
        md, d = 0, []
    elif -128 <= disp < 128:
        md, d = 1, le(disp & 0xff, 1)
    else:
        assert -(1 << 31) <= disp < (1 << 31)
        md, d = 2, le(disp & 0xffffffff, 4)
    out = [(md << 6) | ((reg & 7) << 3) | rm]
    if rm == 4:
        out.append(0x24)
    return out + d


# ---------------------------------------------------------------- encoders
# Each returns (bytes, ins) where ins is the X86Int.Ins the bytes must decode to.

def e_movI(d, imm):
    imm %= W
    if imm < (1 << 32):
        b = ([rex(0, 0, 0, 1)] if d >= 8 else []) + [0xb8 + (d & 7)] + le(imm, 4)
    else:
        b = [rex(1, 0, 0, d >> 3), 0xb8 + (d & 7)] + le(imm, 8)
    return b, ('movI', d, imm)


def e_movI64(d, imm):
    return [rex(1, 0, 0, d >> 3), 0xb8 + (d & 7)] + le(imm % W, 8), ('movI', d, imm % W)


def e_mov(d, s):
    # 8b /r with mod 11: a flipped mod bit makes it a load, never a store
    return [rex(1, d >> 3, 0, s >> 3), 0x8b, 0xc0 | ((d & 7) << 3) | (s & 7)], ('mov', d, s)


def e_mov32(d, s):
    p = [rex(0, d >> 3, 0, s >> 3)] if (d | s) >= 8 else []
    return p + [0x8b, 0xc0 | ((d & 7) << 3) | (s & 7)], ('mov32', d, s)


def e_alu(op, d, s):
    return [rex(1, s >> 3, 0, d >> 3), ALU_RR[op], 0xc0 | ((s & 7) << 3) | (d & 7)], ('alu', op, d, s)


def e_zero(d):
    p = [rex(0, d >> 3, 0, d >> 3)] if d >= 8 else []
    return p + [0x31, 0xc0 | ((d & 7) << 3) | (d & 7)], ('alu', 'xor', d, d)


def _ext(e, d, imm):
    s = imm if imm < (1 << 63) else imm - W
    if -128 <= s < 128:
        return [rex(1, 0, 0, d >> 3), 0x83, 0xc0 | (e << 3) | (d & 7)] + le(s & 0xff, 1)
    assert -(1 << 31) <= s < (1 << 31)
    return [rex(1, 0, 0, d >> 3), 0x81, 0xc0 | (e << 3) | (d & 7)] + le(s & 0xffffffff, 4)


def e_aluI(op, d, imm):
    imm %= W
    return _ext(ALU_EXT[op], d, imm), ('aluI', op, d, imm)


def e_cmpI(a, imm):
    imm %= W
    return _ext(7, a, imm), ('cmpI', a, imm)


def e_cmp(a, b):
    return [rex(1, b >> 3, 0, a >> 3), 0x39, 0xc0 | ((b & 7) << 3) | (a & 7)], ('cmp', a, b)


def e_test(a, b):
    return [rex(1, b >> 3, 0, a >> 3), 0x85, 0xc0 | ((b & 7) << 3) | (a & 7)], ('test', a, b)


def e_shlI(d, c):
    return [rex(1, 0, 0, d >> 3), 0xc1, 0xe0 | (d & 7), c], ('shlI', d, c)


def e_shrI(d, c):
    return [rex(1, 0, 0, d >> 3), 0xc1, 0xe8 | (d & 7), c], ('shrI', d, c)


def e_ld(d, b, disp=0):
    return [rex(1, d >> 3, 0, b >> 3), 0x8b] + memop(d, b, disp), ('ld', d, b, disp % W)


def e_ld32(d, b, disp=0):
    p = [rex(0, d >> 3, 0, b >> 3)] if (d | b) >= 8 else []
    return p + [0x8b] + memop(d, b, disp), ('ld32', d, b, disp % W)


def e_ld8(d, b, disp=0):
    p = [rex(0, d >> 3, 0, b >> 3)] if (d | b) >= 8 else []
    return p + [0x0f, 0xb6] + memop(d, b, disp), ('ld8', d, b, disp % W)


def e_st(b, disp, s):
    return [rex(1, s >> 3, 0, b >> 3), 0x89] + memop(s, b, disp), ('st', b, disp % W, s)


def e_st8(b, disp, s):
    p = [rex(0, s >> 3, 0, b >> 3)] if (s >= 4 or b >= 8) else []
    return p + [0x88] + memop(s, b, disp), ('st8', b, disp % W, s)


def e_lea(d, b, disp):
    return [rex(1, d >> 3, 0, b >> 3), 0x8d] + memop(d, b, disp), ('lea', d, b, disp % W)


def e_syscall():
    return [0x0f, 0x05], ('syscall',)


# ------------------------------------------------------- X86Dec transcription

def sx8(v):
    return v if v < 128 else v + W - 256


def sx32(v):
    return v if v < (1 << 31) else v + W - (1 << 32)


def leB(b, off, n):
    r = 0
    for i in range(n):
        x = b(off + i)
        if x is None:
            return None
        r += x << (8 * i)
    return r


def memOp(b, off, m, bb, x):
    md, rm = m // 64, m % 8
    if md == 3:
        return None
    if rm == 4:
        s = b(off)
        if s is None:
            return None
        if s // 8 % 8 != 4 or x:
            return None
        if md == 0 and s % 8 == 5:
            return None
        base, off1 = s % 8 + bb, off + 1
    elif md == 0 and rm == 5:
        return None
    else:
        base, off1 = rm + bb, off
    if md == 0:
        return (base, 0, off1)
    if md == 1:
        d = b(off1)
        if d is None:
            return None
        return (base, sx8(d), off1 + 1)
    d = leB(b, off1, 4)
    if d is None:
        return None
    return (base, sx32(d), off1 + 4)


def aluOfRR(o):
    return {0x01: 'add', 0x29: 'sub', 0x21: 'and', 0x09: 'or', 0x31: 'xor'}.get(o)


def aluOfExt(e):
    return {0: 'add', 1: 'or', 4: 'and', 5: 'sub', 6: 'xor'}.get(e)


def decode(b, a):
    b0 = b(0)
    if b0 is None:
        return None
    hasRex = 0x40 <= b0 <= 0x4f
    rx = b0 - 0x40 if hasRex else 0
    p = 1 if hasRex else 0
    w = rx // 8 % 2 == 1
    r = (rx // 4 % 2) * 8
    x = rx // 2 % 2 == 1
    bb = (rx % 2) * 8
    o = b(p)
    if o is None:
        return None
    tgt = lambda ln, rel: (a + ln + rel) % W
    if 0xb8 <= o <= 0xbf:
        if r != 0 or x:
            return None
        if w:
            imm = leB(b, p + 1, 8)
            return None if imm is None else (('movI', o - 0xb8 + bb, imm), p + 9)
        imm = leB(b, p + 1, 4)
        return None if imm is None else (('movI', o - 0xb8 + bb, imm), p + 5)
    if o in (0x01, 0x29, 0x21, 0x09, 0x31):
        m = b(p + 1)
        if m is None:
            return None
        if m // 64 != 3 or x:
            return None
        d, s = m % 8 + bb, m // 8 % 8 + r
        op = aluOfRR(o)
        if w:
            return (('alu', op, d, s), p + 2)
        if o == 0x31 and d == s:
            return (('alu', 'xor', d, d), p + 2)
        return None
    if o in (0x89, 0x8b):
        if x:
            return None
        m = b(p + 1)
        if m is None:
            return None
        if not w:
            rm, rg = m % 8 + bb, m // 8 % 8 + r
            if m // 64 == 3:
                return (('mov32', rm, rg), p + 2) if o == 0x89 else (('mov32', rg, rm), p + 2)
            if o == 0x89:
                return None
            mo = memOp(b, p + 2, m, bb, x)
            if mo is None:
                return None
            return (('ld32', rg, mo[0], mo[1]), mo[2])
        if m // 64 == 3:
            rm, rg = m % 8 + bb, m // 8 % 8 + r
            return (('mov', rm, rg), p + 2) if o == 0x89 else (('mov', rg, rm), p + 2)
        mo = memOp(b, p + 2, m, bb, x)
        if mo is None:
            return None
        rg = m // 8 % 8 + r
        return (('st', mo[0], mo[1], rg), mo[2]) if o == 0x89 else (('ld', rg, mo[0], mo[1]), mo[2])
    if o == 0x85:
        if not w or x:
            return None
        m = b(p + 1)
        if m is None or m // 64 != 3:
            return None
        return (('test', m % 8 + bb, m // 8 % 8 + r), p + 2)
    if o == 0x6a:
        if hasRex:
            return None
        i = b(1)
        return None if i is None else (('pushI', sx8(i)), 2)
    if o in (0x39, 0x3b):
        if not w or x:
            return None
        m = b(p + 1)
        if m is None or m // 64 != 3:
            return None
        rm, rg = m % 8 + bb, m // 8 % 8 + r
        return (('cmp', rm, rg), p + 2) if o == 0x39 else (('cmp', rg, rm), p + 2)
    if o in (0x83, 0x81):
        if not w or r != 0 or x:
            return None
        m = b(p + 1)
        if m is None or m // 64 != 3:
            return None
        d, e = m % 8 + bb, m // 8 % 8
        if o == 0x83:
            i = b(p + 2)
            if i is None:
                return None
            imm, ln = sx8(i), p + 3
        else:
            i = leB(b, p + 2, 4)
            if i is None:
                return None
            imm, ln = sx32(i), p + 6
        if e == 7:
            return (('cmpI', d, imm), ln)
        op = aluOfExt(e)
        return None if op is None else (('aluI', op, d, imm), ln)
    if o == 0xc1:
        if not w or r != 0 or x:
            return None
        m = b(p + 1)
        c = b(p + 2)
        if m is None or c is None or m // 64 != 3:
            return None
        d = m % 8 + bb
        if m // 8 % 8 == 4:
            return (('shlI', d, c), p + 3)
        if m // 8 % 8 == 5:
            return (('shrI', d, c), p + 3)
        return None
    if o == 0x8d:
        if not w or x:
            return None
        m = b(p + 1)
        if m is None:
            return None
        mo = memOp(b, p + 2, m, bb, x)
        if mo is None:
            return None
        return (('lea', m // 8 % 8 + r, mo[0], mo[1]), mo[2])
    if o == 0x88:
        if x:
            return None
        m = b(p + 1)
        if m is None:
            return None
        mo = memOp(b, p + 2, m, bb, x)
        if mo is None:
            return None
        reg = m // 8 % 8
        if not hasRex and reg >= 4:
            return None
        return (('st8', mo[0], mo[1], reg + r), mo[2])
    if o == 0x0f:
        o2 = b(p + 1)
        if o2 is None:
            return None
        if o2 == 0x05:
            return None if hasRex else (('syscall',), 2)
        if o2 == 0xb6:
            if x:
                return None
            m = b(p + 2)
            if m is None:
                return None
            mo = memOp(b, p + 3, m, bb, x)
            if mo is None:
                return None
            return (('ld8', m // 8 % 8 + r, mo[0], mo[1]), mo[2])
        if 0x80 <= o2 <= 0x8f:
            if hasRex:
                return None
            d = leB(b, 2, 4)
            return None if d is None else (('jcc', o2 - 0x80, tgt(6, sx32(d))), 6)
        return None
    if 0x70 <= o <= 0x7f:
        if hasRex:
            return None
        d = b(1)
        return None if d is None else (('jcc', o - 0x70, tgt(2, sx8(d))), 2)
    if o == 0xeb:
        if hasRex:
            return None
        d = b(1)
        return None if d is None else (('jmp', tgt(2, sx8(d))), 2)
    if o == 0xe9:
        if hasRex:
            return None
        d = leB(b, 1, 4)
        return None if d is None else (('jmp', tgt(5, sx32(d))), 5)
    if o == 0xe8:
        if hasRex:
            return None
        d = leB(b, 1, 4)
        return None if d is None else (('call', tgt(5, sx32(d))), 5)
    if o == 0xc3:
        return None if hasRex else (('ret',), 1)
    if 0x50 <= o <= 0x57:
        if w or r != 0 or x:
            return None
        return (('push', o - 0x50 + bb), p + 1)
    if 0x58 <= o <= 0x5f:
        if w or r != 0 or x:
            return None
        return (('pop', o - 0x58 + bb), p + 1)
    if o == 0xff:
        if w or r != 0:
            return None
        m = b(p + 1)
        if m is None or m // 8 % 8 != 4 or m // 64 == 3 or m % 8 != 4:
            return None
        s = b(p + 2)
        if s is None or s // 64 != 3:
            return None
        idx = s // 8 % 8 + (8 if x else 0)
        if idx == 4:
            return None
        md = m // 64
        if md == 0 and s % 8 == 5:
            return None
        base = s % 8 + bb
        if md == 0:
            return (('jtab', base, idx, 0), p + 3)
        if md == 1:
            d = b(p + 3)
            return None if d is None else (('jtab', base, idx, sx8(d)), p + 4)
        d = leB(b, p + 3, 4)
        return None if d is None else (('jtab', base, idx, sx32(d)), p + 7)
    return None


# ---------------------------------------------------------------- assembler

class Asm:
    """Labels, instructions, jumps. A jump is short (rel8) when it reaches,
    unless it is marked far; far and out-of-reach jumps are rel32. A far or
    long jne is emitted as je over a jmp rel32, so that 0f 85 never occurs.
    Macro instances are recorded (name, parameters, item range) so that the
    Lean side can state the code of each instance by a template."""
    def __init__(self):
        self.items = []
        self.n = 0
        self.macros = []      # [name, params, start_idx, end_idx]
        self.open = []

    def label(self, name):
        self.items.append(('L', name))

    def fresh(self, stem):
        self.n += 1
        return f'{stem}_{self.n}'

    def i(self, fn, *args, c=''):
        self.items.append(('I', fn, args, c))

    def jmp(self, lab, c='', far=False):
        self.items.append(('J', 'jmp', None, lab, c, far))

    def jcc(self, cc, lab, c='', far=False):
        self.items.append(('J', 'jcc', CC[cc], lab, c, far))

    def raw(self, bs, ins, c=''):
        self.items.append(('R', bs, ins, c))

    def mbegin(self, name, **params):
        self.open.append([name, params, len(self.items), None])

    def mend(self):
        m = self.open.pop()
        m[3] = len(self.items)
        self.macros.append(m)

    def assemble(self, org):
        long_ = {idx for idx, it in enumerate(self.items) if it[0] == 'J' and it[5]}
        while True:
            addr, labels, idx_addr = org, {}, {}
            for idx, it in enumerate(self.items):
                idx_addr[idx] = addr
                if it[0] == 'L':
                    assert it[1] not in labels, it[1]
                    labels[it[1]] = addr
                elif it[0] == 'I':
                    addr += len(it[1](*it[2])[0])
                elif it[0] == 'R':
                    addr += len(it[1])
                else:
                    if idx in long_:
                        addr += 5 if it[1] == 'jmp' else (7 if it[2] == CC['ne'] else 6)
                    else:
                        addr += 2
            idx_addr[len(self.items)] = addr
            grew = False
            out = []
            for idx, it in enumerate(self.items):
                a = idx_addr[idx]
                if it[0] == 'L':
                    continue
                if it[0] == 'I':
                    bs, ins = it[1](*it[2])
                    out.append((a, bs, ins, it[3], idx))
                elif it[0] == 'R':
                    out.append((a, it[1], it[2], it[3], idx))
                else:
                    _, kind, cc, lab, c, _f = it
                    t = labels[lab]
                    if idx not in long_:
                        rel = t - (a + 2)
                        if -128 <= rel < 128:
                            if kind == 'jmp':
                                bs, ins = [0xeb, rel & 0xff], ('jmp', t)
                            else:
                                bs, ins = [0x70 + cc, rel & 0xff], ('jcc', cc, t)
                            out.append((a, bs, ins, c + f' -> {lab}', idx))
                            continue
                        long_.add(idx)
                        grew = True
                        break
                    if kind == 'jmp':
                        rel = t - (a + 5)
                        out.append((a, [0xe9] + le(rel & 0xffffffff, 4), ('jmp', t), c + f' -> {lab}', idx))
                    elif cc == CC['ne']:
                        # never emit 0f 85 (one bit from 0f 05): je over a jmp rel32
                        out.append((a, [0x74, 0x05], ('jcc', CC['e'], a + 7), c + ' (jne: je over the jmp)', idx))
                        rel = t - (a + 7)
                        out.append((a + 2, [0xe9] + le(rel & 0xffffffff, 4), ('jmp', t), f'-> {lab}', idx))
                    else:
                        rel = t - (a + 6)
                        out.append((a, [0x0f, 0x80 + cc] + le(rel & 0xffffffff, 4), ('jcc', cc, t),
                                    c + f' -> {lab}', idx))
            if not grew:
                self.idx_addr = idx_addr
                return out, labels


# ---------------------------------------------------------------- the program
#
# Registers in the passes. rax, rbx, rcx hold the three results and are written
# only by the verdicts; rsp is never touched. rbp is the end of the input copy.
# The macros (skipCode, skipWords, skipData, lookup, the height check, the
# effect table) use only r8-r11, r14, r15, so that every instance of a macro has
# the same encoding lengths; jumps out of a macro are always rel32.
#
#   pass   rdi base (header only)   r12 main / word header   r13 ws
#          rsi nw / p               rdx curD                 rdi curA
#          r8 i                     r9 annOff                r10 r11 r14 r15 scratch

def skipcode(A, p, rej, c=''):
    """skipCode: advance p over r11 instructions; refuse a bad opcode or a
    truncated instruction (bounds against rbp = end of input). Scratch r14."""
    A.mbegin('skipcode', p=p, rej=rej)
    cnt, t1 = R11, R14
    top, done = A.fresh('sc_top'), A.fresh('sc_done')
    l1, l5, l9, ln = A.fresh('sc_l1'), A.fresh('sc_l5'), A.fresh('sc_l9'), A.fresh('sc_len')
    A.label(top)
    A.i(e_test, cnt, cnt, c=f'skipCode{c}: instructions left?')
    A.jcc('e', done)
    A.i(e_cmp, p, RBP, c='opcode byte inside the input?')
    A.jcc('ae', rej, far=True)
    A.i(e_ld8, t1, p, 0, c='opcode')
    A.i(e_cmpI, t1, 57, c='prim 0x00..0x38')
    A.jcc('b', l1)
    A.i(e_cmpI, t1, 0x80)
    A.jcc('b', rej, c='0x39..0x7f: no such opcode', far=True)
    A.jcc('e', l9, c='0x80 lit u64')
    A.i(e_cmpI, t1, 0x8c)
    A.jcc('a', rej, c='0x8d..0xff: no such opcode', far=True)
    A.jcc('e', l1, c='0x8c ret')
    A.i(e_cmpI, t1, 0x83)
    A.jcc('b', l5, c='0x81 str, 0x82 pick')
    A.i(e_cmpI, t1, 0x86)
    A.jcc('b', l1, c='0x83 tor, 0x84 fromr, 0x85 rfetch')
    A.label(l5)
    A.i(e_movI, t1, 5, c='opcode + u32')
    A.jmp(ln)
    A.label(l9)
    A.i(e_movI, t1, 9, c='opcode + u64')
    A.jmp(ln)
    A.label(l1)
    A.i(e_movI, t1, 1, c='opcode alone')
    A.label(ln)
    A.i(e_alu, 'add', p, t1, c='past the instruction')
    A.i(e_cmp, p, RBP)
    A.jcc('a', rej, c='truncated instruction', far=True)
    A.i(e_aluI, 'sub', cnt, 1)
    A.jmp(top)
    A.label(done)
    A.mend()


def skipwords(A, rej, c=''):
    """skipWords: advance r10 over r15 words (header, code, annotations). The
    annotation count is parked in the high half of r15 while the code is
    skipped (r15 < 2^32). Scratch r11, r14."""
    A.mbegin('skipwords', rej=rej)
    p, wc, nc, t1 = R10, R15, R11, R14
    top, done = A.fresh('sw_top'), A.fresh('sw_done')
    A.label(top)
    A.i(e_test, wc, wc, c=f'skipWords{c}: words left?')
    A.jcc('e', done)
    A.i(e_mov, t1, p)
    A.i(e_aluI, 'add', t1, 20)
    A.i(e_cmp, t1, RBP)
    A.jcc('a', rej, c='truncated word header', far=True)
    A.i(e_ld32, t1, p, 16, c='na')
    A.i(e_shlI, t1, 32)
    A.i(e_alu, 'add', wc, t1, c='park na in the high half')
    A.i(e_ld32, nc, p, 12, c='nc')
    A.i(e_aluI, 'add', p, 20, c='first instruction')
    skipcode(A, p, rej, c)
    A.i(e_mov, t1, wc)
    A.i(e_shrI, t1, 32, c='na')
    A.i(e_mov32, wc, wc, c='words left, high half cleared')
    A.i(e_mov, nc, t1)
    A.i(e_shlI, nc, 3)
    A.i(e_shlI, t1, 2)
    A.i(e_alu, 'add', nc, t1, c='12 * na')
    A.i(e_alu, 'add', p, nc, c='past the annotations')
    A.i(e_cmp, p, RBP)
    A.jcc('a', rej, c='truncated annotations', far=True)
    A.i(e_aluI, 'sub', wc, 1)
    A.jmp(top)
    A.label(done)
    A.mend()


def skipdata(A, rej):
    """skipData: advance r10 over r15 data items. Scratch r11, r14."""
    A.mbegin('skipdata', rej=rej)
    p, cnt, t1, t2 = R10, R15, R11, R14
    top, done = A.fresh('sd_top'), A.fresh('sd_done')
    A.label(top)
    A.i(e_test, cnt, cnt, c='skipData: items left?')
    A.jcc('e', done)
    A.i(e_mov, t1, p)
    A.i(e_aluI, 'add', t1, 4)
    A.i(e_cmp, t1, RBP)
    A.jcc('a', rej, c='truncated length', far=True)
    A.i(e_ld32, t2, p, 0, c='len')
    A.i(e_mov, p, t1)
    A.i(e_mov, t1, RBP)
    A.i(e_alu, 'sub', t1, p, c='bytes left')
    A.i(e_cmp, t1, t2)
    A.jcc('b', rej, c='truncated data', far=True)
    A.i(e_alu, 'add', p, t2)
    A.i(e_aluI, 'sub', cnt, 1)
    A.jmp(top)
    A.label(done)
    A.mend()


def chk(A, r, ovf):
    """The height in r (rdx or rdi) must stay below 2^48 (HMAX). Scratch r10."""
    A.mbegin('chk', r=r, ovf=ovf)
    A.i(e_mov, R10, r)
    A.i(e_shrI, R10, HMAX_SHIFT)
    A.i(e_test, R10, R10)
    A.jcc('ne', ovf, c='height >= 2^48', far=True)
    A.mend()


def lookup(A, t, c=''):
    """lookupAnn: r10 := address of the first annotation of the current word
    (r12 header, r9 first annotation) whose index is t, or r10 = r11 = end of
    the annotations when there is none. Scratch r14."""
    A.mbegin('lookup', t=t)
    top, done = A.fresh('lk_top'), A.fresh('lk_done')
    A.i(e_ld32, R14, R12, 16, c=f'lookup{c}: na')
    A.i(e_mov, R11, R14)
    A.i(e_shlI, R11, 3)
    A.i(e_shlI, R14, 2)
    A.i(e_alu, 'add', R11, R14)
    A.i(e_alu, 'add', R11, R9, c='r11 = end of the annotations')
    A.i(e_mov, R10, R9)
    A.label(top)
    A.i(e_cmp, R10, R11)
    A.jcc('e', done, c='none')
    A.i(e_ld32, R14, R10, 0)
    A.i(e_cmp, R14, t)
    A.jcc('e', done, c='found')
    A.i(e_aluI, 'add', R10, 12)
    A.jmp(top)
    A.label(done)
    A.mend()


# Prim.eff, by opcode: (operands taken, results pushed). exit (50) is handled apart.
EFF = [(1, 2), (1, 0), (2, 2), (2, 3), (3, 3), (2, 1), (2, 3),
       (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (1, 1),
       (2, 1), (2, 1), (2, 1), (1, 1), (2, 1), (2, 1), (2, 1),
       (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1),
       (1, 1), (2, 0), (1, 1), (2, 0), (1, 1), (2, 0), (1, 1), (2, 0),
       (3, 0), (3, 0), (3, 1),
       (1, 1), (0, 1), (1, 0),
       (4, 1), (2, 1),
       (0, 1), (3, 0), (2, 2),
       (0, 1), (1, 0), (1, 0), (1, 0),
       (0, 1), (0, 1), (1, 1),
       (1, 2), (1, 2), (2, 1)]
assert len(EFF) == 57


def effchain(A, have):
    """r10, r11 := Prim.eff of opcode r15 (< 57), by a binary search over the
    runs of equal effects."""
    A.mbegin('eff', tgt=have)
    runs = []
    for op in range(57):
        if runs and runs[-1][2] == EFF[op]:
            runs[-1][1] = op
        else:
            runs.append([op, op, EFF[op]])

    def tree(rs):
        if len(rs) == 1:
            i_, o_ = rs[0][2]
            A.i(e_movI, R10, i_, c=f'ops {rs[0][0]}..{rs[0][1]}: takes {i_}')
            A.i(e_movI, R11, o_, c=f'pushes {o_}')
            A.jmp(have, far=True)
            return
        mid = len(rs) // 2
        right = A.fresh('eff')
        A.i(e_cmpI, R15, rs[mid][0])
        A.jcc('ae', right, far=True)
        tree(rs[:mid])
        A.label(right)
        tree(rs[mid:])
    tree(runs)
    A.mend()


def gen_pass(A, k):
    D, R = DESC[k], RES[k]
    P = f'p{k + 1}'
    rej, ovf, acc = P + '_rej', P + '_ovf', P + '_acc'
    nxt = f'p{k + 2}' if k < 2 else 'vote'
    A.label(P)
    # ---- P1: parse the whole file
    A.i(e_movI, R15, D, c=f'pass {k + 1}: descriptor {k + 1}')
    A.i(e_ld, RDI, R15, 0, c='base of input copy')
    A.i(e_ld, RDX, R15, 8, c='N')
    A.i(e_mov, RBP, RDI)
    A.i(e_alu, 'add', RBP, RDX, c='rbp = end of input')
    A.i(e_cmpI, RDX, 16)
    A.jcc('b', rej, c='shorter than the file header', far=True)
    A.i(e_ld32, RDX, RDI, 0)
    A.i(e_cmpI, RDX, 0x31525053)
    A.jcc('ne', rej, c='not "SPR1"', far=True)
    A.i(e_ld32, R12, RDI, 4, c='main')
    A.i(e_ld32, RSI, RDI, 8, c='nw')
    A.i(e_ld32, R15, RDI, 12, c='nd')
    A.i(e_lea, R10, RDI, 16, c='first data item')
    skipdata(A, rej)
    A.i(e_mov, R13, R10, c='ws = first word')
    A.i(e_mov, R15, RSI, c='nw words')
    skipwords(A, rej, ' (parse)')
    A.i(e_cmp, R10, RBP)
    A.jcc('ne', rej, c='trailing bytes', far=True)
    # ---- P2: main
    A.i(e_cmp, R12, RSI)
    A.jcc('ae', rej, c='main >= nw', far=True)
    A.i(e_mov, R10, R13)
    A.i(e_mov, R15, R12)
    skipwords(A, rej, ' (main)')
    A.i(e_ld32, RDX, R10, 0, c='ins of main')
    A.i(e_test, RDX, RDX)
    A.jcc('ne', rej, c='main takes arguments', far=True)
    # ---- P3: every word
    A.i(e_mov, R12, R13, c='r12 = word header')
    wl = P + '_word'
    A.label(wl)
    A.i(e_cmp, R12, RBP)
    A.jcc('e', acc, c='all words checked', far=True)
    A.i(e_ld32, R11, R12, 12, c='nc')
    A.i(e_test, R11, R11)
    A.jcc('e', rej, c='empty code', far=True)
    A.i(e_lea, R9, R12, 20)
    skipcode(A, R9, rej, ' (annotations)')
    A.i(e_zero, R8, c='i = 0')
    lookup(A, R8, ' (0)')
    init = A.fresh('init')
    A.i(e_cmp, R10, R11)
    A.jcc('e', init)
    A.i(e_ld32, R14, R10, 4)
    A.i(e_ld32, R15, R12, 0)
    A.i(e_cmp, R14, R15)
    A.jcc('ne', rej, c='H[0].d != ins', far=True)
    A.i(e_ld32, R14, R10, 8)
    A.i(e_test, R14, R14)
    A.jcc('ne', rej, c='H[0].a != 0', far=True)
    A.label(init)
    A.i(e_ld32, RDX, R12, 0, c='curD = ins')
    A.i(e_zero, RDI, c='curA = 0')
    A.i(e_lea, RSI, R12, 20, c='p = first instruction')
    walk, wdone = P + '_walk', P + '_wdone'
    fall, nofall, adv = P + '_fall', P + '_nofall', P + '_adv'
    A.label(walk)
    A.i(e_ld32, R15, R12, 12)
    A.i(e_cmp, R8, R15)
    A.jcc('e', wdone, c='i == nc', far=True)
    lookup(A, R8, ' (i)')
    hr = A.fresh('hready')
    A.i(e_cmp, R10, R11)
    A.jcc('e', hr)
    A.i(e_ld32, RDX, R10, 4, c='h = annotation at i')
    A.i(e_ld32, RDI, R10, 8)
    A.label(hr)
    A.i(e_ld8, R15, RSI, 0, c='opcode')
    names = {}
    for nm in ['prim', 'lit', 'str', 'pick', 'tor', 'fromr', 'rfetch', 'locget', 'locset',
               'jmp', 'jz', 'call', 'ret', 'have', 'exit']:
        names[nm] = f'{P}_{nm}'
    A.i(e_cmpI, R15, 57)
    A.jcc('b', names['prim'], far=True)
    for op, nm in [(0x80, 'lit'), (0x81, 'str'), (0x82, 'pick'), (0x83, 'tor'), (0x84, 'fromr'),
                   (0x85, 'rfetch'), (0x86, 'locget'), (0x87, 'locset'), (0x88, 'jmp'),
                   (0x89, 'jz'), (0x8a, 'jz'), (0x8b, 'call'), (0x8c, 'ret')]:
        A.i(e_cmpI, R15, op)
        A.jcc('e', names[nm], far=True)
    A.jmp(rej, c='unreachable after parse', far=True)
    # prim
    A.label(names['prim'])
    A.i(e_cmpI, R15, 50)
    A.jcc('e', names['exit'], far=True)
    effchain(A, names['have'])
    A.label(names['have'])
    A.i(e_cmp, RDX, R10)
    A.jcc('b', rej, c='too few operands', far=True)
    A.i(e_alu, 'sub', RDX, R10)
    A.i(e_alu, 'add', RDX, R11)
    chk(A, RDX, ovf)
    A.i(e_movI, R15, 1)
    A.jmp(fall, far=True)
    A.label(names['exit'])
    A.i(e_test, RDX, RDX)
    A.jcc('e', rej, c='exit needs one operand', far=True)
    A.i(e_movI, R15, 1)
    A.jmp(nofall, far=True)
    # lit / str
    A.label(names['lit'])
    A.i(e_aluI, 'add', RDX, 1)
    chk(A, RDX, ovf)
    A.i(e_movI, R15, 9)
    A.jmp(fall, far=True)
    A.label(names['str'])
    A.i(e_aluI, 'add', RDX, 2)
    chk(A, RDX, ovf)
    A.i(e_movI, R15, 5)
    A.jmp(fall, far=True)
    # pick n: n < d
    A.label(names['pick'])
    A.i(e_ld32, R10, RSI, 1)
    A.i(e_cmp, R10, RDX)
    A.jcc('ae', rej, c='pick beyond the stack', far=True)
    A.i(e_aluI, 'add', RDX, 1)
    chk(A, RDX, ovf)
    A.i(e_movI, R15, 5)
    A.jmp(fall, far=True)
    # tor
    A.label(names['tor'])
    A.i(e_test, RDX, RDX)
    A.jcc('e', rej, far=True)
    A.i(e_aluI, 'sub', RDX, 1)
    A.i(e_aluI, 'add', RDI, 1)
    chk(A, RDI, ovf)
    A.i(e_movI, R15, 1)
    A.jmp(fall, far=True)
    # fromr
    A.label(names['fromr'])
    A.i(e_test, RDI, RDI)
    A.jcc('e', rej, far=True)
    A.i(e_aluI, 'sub', RDI, 1)
    A.i(e_aluI, 'add', RDX, 1)
    chk(A, RDX, ovf)
    A.i(e_movI, R15, 1)
    A.jmp(fall, far=True)
    # rfetch
    A.label(names['rfetch'])
    A.i(e_test, RDI, RDI)
    A.jcc('e', rej, far=True)
    A.i(e_aluI, 'add', RDX, 1)
    chk(A, RDX, ovf)
    A.i(e_movI, R15, 1)
    A.jmp(fall, far=True)
    # locget i: i < locals
    A.label(names['locget'])
    A.i(e_ld32, R10, RSI, 1)
    A.i(e_ld32, R11, R12, 8, c='locals')
    A.i(e_cmp, R10, R11)
    A.jcc('ae', rej, far=True)
    A.i(e_aluI, 'add', RDX, 1)
    chk(A, RDX, ovf)
    A.i(e_movI, R15, 5)
    A.jmp(fall, far=True)
    # locset i: 1 <= d and i < locals
    A.label(names['locset'])
    A.i(e_test, RDX, RDX)
    A.jcc('e', rej, far=True)
    A.i(e_ld32, R10, RSI, 1)
    A.i(e_ld32, R11, R12, 8, c='locals')
    A.i(e_cmp, R10, R11)
    A.jcc('ae', rej, far=True)
    A.i(e_aluI, 'sub', RDX, 1)
    A.i(e_movI, R15, 5)
    A.jmp(fall, far=True)
    # jmp t: ann(t) = h
    A.label(names['jmp'])
    A.i(e_ld32, R15, RSI, 1, c='t')
    lookup(A, R15, ' (jmp t)')
    A.i(e_cmp, R10, R11)
    A.jcc('e', rej, c='no annotation at t', far=True)
    A.i(e_ld32, R14, R10, 4)
    A.i(e_cmp, R14, RDX)
    A.jcc('ne', rej, far=True)
    A.i(e_ld32, R14, R10, 8)
    A.i(e_cmp, R14, RDI)
    A.jcc('ne', rej, far=True)
    A.i(e_movI, R15, 5)
    A.jmp(nofall, far=True)
    # jz/jnz t: 1 <= d and ann(t) = (d - 1, a)
    A.label(names['jz'])
    A.i(e_test, RDX, RDX)
    A.jcc('e', rej, far=True)
    A.i(e_aluI, 'sub', RDX, 1)
    A.i(e_ld32, R15, RSI, 1, c='t')
    lookup(A, R15, ' (jz t)')
    A.i(e_cmp, R10, R11)
    A.jcc('e', rej, c='no annotation at t', far=True)
    A.i(e_ld32, R14, R10, 4)
    A.i(e_cmp, R14, RDX)
    A.jcc('ne', rej, far=True)
    A.i(e_ld32, R14, R10, 8)
    A.i(e_cmp, R14, RDI)
    A.jcc('ne', rej, far=True)
    A.i(e_movI, R15, 5)
    A.jmp(fall, far=True)
    # call w: w < nw, c.ins <= d, d - c.ins + c.outs
    A.label(names['call'])
    A.i(e_ld32, R15, RSI, 1, c='w')
    A.i(e_movI, R14, D)
    A.i(e_ld, R14, R14, 0, c='base')
    A.i(e_ld32, R14, R14, 8, c='nw')
    A.i(e_cmp, R15, R14)
    A.jcc('ae', rej, c='no such word', far=True)
    A.i(e_mov, R10, R13, c='rescan from ws')
    skipwords(A, rej, ' (call)')
    A.i(e_ld32, R14, R10, 0, c='c.ins')
    A.i(e_cmp, RDX, R14)
    A.jcc('b', rej, c='too few arguments', far=True)
    A.i(e_alu, 'sub', RDX, R14)
    A.i(e_ld32, R14, R10, 4, c='c.outs')
    A.i(e_alu, 'add', RDX, R14)
    chk(A, RDX, ovf)
    A.i(e_movI, R15, 5)
    A.jmp(fall, far=True)
    # ret: h = (outs, 0)
    A.label(names['ret'])
    A.i(e_ld32, R14, R12, 4, c='outs')
    A.i(e_cmp, RDX, R14)
    A.jcc('ne', rej, far=True)
    A.i(e_test, RDI, RDI)
    A.jcc('ne', rej, far=True)
    A.i(e_movI, R15, 1)
    A.jmp(nofall, far=True)
    # falls through to h' = (rdx, rdi): i + 1 < nc and ann(i + 1) in {none, h'}
    A.label(fall)
    A.i(e_aluI, 'add', R8, 1)
    A.i(e_ld32, R14, R12, 12)
    A.i(e_cmp, R8, R14)
    A.jcc('ae', rej, c='falls off the end', far=True)
    lookup(A, R8, ' (i + 1)')
    A.i(e_cmp, R10, R11)
    A.jcc('e', adv)
    A.i(e_ld32, R14, R10, 4)
    A.i(e_cmp, R14, RDX)
    A.jcc('ne', rej, far=True)
    A.i(e_ld32, R14, R10, 8)
    A.i(e_cmp, R14, RDI)
    A.jcc('ne', rej, far=True)
    A.label(adv)
    A.i(e_alu, 'add', RSI, R15, c='next instruction')
    A.jmp(walk, far=True)
    A.label(nofall)
    A.i(e_aluI, 'add', R8, 1)
    lookup(A, R8, ' (i + 1)')
    nz = A.fresh('nfzero')
    A.i(e_cmp, R10, R11)
    A.jcc('e', nz)
    A.i(e_ld32, RDX, R10, 4)
    A.i(e_ld32, RDI, R10, 8)
    A.jmp(adv, far=True)
    A.label(nz)
    A.i(e_zero, RDX)
    A.i(e_zero, RDI)
    A.jmp(adv, far=True)
    # annotations: every (t, d, a) has t < nc and lookup t = (d, a)
    A.label(wdone)
    A.i(e_ld32, R14, R12, 16, c='na')
    A.i(e_mov, RDI, R14)
    A.i(e_shlI, RDI, 3)
    A.i(e_shlI, R14, 2)
    A.i(e_alu, 'add', RDI, R14)
    A.i(e_alu, 'add', RDI, R9, c='rdi = end of the annotations = next word')
    A.i(e_mov, RSI, R9)
    al, ad = P + '_anns', P + '_annsok'
    A.label(al)
    A.i(e_cmp, RSI, RDI)
    A.jcc('e', ad)
    A.i(e_ld32, R15, RSI, 0, c='t')
    A.i(e_ld32, R14, R12, 12)
    A.i(e_cmp, R15, R14)
    A.jcc('ae', rej, c='annotation past the code', far=True)
    lookup(A, R15, ' (ann t)')
    A.i(e_cmp, R10, R11)
    A.jcc('e', rej, far=True)
    A.i(e_ld32, R14, R10, 4)
    A.i(e_ld32, RDX, RSI, 4)
    A.i(e_cmp, R14, RDX)
    A.jcc('ne', rej, c='conflicting annotations', far=True)
    A.i(e_ld32, R14, R10, 8)
    A.i(e_ld32, RDX, RSI, 8)
    A.i(e_cmp, R14, RDX)
    A.jcc('ne', rej, c='conflicting annotations', far=True)
    A.i(e_aluI, 'add', RSI, 12)
    A.jmp(al, far=True)
    A.label(ad)
    A.i(e_mov, R12, RDI, c='next word')
    A.jmp(wl, far=True)
    # verdicts
    A.label(acc)
    A.i(e_movI64, R, ACCEPT, c=f'pass {k + 1}: ACCEPT')
    A.jmp(nxt, far=True)
    A.label(ovf)
    A.i(e_movI64, R, OVERFLOW, c=f'pass {k + 1}: OVERFLOW')
    A.jmp(nxt, far=True)
    A.label(rej)
    A.i(e_movI64, R, REJECT, c=f'pass {k + 1}: REJECT')
    A.jmp(nxt, far=True)
    A.label(P + '_end')


def gen_program():
    A = Asm()
    A.raw([0xf4] * PAD0, ('pad0',), 'barrier between ELF header and tripled region')
    A.label('tripled')
    for k in range(3):
        gen_pass(A, k)
    # ---- the voter: TMRVoter.voterCode with the park (eb fe) replaced by eb 0c
    A.label('vote')
    voter = [0x48, 0x39, 0xd8, 0x74, 0x0d, 0x48, 0x39, 0xc8, 0x74, 0x08, 0x48, 0x39, 0xcb,
             0x75, 0x0b, 0x48, 0x89, 0xd8, 0x48, 0x89, 0xc3, 0x48, 0x89, 0xc1, 0xeb, 0x0c,
             0xbf, 0x03, 0x00, 0x00, 0x00, 0xb8, 0x3c, 0x00, 0x00, 0x00, 0x0f, 0x05]
    A.raw(voter, ('voter',), 'TMRVoter.voterCode, park eb fe -> eb 0c (to guard)')
    # ---- the dual guard
    A.label('guard')
    A.i(e_movI, RDI, 5, c='status 5 unless both guards pass')
    A.i(e_movI64, RDX, NEG_ACC, c='-ACCEPT')
    A.i(e_alu, 'add', RDX, RAX, c='rdx = rax - ACCEPT')
    A.jcc('ne', 'not_acc', c='guard 1')
    A.i(e_movI64, RSI, NEG_ACC, c='-ACCEPT')
    A.i(e_alu, 'add', RSI, RBX, c='rsi = rbx - ACCEPT')
    A.jcc('ne', 'not_acc', c='guard 2')
    A.i(e_movI, RAX, 60, c='exit(0): rax first,')
    A.i(e_lea, RDI, RBX, -(ACCEPT & 0xff), c='then status = (rbx - 0x96) mod 256, zero only for ACCEPT')
    A.i(e_syscall)
    A.jmp('rej_exit')
    A.label('not_acc')
    A.i(e_movI, RDI, 5)
    A.i(e_movI64, RDX, NEG_OVF, c='-OVERFLOW')
    A.i(e_alu, 'add', RDX, RAX)
    A.jcc('ne', 'rej_exit', c='not OVERFLOW')
    A.label('nomem')
    A.i(e_movI, RDI, 6, c='exit(6)')
    A.i(e_movI, RAX, 60)
    A.i(e_movI, RDI, 6)
    A.i(e_syscall)
    A.label('rej_exit')
    A.i(e_movI, RDI, 5, c='exit(5)')
    A.i(e_movI, RAX, 60)
    A.i(e_movI, RDI, 5)
    A.i(e_syscall)
    A.label('core_end')
    A.raw([0xf4] * PAD1, ('pad1',), 'barrier between tripled region and acquisition')
    A.label('start')
    # ---- acquisition: mmap a buffer, read all of stdin, growing with mremap
    A.i(e_zero, RDI, c='mmap(0, 64 KiB, RW, PRIVATE|ANON, -1, 0)')
    A.i(e_movI, RSI, CAP0)
    A.i(e_movI, RDX, 3)
    A.i(e_movI, R10, 0x22)
    A.i(e_movI64, R8, W - 1)
    A.i(e_zero, R9)
    A.i(e_movI, RAX, 9)
    A.i(e_syscall)
    A.i(e_test, RAX, RAX)
    A.jcc('s', 'nomem', c='refused')
    A.i(e_mov, R14, RAX, c='r14 = buffer')
    A.i(e_movI, R12, CAP0, c='r12 = capacity')
    A.i(e_zero, R13, c='r13 = length')
    A.label('rd')
    A.i(e_cmp, R13, R12)
    A.jcc('ne', 'rd_go', c='room left')
    A.i(e_mov, RDI, R14, c='mremap(buf, cap, 2 cap, MAYMOVE)')
    A.i(e_mov, RSI, R12)
    A.i(e_mov, RDX, R12)
    A.i(e_shlI, RDX, 1)
    A.i(e_movI, R10, 1)
    A.i(e_movI, RAX, 25)
    A.i(e_syscall)
    A.i(e_test, RAX, RAX)
    A.jcc('s', 'nomem', c='refused')
    A.i(e_mov, R14, RAX)
    A.i(e_shlI, R12, 1)
    A.label('rd_go')
    A.i(e_zero, RDI, c='read(0, buf + len, cap - len)')
    A.i(e_mov, RSI, R14)
    A.i(e_alu, 'add', RSI, R13)
    A.i(e_mov, RDX, R12)
    A.i(e_alu, 'sub', RDX, R13)
    A.i(e_zero, RAX)
    A.i(e_syscall)
    A.i(e_test, RAX, RAX)
    A.jcc('s', 'nomem', c='read failed')
    A.jcc('e', 'eof')
    A.i(e_alu, 'add', R13, RAX)
    A.jmp('rd')
    # ---- triplication: two more copies, then the three descriptors
    A.label('eof')
    for k, dst in [(1, R15), (2, RBP)]:
        A.i(e_zero, RDI, c=f'mmap copy {k + 1}: len + 1 bytes')
        A.i(e_lea, RSI, R13, 1)
        A.i(e_movI, RDX, 3)
        A.i(e_movI, R10, 0x22)
        A.i(e_zero, R9)
        A.i(e_movI, RAX, 9)
        A.i(e_syscall)
        A.i(e_test, RAX, RAX)
        A.jcc('s', 'nomem')
        A.i(e_mov, dst, RAX)
        top, done = A.fresh('cp_top'), A.fresh('cp_done')
        A.i(e_mov, RSI, R14, c='copy len bytes')
        A.i(e_mov, RDI, dst)
        A.i(e_mov, R11, R13)
        A.label(top)
        A.i(e_test, R11, R11)
        A.jcc('e', done)
        A.i(e_ld8, RDX, RSI, 0)
        A.i(e_st8, RDI, 0, RDX)
        A.i(e_aluI, 'add', RSI, 1)
        A.i(e_aluI, 'add', RDI, 1)
        A.i(e_aluI, 'sub', R11, 1)
        A.jmp(top)
        A.label(done)
    for k, src in enumerate([R14, R15, RBP]):
        A.i(e_movI, RDI, DESC[k], c=f'descriptor {k + 1}')
        A.i(e_st, RDI, 0, src)
        A.i(e_st, RDI, 8, R13)
    A.i(e_movI64, RAX, REJECT, c='results start at REJECT')
    A.i(e_movI64, RBX, REJECT)
    A.i(e_movI64, RCX, REJECT)
    A.jmp('tripled')
    return A


def elf_header(entry, filesz):
    h = []
    h += [0x7f, 0x45, 0x4c, 0x46, 2, 1, 1, 0] + [0] * 8
    h += le(2, 2) + le(0x3e, 2) + le(1, 4) + le(entry, 8) + le(0x40, 8) + le(0, 8)
    h += le(0, 4) + le(64, 2) + le(56, 2) + le(2, 2) + le(0, 2) + le(0, 2) + le(0, 2)
    # PT_LOAD code R+X
    h += le(1, 4) + le(5, 4) + le(0, 8) + le(BASE, 8) + le(BASE, 8) + le(filesz, 8) + le(filesz, 8) + le(0x1000, 8)
    # PT_LOAD bss R+W (descriptors)
    h += le(1, 4) + le(6, 4) + le(0, 8) + le(BSS, 8) + le(BSS, 8) + le(0, 8) + le(BSS_SIZE, 8) + le(0x1000, 8)
    assert len(h) == 0xb0
    return h


HDR_COMMENTS = [
    (0x00, 4, 'e_ident: magic 0x7f E L F'), (0x04, 1, 'ELFCLASS64'), (0x05, 1, 'little endian'),
    (0x06, 1, 'EI_VERSION = 1'), (0x07, 1, 'System V'), (0x08, 8, 'padding'),
    (0x10, 2, 'e_type = ET_EXEC'), (0x12, 2, 'e_machine = EM_X86_64'), (0x14, 4, 'e_version = 1'),
    (0x18, 8, 'e_entry'), (0x20, 8, 'e_phoff = 0x40'), (0x28, 8, 'e_shoff = 0'),
    (0x30, 4, 'e_flags = 0'), (0x34, 2, 'e_ehsize = 64'), (0x36, 2, 'e_phentsize = 56'),
    (0x38, 2, 'e_phnum = 2'), (0x3a, 2, 'e_shentsize = 0'), (0x3c, 2, 'e_shnum = 0'),
    (0x3e, 2, 'e_shstrndx = 0'),
    (0x40, 4, 'PT_LOAD (code)'), (0x44, 4, 'PF_R | PF_X'), (0x48, 8, 'p_offset = 0'),
    (0x50, 8, 'p_vaddr = 0x400000'), (0x58, 8, 'p_paddr'), (0x60, 8, 'p_filesz = whole file'),
    (0x68, 8, 'p_memsz'), (0x70, 8, 'p_align = 0x1000'),
    (0x78, 4, 'PT_LOAD (descriptors)'), (0x7c, 4, 'PF_R | PF_W'), (0x80, 8, 'p_offset = 0'),
    (0x88, 8, 'p_vaddr = 0x600000'), (0x90, 8, 'p_paddr'), (0x98, 8, 'p_filesz = 0'),
    (0xa0, 8, 'p_memsz = 0x3000 (three pages)'), (0xa8, 8, 'p_align = 0x1000'),
]


def ins_str(ins):
    k = ins[0]
    if k == 'movI':
        return f'mov {RN[ins[1]]}, {ins[2]:#x}'
    if k in ('mov', 'mov32'):
        return f'{k} {RN[ins[1]]}, {RN[ins[2]]}'
    if k == 'alu':
        return f'{ins[1]} {RN[ins[2]]}, {RN[ins[3]]}'
    if k == 'aluI':
        return f'{ins[1]} {RN[ins[2]]}, {ins[3]:#x}'
    if k == 'cmpI':
        return f'cmp {RN[ins[1]]}, {ins[2]:#x}'
    if k in ('cmp', 'test'):
        return f'{k} {RN[ins[1]]}, {RN[ins[2]]}'
    if k in ('shlI', 'shrI'):
        return f'{k[:3]} {RN[ins[1]]}, {ins[2]}'
    if k in ('ld', 'ld32', 'ld8', 'lea'):
        d = ins[3] if ins[3] < (1 << 63) else ins[3] - W
        return f'{k} {RN[ins[1]]}, [{RN[ins[2]]}{d:+#x}]'
    if k in ('st', 'st8'):
        d = ins[2] if ins[2] < (1 << 63) else ins[2] - W
        return f'{k} [{RN[ins[1]]}{d:+#x}], {RN[ins[3]]}'
    if k == 'jmp':
        return f'jmp {ins[1]:#x}'
    if k == 'jcc':
        return f'j{CCN[ins[1]]} {ins[2]:#x}'
    return k


def lean_ins(ins):
    k = ins[0]
    if k == 'alu':
        return f'.alu .{ins[1]} {ins[2]} {ins[3]}'
    if k == 'aluI':
        return f'.aluI .{ins[1]} {ins[2]} {ins[3]}'
    if k == 'syscall':
        return '.syscall'
    return '.' + k + ''.join(f' {a}' for a in ins[1:])



# ------------------------------------------------------- Lean templates
# Each macro is stated in Lean as a function of its address and parameters;
# the template is read off two instances of the macro, and every instance is
# checked against it here (the Lean build checks it again, through the table).

FIELDS = {'movI': 'ri', 'mov': 'rr', 'mov32': 'rr', 'alu': 'orr', 'aluI': 'ori', 'cmp': 'rr',
          'cmpI': 'ri', 'test': 'rr', 'shlI': 'ri', 'shrI': 'ri', 'ld': 'rri', 'st': 'rir',
          'ld32': 'rri', 'ld8': 'rri', 'st8': 'rir', 'lea': 'rri', 'jmp': 'a', 'jcc': 'ia',
          'syscall': ''}
SENT = {ACCEPT: 'ACCEPT', REJECT: 'REJECT', OVERFLOW: 'OVERFLOW'}


def render(ins, L, size, aparams, rparams, iparams=None, Lname='L'):
    """Lean term of `ins` inside an instance at L of the given size. aparams:
    address -> name; rparams: (field index, reg) -> name; iparams: imm -> name."""
    k = ins[0]
    out = []
    for j, (f, v) in enumerate(zip(FIELDS[k], ins[1:])):
        if f == 'o':
            out.append('.' + v)
        elif f == 'r':
            out.append(rparams.get((j, v), str(v)))
        elif f == 'i':
            if iparams and v in iparams:
                out.append(iparams[v])
            else:
                out.append(str(v))
        elif f == 'a':
            if L <= v <= L + size and v not in aparams:
                out.append(f'({Lname} + {v - L})' if v != L else Lname)
            elif v in aparams:
                out.append(aparams[v])
            else:
                raise ValueError(f'address {v:#x} outside instance at {L:#x}')
    return '.' + k + ''.join(' ' + x for x in out)


def instances(A, out):
    """name -> list of (L, size, params, [(addr, ins, len)]) for every macro
    instance, in order."""
    by_idx = {}
    for (a, bs, ins, c, idx) in out:
        by_idx.setdefault(idx, []).append((a, ins, len(bs)))
    res = {}
    for (name, params, st, en) in A.macros:
        L = A.idx_addr[st]
        size = A.idx_addr[en] - L
        lst = [e for idx in range(st, en) for e in by_idx.get(idx, [])]
        res.setdefault(name, []).append((L, size, params, lst, st, en))
    return res


TEMPLATE_PARAMS = {
    'skipcode': (['p'], ['rej']), 'skipwords': ([], ['rej']), 'skipdata': ([], ['rej']),
    'chk': (['r'], ['ovf']), 'lookup': (['t'], []), 'eff': ([], ['tgt'])}


def emit_templates(A, out, labels):
    insts = instances(A, out)
    lines = []
    for name in ['skipcode', 'skipdata', 'skipwords', 'chk', 'lookup', 'eff']:
        regp, addrp = TEMPLATE_PARAMS[name]
        ilist = insts[name]
        ref = ilist[0]
        # nested macros (skipcode inside skipwords) are template calls
        L, size, params, lst, st, en = ref
        inner = [m for m in A.macros if st <= m[2] and m[3] <= en and (m[2], m[3]) != (st, en)]
        sig = ' '.join(['(L : Nat)'] + [f'({p} : Nat)' for p in regp] + [f'({p} : Nat)' for p in addrp])
        terms = []

        def body(inst, pnames):
            L, size, params, lst, st, en = inst
            aparams = {labels[params[p]]: p for p in addrp}
            ts = []
            j = 0
            covered = set()
            for m in A.macros:
                if st <= m[2] and m[3] <= en and (m[2], m[3]) != (st, en):
                    mL = A.idx_addr[m[2]]
                    mparams = m[1]
                    rp, ap = TEMPLATE_PARAMS[m[0]]
                    args = []
                    for p in rp:
                        v = mparams[p]
                        args.append(next((q for q in regp if params[q] == v), str(v)))
                    for p in ap:
                        v = labels[mparams[p]]
                        args.append(aparams.get(v, f'(L + {v - L})'))
                    covered |= set(range(m[2], m[3]))
                    ts.append(('call', m[2], f'{m[0]}T (L + {mL - L}) ' + ' '.join(args)))
            by_idx = {}
            for (a, bs, ins, c, idx) in out:
                if st <= idx < en and idx not in covered:
                    by_idx.setdefault(idx, []).append((a, ins, len(bs)))
            for idx, es in by_idx.items():
                for (a, ins, ln) in es:
                    rparams = {}
                    for jj, (f, v) in enumerate(zip(FIELDS[ins[0]], ins[1:])):
                        if f == 'r':
                            for q in regp:
                                if params[q] == v:
                                    rparams[(jj, v)] = q
                    ts.append(('ins', idx, f'(L + {a - L}, {render(ins, L, size, aparams, rparams)}, {ln})'))
            ts.sort(key=lambda t: t[1])
            # group consecutive raw instructions into lists
            groups, cur = [], []
            for kind, _, t in ts:
                if kind == 'ins':
                    cur.append(t)
                else:
                    if cur:
                        groups.append('[' + ', '.join(cur) + ']')
                        cur = []
                    groups.append(t)
            if cur:
                groups.append('[' + ', '.join(cur) + ']')
            return ' ++\n  '.join(groups)
        b0 = body(ref, None)
        # every instance must render to the same template text
        for inst in ilist[1:]:
            bi = body(inst, None)
            assert bi == b0, (name, inst[0])
        lines.append(f'/-- The {name} macro at L ({len(ilist)} instances). -/')
        lines.append(f'def {name}T {sig} : List (Nat × Ins × Nat) :=\n  {b0}\n')
    # the pass, with its macros as template calls; B its address, D its
    # descriptor, R its result register
    B = labels['p1']
    size = labels['p1_end'] - B
    st_idx = next(i for i, it in enumerate(A.items) if it[0] == 'L' and it[1] == 'p1')
    en_idx = next(i for i, it in enumerate(A.items) if it[0] == 'L' and it[1] == 'p1_end')
    top = [m for m in A.macros if st_idx <= m[2] and m[3] <= en_idx
           and not any(n is not m and n[2] <= m[2] and m[3] <= n[3] and (n[2], n[3]) != (m[2], m[3]) for n in A.macros)]
    covered = set()
    ts = []
    for m in top:
        mL = A.idx_addr[m[2]]
        rp, ap = TEMPLATE_PARAMS[m[0]]
        args = [str(m[1][p]) for p in rp] + [f'(B + {labels[m[1][p]] - B})' for p in ap]
        covered |= set(range(m[2], m[3]))
        ts.append(('call', m[2], f'{m[0]}T (B + {mL - B}) ' + ' '.join(args)))
    for (a, bs, ins, c, idx) in out:
        if st_idx <= idx < en_idx and idx not in covered:
            rparams = {}
            iparams = {DESC[0]: 'D'}
            if ins[0] == 'movI' and ins[2] in SENT:
                rparams[(0, ins[1])] = 'R'
                iparams[ins[2]] = SENT[ins[2]]
            ts.append(('ins', idx, f'(B + {a - B}, {render(ins, B, size, {}, rparams, iparams, "B")}, {len(bs)})'))
    ts.sort(key=lambda t: t[1])
    groups, cur = [], []
    for kind, _, t in ts:
        if kind == 'ins':
            cur.append(t)
        else:
            if cur:
                groups.append('[' + ', '.join(cur) + ']')
                cur = []
            groups.append(t)
    if cur:
        groups.append('[' + ', '.join(cur) + ']')
    lines.append(f'/-- Pass code at B ({size} bytes), descriptor D, result register R. -/')
    lines.append('def passCode (B D R : Nat) : List (Nat × Ins × Nat) :=\n  ' + ' ++\n  '.join(groups) + '\n')
    lines.append(f'def passSize : Nat := {size}\n')
    return '\n'.join(lines) + '\n'


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    A = gen_program()
    hdr_len = 0xb0
    out, labels = A.assemble(BASE + hdr_len)
    code = []
    for (a, bs, ins, c, _) in out:
        assert a == BASE + hdr_len + len(code)
        code += bs
    filesz = hdr_len + len(code)
    image = elf_header(labels['start'], filesz) + code
    # every instruction decodes to what was meant
    b_at = lambda a: (lambda i: image[a - BASE + i] if 0 <= a - BASE + i < len(image) else None)
    table = []
    for (a, bs, ins, c, _) in out:
        if ins in (('pad0',), ('pad1',)):
            continue
        if ins == ('voter',):
            off = 0
            while off < len(bs):
                d = decode(b_at(a + off), a + off)
                assert d is not None
                table.append((a + off, d[0], d[1]))
                off += d[1]
            continue
        d = decode(b_at(a), a)
        assert d == (ins, len(bs)), (hex(a), d, ins, bs)
        table.append((a, ins, len(bs)))
    # no 0f 85 anywhere, and 0f 05 only at the syscalls
    sys_at = {a for (a, ins, _) in table if ins == ('syscall',)}
    for j in range(len(image) - 1):
        if image[j] == 0x0f and image[j + 1] == 0x05:
            assert BASE + j in sys_at, hex(BASE + j)
        assert not (image[j] == 0x0f and image[j + 1] == 0x85), hex(BASE + j)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'spr1_check'), 'wb') as f:
        f.write(bytes(image))
    lab_at = {}
    for k_, v in labels.items():
        lab_at.setdefault(v, []).append(k_)
    with open(os.path.join(outdir, 'spr1_check.hex'), 'w') as f:
        f.write(LISTING_HEAD.format(size=filesz, entry=labels['start'], tripled=labels['tripled'],
                                    vote=labels['vote'], guard=labels['guard'], overflow=OVERFLOW))
        f.write('\n# ---------------------------------------------------------------- ELF header\n')
        for (o, n, c) in HDR_COMMENTS:
            if o == 0x40:
                f.write('\n# ----------------------------------------------------------- program headers\n')
            f.write(f'{" ".join(f"{x:02x}" for x in image[o:o + n]):<32}# {o:03x}  {c}\n')
        f.write('\n# --------------------------------------------------------------------- code\n')
        f.write('# Offsets on the left are file offsets; address = 0x400000 + offset.\n')
        for (a, bs, ins, c, _) in out:
            for l in lab_at.get(a, []):
                if l != 'core_end':
                    f.write(f'\n# {l}:\n')
            if ins in (('pad0',), ('pad1',)):
                f.write(f'# {ins[0]} ({len(bs)} bytes of 0xf4)\n')
                for j in range(0, len(bs), 16):
                    seg = bs[j:j + 16]
                    f.write(f'{" ".join(f"{x:02x}" for x in seg)}\n')
                continue
            if ins == ('voter',):
                f.write(f'# {c}\n')
                off = 0
                while off < len(bs):
                    d = decode(b_at(a + off), a + off)
                    seg = bs[off:off + d[1]]
                    f.write(f'{" ".join(f"{x:02x}" for x in seg):<32}# {a - BASE + off:03x}  {ins_str(d[0])}\n')
                    off += d[1]
                continue
            txt = ins_str(ins)
            hexs = " ".join(f"{x:02x}" for x in bs)
            f.write(f'{hexs:<32}# {a - BASE:03x}  {txt}{("   " + c) if c else ""}\n')
    with open(os.path.join(outdir, 'symbols.txt'), 'w') as f:
        for k_, v in sorted(labels.items(), key=lambda kv: kv[1]):
            f.write(f'{v:#x} {k_}\n')
    core_start = labels['tripled'] - BASE
    core_end_off = labels['core_end'] - BASE
    init_start = labels['start'] - BASE
    hdr_bytes = image[:hdr_len]
    core_bytes = image[core_start:core_end_off]
    init_bytes = image[init_start:]
    core_table = [e for e in table if e[0] < labels['core_end']]
    init_table = [e for e in table if e[0] >= labels['start']]
    with open(os.path.join(outdir, 'SporosCheckBinData.lean'), 'w') as f:
        f.write(LEAN_HEAD)
        f.write(f'def hdrBytes : List Nat := [' + ', '.join(str(x) for x in hdr_bytes) + ']\n')
        c_chunks = [core_bytes[i:i + 256] for i in range(0, len(core_bytes), 256)]
        for j, ch in enumerate(c_chunks):
            f.write(f'def core_{j} : List Nat := [' + ', '.join(str(x) for x in ch) + ']\n')
        f.write('def coreBytes : List Nat :=\n  ' + ' ++\n  '.join(f'core_{j}' for j in range(len(c_chunks))) + '\n')
        i_chunks = [init_bytes[i:i + 256] for i in range(0, len(init_bytes), 256)]
        for j, ch in enumerate(i_chunks):
            f.write(f'def init_{j} : List Nat := [' + ', '.join(str(x) for x in ch) + ']\n')
        f.write('def initBytes : List Nat :=\n  ' + ' ++\n  '.join(f'init_{j}' for j in range(len(i_chunks))) + '\n\n')
        f.write(f'/-- bootstrap/model/spr1_check -/\ndef spr1Bytes : List Nat :=\n'
                f'  hdrBytes ++ List.replicate {PAD0} 0xf4 ++ coreBytes ++ List.replicate {PAD1} 0xf4 ++ initBytes\n\n')
        def emit_tbl(name, tbl):
            tch = [tbl[i:i + 64] for i in range(0, len(tbl), 64)]
            for j, ch in enumerate(tch):
                f.write(f'def {name}_{j} : List (Nat × Ins × Nat) := [\n')
                f.write(',\n'.join(f'  ({a}, {lean_ins(i)}, {l})' for (a, i, l) in ch))
                f.write(']\n')
            f.write(f'def {name} : List (Nat × Ins × Nat) :=\n  ' + ' ++\n  '.join(f'{name}_{j}' for j in range(len(tch))) + '\n\n')
        emit_tbl('coreTable', core_table)
        emit_tbl('initTable', init_table)
        f.write('/-- Every instruction of the code, at its address, with its length. -/\n'
                'def spr1Table : List (Nat × Ins × Nat) := coreTable ++ initTable\n\n')
        f.write(f'def ACCEPT : Nat := {ACCEPT:#x}\ndef REJECT : Nat := {REJECT:#x}\ndef OVERFLOW : Nat := {OVERFLOW:#x}\n\n')
        f.write(emit_templates(A, out, labels))
        f.write('/-- Addresses named in the listing. -/\n')
        for k_, v in sorted(labels.items(), key=lambda kv: kv[1]):
            if '_' not in k_ or k_.startswith('p1_') or k_.startswith('p2_') or k_.startswith('p3_') or k_ in ('not_acc', 'rej_exit', 'rd_go', 'core_end'):
                f.write(f'def L_{k_} : Nat := {v:#x}\n')
        f.write('\nend SporosCheckBin\n')
    print(f'size {filesz} bytes, {len(table)} instructions, entry {labels["start"]:#x}, '
          f'tripled {labels["tripled"]:#x}, vote {labels["vote"]:#x}, guard {labels["guard"]:#x}')


LEAN_HEAD = """/-
Generated by bootstrap/model/spr1_check_gen.py from the listing
bootstrap/model/spr1_check.hex; do not edit. `spr1Bytes` is the file
bootstrap/model/spr1_check, byte for byte, and `spr1Table` the instruction at
every address of its code (checked against `X86Dec.decode` in
Bootstrap/SporosCheckBin.lean).
-/
import Bootstrap.X86Int

namespace SporosCheckBin

open X86Int

"""

LISTING_HEAD = """# spr1_check: the checker of Sporos bytecode, stage 1 of bootstrap root A.
#
# What it does: reads a Sporos bytecode file (formal/Sporos/Bytecode.lean) from
# standard input and checks it as Sporos.decode and Sporos.checkProg
# (formal/Sporos/Check.lean) do, three times, on three copies of the input.
#
# Exit status: 0 when the file decodes and passes the checker; 5 when it is
# malformed or ill-typed; 6 when memory is refused or a stack height reaches
# 2^64; 3 when the three passes all disagree. Every non-zero status is at
# Hamming distance 2 or more from 0.
#
# Structure (addresses are 0x400000 + file offset):
#   start   ({entry:#x})  mmap a 64 KiB buffer, read all of stdin, doubling it
#                        with mremap; triplicate the input into two more
#                        mappings; descriptors (base, N) at 0x600000,
#                        0x601000, 0x602000; rax = rbx = rcx = REJECT.
#   tripled ({tripled:#x})  pass 1 (result in rax), pass 2 (rbx), pass 3 (rcx).
#                        Each pass reads only its own copy and descriptor, makes
#                        no store, call, push or syscall, and writes only its
#                        result register: ACCEPT 0x5acce970a55a3c96, REJECT
#                        0xdeadbeef11110001 or OVERFLOW {overflow:#x}.
#   vote    ({vote:#x})  the 38-byte voter of Bootstrap/TMRVoter.lean, with its
#                        park (eb fe) replaced by eb 0c, a jump to the guard.
#   guard   ({guard:#x})  exit(0) only when rax - ACCEPT and rbx - ACCEPT are both
#                        zero, tested by two conditional branches; the exit
#                        status is computed from the two differences.
#
# The bytes 0f 85 (jne rel32, one bit from syscall 0f 05) never occur, and
# 0f 05 occurs only at the syscalls.
#
# Total {size} bytes.
"""

if __name__ == '__main__':
    main()
