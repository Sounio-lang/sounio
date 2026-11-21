#!/usr/bin/env python3
"""Host interpreter for the PIREUS lowering PTX, used only by the tests.

It runs the exact text materialize_ptx.sio emits: grid = vector count and 16
threads per block in lockstep, as gpu_probe.py launches it (or, for a
`.maxntid 32/64/128` body, grid = ceil(n / (threads/16)) with n in the fourth
parameter pn, as r740_wide_launch.py launches it), with binary64
round-to-nearest on the host (Python floats are IEEE binary64 and never fuse). It knows only the instructions that emitter
writes and refuses anything else, so a new opcode in the emitter fails the test
instead of being skipped. It is not a GPU: it shows the PTX text computes the
reference, and the probe on real hardware is still the material evidence.
"""
import re
import struct

GLOBAL = {"pa": 1, "pb": 2, "pc": 3}
THREADS = (16, 32, 64, 128)
CONST_BASE = 8
SHARED_BASE = 16
M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF


def f(bits):
    return struct.unpack("<d", struct.pack("<Q", bits & M64))[0]


def b(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


class Kernel:
    def __init__(self, text):
        self.consts = {}
        self.shared = {}
        self.code = []
        self.labels = {}
        self.threads = None
        self.params = None
        body = False
        for raw in text.splitlines():
            line = raw.split("//")[0].strip()
            if not line:
                continue
            m = re.match(r"\.const \.align 4 \.u32 (\w+)\[16\] = \{([0-9,]+)\};$", line)
            if m:
                self.consts[m.group(1)] = (CONST_BASE + len(self.consts), [int(x) for x in m.group(2).split(",")])
                continue
            m = re.match(r"\.visible \.entry pireus_candidate\((.*)\)$", line)
            if m:
                self.params = m.group(1)
                continue
            m = re.match(r"\.maxntid (\d+),1,1$", line)
            if m:
                self.threads = int(m.group(1))
                continue
            m = re.match(r"\.shared \.align 16 \.f64 (\w+)\[(16|32|64|128)\];$", line)
            if m:
                self.shared[m.group(1)] = (SHARED_BASE + len(self.shared), int(m.group(2)))
                continue
            if line == "{":
                body = True
                continue
            if not body or line == "}" or line.startswith(".reg"):
                continue
            if line.endswith(":"):
                self.labels[line[:-1]] = len(self.code)
                continue
            assert line.endswith(";"), line
            line = line[:-1]
            guard = None
            if line.startswith("@"):
                g, line = line.split(None, 1)
                guard = (g[1] == "!", g.lstrip("@!"))
            op, _, rest = line.partition(" ")
            args = [a.strip() for a in re.split(r",(?![^{]*\})", rest)] if rest else []
            self.code.append((guard, op, args))
        self.const_tables = {base: words for base, words in self.consts.values()}
        # The two launch contracts: gpu_probe.py's (16 threads, three pointers) and the
        # r740 wide one (32/64/128 threads, three pointers and the vector count).
        narrow = ".param .u64 pa, .param .u64 pb, .param .u64 pc"
        assert self.threads in THREADS, self.threads
        assert self.params == (narrow if self.threads == 16 else narrow + ", .param .u32 pn"), self.params

    def run(self, a, b_in, vectors):
        # A finite sentinel, so an output the kernel never stores cannot pass as a result.
        c = [0x5A5A5A5A5A5A5A5A] * (16 * vectors)
        mem = {1: a, 2: b_in, 3: c}
        threads = self.threads
        grid = vectors if threads == 16 else -(-vectors // (threads // 16))
        for block in range(grid):
            # Shared memory is per block and starts as a finite sentinel, like the output.
            for base, size in self.shared.values():
                mem[base] = [0x5A5A5A5A5A5A5A5A] * size
            regs = [dict() for _ in range(threads)]
            for t in range(threads):
                regs[t].update({"%tid.x": t, "%ctaid.x": block, "%nctaid.x": grid, "pn": vectors})
            pc = 0
            while True:
                guard, op, args = self.code[pc]
                pc += 1
                if op == "ret":
                    break
                if op == "bra":
                    taken = {self._pred(r, guard) for r in regs}
                    assert len(taken) == 1, "divergent branch"
                    if taken.pop():
                        pc = self.labels[args[0]]
                    continue
                if op == "bar.sync":
                    # Lockstep execution already completes every thread's earlier
                    # instructions before this one; the barrier must be unguarded.
                    assert guard is None and args == ["0"], args
                    continue
                if op == "shfl.sync.idx.b32":
                    # 16 threads: one half warp, clamp 15, mask 0xffff. Wider blocks: full
                    # warps, 16-lane segments (c = 0x101f), mask 0xffffffff.
                    if threads == 16:
                        assert args[3] == "15" and args[4] == "65535", args
                    else:
                        assert args[3] == "4127" and args[4] == "-1", args
                    assert guard is None
                    control = int(args[3])
                    clamp, seg = control & 31, (control >> 8) & 31
                    src = [r[args[1]] for r in regs]
                    out = []
                    for t, r in enumerate(regs):
                        warp, lane = t - t % 32, t % 32
                        low = lane & seg
                        high = low | (clamp & ~seg)
                        j = low | (self._val(r, args[2]) & ~seg & 31)
                        if j > high:
                            j = lane
                        assert warp + j < len(regs), "shuffle from a missing thread"
                        out.append(src[warp + j])
                    for r, value in zip(regs, out):
                        r[args[0]] = value
                    continue
                for r in regs:
                    if self._pred(r, guard):
                        self._step(r, op, args, mem)
        return c

    def _pred(self, r, guard):
        if guard is None:
            return True
        neg, name = guard
        return bool(r[name]) != neg

    def _val(self, r, tok):
        if tok.startswith("%"):
            return r[tok]
        if tok.startswith("0d"):
            return int(tok[2:], 16)
        if tok in self.consts:
            return self.consts[tok][0] << 40
        if tok in self.shared:
            return self.shared[tok][0] << 40
        return int(tok)

    def _addr(self, r, tok):
        return r[tok.strip("[]")]

    def _step(self, r, op, a, mem):
        v = lambda i: self._val(r, a[i])
        if op == "ld.param.u64":
            r[a[0]] = GLOBAL[a[1].strip("[]")] << 40
        elif op == "ld.param.u32":
            assert a[1] == "[pn]", a
            r[a[0]] = r["pn"]
        elif op in ("mov.u32", "mov.u64", "mov.f64"):
            r[a[0]] = v(1)
        elif op == "setp.gt.u32":
            r[a[0]] = v(1) > v(2)
        elif op == "setp.ge.u32":
            r[a[0]] = v(1) >= v(2)
        elif op == "min.u32":
            r[a[0]] = min(v(1), v(2))
        elif op == "shl.b32":
            r[a[0]] = (v(1) << v(2)) & M32 if v(2) < 32 else 0
        elif op == "setp.lt.u32":
            r[a[0]] = v(1) < v(2)
        elif op == "setp.ne.u32":
            r[a[0]] = v(1) != v(2)
        elif op == "mad.lo.u32":
            r[a[0]] = (v(1) * v(2) + v(3)) & M32
        elif op == "mul.lo.u32":
            r[a[0]] = (v(1) * v(2)) & M32
        elif op == "add.u32":
            r[a[0]] = (v(1) + v(2)) & M32
        elif op == "sub.u32":
            r[a[0]] = (v(1) - v(2)) & M32
        elif op == "and.b32":
            r[a[0]] = v(1) & v(2) & M32
        elif op == "xor.b32":
            r[a[0]] = (v(1) ^ v(2)) & M32
        elif op == "shr.u32":
            r[a[0]] = (v(1) & M32) >> (v(2) & 31) if v(2) < 32 else 0
        elif op == "mul.wide.u32":
            r[a[0]] = (v(1) * v(2)) & M64
        elif op == "add.u64":
            r[a[0]] = (v(1) + v(2)) & M64
        elif op == "ld.global.f64":
            addr = self._addr(r, a[1])
            assert addr % 8 == 0
            r[a[0]] = mem[addr >> 40][(addr & ((1 << 40) - 1)) // 8]
        elif op == "st.global.f64":
            addr = self._addr(r, a[0])
            assert addr % 8 == 0
            mem[addr >> 40][(addr & ((1 << 40) - 1)) // 8] = r[a[1]]
        elif op == "ld.global.v2.f64":
            addr = self._addr(r, a[1])
            assert addr % 16 == 0, "misaligned v2 load"
            lo, hi = [x.strip() for x in a[0].strip("{}").split(",")]
            buf, at = mem[addr >> 40], (addr & ((1 << 40) - 1)) // 8
            r[lo], r[hi] = buf[at], buf[at + 1]
        elif op in ("ld.shared.f64", "st.shared.f64"):
            addr = self._addr(r, a[1] if op[0] == "l" else a[0])
            assert addr % 8 == 0
            buf, at = mem[addr >> 40], (addr & ((1 << 40) - 1)) // 8
            assert at < len(buf), "shared index out of range"
            if op[0] == "l":
                r[a[0]] = buf[at]
            else:
                buf[at] = r[a[1]]
        elif op == "ld.const.u32":
            addr = self._addr(r, a[1])
            assert addr % 4 == 0
            r[a[0]] = self.const_tables[addr >> 40][(addr & ((1 << 40) - 1)) // 4]
        elif op == "mov.b64" and a[0].startswith("{"):
            lo, hi = [x.strip() for x in a[0].strip("{}").split(",")]
            r[lo], r[hi] = r[a[1]] & M32, r[a[1]] >> 32
        elif op == "mov.b64" and a[1].startswith("{"):
            lo, hi = [x.strip() for x in a[1].strip("{}").split(",")]
            r[a[0]] = r[lo] | (r[hi] << 32)
        elif op == "selp.f64":
            r[a[0]] = v(1) if r[a[3]] else v(2)
        elif op == "mul.rn.f64":
            r[a[0]] = b(f(v(1)) * f(v(2)))
        elif op == "add.rn.f64":
            r[a[0]] = b(f(v(1)) + f(v(2)))
        elif op == "sub.rn.f64":
            r[a[0]] = b(f(v(1)) - f(v(2)))
        else:
            raise AssertionError("ptx_sim does not know " + op)


def run_rows(text, rows, layout):
    """Lay the fixture rows out like the probe does and return per-row output bits."""
    n = len(rows)
    a = [0] * (16 * n)
    bb = [0] * (16 * n)
    for v, row in enumerate(rows):
        for i in range(16):
            at = v * 16 + i if layout == 0 else i * n + v
            a[at] = row["a_bits"][i] & M64
            bb[at] = row["b_bits"][i] & M64
    c = Kernel(text).run(a, bb, n)
    out = []
    for v in range(n):
        words = []
        for k in range(16):
            at = v * 16 + k if layout == 0 else k * n + v
            x = c[at]
            words.append(x - (1 << 64) if x >> 63 else x)
        out.append(words)
    return out
