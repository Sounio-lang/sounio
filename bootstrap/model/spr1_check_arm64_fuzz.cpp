// spr1_check_arm64_fuzz: inputs, runner (Linux native / Darwin MAP_JIT), and
// Linux aarch64 ptrace single-fault injector for the gate of spr1_check_arm64
// (bootstrap/model/spr1_check_arm64_gate.sh).
//
// Subcommands:
//   gen N SEED DIR
//   examples DIR
//   vectors DIR
//   run BIN LISTFILE [TIMEOUT_MS]
//   exec1 BIN INPUT [TIMEOUT_MS]
//   syms LISTING BIN
//   patch IN OUT ADDR HEX [ADDR HEX]...
//   inject BIN INPUT START_ADDR [options]   (Linux aarch64 ptrace)

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <map>
#include <new>
#include <random>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(__APPLE__) && defined(__aarch64__)
#include <csetjmp>
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/ucontext.h>
#define SPR1_DARWIN_JIT 1
#endif

#if defined(__linux__) && defined(__aarch64__)
#include <asm/ptrace.h>
#include <atomic>
#include <elf.h>
#include <sched.h>
#include <stddef.h>
#include <sys/mman.h>
#include <sys/personality.h>
#include <sys/ptrace.h>
#include <sys/uio.h>
#include <sys/user.h>
#define SPR1_INJECT 1
#endif

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Bytes = std::vector<u8>;

namespace {

[[noreturn]] void die(const char* fmt, const char* a = "") {
  std::fprintf(stderr, "spr1_check_arm64_fuzz: ");
  std::fprintf(stderr, fmt, a);
  std::fprintf(stderr, "\n");
  std::exit(2);
}

// ------------------------------------------------------------------------
// The program model and its encoding (formal/Sporos/Bytecode.lean)

constexpr int kNPrim = 57;
const int effIn[kNPrim] = {1,1,2,2,3,2,2, 2,2,2,2,2,2,2,1, 2,2,2,1,2,2,2, 2,2,2,2,2,2,
                           1,2,1,2,1,2,1,2, 3,3,3, 1,0,1, 4,2, 0,3,2, 0,1,1,1, 0,0,1, 1,1,2};
const int effOut[kNPrim] = {2,0,2,3,3,1,3, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1, 1,1,1,1,1,1,
                            1,0,1,0,1,0,1,0, 0,0,1, 1,1,0, 1,1, 1,0,2, 1,0,0,0, 1,1,1, 2,2,1};
enum : u8 {
  pDup = 0, pDrop = 1, pAdd = 7, pSub = 8, pDivu = 10, pModu = 11, pEq = 22, pLtu = 24,
  pLd8 = 28, pGetb = 47, pPutb = 48, pExit = 50,
  oLit = 0x80, oStr, oPick, oTor, oFromr, oRfetch, oLocget, oLocset, oJmp, oJz, oJnz, oCall, oRet
};

int immLen(u8 op) {
  if (op == oLit) return 8;
  if (op == oStr || op == oPick || (op >= oLocget && op <= oCall)) return 4;
  return 0;
}

struct In { u8 op; u64 imm; };
struct Ann { u32 t, d, a; };
struct Word { u32 ins = 0, outs = 0, locals = 0; std::vector<In> code; std::vector<Ann> ann; };
struct Prog { u32 main = 0; std::vector<Word> words; std::vector<Bytes> data; };

In P(u8 p) { return {p, 0}; }
In I(u8 op, u64 imm) { return {op, imm}; }

struct Enc {
  Bytes b;
  std::vector<size_t> u32s;
  std::vector<size_t> counts;
  std::vector<size_t> ops;
  size_t mainOff = 4;
  std::vector<size_t> insOffs;
};

void le(Bytes& b, u64 v, int k) { for (int i = 0; i < k; ++i) b.push_back(static_cast<u8>(v >> (8 * i))); }

Enc encode(const Prog& p) {
  Enc e;
  auto u32f = [&](u64 v, bool count) {
    e.u32s.push_back(e.b.size());
    if (count) e.counts.push_back(e.b.size());
    le(e.b, v, 4);
  };
  e.b = {0x53, 0x50, 0x52, 0x31};
  u32f(p.main, false);
  u32f(p.words.size(), true);
  u32f(p.data.size(), true);
  for (const auto& d : p.data) { u32f(d.size(), true); e.b.insert(e.b.end(), d.begin(), d.end()); }
  for (const auto& w : p.words) {
    e.insOffs.push_back(e.b.size());
    u32f(w.ins, false); u32f(w.outs, false); u32f(w.locals, false);
    u32f(w.code.size(), true); u32f(w.ann.size(), true);
    for (const auto& x : w.code) {
      e.ops.push_back(e.b.size());
      e.b.push_back(x.op);
      int k = immLen(x.op);
      if (k == 4) u32f(x.imm, false);
      else if (k) le(e.b, x.imm, k);
    }
    for (const auto& a : w.ann) { u32f(a.t, false); u32f(a.d, false); u32f(a.a, false); }
  }
  return e;
}

constexpr u64 kM1 = ~u64{0};

Prog exEcho() {
  Prog p;
  p.words.push_back({0, 0, 0,
    {P(pGetb), P(pDup), I(oLit, kM1), P(pEq), I(oJnz, 7), P(pPutb), I(oJmp, 0),
     P(pDrop), I(oLit, 0), P(pExit)},
    {{0, 0, 0}, {7, 1, 0}}});
  return p;
}

Prog exHello() {
  Prog p;
  p.words.push_back({0, 0, 2,
    {I(oStr, 0), I(oLocset, 1), I(oLocset, 0),
     I(oLocget, 1), I(oJz, 17),
     I(oLocget, 0), P(pLd8), P(pPutb),
     I(oLocget, 0), I(oLit, 1), P(pAdd), I(oLocset, 0),
     I(oLocget, 1), I(oLit, 1), P(pSub), I(oLocset, 1),
     I(oJmp, 3),
     I(oLit, 0), P(pExit)},
    {{3, 0, 0}, {17, 0, 0}}});
  const char* s = "Hello from Sporos!\n";
  p.data.push_back(Bytes(s, s + std::strlen(s)));
  return p;
}

Prog exUpper() {
  Prog p;
  p.words.push_back({0, 0, 0,
    {P(pGetb), P(pDup), I(oLit, kM1), P(pEq), I(oJnz, 17),
     P(pDup), I(oLit, 97), P(pLtu), I(oJnz, 15),
     P(pDup), I(oLit, 123), P(pLtu), I(oJz, 15),
     I(oLit, 32), P(pSub),
     P(pPutb), I(oJmp, 0),
     P(pDrop), I(oLit, 0), P(pExit)},
    {{0, 0, 0}, {15, 1, 0}, {17, 1, 0}}});
  return p;
}

Prog exFib() {
  Prog p;
  p.words.push_back({0, 0, 3,
    {I(oLit, 0), I(oLocset, 0), I(oLit, 1), I(oLocset, 1), I(oLit, 20), I(oLocset, 2),
     I(oLocget, 2), I(oJz, 23),
     I(oLocget, 0), I(oCall, 1), I(oLit, 10), P(pPutb),
     I(oLocget, 0), I(oLocget, 1), P(pAdd), I(oLocget, 1), I(oLocset, 0), I(oLocset, 1),
     I(oLocget, 2), I(oLit, 1), P(pSub), I(oLocset, 2),
     I(oJmp, 6),
     I(oLit, 0), P(pExit)},
    {{6, 0, 0}, {23, 0, 0}}});
  p.words.push_back({1, 0, 0,
    {P(pDup), I(oLit, 10), P(pLtu), I(oJnz, 8),
     P(pDup), I(oLit, 10), P(pDivu), I(oCall, 1),
     I(oLit, 10), P(pModu), I(oLit, 48), P(pAdd), P(pPutb), I(oRet, 0)},
    {{8, 1, 0}}});
  return p;
}

std::vector<std::pair<const char*, Prog>> examples() {
  return {{"echo", exEcho()}, {"hello", exHello()}, {"upper", exUpper()}, {"fib", exFib()}};
}

struct H { u64 d, a; bool operator==(const H&) const = default; };

struct Rng {
  std::mt19937_64 g;
  explicit Rng(u64 s) : g(s) {}
  u64 operator()(u64 n) { return n ? g() % n : 0; }
  bool chance(int pct) { return static_cast<int>(g() % 100) < pct; }
  u64 raw() { return g(); }
};

const u64 litEdges[] = {0, 1, 2, 10, 48, 255, 256, 0xffffffffull, 0x100000000ull,
                        0x7fffffffffffffffull, 0x8000000000000000ull, kM1};
const u32 u32Edges[] = {0, 1, 2, 3, 7, 0xff, 0x100, 0xffff, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff};

struct Sig { u32 ins, outs, locals; };

Word genWord(Rng& r, const Sig& s, const std::vector<Sig>& sigs, u32 ndata) {
  Word w;
  w.ins = s.ins; w.outs = s.outs; w.locals = s.locals;
  std::vector<H> Hs;
  std::map<u32, H> annAt;
  struct Pend { size_t i; H req; };
  std::vector<Pend> pend;
  auto annotate = [&](u32 t, H h) {
    if (!annAt.count(t)) { annAt[t] = h; w.ann.push_back({t, static_cast<u32>(h.d), static_cast<u32>(h.a)}); }
    else if (r.chance(10)) w.ann.push_back({t, static_cast<u32>(h.d), static_cast<u32>(h.a)});
  };
  H h{s.ins, 0};
  bool live = true;
  size_t target = 1 + r(r.chance(20) ? 60 : 24);
  while (w.code.size() < target) {
    u32 i = static_cast<u32>(w.code.size());
    if (!live) {
      if (r.chance(50)) { h = {0, 0}; }
      else { h = {r(5), r(3)}; annotate(i, h); }
      live = true;
    } else if (r.chance(8)) {
      annotate(i, h);
    }
    Hs.push_back(h);
    std::vector<int> kinds;
    bool high = h.d > 6 || h.a > 3;
    for (int rep = 0; rep < (high ? 1 : 4); ++rep) { kinds.push_back(1); kinds.push_back(0); }
    kinds.push_back(0); kinds.push_back(0);
    if (!high) kinds.push_back(2);
    if (h.d >= 1) { kinds.push_back(3); kinds.push_back(10); kinds.push_back(10); }
    if (h.d >= 1 && h.a < 3) kinds.push_back(4);
    if (h.a >= 1) { kinds.push_back(5); kinds.push_back(5); kinds.push_back(6); }
    if (s.locals) { kinds.push_back(7); if (h.d >= 1) { kinds.push_back(8); kinds.push_back(8); } }
    kinds.push_back(11);
    if (r.chance(15)) kinds.push_back(9);
    if (h == H{s.outs, 0} && r.chance(30)) kinds.push_back(12);
    if (h.d >= 1 && r.chance(8)) kinds.push_back(13);
    int k = kinds[r(kinds.size())];
    switch (k) {
      case 0: {
        std::vector<u8> ok;
        for (int p = 0; p < kNPrim; ++p)
          if (p != pExit && static_cast<u64>(effIn[p]) <= h.d && (!high || effOut[p] < effIn[p])) ok.push_back(static_cast<u8>(p));
        if (ok.empty()) { w.code.push_back(I(oLit, litEdges[r(12)])); h.d += 1; break; }
        u8 p = ok[r(ok.size())];
        w.code.push_back(P(p));
        h.d = h.d - static_cast<u64>(effIn[p]) + static_cast<u64>(effOut[p]);
        break;
      }
      case 1: w.code.push_back(I(oLit, r.chance(70) ? litEdges[r(12)] : r.raw())); h.d += 1; break;
      case 2: w.code.push_back(I(oStr, ndata && r.chance(80) ? r(ndata) : u32Edges[r(12)])); h.d += 2; break;
      case 3: w.code.push_back(I(oPick, r(h.d))); h.d += 1; break;
      case 4: w.code.push_back(I(oTor, 0)); h.d -= 1; h.a += 1; break;
      case 5: w.code.push_back(I(oFromr, 0)); h.d += 1; h.a -= 1; break;
      case 6: w.code.push_back(I(oRfetch, 0)); h.d += 1; break;
      case 7: w.code.push_back(I(oLocget, r(s.locals))); h.d += 1; break;
      case 8: w.code.push_back(I(oLocset, r(s.locals))); h.d -= 1; break;
      case 9: case 10: {
        u8 op = k == 9 ? oJmp : (r.chance(50) ? oJz : oJnz);
        H req = k == 9 ? h : H{h.d - 1, h.a};
        std::vector<u32> back;
        for (u32 t = 0; t <= i; ++t)
          if (Hs[t] == req && (!annAt.count(t) || annAt[t] == req)) back.push_back(t);
        if (!back.empty() && r.chance(50)) {
          u32 t = back[r(back.size())];
          annotate(t, req);
          w.code.push_back(I(op, t));
        } else {
          pend.push_back({w.code.size(), req});
          w.code.push_back(I(op, 0));
        }
        if (k == 9) live = false; else h = req;
        break;
      }
      case 11: {
        std::vector<u32> ok;
        for (u32 c = 0; c < sigs.size(); ++c)
          if (sigs[c].ins <= h.d && (!high || sigs[c].outs < sigs[c].ins + 1)) ok.push_back(c);
        if (ok.empty()) { w.code.push_back(I(oLit, 0)); h.d += 1; break; }
        u32 c = ok[r(ok.size())];
        w.code.push_back(I(oCall, c));
        h.d = h.d - sigs[c].ins + sigs[c].outs;
        break;
      }
      case 12: w.code.push_back(I(oRet, 0)); live = false; break;
      case 13: w.code.push_back(P(pExit)); live = false; break;
    }
  }
  if (live) {
    auto push = [&](In x, H after) { Hs.push_back(h); w.code.push_back(x); h = after; };
    while (h.a > 0) { push(I(oFromr, 0), {h.d + 1, h.a - 1}); push(P(pDrop), {h.d - 1, h.a}); }
    if (r.chance(50)) {
      while (h.d > s.outs) push(P(pDrop), {h.d - 1, h.a});
      while (h.d < s.outs) push(I(oLit, r.raw()), {h.d + 1, h.a});
      push(I(oRet, 0), h);
    } else {
      push(I(oLit, 0), {h.d + 1, h.a});
      push(P(pExit), h);
    }
  }
  std::vector<std::pair<H, u32>> pads;
  for (const auto& pj : pend) {
    std::vector<u32> fwd;
    for (u32 t = static_cast<u32>(pj.i) + 1; t < Hs.size(); ++t)
      if (Hs[t] == pj.req && (!annAt.count(t) || annAt[t] == pj.req)) fwd.push_back(t);
    u32 t;
    if (!fwd.empty() && r.chance(85)) {
      t = fwd[r(fwd.size())];
    } else {
      auto it = std::find_if(pads.begin(), pads.end(), [&](auto& x) { return x.first == pj.req; });
      if (it != pads.end() && r.chance(70)) t = it->second;
      else {
        t = static_cast<u32>(w.code.size());
        Hs.push_back(pj.req); w.code.push_back(I(oLit, 0));
        Hs.push_back({pj.req.d + 1, pj.req.a}); w.code.push_back(P(pExit));
        pads.push_back({pj.req, t});
      }
    }
    annotate(t, pj.req);
    w.code[pj.i].imm = t;
  }
  for (size_t j = w.ann.size(); j > 1; --j) std::swap(w.ann[j - 1], w.ann[r(j)]);
  return w;
}

Prog genProg(Rng& r) {
  Prog p;
  size_t nw = 1 + r(r.chance(20) ? 8 : 4);
  size_t nd = r(4);
  for (size_t j = 0; j < nd; ++j) {
    Bytes d(r(r.chance(10) ? 64 : 12));
    for (auto& x : d) x = static_cast<u8>(r.raw());
    p.data.push_back(d);
  }
  p.main = static_cast<u32>(r(nw));
  std::vector<Sig> sigs;
  for (size_t j = 0; j < nw; ++j)
    sigs.push_back({j == p.main ? 0u : static_cast<u32>(r(4)), static_cast<u32>(r(4)), static_cast<u32>(r(5))});
  for (size_t j = 0; j < nw; ++j) p.words.push_back(genWord(r, sigs[j], sigs, static_cast<u32>(nd)));
  return p;
}

In randomIn(Rng& r, const Prog& p) {
  u8 op = r.chance(50) ? static_cast<u8>(r(kNPrim)) : static_cast<u8>(oLit + r(13));
  u64 imm = 0;
  if (op == oLit) imm = litEdges[r(12)];
  else if (immLen(op) == 4) imm = r.chance(70) ? r(8) : u32Edges[r(12)];
  if (op == oCall && r.chance(70)) imm = r(p.words.size() + 1);
  return {op, imm};
}

const char* mutateProg(Rng& r, Prog& p) {
  Word& w = p.words[r(p.words.size())];
  for (int tries = 0; tries < 8; ++tries) {
    switch (r(19)) {
      case 0: if (w.ann.empty()) break; else {
        Ann& a = w.ann[r(w.ann.size())];
        if (r.chance(50)) a.d += r.chance(50) ? 1 : -1; else a.a += r.chance(50) ? 1 : -1;
        return "ann_height";
      }
      case 1: if (w.ann.empty()) break; else {
        Ann a = w.ann[r(w.ann.size())];
        a.d += 1 + static_cast<u32>(r(2));
        w.ann.insert(w.ann.begin() + static_cast<long>(r(w.ann.size() + 1)), a);
        return "ann_conflict";
      }
      case 2: if (w.ann.empty()) break; else {
        Ann a = w.ann[r(w.ann.size())];
        w.ann.push_back(a);
        return "ann_dup";
      }
      case 3: w.ann.push_back({static_cast<u32>(w.code.size() + r(3)), 0, 0}); return "ann_out_of_range";
      case 4: if (w.ann.empty()) break; else { w.ann.erase(w.ann.begin() + static_cast<long>(r(w.ann.size()))); return "ann_removed"; }
      case 5: for (auto& x : w.code) if (x.op >= oJmp && x.op <= oJnz && r.chance(60)) {
          x.imm = r.chance(50) ? x.imm + (r.chance(50) ? 1 : -1) : r(w.code.size() + 2);
          return "jump_target";
        }
        break;
      case 6:
        if (w.locals && r.chance(50)) { w.locals -= 1; return "locals_shrunk"; }
        for (auto& x : w.code) if ((x.op == oLocget || x.op == oLocset) && r.chance(60)) { x.imm = w.locals + r(2); return "local_index"; }
        break;
      case 7: for (auto& x : w.code) if (x.op == oCall && r.chance(60)) {
          x.imm = r.chance(50) ? p.words.size() + r(2) : r(p.words.size());
          return "call_target";
        }
        break;
      case 8: p.main = r.chance(70) ? static_cast<u32>(p.words.size() + r(2)) : u32Edges[r(12)]; return "main_out_of_range";
      case 9: if (p.main >= p.words.size()) break; else { p.words[p.main].ins = 1 + static_cast<u32>(r(2)); return "main_ins"; }
      case 10: w.outs += r.chance(50) ? 1 : -1; return "outs";
      case 11: w.ins += r.chance(50) ? 1 : -1; return "ins";
      case 12: if (w.code.size() < 2) break; else { w.code.erase(w.code.begin() + static_cast<long>(r(w.code.size()))); return "instr_deleted"; }
      case 13: if (w.code.size() < 2) break; else {
        size_t j = r(w.code.size() - 1); std::swap(w.code[j], w.code[j + 1]); return "instr_swapped";
      }
      case 14: if (w.code.empty()) break; else { w.code[r(w.code.size())] = randomIn(r, p); return "instr_replaced"; }
      case 15: for (auto& x : w.code) if (x.op == oPick && r.chance(60)) { x.imm += 1 + r(2); return "pick_depth"; }
        break;
      case 16: if (w.code.empty()) break; else { w.code.pop_back(); return "terminator_removed"; }
      case 17: w.code.insert(w.code.begin() + static_cast<long>(r(w.code.size() + 1)), randomIn(r, p)); return "instr_inserted";
      case 18: if (r.chance(30)) { w.code.clear(); return "code_empty"; } break;
    }
  }
  w.ann.push_back({0, w.ins + 1, 0});
  return "ann_entry";
}

const char* mutateBytes(Rng& r, Enc& e) {
  Bytes& b = e.b;
  switch (r(11)) {
    case 0: {
      int n = 1 + static_cast<int>(r(3));
      for (int j = 0; j < n; ++j) b[r(b.size())] ^= static_cast<u8>(1u << r(8));
      return "bit_flip";
    }
    case 1: b.resize(r(b.size())); return "truncated";
    case 2: {
      size_t n = 1 + r(8);
      bool z = r.chance(40);
      for (size_t j = 0; j < n; ++j) b.push_back(z ? 0 : static_cast<u8>(r.raw()));
      return "extended";
    }
    case 3: {
      size_t o = e.u32s[r(e.u32s.size())];
      u32 v = r.chance(60) ? u32Edges[r(12)] : static_cast<u32>(r.raw());
      if (r.chance(30)) { v = static_cast<u32>(b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (static_cast<u32>(b[o + 3]) << 24)) + (r.chance(50) ? 1 : static_cast<u32>(-1)); }
      for (int k = 0; k < 4; ++k) b[o + static_cast<size_t>(k)] = static_cast<u8>(v >> (8 * k));
      return "u32_field";
    }
    case 4: {
      if (e.ops.empty()) break;
      size_t o = e.ops[r(e.ops.size())];
      b[o] = r.chance(50) ? static_cast<u8>(0x39 + r(0x7f - 0x39 + 1)) : static_cast<u8>(0x8d + r(0xff - 0x8d + 1));
      return "bad_opcode";
    }
    case 5: {
      size_t o = e.counts[r(e.counts.size())];
      const u32 huge[] = {0xffffffff, 0xfffffffe, 0x80000000, 0x7fffffff, 0x10000000, 0x01000000};
      u32 v = huge[r(6)];
      for (int k = 0; k < 4; ++k) b[o + static_cast<size_t>(k)] = static_cast<u8>(v >> (8 * k));
      return "huge_count";
    }
    case 6: b[r(b.size())] = static_cast<u8>(r.raw()); return "byte_replaced";
    case 7: {
      size_t i = r(b.size()), n = 1 + r(std::min<size_t>(16, b.size() - i));
      Bytes s(b.begin() + static_cast<long>(i), b.begin() + static_cast<long>(i + n));
      b.insert(b.begin() + static_cast<long>(i), s.begin(), s.end());
      return "slice_duplicated";
    }
    case 8: {
      size_t i = r(b.size()), n = 1 + r(std::min<size_t>(16, b.size() - i));
      b.erase(b.begin() + static_cast<long>(i), b.begin() + static_cast<long>(i + n));
      return "slice_deleted";
    }
    case 9: b[r(4)] ^= static_cast<u8>(1u << r(8)); return "magic";
    case 10: {
      if (e.ops.empty()) break;
      size_t o = e.ops[r(e.ops.size())];
      b[o] = static_cast<u8>(r(256));
      return "opcode_random";
    }
  }
  b[r(b.size())] ^= 0x80;
  return "bit_flip";
}

void writeFile(const std::string& path, const Bytes& b) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) die("cannot write %s", path.c_str());
  if (!b.empty() && std::fwrite(b.data(), 1, b.size(), f) != b.size()) die("short write %s", path.c_str());
  std::fclose(f);
}

int cmdGen(u64 n, u64 seed, const std::string& dir) {
  ::mkdir(dir.c_str(), 0755);
  Rng r(seed * 0x9e3779b97f4a7c15ull + 1);
  auto exs = examples();
  std::string man;
  std::map<std::string, u64> kinds;
  for (u64 k = 0; k < n; ++k) {
    char name[64];
    std::snprintf(name, sizeof name, "case_%06" PRIu64 ".spb", k);
    std::string kind;
    Bytes out;
    if (k < exs.size()) {
      out = encode(exs[k].second).b;
      kind = std::string("example:") + exs[k].first;
    } else {
      u64 c = r(100);
      Prog p = r.chance(5) ? exs[r(exs.size())].second : genProg(r);
      if (c < 45) {
        out = encode(p).b;
        kind = "valid";
      } else if (c < 68) {
        kind = std::string("struct:") + mutateProg(r, p);
        if (r.chance(20)) kind += std::string("+") + mutateProg(r, p);
        out = encode(p).b;
      } else {
        Enc e = encode(p);
        kind = std::string("byte:") + mutateBytes(r, e);
        if (r.chance(10)) { e.b.resize(std::min<size_t>(e.b.size(), 4)); kind = "byte:tiny"; }
        out = e.b;
      }
    }
    writeFile(dir + "/" + name, out);
    man += std::string(name) + " " + kind + "\n";
    kinds[kind.substr(0, kind.find(':'))]++;
  }
  FILE* f = std::fopen((dir + "/manifest.txt").c_str(), "w");
  if (!f) die("cannot write manifest");
  std::fputs(man.c_str(), f);
  std::fclose(f);
  for (auto& [k, v] : kinds) std::printf("gen: %s=%" PRIu64 "\n", k.c_str(), v);
  return 0;
}

int cmdExamples(const std::string& dir) {
  ::mkdir(dir.c_str(), 0755);
  for (auto& [name, p] : examples()) writeFile(dir + "/" + name + ".spb", encode(p).b);
  return 0;
}

int cmdVectors(const std::string& dir) {
  ::mkdir(dir.c_str(), 0755);
  auto one = [](std::vector<In> code, std::vector<Ann> ann = {}, u32 ins = 0, u32 outs = 0, u32 locals = 0) {
    Prog p; p.words.push_back({ins, outs, locals, std::move(code), std::move(ann)}); return p;
  };
  const In lit0 = I(oLit, 0), ex = P(pExit);
  std::vector<std::pair<std::string, Bytes>> acc, rej;
  acc.push_back({"tiny", encode(one({lit0, ex})).b});
  acc.push_back({"jump_annotated", encode(one({I(oJmp, 1), lit0, ex}, {{1, 0, 0}})).b});
  acc.push_back({"jump_dup_ann", encode(one({I(oJmp, 1), lit0, ex}, {{1, 0, 0}, {1, 0, 0}})).b});
  acc.push_back({"ret_outs", encode(one({I(oLit, 5), I(oLit, 6), I(oRet, 0)}, {}, 0, 2)).b});
  acc.push_back({"aux_roundtrip", encode(one({lit0, I(oTor, 0), I(oRfetch, 0), I(oFromr, 0), P(pAdd), ex})).b});
  {
    Prog p = one({I(oCall, 1), ex});
    p.words.push_back({0, 1, 1, {lit0, I(oLocset, 0), I(oLocget, 0), I(oRet, 0)}, {}});
    acc.push_back({"call_local", encode(p).b});
  }
  {
    Prog p = one({lit0, ex}); p.data.push_back({1, 2, 3}); p.data.push_back({});
    acc.push_back({"data_items", encode(p).b});
  }
  {
    Prog p = one({lit0, ex}); p.words.insert(p.words.begin(), Word{3, 0, 0, {P(pDrop), P(pDrop), P(pDrop), I(oRet, 0)}, {}});
    p.main = 1;
    acc.push_back({"main_second", encode(p).b});
  }
  Bytes tiny = encode(one({lit0, ex})).b;
  rej.push_back({"empty", {}});
  rej.push_back({"magic_only", {0x53, 0x50, 0x52, 0x31}});
  { Bytes b = tiny; b[3] = 0x32; rej.push_back({"bad_magic", b}); }
  { Prog p; rej.push_back({"no_words", encode(p).b}); }
  { Prog p = one({lit0, ex}); p.main = 1; rej.push_back({"main_out_of_range", encode(p).b}); }
  { Prog p = one({lit0, ex}); p.main = 0xffffffff; rej.push_back({"main_huge", encode(p).b}); }
  rej.push_back({"main_ins", encode(one({lit0, ex}, {}, 1)).b});
  { Bytes b = tiny; b.push_back(0); rej.push_back({"trailing_byte", b}); }
  { Bytes b = tiny; b.pop_back(); rej.push_back({"truncated", b}); }
  { Bytes b = tiny; b.back() = 0x39; rej.push_back({"opcode_39", b}); }
  { Bytes b = tiny; b.back() = 0x7f; rej.push_back({"opcode_7f", b}); }
  { Bytes b = tiny; b.back() = 0x8d; rej.push_back({"opcode_8d", b}); }
  { Bytes b = tiny; b.back() = 0xff; rej.push_back({"opcode_ff", b}); }
  rej.push_back({"exit_empty_stack", encode(one({ex})).b});
  rej.push_back({"empty_word", encode(one({})).b});
  rej.push_back({"falls_off_end", encode(one({lit0})).b});
  {
    Enc e = encode(one({lit0, ex}));
    for (int k = 0; k < 4; ++k) e.b[e.counts[2] + static_cast<size_t>(k)] = 0xff;
    rej.push_back({"huge_ncode", e.b});
    Enc f = encode(one({lit0, ex}));
    for (int k = 0; k < 4; ++k) f.b[f.counts[3] + static_cast<size_t>(k)] = 0xff;
    rej.push_back({"huge_nann", f.b});
    Enc g = encode(one({lit0, ex}));
    for (int k = 0; k < 4; ++k) g.b[g.counts[0] + static_cast<size_t>(k)] = 0xff;
    rej.push_back({"huge_nwords", g.b});
    Enc h = encode(one({lit0, ex}));
    for (int k = 0; k < 4; ++k) h.b[h.counts[1] + static_cast<size_t>(k)] = 0xff;
    rej.push_back({"huge_ndata", h.b});
  }
  { Prog p = one({lit0, ex}); p.data.push_back({1, 2, 3}); Bytes b = encode(p).b; b[16] = 0xff; b[17] = 0xff; rej.push_back({"huge_data_len", b}); }
  rej.push_back({"bad_local", encode(one({I(oLocget, 0), ex})).b});
  rej.push_back({"locset_empty", encode(one({I(oLocset, 0), lit0, ex}, {}, 0, 0, 1)).b});
  rej.push_back({"jump_unannotated", encode(one({I(oJmp, 1), lit0, ex})).b});
  rej.push_back({"jump_ann_conflict", encode(one({I(oJmp, 1), lit0, ex}, {{1, 0, 0}, {1, 1, 0}})).b});
  rej.push_back({"jump_wrong_height", encode(one({I(oJmp, 1), lit0, ex}, {{1, 1, 0}})).b});
  rej.push_back({"ann_out_of_range", encode(one({lit0, ex}, {{2, 0, 0}})).b});
  rej.push_back({"ann_entry", encode(one({lit0, ex}, {{0, 1, 0}})).b});
  rej.push_back({"jz_empty", encode(one({I(oJz, 1), lit0, ex}, {{1, 0, 0}})).b});
  rej.push_back({"pick_deep", encode(one({lit0, I(oPick, 1), ex})).b});
  rej.push_back({"fromr_empty", encode(one({I(oFromr, 0), ex})).b});
  rej.push_back({"rfetch_empty", encode(one({I(oRfetch, 0), ex})).b});
  rej.push_back({"call_missing", encode(one({I(oCall, 1), lit0, ex})).b});
  {
    Prog p = one({I(oCall, 1), lit0, ex});
    p.words.push_back({1, 0, 0, {P(pDrop), I(oRet, 0)}, {}});
    rej.push_back({"call_underflow", encode(p).b});
  }
  rej.push_back({"ret_wrong_outs", encode(one({lit0, I(oRet, 0)})).b});
  rej.push_back({"ret_with_aux", encode(one({lit0, I(oTor, 0), I(oRet, 0)})).b});
  {
    Prog p = exEcho(); p.words[0].ann[0] = {0, 1, 0};
    rej.push_back({"echo_bad_ann", encode(p).b});
    Prog q = exFib(); q.words[1].ins = 2;
    rej.push_back({"fib_callee_ins", encode(q).b});
    Prog s = exUpper(); s.words[0].code[16].imm = 1;
    rej.push_back({"upper_bad_jump", encode(s).b});
  }
  for (int op = 0x39; op <= 0xff; ++op) {
    if (op >= 0x80 && op <= 0x8c) continue;
    if (op != 0x39 && op != 0x7f && op != 0x8d && op != 0xff && op % 16) continue;
    Bytes b = tiny; b[b.size() - 1] = static_cast<u8>(op);
    char nm[32]; std::snprintf(nm, sizeof nm, "op_%02x", op);
    rej.push_back({nm, b});
  }
  for (auto& [n, b] : acc) writeFile(dir + "/accept_" + n + ".spb", b);
  for (auto& [n, b] : rej) writeFile(dir + "/reject_" + n + ".spb", b);
  std::printf("vectors: accept=%zu reject=%zu\n", acc.size(), rej.size());
  return 0;
}

Bytes readFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("cannot read %s", path.c_str());
  return Bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

u64 rdLe(const Bytes& b, size_t o, int k) {
  if (o + static_cast<size_t>(k) > b.size()) die("truncated ELF");
  u64 v = 0;
  for (int i = 0; i < k; ++i) v |= static_cast<u64>(b[o + static_cast<size_t>(i)]) << (8 * i);
  return v;
}

// ------------------------------------------------------------------------
// run / exec1: native on Linux, MAP_JIT + brk #0x517 trap runner on Darwin arm64

#ifdef SPR1_DARWIN_JIT

struct GuestAlloc {
  void* ptr;
  size_t len;
};

struct JitGuest {
  u8* jit_base = nullptr;
  size_t jit_sz = 0;
  u8* bss_ptr = nullptr;
  size_t bss_sz = 0;
  u64 entry_off = 0;
  void* tramp_mem = nullptr;
  size_t tramp_sz = 0;
  u8* stack_base = nullptr;
  size_t stack_sz = 0;
  const Bytes* input = nullptr;
  size_t in_pos = 0;
  std::vector<GuestAlloc> allocs;
  int exit_code = 0;
  bool exited = false;
  u64 bad_syscall = 0;
};

static JitGuest g_jit;
static sigjmp_buf g_jb;
static stack_t g_alt_ss{};

static constexpr u32 INSN_SVC0 = 0xd4000001u;
static constexpr u32 INSN_BRK_SVC = 0xd420a2e0u;  // brk #0x517

static bool valid_guest_write(u64 addr, u64 len) {
  if (addr + len < addr) return false;
  for (const auto& a : g_jit.allocs) {
    u64 lo = reinterpret_cast<u64>(a.ptr);
    u64 hi = lo + a.len;
    if (addr >= lo && addr + len <= hi) return true;
  }
  return false;
}

static void jit_trap_handler(int sig, siginfo_t*, void* uctx) {
  if (sig == SIGALRM) siglongjmp(g_jb, 4);
  if (sig != SIGTRAP) siglongjmp(g_jb, 100 + sig);
  auto* uc = static_cast<ucontext_t*>(uctx);
  auto& ss = uc->uc_mcontext->__ss;
  u64 pc = ss.__pc;
  u64 j_lo = reinterpret_cast<u64>(g_jit.jit_base);
  u64 j_hi = j_lo + g_jit.jit_sz;
  if (pc < j_lo || pc + 4 > j_hi || *reinterpret_cast<const u32*>(pc) != INSN_BRK_SVC) {
    siglongjmp(g_jb, 100 + sig);
  }
  u64 sysno = ss.__x[8];
  if (sysno == 222) {  // Linux __NR_mmap(addr, len, prot, flags, fd, off)
    u64 addr = ss.__x[0];
    u64 len = ss.__x[1];
    if (addr != 0 || len == 0 || len > (64ULL << 20)) {
      ss.__x[0] = static_cast<u64>(-12LL);  // -ENOMEM
    } else {
      void* p = ::mmap(nullptr, static_cast<size_t>(len), PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0);
      if (p == MAP_FAILED) {
        ss.__x[0] = static_cast<u64>(-12LL);
      } else {
        g_jit.allocs.push_back({p, static_cast<size_t>(len)});
        ss.__x[0] = reinterpret_cast<u64>(p);
      }
    }
    ss.__pc += 4;
    return;
  }
  if (sysno == 216) {  // Linux __NR_mremap(old, oldlen, newlen, flags)
    u64 oldp = ss.__x[0];
    u64 oldlen = ss.__x[1];
    u64 newlen = ss.__x[2];
    int idx = -1;
    for (size_t i = 0; i < g_jit.allocs.size(); ++i) {
      if (reinterpret_cast<u64>(g_jit.allocs[i].ptr) == oldp && g_jit.allocs[i].len == oldlen) {
        idx = static_cast<int>(i);
        break;
      }
    }
    if (idx < 0 || newlen == 0 || newlen > (64ULL << 20)) {
      ss.__x[0] = static_cast<u64>(-12LL);
    } else {
      void* np = ::mmap(nullptr, static_cast<size_t>(newlen), PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANON, -1, 0);
      if (np == MAP_FAILED) {
        ss.__x[0] = static_cast<u64>(-12LL);
      } else {
        size_t cp = std::min(static_cast<size_t>(oldlen), static_cast<size_t>(newlen));
        std::memcpy(np, reinterpret_cast<const void*>(oldp), cp);
        ::munmap(reinterpret_cast<void*>(oldp), static_cast<size_t>(oldlen));
        g_jit.allocs[static_cast<size_t>(idx)] = {np, static_cast<size_t>(newlen)};
        ss.__x[0] = reinterpret_cast<u64>(np);
      }
    }
    ss.__pc += 4;
    return;
  }
  if (sysno == 63) {  // Linux __NR_read(fd, buf, count)
    u64 fd = ss.__x[0], buf = ss.__x[1], cnt = ss.__x[2];
    if (fd != 0) {
      ss.__x[0] = static_cast<u64>(-9LL);
    } else if (!valid_guest_write(buf, cnt)) {
      ss.__x[0] = static_cast<u64>(-14LL);
    } else {
      size_t avail = g_jit.input->size() - g_jit.in_pos;
      size_t take = cnt < avail ? static_cast<size_t>(cnt) : avail;
      if (take > 0) {
        std::memcpy(reinterpret_cast<void*>(buf), g_jit.input->data() + g_jit.in_pos, take);
        g_jit.in_pos += take;
      }
      ss.__x[0] = static_cast<u64>(take);
    }
    ss.__pc += 4;
    return;
  }
  if (sysno == 93 || sysno == 94) {  // __NR_exit / __NR_exit_group
    g_jit.exit_code = static_cast<int>(ss.__x[0] & 0xffULL);
    g_jit.exited = true;
    siglongjmp(g_jb, 1);
  }
  g_jit.bad_syscall = sysno;
  siglongjmp(g_jb, 2);
}

static void jit_cleanup_allocs() {
  for (const auto& a : g_jit.allocs) ::munmap(a.ptr, a.len);
  g_jit.allocs.clear();
}

static bool jit_init(const char* bin_path) {
  Bytes raw = readFile(bin_path);
  if (raw.size() < 64 || raw[0] != 0x7f || raw[1] != 'E' || raw[2] != 'L' || raw[3] != 'F' ||
      raw[4] != 2 || raw[5] != 1 || raw[6] != 1) {
    die("invalid ELF64 binary: %s", bin_path);
  }
  u16 e_type = static_cast<u16>(rdLe(raw, 16, 2));
  u16 e_machine = static_cast<u16>(rdLe(raw, 18, 2));
  u64 e_entry = rdLe(raw, 24, 8);
  u64 e_phoff = rdLe(raw, 32, 8);
  u16 e_phentsize = static_cast<u16>(rdLe(raw, 54, 2));
  u16 e_phnum = static_cast<u16>(rdLe(raw, 56, 2));
  if (e_type != 2 || e_machine != 183 || e_phentsize != 56 || e_phnum == 0 ||
      e_phoff + static_cast<u64>(e_phnum) * 56ULL > raw.size()) {
    die("unsupported EM_AARCH64 ELF64: %s", bin_path);
  }
  u64 min_va = ~0ULL, max_va = 0;
  for (u16 i = 0; i < e_phnum; ++i) {
    size_t ph = static_cast<size_t>(e_phoff) + static_cast<size_t>(i) * 56;
    if (rdLe(raw, ph, 4) != 1) continue;
    u64 p_va = rdLe(raw, ph + 16, 8);
    u64 p_memsz = rdLe(raw, ph + 40, 8);
    if (p_va < min_va) min_va = p_va;
    if (p_va + p_memsz > max_va) max_va = p_va + p_memsz;
  }
  size_t page_sz = static_cast<size_t>(getpagesize());
  size_t span = static_cast<size_t>(max_va - min_va);
  size_t map_sz = ((span + page_sz - 1) / page_sz) * page_sz;

  void* mem = ::mmap(nullptr, map_sz, PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
  if (mem == MAP_FAILED) die("mmap MAP_JIT failed");

  u64 hdr_end = e_phoff + static_cast<u64>(e_phnum) * 56ULL;
  pthread_jit_write_protect_np(0);
  std::memset(mem, 0, map_sz);
  for (u16 i = 0; i < e_phnum; ++i) {
    size_t ph = static_cast<size_t>(e_phoff) + static_cast<size_t>(i) * 56;
    if (rdLe(raw, ph, 4) != 1) continue;
    u32 p_flags = static_cast<u32>(rdLe(raw, ph + 4, 4));
    u64 p_off = rdLe(raw, ph + 8, 8);
    u64 p_va = rdLe(raw, ph + 16, 8);
    u64 p_filesz = rdLe(raw, ph + 32, 8);
    u64 p_memsz = rdLe(raw, ph + 40, 8);
    if (p_filesz > 0) {
      u8* dst = static_cast<u8*>(mem) + (p_va - min_va);
      std::memcpy(dst, raw.data() + p_off, static_cast<size_t>(p_filesz));
      if (p_flags & 1U) {
        u64 scan_start = (p_off == 0 && p_filesz > hdr_end) ? ((hdr_end + 3ULL) & ~3ULL) : 0ULL;
        for (u64 off = scan_start; off + 4 <= p_filesz; off += 4) {
          u32 w = static_cast<u32>(dst[off]) | (static_cast<u32>(dst[off + 1]) << 8) |
                  (static_cast<u32>(dst[off + 2]) << 16) | (static_cast<u32>(dst[off + 3]) << 24);
          if (w == INSN_SVC0) {
            dst[off + 0] = static_cast<u8>(INSN_BRK_SVC & 0xff);
            dst[off + 1] = static_cast<u8>((INSN_BRK_SVC >> 8) & 0xff);
            dst[off + 2] = static_cast<u8>((INSN_BRK_SVC >> 16) & 0xff);
            dst[off + 3] = static_cast<u8>((INSN_BRK_SVC >> 24) & 0xff);
          }
        }
      }
    } else if ((p_flags & 2U) && p_memsz > 0) {
      u64 va_off = p_va - min_va;
      size_t bss_sz = ((static_cast<size_t>(p_memsz) + page_sz - 1) / page_sz) * page_sz;
      void* bss = static_cast<u8*>(mem) + va_off;
      void* mapped = ::mmap(bss, bss_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
      if (mapped == MAP_FAILED) die("mmap FIXED BSS failed");
      g_jit.bss_ptr = static_cast<u8*>(bss);
      g_jit.bss_sz = bss_sz;
    }
  }
  pthread_jit_write_protect_np(1);
  sys_icache_invalidate(mem, map_sz);

  void* tramp = ::mmap(nullptr, page_sz, PROT_READ | PROT_WRITE | PROT_EXEC,
                       MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
  if (tramp == MAP_FAILED) die("mmap MAP_JIT tramp failed");
  auto* tc = static_cast<u32*>(tramp);
  u32 tn = 0;
  pthread_jit_write_protect_np(0);
  tc[tn++] = 0x9100001fu;  // mov sp, x0
  tc[tn++] = 0xaa0103f0u;  // mov x16, x1 (entry pc)
  for (u32 r = 0; r <= 30; ++r) {
    if (r == 16 || r == 18) continue;
    tc[tn++] = 0xd2e00000u | (0xdeadu << 5) | r;
    tc[tn++] = 0xf2c00000u | (0xbeefu << 5) | r;
    tc[tn++] = 0xf2800000u | (r << 5) | r;
  }
  tc[tn++] = 0xd61f0200u;  // br x16
  pthread_jit_write_protect_np(1);
  sys_icache_invalidate(tramp, page_sz);

  size_t stack_sz = 4 * page_sz;
  void* stk = ::mmap(nullptr, stack_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (stk == MAP_FAILED) die("mmap guest stack failed");

  g_alt_ss.ss_sp = std::malloc(SIGSTKSZ * 4);
  g_alt_ss.ss_size = SIGSTKSZ * 4;
  g_alt_ss.ss_flags = 0;
  if (sigaltstack(&g_alt_ss, nullptr) != 0) die("sigaltstack failed");

  struct sigaction sa{};
  sa.sa_sigaction = jit_trap_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
  sigaction(SIGTRAP, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
  sigaction(SIGALRM, &sa, nullptr);

  g_jit.jit_base = static_cast<u8*>(mem);
  g_jit.jit_sz = map_sz;
  g_jit.entry_off = e_entry - min_va;
  g_jit.tramp_mem = tramp;
  g_jit.tramp_sz = page_sz;
  g_jit.stack_base = static_cast<u8*>(stk);
  g_jit.stack_sz = stack_sz;
  return true;
}

static std::string jit_run_bytes(const Bytes& input, int timeoutMs, int* raw_rc = nullptr) {
  if (g_jit.bss_ptr) std::memset(g_jit.bss_ptr, 0, g_jit.bss_sz);
  g_jit.input = &input;
  g_jit.in_pos = 0;
  g_jit.exited = false;
  g_jit.exit_code = 0;
  g_jit.bad_syscall = 0;
  g_jit.allocs.clear();

  struct itimerval it{}, zero_it{};
  if (timeoutMs > 0) {
    it.it_value.tv_sec = timeoutMs / 1000;
    it.it_value.tv_usec = (timeoutMs % 1000) * 1000;
    setitimer(ITIMER_REAL, &it, nullptr);
  }
  u64 guest_sp = reinterpret_cast<u64>(g_jit.stack_base) + g_jit.stack_sz / 2;
  u64 entry_pc = reinterpret_cast<u64>(g_jit.jit_base) + g_jit.entry_off;

  int jrc = sigsetjmp(g_jb, 1);
  if (jrc == 0) {
    auto fn = reinterpret_cast<void (*)(u64, u64)>(g_jit.tramp_mem);
    fn(guest_sp, entry_pc);
    jrc = 126;
  }
  if (timeoutMs > 0) setitimer(ITIMER_REAL, &zero_it, nullptr);
  jit_cleanup_allocs();
  if (jrc == 1 && g_jit.exited) {
    if (raw_rc) *raw_rc = g_jit.exit_code;
    return std::to_string(g_jit.exit_code);
  }
  if (jrc == 4) {
    if (raw_rc) *raw_rc = 124;
    return "timeout";
  }
  if (jrc >= 100) {
    int sig = jrc - 100;
    if (raw_rc) *raw_rc = 128 + sig;
    return "sig" + std::to_string(sig);
  }
  if (raw_rc) *raw_rc = 125;
  return "sig" + std::to_string(SIGSYS);
}

static void jit_destroy() {
  if (g_jit.jit_base) ::munmap(g_jit.jit_base, g_jit.jit_sz);
  if (g_jit.tramp_mem) ::munmap(g_jit.tramp_mem, g_jit.tramp_sz);
  if (g_jit.stack_base) ::munmap(g_jit.stack_base, g_jit.stack_sz);
  std::free(g_alt_ss.ss_sp);
}

#else

u64 nowNs() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return static_cast<u64>(t.tv_sec) * 1000000000ull + static_cast<u64>(t.tv_nsec);
}

std::string runOne(const char* bin, const std::string& file, int timeoutMs, int* raw_rc = nullptr) {
  int fd = ::open(file.c_str(), O_RDONLY);
  if (fd < 0) {
    if (raw_rc) *raw_rc = 127;
    return "noinput";
  }
  pid_t pid = fork();
  if (pid < 0) die("fork failed");
  if (pid == 0) {
    dup2(fd, 0);
    int nul = ::open("/dev/null", O_WRONLY);
    if (nul >= 0) { dup2(nul, 1); dup2(nul, 2); }
    execl(bin, bin, static_cast<char*>(nullptr));
    _exit(127);
  }
  ::close(fd);
  u64 deadline = nowNs() + static_cast<u64>(timeoutMs) * 1000000ull;
  int st = 0;
  for (;;) {
    pid_t w = waitpid(pid, &st, WNOHANG);
    if (w == pid) break;
    if (nowNs() > deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &st, 0);
      if (raw_rc) *raw_rc = 124;
      return "timeout";
    }
    struct timespec ts = {0, 20000};
    nanosleep(&ts, nullptr);
  }
  if (WIFEXITED(st)) {
    if (raw_rc) *raw_rc = WEXITSTATUS(st);
    return std::to_string(WEXITSTATUS(st));
  }
  int sig = WTERMSIG(st);
  if (raw_rc) *raw_rc = 128 + sig;
  return "sig" + std::to_string(sig);
}

#endif

int cmdRun(const char* bin, const std::string& list, int timeoutMs) {
  std::ifstream in(list);
  if (!in) die("cannot read %s", list.c_str());
#ifdef SPR1_DARWIN_JIT
  jit_init(bin);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    Bytes input = readFile(line);
    std::printf("%s %s\n", jit_run_bytes(input, timeoutMs).c_str(), line.c_str());
  }
  jit_destroy();
#else
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::printf("%s %s\n", runOne(bin, line, timeoutMs).c_str(), line.c_str());
  }
#endif
  std::fflush(stdout);
  return 0;
}

int cmdExec1(const char* bin, const std::string& file, int timeoutMs) {
  int rc = 0;
#ifdef SPR1_DARWIN_JIT
  jit_init(bin);
  Bytes input = readFile(file);
  jit_run_bytes(input, timeoutMs, &rc);
  jit_destroy();
#else
  runOne(bin, file, timeoutMs, &rc);
#endif
  return rc;
}

// ------------------------------------------------------------------------
// syms and patch: supports both Format C (`:`) and Format A (`#`) listings

std::vector<std::array<u64, 3>> loads(const Bytes& e) {
  if (e.size() < 64 || e[0] != 0x7f || e[1] != 'E' || e[2] != 'L' || e[3] != 'F' || e[4] != 2) die("not an ELF64 file");
  u64 phoff = rdLe(e, 32, 8), phentsize = rdLe(e, 54, 2), phnum = rdLe(e, 56, 2);
  std::vector<std::array<u64, 3>> v;
  for (u64 i = 0; i < phnum; ++i) {
    size_t p = static_cast<size_t>(phoff + i * phentsize);
    if (rdLe(e, p, 4) == 1) v.push_back({rdLe(e, p + 8, 8), rdLe(e, p + 16, 8), rdLe(e, p + 32, 8)});
  }
  return v;
}

u64 offToAddr(const Bytes& e, u64 off) {
  for (auto& l : loads(e)) if (off >= l[0] && off < l[0] + l[2]) return l[1] + (off - l[0]);
  die("file offset outside every PT_LOAD segment");
}

u64 addrToOff(const Bytes& e, u64 addr) {
  for (auto& l : loads(e)) if (addr >= l[1] && addr < l[1] + l[2]) return l[0] + (addr - l[1]);
  die("address outside every PT_LOAD segment");
}

int cmdSyms(const std::string& listing, const std::string& binPath) {
  Bytes e = readFile(binPath);
  std::ifstream in(listing);
  if (!in) die("cannot read %s", listing.c_str());
  std::string line;
  std::vector<std::string> pend;
  while (std::getline(in, line)) {
    if (line.size() > 3 && (line[0] == '#' || line[0] == ':') && line[1] == ' ' &&
        line.back() == ':' && line.find(' ', 2) == std::string::npos) {
      pend.push_back(line.substr(2, line.size() - 3));
      continue;
    }
    if (pend.empty() || line.size() < 2) continue;
    char c0 = line[0];
    if (!std::isxdigit(static_cast<unsigned char>(c0))) continue;
    size_t h = line.find(':');
    if (h == std::string::npos) h = line.find('#');
    if (h == std::string::npos) continue;
    char* end = nullptr;
    u64 off = std::strtoull(line.c_str() + h + 1, &end, 16);
    if (end == line.c_str() + h + 1) continue;
    u64 addr = offToAddr(e, off);
    for (auto& n : pend) {
      std::printf("0x%" PRIx64 " %s\n", addr, n.c_str());
      if (n.size() == 2 && n[0] == 'p' && n[1] >= '1' && n[1] <= '3') {
        size_t pos = line.find("0x", h);
        if (pos != std::string::npos) {
          std::printf("0x%" PRIx64 " desc%c\n", static_cast<u64>(std::strtoull(line.c_str() + pos, nullptr, 0)), n[1]);
        }
      }
    }
    pend.clear();
  }
  return 0;
}

int cmdPatch(int argc, char** argv) {
  if (argc < 6 || (argc - 4) % 2) die("usage: patch IN OUT ADDR HEX [ADDR HEX]...");
  Bytes e = readFile(argv[2]);
  for (int i = 4; i + 1 < argc; i += 2) {
    u64 off = addrToOff(e, std::strtoull(argv[i], nullptr, 0));
    std::string hx, old;
    for (const char* p = argv[i + 1]; *p; ++p) if (!std::isspace(static_cast<unsigned char>(*p))) hx += *p;
    if (size_t gt = hx.find('>'); gt != std::string::npos) { old = hx.substr(0, gt); hx = hx.substr(gt + 1); }
    if (hx.size() % 2 || old.size() % 2) die("odd hex string %s", argv[i + 1]);
    for (size_t k = 0; k < old.size(); k += 2) {
      if (off + k / 2 >= e.size()) die("patch past the end of the file");
      if (old.substr(k, 2) != "??" && e[off + k / 2] != std::stoul(old.substr(k, 2), nullptr, 16))
        die("patch at %s: the bytes there are not the expected ones", argv[i]);
    }
    for (size_t k = 0; k < hx.size(); k += 2) {
      if (off + k / 2 >= e.size()) die("patch past the end of the file");
      e[off + k / 2] = static_cast<u8>(std::stoul(hx.substr(k, 2), nullptr, 16));
    }
  }
  writeFile(argv[3], e);
  ::chmod(argv[3], 0755);
  return 0;
}

}  // namespace

// ------------------------------------------------------------------------
// inject: single-fault injection under ptrace (Linux aarch64)

#ifdef SPR1_INJECT
namespace {

enum Cls { cGpr, cFlags, cFetch, cMem, kNCls };
const char* clsName[kNCls] = {"gpr", "flags", "fetch", "mem"};
enum Out { oExit0, oExit3, oExit5, oExit6, oExitOther, oSignal, oTimeout, oHarness, kNOut };
const char* outName[kNOut] = {"exit0", "exit3", "exit5", "exit6", "exit_other", "signal", "timeout", "harness"};

struct Region { std::string name; u64 addr, len; };
struct Desc { std::string name; u64 addr, len, ptrOff, lenOff; };

struct Cfg {
  const char* bin;
  const char* input;
  u64 start;
  int expect = -1;
  std::vector<Region> regions;
  std::vector<Desc> descs;
  u64 bufExtra = 0;
  int jobs = 0;
  int timeoutMs = 500;
  u64 stride = 1;
  u64 maxSteps = ~u64{0};
  bool cls[kNCls] = {true, true, true, true};
};

constexpr int kMaxSdc = 16;
struct SdcRec { u64 step, pc; int cls; u64 a, b; };
struct Shared {
  std::atomic<u64> nextChunk;
  std::atomic<u64> sdc;
  std::atomic<int> nsdc;
  std::atomic<int> err;
  SdcRec sdcs[kMaxSdc];
};

const int flagBit[4] = {31, 30, 29, 28};  // N, Z, C, V in AArch64 PSTATE/CPSR
const char* flagName[4] = {"N", "Z", "C", "V"};

static u64 nowNsInj() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return static_cast<u64>(t.tv_sec) * 1000000000ull + static_cast<u64>(t.tv_nsec);
}

bool waitStop(pid_t pid, int& st) {
  for (;;) {
    pid_t w = waitpid(pid, &st, __WALL);
    if (w == pid) return true;
    if (w < 0 && errno != EINTR) return false;
  }
}

bool getRegs(pid_t pid, struct user_pt_regs& r) {
  struct iovec iov{&r, sizeof(r)};
  return ptrace(PTRACE_GETREGSET, pid, reinterpret_cast<void*>(NT_PRSTATUS), &iov) == 0;
}

bool setRegs(pid_t pid, const struct user_pt_regs& r) {
  struct user_pt_regs copy = r;
  struct iovec iov{&copy, sizeof(copy)};
  return ptrace(PTRACE_SETREGSET, pid, reinterpret_cast<void*>(NT_PRSTATUS), &iov) == 0;
}

u64 getPc(pid_t pid) {
  struct user_pt_regs r{};
  if (!getRegs(pid, r)) die("PTRACE_GETREGSET failed");
  return r.pc;
}

pid_t spawnTraced(const Cfg& c) {
  pid_t pid = fork();
  if (pid < 0) die("fork failed");
  if (pid == 0) {
    int fd = ::open(c.input, O_RDONLY);
    if (fd < 0) _exit(126);
    dup2(fd, 0);
    ::close(fd);
    int nul = ::open("/dev/null", O_WRONLY);
    if (nul >= 0) { dup2(nul, 1); dup2(nul, 2); }
    personality(ADDR_NO_RANDOMIZE);
    if (ptrace(PTRACE_TRACEME, 0, nullptr, nullptr) != 0) _exit(125);
    execl(c.bin, c.bin, static_cast<char*>(nullptr));
    _exit(127);
  }
  int st;
  if (!waitStop(pid, st) || !WIFSTOPPED(st)) die("tracee did not stop at exec");
  ptrace(PTRACE_SETOPTIONS, pid, nullptr,
         reinterpret_cast<void*>(PTRACE_O_EXITKILL | PTRACE_O_TRACEFORK));
  return pid;
}

bool step1(pid_t pid, int& st) {
  if (ptrace(PTRACE_SINGLESTEP, pid, nullptr, nullptr) != 0) return false;
  if (!waitStop(pid, st)) return false;
  return WIFSTOPPED(st) && WSTOPSIG(st) == SIGTRAP && (st >> 16) == 0;
}

bool peek(pid_t pid, u64 addr, u64& v) {
  errno = 0;
  long x = ptrace(PTRACE_PEEKDATA, pid, reinterpret_cast<void*>(addr), nullptr);
  if (errno) return false;
  v = static_cast<u64>(x);
  return true;
}

bool poke(pid_t pid, u64 addr, u64 v) {
  return ptrace(PTRACE_POKEDATA, pid, reinterpret_cast<void*>(addr), reinterpret_cast<void*>(v)) == 0;
}

bool flipBit(pid_t pid, u64 addr, int bit, u64* orig = nullptr, u64* alignedAddr = nullptr) {
  u64 a = addr & ~u64{7}, w;
  if (!peek(pid, a, w)) return false;
  if (orig) *orig = w;
  if (alignedAddr) *alignedAddr = a;
  return poke(pid, a, w ^ (u64{1} << (8 * (addr - a) + static_cast<u64>(bit))));
}

struct Trace { std::vector<u64> pc; int exitCode = -1; int exitSig = 0; u64 pre = 0; };

Trace golden(const Cfg& c, std::vector<Region>& regs) {
  Trace t;
  pid_t pid = spawnTraced(c);
  int st = 0;
  u64 n = 0;
  for (;;) {
    if (getPc(pid) == c.start) break;
    if (!step1(pid, st)) die("program ended before reaching START_ADDR");
    if (++n > 2000000000ull) die("START_ADDR not reached");
  }
  t.pre = n;
  for (auto& d : c.descs) {
    regs.push_back({d.name, d.addr, d.len});
    u64 ptr, len;
    if (!peek(pid, d.addr + d.ptrOff, ptr) || !peek(pid, d.addr + d.lenOff, len)) die("cannot read descriptor %s", d.name.c_str());
    if (len > (1u << 20)) die("descriptor %s: implausible length", d.name.c_str());
    regs.push_back({d.name + ".buf", ptr, len + c.bufExtra});
  }
  for (;;) {
    t.pc.push_back(getPc(pid));
    if (!step1(pid, st)) break;
  }
  if (WIFEXITED(st)) t.exitCode = WEXITSTATUS(st);
  else if (WIFSIGNALED(st)) t.exitSig = WTERMSIG(st);
  else if (WIFSTOPPED(st)) { t.exitSig = WSTOPSIG(st); kill(pid, SIGKILL); waitStop(pid, st); }
  return t;
}

// Create traced child snapshot at current pc by executing svc #0 (__NR_clone=220).
pid_t cloneAt(pid_t m, const struct user_pt_regs& R) {
  u64 a = R.pc & ~u64{7}, w0 = 0;
  int sh = static_cast<int>(8 * (R.pc - a));
  if (!peek(m, a, w0)) return -1;
  u64 p0 = (w0 & ~(u64{0xffffffffULL} << sh)) | (u64{0xd4000001ULL} << sh);
  if (!poke(m, a, p0)) return -1;
  struct user_pt_regs T = R;
  T.regs[8] = 220;                    // __NR_clone
  T.regs[0] = 0x00008000 | SIGCHLD;   // CLONE_PARENT | SIGCHLD
  T.regs[1] = 0;
  T.regs[2] = 0;
  T.regs[3] = 0;
  T.regs[4] = 0;
  setRegs(m, T);
  int st;
  pid_t child = -1;
  if (ptrace(PTRACE_SINGLESTEP, m, nullptr, nullptr) == 0 && waitStop(m, st) && WIFSTOPPED(st) &&
      (st >> 8) == (SIGTRAP | (PTRACE_EVENT_FORK << 8))) {
    unsigned long msg = 0;
    ptrace(PTRACE_GETEVENTMSG, m, nullptr, &msg);
    child = static_cast<pid_t>(msg);
    step1(m, st);
  }
  poke(m, a, w0);
  setRegs(m, R);
  if (child <= 0) return -1;
  if (!waitStop(child, st) || !WIFSTOPPED(st)) return -1;
  poke(child, a, w0);
  setRegs(child, R);
  return child;
}

int runFree(pid_t pid, int timeoutMs) {
  if (ptrace(PTRACE_CONT, pid, nullptr, nullptr) != 0) { kill(pid, SIGKILL); int st; waitStop(pid, st); return oHarness; }
  u64 deadline = nowNsInj() + static_cast<u64>(timeoutMs) * 1000000ull;
  int st = 0;
  for (;;) {
    pid_t w = waitpid(pid, &st, __WALL | WNOHANG);
    if (w == pid) {
      if (WIFEXITED(st)) {
        int code = WEXITSTATUS(st);
        return code == 0 ? oExit0 : code == 3 ? oExit3 : code == 5 ? oExit5 : code == 6 ? oExit6 : oExitOther;
      }
      if (WIFSIGNALED(st)) return oSignal;
      if (WIFSTOPPED(st)) {
        int sig = WSTOPSIG(st);
        if ((st >> 16) != 0 || sig == SIGSTOP || sig == SIGTRAP) {
          if ((st >> 16) == PTRACE_EVENT_FORK) {
            unsigned long gc = 0;
            ptrace(PTRACE_GETEVENTMSG, pid, nullptr, &gc);
            if (gc) { kill(static_cast<pid_t>(gc), SIGKILL); int s2; waitpid(static_cast<pid_t>(gc), &s2, __WALL); }
          }
          ptrace(PTRACE_CONT, pid, nullptr, nullptr);
          continue;
        }
        kill(pid, SIGKILL);
        waitStop(pid, st);
        while (!(WIFEXITED(st) || WIFSIGNALED(st))) { if (!waitStop(pid, st)) break; }
        return oSignal;
      }
      continue;
    }
    if (w < 0 && errno != EINTR) return oHarness;
    if (nowNsInj() > deadline) {
      kill(pid, SIGKILL);
      while (waitStop(pid, st) && !(WIFEXITED(st) || WIFSIGNALED(st))) {}
      return oTimeout;
    }
    struct timespec ts = {0, 10000};
    nanosleep(&ts, nullptr);
  }
}

int afterStep(pid_t pid, int st) {
  if (WIFEXITED(st)) {
    int code = WEXITSTATUS(st);
    return code == 0 ? oExit0 : code == 3 ? oExit3 : code == 5 ? oExit5 : code == 6 ? oExit6 : oExitOther;
  }
  if (WIFSIGNALED(st)) return oSignal;
  if (WIFSTOPPED(st) && WSTOPSIG(st) == SIGTRAP && (st >> 16) == 0) return -1;
  kill(pid, SIGKILL);
  while (waitStop(pid, st) && !(WIFEXITED(st) || WIFSIGNALED(st))) {}
  return oSignal;
}

struct Worker {
  const Cfg& c;
  const Trace& t;
  const std::vector<Region>& regs;
  Shared* sh;
  u64* counts;
  u64 nsteps;

  void record(int cls, int out, u64 s, u64 pc, u64 a, u64 b) {
    counts[cls * kNOut + out]++;
    if (out == oExit0 && c.expect == 0) {
      sh->sdc.fetch_add(1);
      int k = sh->nsdc.fetch_add(1);
      if (k < kMaxSdc) sh->sdcs[k] = {s, pc, cls, a, b};
    }
  }

  void injectAll(pid_t m, u64 s) {
    struct user_pt_regs R{};
    if (!getRegs(m, R)) { sh->err.store(1); return; }
    auto fresh = [&]() {
      pid_t ch = cloneAt(m, R);
      if (ch < 0) sh->err.store(1);
      return ch;
    };
    if (c.cls[cGpr]) {
      for (int k = 0; k < 32; ++k) {
        for (int b = 0; b < 64; ++b) {
          pid_t ch = fresh();
          if (ch < 0) return;
          struct user_pt_regs F = R;
          if (k < 31) F.regs[k] ^= 1ull << b;
          else F.sp ^= 1ull << b;
          setRegs(ch, F);
          record(cGpr, runFree(ch, c.timeoutMs), s, R.pc, static_cast<u64>(k), static_cast<u64>(b));
        }
      }
    }
    if (c.cls[cFlags]) {
      for (int k = 0; k < 4; ++k) {
        pid_t ch = fresh();
        if (ch < 0) return;
        struct user_pt_regs F = R;
        F.pstate ^= 1ull << flagBit[k];
        setRegs(ch, F);
        record(cFlags, runFree(ch, c.timeoutMs), s, R.pc, static_cast<u64>(k), 0);
      }
    }
    if (c.cls[cFetch]) {
      for (int i = 0; i < 4; ++i) {
        for (int b = 0; b < 8; ++b) {
          pid_t ch = fresh();
          if (ch < 0) return;
          u64 orig = 0, aa = 0;
          if (!flipBit(ch, R.pc + static_cast<u64>(i), b, &orig, &aa)) {
            kill(ch, SIGKILL); int st; waitStop(ch, st); sh->err.store(2); return;
          }
          int st = 0, out;
          if (ptrace(PTRACE_SINGLESTEP, ch, nullptr, nullptr) != 0 || !waitStop(ch, st)) out = oHarness;
          else out = afterStep(ch, st);
          if (out < 0) {
            poke(ch, aa, orig);
            out = runFree(ch, c.timeoutMs);
          }
          record(cFetch, out, s, R.pc, static_cast<u64>(i), static_cast<u64>(b));
        }
      }
    }
    if (c.cls[cMem]) {
      for (size_t g = 0; g < regs.size(); ++g) {
        for (u64 i = 0; i < regs[g].len; ++i) {
          for (int b = 0; b < 8; ++b) {
            pid_t ch = fresh();
            if (ch < 0) return;
            if (!flipBit(ch, regs[g].addr + i, b)) {
              kill(ch, SIGKILL); int st; waitStop(ch, st); sh->err.store(3); return;
            }
            record(cMem, runFree(ch, c.timeoutMs), s, R.pc, g, i * 8 + static_cast<u64>(b));
          }
        }
      }
    }
  }

  void run() {
    pid_t m = spawnTraced(c);
    int st;
    for (u64 n = 0; n < t.pre; ++n) {
      if (!step1(m, st)) { sh->err.store(4); return; }
    }
    if (getPc(m) != c.start) { sh->err.store(5); return; }
    u64 at = 0;
    constexpr u64 kChunk = 4;
    for (;;) {
      u64 ck = sh->nextChunk.fetch_add(1);
      u64 lo = ck * kChunk;
      if (lo >= nsteps || sh->err.load()) break;
      u64 hi = std::min(nsteps, lo + kChunk);
      for (u64 s = lo; s < hi; ++s) {
        while (at < s) {
          if (!step1(m, st)) { sh->err.store(6); return; }
          ++at;
        }
        if (getPc(m) != t.pc[s]) { sh->err.store(7); return; }
        if (s % c.stride == 0) injectAll(m, s);
      }
    }
    kill(m, SIGKILL);
    waitStop(m, st);
  }
};

u64 num(const char* s) {
  char* e;
  u64 v = std::strtoull(s, &e, 0);
  if (*e) die("not a number: %s", s);
  return v;
}

int cmdInject(int argc, char** argv) {
  if (argc < 5) die("usage: inject BIN INPUT START_ADDR [options]");
  Cfg c;
  c.bin = argv[2];
  c.input = argv[3];
  c.start = num(argv[4]);
  for (int i = 5; i < argc; ++i) {
    std::string o = argv[i];
    auto need = [&](int k) { if (i + k >= argc) die("missing arguments to %s", o.c_str()); };
    if (o == "--expect") { need(1); std::string v = argv[++i]; c.expect = v == "accept" ? 1 : v == "reject" ? 0 : (die("--expect accept|reject"), 0); }
    else if (o == "--region") { need(3); c.regions.push_back({argv[i + 1], num(argv[i + 2]), num(argv[i + 3])}); i += 3; }
    else if (o == "--desc") { need(5); c.descs.push_back({argv[i + 1], num(argv[i + 2]), num(argv[i + 3]), num(argv[i + 4]), num(argv[i + 5])}); i += 5; }
    else if (o == "--buf-extra") { need(1); c.bufExtra = num(argv[++i]); }
    else if (o == "--jobs") { need(1); c.jobs = static_cast<int>(num(argv[++i])); }
    else if (o == "--timeout-ms") { need(1); c.timeoutMs = static_cast<int>(num(argv[++i])); }
    else if (o == "--stride") { need(1); c.stride = std::max<u64>(1, num(argv[++i])); }
    else if (o == "--max-steps") { need(1); c.maxSteps = num(argv[++i]); }
    else if (o == "--classes") {
      need(1);
      std::string v = argv[++i];
      for (auto& b : c.cls) b = false;
      size_t p = 0;
      while (p <= v.size()) {
        size_t q = v.find(',', p);
        if (q == std::string::npos) q = v.size();
        std::string w = v.substr(p, q - p);
        bool ok = false;
        for (int k = 0; k < kNCls; ++k) if (w == clsName[k]) { c.cls[k] = true; ok = true; }
        if (!ok) die("unknown class %s", w.c_str());
        p = q + 1;
      }
    } else die("unknown option %s", o.c_str());
  }
  if (c.jobs <= 0) c.jobs = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
  std::vector<Region> regs = c.regions;
  Trace t = golden(c, regs);
  u64 nsteps = std::min<u64>(t.pc.size(), c.maxSteps);
  std::printf("inject: bin=%s input=%s start=0x%" PRIx64 " expect=%s\n", c.bin, c.input, c.start,
              c.expect == 1 ? "accept" : c.expect == 0 ? "reject" : "unknown");
  std::printf("golden: exit=%d signal=%d steps_before_start=%" PRIu64 " steps_from_start=%zu injected_steps=%" PRIu64 "\n",
              t.exitCode, t.exitSig, t.pre, t.pc.size(), (nsteps + c.stride - 1) / c.stride);
  for (auto& r : regs) std::printf("region: %s addr=0x%" PRIx64 " len=%" PRIu64 "\n", r.name.c_str(), r.addr, r.len);
  if (c.expect == 0 && t.exitCode == 0) std::printf("golden: ACCEPTS an input the oracle rejects\n");
  std::fflush(stdout);

  size_t cnt = static_cast<size_t>(c.jobs) * kNCls * kNOut;
  size_t bytes = sizeof(Shared) + cnt * sizeof(u64);
  void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (mem == MAP_FAILED) die("mmap failed");
  std::memset(mem, 0, bytes);
  Shared* sh = new (mem) Shared;
  u64* counts = reinterpret_cast<u64*>(static_cast<char*>(mem) + sizeof(Shared));
  u64 t0 = nowNsInj();
  std::vector<pid_t> ws;
  for (int j = 0; j < c.jobs; ++j) {
    std::fflush(nullptr);
    pid_t p = fork();
    if (p < 0) die("fork failed");
    if (p == 0) {
      Worker w{c, t, regs, sh, counts + static_cast<size_t>(j) * kNCls * kNOut, nsteps};
      w.run();
      _exit(0);
    }
    ws.push_back(p);
  }
  for (pid_t p : ws) { int st; while (waitpid(p, &st, 0) < 0 && errno == EINTR) {} }
  double secs = static_cast<double>(nowNsInj() - t0) / 1e9;
  u64 tot[kNCls][kNOut] = {};
  for (int j = 0; j < c.jobs; ++j)
    for (int k = 0; k < kNCls; ++k)
      for (int o = 0; o < kNOut; ++o) tot[k][o] += counts[(static_cast<size_t>(j) * kNCls + k) * kNOut + o];
  u64 all = 0;
  for (int k = 0; k < kNCls; ++k) {
    u64 n = 0;
    for (int o = 0; o < kNOut; ++o) n += tot[k][o];
    all += n;
    std::printf("class=%s injections=%" PRIu64, clsName[k], n);
    for (int o = 0; o < kNOut; ++o) std::printf(" %s=%" PRIu64, outName[o], tot[k][o]);
    std::printf("\n");
  }
  int nshow = std::min(sh->nsdc.load(), kMaxSdc);
  for (int k = 0; k < nshow; ++k) {
    const SdcRec& r = sh->sdcs[k];
    std::string what;
    if (r.cls == cGpr) what = (r.a < 31 ? ("x" + std::to_string(r.a)) : "sp") + " bit " + std::to_string(r.b);
    else if (r.cls == cFlags) what = flagName[r.a];
    else if (r.cls == cFetch) what = "byte " + std::to_string(r.a) + " bit " + std::to_string(r.b);
    else what = regs[r.a].name + " byte " + std::to_string(r.b / 8) + " bit " + std::to_string(r.b % 8);
    std::printf("sdc: step=%" PRIu64 " pc=0x%" PRIx64 " class=%s fault=%s\n", r.step, r.pc, clsName[r.cls], what.c_str());
  }
  u64 harness = 0;
  for (int k = 0; k < kNCls; ++k) harness += tot[k][oHarness];
  std::printf("total_injections=%" PRIu64 " seconds=%.1f jobs=%d harness_errors=%" PRIu64 " worker_error=%d\n",
              all, secs, c.jobs, harness, sh->err.load());
  u64 sdc = sh->sdc.load();
  std::printf("SDC_accept_invalid=%" PRIu64 "\n", c.expect == 0 ? sdc : 0);
  if (sh->err.load() || harness || all == 0) return 2;
  return sdc ? 1 : 0;
}

}  // namespace
#endif

int main(int argc, char** argv) {
  std::string cmd = argc > 1 ? argv[1] : "";
  if (cmd == "gen" && argc == 5) return cmdGen(std::strtoull(argv[2], nullptr, 10), std::strtoull(argv[3], nullptr, 10), argv[4]);
  if (cmd == "examples" && argc == 3) return cmdExamples(argv[2]);
  if (cmd == "vectors" && argc == 3) return cmdVectors(argv[2]);
  if (cmd == "run" && (argc == 4 || argc == 5)) return cmdRun(argv[2], argv[3], argc == 5 ? std::atoi(argv[4]) : 5000);
  if (cmd == "exec1" && (argc == 4 || argc == 5)) return cmdExec1(argv[2], argv[3], argc == 5 ? std::atoi(argv[4]) : 5000);
  if (cmd == "syms" && argc == 4) return cmdSyms(argv[2], argv[3]);
  if (cmd == "patch") return cmdPatch(argc, argv);
  if (cmd == "inject") {
#ifdef SPR1_INJECT
    return cmdInject(argc, argv);
#else
    die("inject needs Linux on aarch64");
#endif
  }
  std::fprintf(stderr,
               "usage: spr1_check_arm64_fuzz gen N SEED DIR | examples DIR | vectors DIR | run BIN LISTFILE [TIMEOUT_MS]\n"
               "       | exec1 BIN INPUT [TIMEOUT_MS] | syms LISTING BIN | patch IN OUT ADDR HEX [ADDR HEX]...\n"
               "       | inject BIN INPUT START_ADDR [--expect accept|reject] [--region NAME ADDR LEN]\n"
               "         [--desc NAME ADDR LEN PTROFF LENOFF] [--buf-extra K] [--jobs N] [--timeout-ms T] [--classes LIST]\n"
               "         [--stride K] [--max-steps M]\n");
  return 2;
}
