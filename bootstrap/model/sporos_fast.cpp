// sporos_fast: a fast Sporos interpreter in C++23, tested against the Lean
// reference interpreter (`formal/Sporos/Main.lean`, `lake build sporos`).
//
//   sporos_fast PROGRAM.spb [MEMORY-BYTES] < input > output
//
// Same bytecode (formal/Sporos/Bytecode.lean), same checker
// (formal/Sporos/Check.lean), same semantics (formal/Sporos/Step.lean), same
// exit statuses and error names.
//
// Provenance. The reference runs with `noPhys`, under which the only way a
// program can observe a provenance is its length: `provget` pushes
// `a.p.length`, and `confget`/`demand` see a confidence that is constantly 0. So
// a value here is its word and the length of its provenance list modulo 2^64,
// which is exactly what `provget` reports after `wrap`. Every list
// concatenation of the reference becomes an addition of lengths. This is a
// claim about the reference, checked by the differential harness
// (sporos_diff.sh), not yet a theorem.
//
// Not trusted: the Lean reference is. This file is judged only by agreement.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sys/mman.h>

using u64 = std::uint64_t;
using i64 = std::int64_t;

namespace {

// ---------------------------------------------------------------- syntax ----

enum class Prim : std::uint8_t {
  dup, drop, swap, over, rot, nip, tuck,
  add, sub, mul, divu, modu, divs, mods, neg,
  band, bor, bxor, bnot, shl, shr, sar,
  eq, ne, ltu, lts, leu, les,
  ld8, st8, ld16, st16, ld32, st32, ld64, st64,
  copy, fill, mcmp,
  alloc, mark, release,
  streq, strhash,
  tnew, tput, tget,
  getb, putb, putn, exit_,
  clock, core, sensor,
  provget, confget, demand,
  count_
};

constexpr int kPrims = static_cast<int>(Prim::count_);

struct Eff { int in, out; };

constexpr Eff eff(Prim p) {
  switch (p) {
    case Prim::dup: return {1, 2}; case Prim::drop: return {1, 0}; case Prim::swap: return {2, 2};
    case Prim::over: return {2, 3}; case Prim::rot: return {3, 3}; case Prim::nip: return {2, 1};
    case Prim::tuck: return {2, 3};
    case Prim::add: case Prim::sub: case Prim::mul: case Prim::divu: case Prim::modu:
    case Prim::divs: case Prim::mods: return {2, 1};
    case Prim::neg: return {1, 1};
    case Prim::band: case Prim::bor: case Prim::bxor: case Prim::shl: case Prim::shr:
    case Prim::sar: return {2, 1};
    case Prim::bnot: return {1, 1};
    case Prim::eq: case Prim::ne: case Prim::ltu: case Prim::lts: case Prim::leu:
    case Prim::les: return {2, 1};
    case Prim::ld8: case Prim::ld16: case Prim::ld32: case Prim::ld64: return {1, 1};
    case Prim::st8: case Prim::st16: case Prim::st32: case Prim::st64: return {2, 0};
    case Prim::copy: return {3, 0}; case Prim::fill: return {3, 0}; case Prim::mcmp: return {3, 1};
    case Prim::alloc: return {1, 1}; case Prim::mark: return {0, 1}; case Prim::release: return {1, 0};
    case Prim::streq: return {4, 1}; case Prim::strhash: return {2, 1};
    case Prim::tnew: return {0, 1}; case Prim::tput: return {3, 0}; case Prim::tget: return {2, 2};
    case Prim::getb: return {0, 1}; case Prim::putb: return {1, 0}; case Prim::putn: return {1, 0};
    case Prim::exit_: return {1, 0};
    case Prim::clock: return {0, 1}; case Prim::core: return {0, 1}; case Prim::sensor: return {1, 1};
    case Prim::provget: return {1, 2}; case Prim::confget: return {1, 2}; case Prim::demand: return {2, 1};
    default: return {0, 0};
  }
}

enum class Op : std::uint8_t { prim, lit, str, pick, tor, fromr, rfetch, locget, locset, jmp, jz, jnz, call, ret };

struct Instr { Op op; Prim p; u64 n; };
struct Height { u64 d, a; bool operator==(const Height&) const = default; };
struct Word {
  u64 ins, outs, locals;
  std::vector<Instr> code;
  std::vector<std::pair<u64, Height>> ann;
  std::optional<Height> lookup(u64 i) const {
    for (auto& [k, h] : ann) if (k == i) return h;
    return std::nullopt;
  }
};
struct Prog { std::vector<Word> words; std::vector<std::vector<std::uint8_t>> data; u64 main; };

// -------------------------------------------------------------- decoding ----

struct Reader {
  const std::vector<std::uint8_t>& b; std::size_t i = 0;
  bool le(int k, u64& out) {
    if (b.size() - i < static_cast<std::size_t>(k)) return false;
    // the reference reads unbounded naturals; a u32 or u64 field always fits
    out = 0;
    for (int j = 0; j < k; ++j) out |= static_cast<u64>(b[i + j]) << (8 * j);
    i += k; return true;
  }
};

bool decInstr(Reader& r, Instr& out) {
  if (r.i >= r.b.size()) return false;
  std::uint8_t op = r.b[r.i++];
  if (op < kPrims) { out = {Op::prim, static_cast<Prim>(op), 0}; return true; }
  auto imm = [&](Op o, int k) { out = {o, Prim::dup, 0}; return r.le(k, out.n); };
  switch (op) {
    case 0x80: return imm(Op::lit, 8);
    case 0x81: return imm(Op::str, 4);
    case 0x82: return imm(Op::pick, 4);
    case 0x83: out = {Op::tor, Prim::dup, 0}; return true;
    case 0x84: out = {Op::fromr, Prim::dup, 0}; return true;
    case 0x85: out = {Op::rfetch, Prim::dup, 0}; return true;
    case 0x86: return imm(Op::locget, 4);
    case 0x87: return imm(Op::locset, 4);
    case 0x88: return imm(Op::jmp, 4);
    case 0x89: return imm(Op::jz, 4);
    case 0x8a: return imm(Op::jnz, 4);
    case 0x8b: return imm(Op::call, 4);
    case 0x8c: out = {Op::ret, Prim::dup, 0}; return true;
    default: return false;
  }
}

std::optional<Prog> decode(const std::vector<std::uint8_t>& b) {
  if (b.size() < 4 || b[0] != 0x53 || b[1] != 0x50 || b[2] != 0x52 || b[3] != 0x31) return std::nullopt;
  Reader r{b, 4};
  Prog p; u64 nw, nd;
  if (!r.le(4, p.main) || !r.le(4, nw) || !r.le(4, nd)) return std::nullopt;
  for (u64 k = 0; k < nd; ++k) {
    u64 len; if (!r.le(4, len) || b.size() - r.i < len) return std::nullopt;
    p.data.emplace_back(b.begin() + r.i, b.begin() + r.i + len); r.i += len;
  }
  for (u64 k = 0; k < nw; ++k) {
    Word w; u64 nc, na;
    if (!r.le(4, w.ins) || !r.le(4, w.outs) || !r.le(4, w.locals) || !r.le(4, nc) || !r.le(4, na)) return std::nullopt;
    for (u64 j = 0; j < nc; ++j) { Instr in; if (!decInstr(r, in)) return std::nullopt; w.code.push_back(in); }
    for (u64 j = 0; j < na; ++j) {
      u64 i, d, a; if (!r.le(4, i) || !r.le(4, d) || !r.le(4, a)) return std::nullopt;
      w.ann.push_back({i, {d, a}});
    }
    p.words.push_back(std::move(w));
  }
  if (r.i != b.size()) return std::nullopt;
  return p;
}

// --------------------------------------------------------------- checker ----

// hstep: nullopt = ill-typed; Some(nullopt) = does not fall through.
std::optional<std::optional<Height>> hstep(const Prog& pr, const Word& wd, Height h, const Instr& in) {
  using R = std::optional<std::optional<Height>>;
  std::optional<Height> none;
  switch (in.op) {
    case Op::prim: {
      auto e = eff(in.p);
      if (static_cast<u64>(e.in) > h.d) return std::nullopt;
      if (in.p == Prim::exit_) return R{none};
      return R{Height{h.d - e.in + e.out, h.a}};
    }
    case Op::lit: return R{Height{h.d + 1, h.a}};
    case Op::str: return R{Height{h.d + 2, h.a}};
    case Op::pick: return in.n < h.d ? R{Height{h.d + 1, h.a}} : std::nullopt;
    case Op::tor: return h.d >= 1 ? R{Height{h.d - 1, h.a + 1}} : std::nullopt;
    case Op::fromr: return h.a >= 1 ? R{Height{h.d + 1, h.a - 1}} : std::nullopt;
    case Op::rfetch: return h.a >= 1 ? R{Height{h.d + 1, h.a}} : std::nullopt;
    case Op::locget: return in.n < wd.locals ? R{Height{h.d + 1, h.a}} : std::nullopt;
    case Op::locset: return (h.d >= 1 && in.n < wd.locals) ? R{Height{h.d - 1, h.a}} : std::nullopt;
    case Op::jmp: { auto t = wd.lookup(in.n); return (t && *t == h) ? R{none} : std::nullopt; }
    case Op::jz: case Op::jnz: {
      if (h.d < 1) return std::nullopt;
      auto t = wd.lookup(in.n); Height h2{h.d - 1, h.a};
      return (t && *t == h2) ? R{h2} : std::nullopt;
    }
    case Op::call: {
      if (in.n >= pr.words.size()) return std::nullopt;
      auto& c = pr.words[in.n];
      return c.ins <= h.d ? R{Height{h.d - c.ins + c.outs, h.a}} : std::nullopt;
    }
    case Op::ret: return (h == Height{wd.outs, 0}) ? R{none} : std::nullopt;
  }
  return std::nullopt;
}

std::vector<Height> infer(const Prog& pr, const Word& wd) {
  std::vector<Height> H; Height cur{wd.ins, 0};
  for (u64 i = 0; i < wd.code.size(); ++i) {
    Height h = wd.lookup(i).value_or(cur);
    auto r = hstep(pr, wd, h, wd.code[i]);
    Height nxt = (r && *r) ? **r : wd.lookup(i + 1).value_or(Height{0, 0});
    H.push_back(h); cur = nxt;
  }
  return H;
}

bool checkWord(const Prog& pr, const Word& wd, const std::vector<Height>& H) {
  if (H.size() != wd.code.size() || wd.code.empty() || !(H[0] == Height{wd.ins, 0})) return false;
  for (u64 i = 0; i < wd.code.size(); ++i) {
    auto r = hstep(pr, wd, H[i], wd.code[i]);
    if (!r) return false;
    if (*r) { if (!(i + 1 < wd.code.size() && H[i + 1] == **r)) return false; }
  }
  for (auto& [t, a] : wd.ann) if (!(t < wd.code.size() && H[t] == a)) return false;
  return true;
}

bool checkProg(const Prog& pr) {
  for (auto& wd : pr.words) if (!checkWord(pr, wd, infer(pr, wd))) return false;
  return pr.main < pr.words.size() && pr.words[pr.main].ins == 0;
}

// ------------------------------------------------------------- semantics ----

struct Val { u64 w; u64 p; };  // word, provenance length mod 2^64

struct Frame { u64 word, pc; std::vector<Val> locs; };

enum class Err { underflow, auxUnderflow, badLocal, badJump, badWord, noFrame, bounds, divZero, badTable, demand };

const char* errName(Err e) {
  switch (e) {
    case Err::underflow: return "underflow"; case Err::auxUnderflow: return "aux-underflow";
    case Err::badLocal: return "bad-local"; case Err::badJump: return "bad-jump";
    case Err::badWord: return "bad-word"; case Err::noFrame: return "no-frame";
    case Err::bounds: return "bounds"; case Err::divZero: return "div-zero";
    case Err::badTable: return "bad-table"; case Err::demand: return "demand";
  }
  return "?";
}

struct Machine {
  const Prog& pr;
  std::vector<Val> ds, aux;      // top at back
  std::vector<Frame> rs;          // current frame at back
  std::uint8_t* memw = nullptr;   // bytes, lazily zero (mmap)
  u64* memp = nullptr;            // provenance length per byte
  u64 msize, hp;
  std::vector<std::vector<std::pair<u64, Val>>> tabs;  // newest entry at back
  std::vector<std::uint8_t> inp; std::size_t ipos = 0;
  std::vector<std::uint8_t> out;
  u64 pc = 0;                     // length of the context label

  // addresses below msize live in the arrays; data may extend beyond msize, and
  // bytes there can never be read (every access is bounds-checked against msize)
  Machine(const Prog& p, u64 ms, std::vector<std::uint8_t> in) : pr(p), msize(ms), inp(std::move(in)) {
    std::size_t bytes = ms ? ms : 1;
    memw = static_cast<std::uint8_t*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    memp = static_cast<u64*>(mmap(nullptr, bytes * 8, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    if (memw == MAP_FAILED || memp == MAP_FAILED) { std::fputs("sporos_fast: out of memory\n", stderr); std::exit(2); }
    u64 a = 0;
    for (auto& d : pr.data) for (auto b : d) { if (a < ms) memw[a] = b; ++a; }
    hp = a;
    u64 locals = pr.main < pr.words.size() ? pr.words[pr.main].locals : 0;
    rs.push_back({pr.main, 0, std::vector<Val>(locals, Val{0, 0})});
  }

  bool inb(u64 a, u64 n) const { return a <= msize && n <= msize - a; }  // a + n ≤ msize, no overflow
};

static inline i64 sgn(u64 x) { return static_cast<i64>(x); }

enum class Out { next, halt, fail };
struct StepRes { Out k; int code = 0; Err e = Err::underflow; };

StepRes step(Machine& m) {
  if (m.rs.empty()) return {Out::fail, 0, Err::noFrame};
  Frame& f = m.rs.back();
  if (f.word >= m.pr.words.size()) return {Out::fail, 0, Err::badJump};
  const Word& wd = m.pr.words[f.word];
  if (f.pc >= wd.code.size()) return {Out::fail, 0, Err::badJump};
  const Instr& in = wd.code[f.pc];
  auto tag = [&](Val v) { return Val{v.w, v.p + m.pc + 1}; };
  auto& ds = m.ds;
  switch (in.op) {
    case Op::prim: {
      auto e = eff(in.p);
      if (ds.size() < static_cast<std::size_t>(e.in)) return {Out::fail, 0, Err::underflow};
      // arguments, top first, as in the reference
      Val a0{}, a1{}, a2{}, a3{};
      std::size_t n = ds.size();
      if (e.in > 0) a0 = ds[n - 1];
      if (e.in > 1) a1 = ds[n - 2];
      if (e.in > 2) a2 = ds[n - 3];
      if (e.in > 3) a3 = ds[n - 4];
      u64 qa = 0;
      if (e.in > 0) qa += a0.p;
      if (e.in > 1) qa += a1.p;
      if (e.in > 2) qa += a2.p;
      if (e.in > 3) qa += a3.p;
      const u64 q = qa + m.pc + 1;
      auto mk = [&](u64 w) { return Val{w, q}; };
      std::vector<Val> res;  // top first
      auto fail = [](Err er) { return StepRes{Out::fail, 0, er}; };
      auto ld = [&](u64 addr, int k, Val& out) -> bool {
        if (!m.inb(addr, k)) return false;
        u64 v = 0, pm = 0;
        for (int i = k - 1; i >= 0; --i) v = v * 256 + m.memw[addr + i];
        for (int i = 0; i < k; ++i) pm += m.memp[addr + i];
        out = Val{v, q + pm}; return true;
      };
      auto st = [&](u64 addr, int k, u64 v) -> bool {
        if (!m.inb(addr, k)) return false;
        for (int i = 0; i < k; ++i) { m.memw[addr + i] = static_cast<std::uint8_t>(v >> (8 * i)); m.memp[addr + i] = q; }
        return true;
      };
      switch (in.p) {
        // stack (b = a0 is the top, a = a1 below it)
        case Prim::dup: res = {tag(a0), tag(a0)}; break;
        case Prim::drop: break;
        case Prim::swap: res = {tag(a1), tag(a0)}; break;
        case Prim::over: res = {tag(a1), tag(a0), tag(a1)}; break;
        case Prim::rot: res = {tag(a2), tag(a0), tag(a1)}; break;
        case Prim::nip: res = {tag(a0)}; break;
        case Prim::tuck: res = {tag(a0), tag(a1), tag(a0)}; break;
        case Prim::add: res = {mk(a1.w + a0.w)}; break;
        case Prim::sub: res = {mk(a1.w - a0.w)}; break;
        case Prim::mul: res = {mk(a1.w * a0.w)}; break;
        case Prim::divu: if (a0.w == 0) return fail(Err::divZero); res = {mk(a1.w / a0.w)}; break;
        case Prim::modu: if (a0.w == 0) return fail(Err::divZero); res = {mk(a1.w % a0.w)}; break;
        case Prim::divs:
          if (a0.w == 0) return fail(Err::divZero);
          res = {mk((sgn(a1.w) == INT64_MIN && sgn(a0.w) == -1) ? a1.w : static_cast<u64>(sgn(a1.w) / sgn(a0.w)))};
          break;
        case Prim::mods:
          if (a0.w == 0) return fail(Err::divZero);
          res = {mk((sgn(a0.w) == -1) ? 0 : static_cast<u64>(sgn(a1.w) % sgn(a0.w)))};
          break;
        case Prim::neg: res = {mk(0 - a0.w)}; break;
        case Prim::band: res = {mk(a1.w & a0.w)}; break;
        case Prim::bor: res = {mk(a1.w | a0.w)}; break;
        case Prim::bxor: res = {mk(a1.w ^ a0.w)}; break;
        case Prim::bnot: res = {mk(~a0.w)}; break;
        case Prim::shl: res = {mk(a1.w << (a0.w % 64))}; break;
        case Prim::shr: res = {mk(a1.w >> (a0.w % 64))}; break;
        case Prim::sar: res = {mk(static_cast<u64>(sgn(a1.w) >> (a0.w % 64)))}; break;
        case Prim::eq: res = {mk(a1.w == a0.w)}; break;
        case Prim::ne: res = {mk(a1.w != a0.w)}; break;
        case Prim::ltu: res = {mk(a1.w < a0.w)}; break;
        case Prim::lts: res = {mk(sgn(a1.w) < sgn(a0.w))}; break;
        case Prim::leu: res = {mk(a1.w <= a0.w)}; break;
        case Prim::les: res = {mk(sgn(a1.w) <= sgn(a0.w))}; break;
        case Prim::ld8: case Prim::ld16: case Prim::ld32: case Prim::ld64: {
          int k = in.p == Prim::ld8 ? 1 : in.p == Prim::ld16 ? 2 : in.p == Prim::ld32 ? 4 : 8;
          Val v; if (!ld(a0.w, k, v)) return fail(Err::bounds); res = {v}; break;
        }
        case Prim::st8: case Prim::st16: case Prim::st32: case Prim::st64: {
          // ( v a -- ): a is the top
          int k = in.p == Prim::st8 ? 1 : in.p == Prim::st16 ? 2 : in.p == Prim::st32 ? 4 : 8;
          if (!st(a0.w, k, a1.w)) return fail(Err::bounds); break;
        }
        case Prim::copy: {  // ( src dst n -- ), memmove
          u64 nn = a0.w, dst = a1.w, src = a2.w;
          if (!(m.inb(src, nn) && m.inb(dst, nn))) return fail(Err::bounds);
          std::vector<std::uint8_t> tw(m.memw + src, m.memw + src + nn);
          std::vector<u64> tp(m.memp + src, m.memp + src + nn);
          for (u64 i = 0; i < nn; ++i) { m.memw[dst + i] = tw[i]; m.memp[dst + i] = tp[i] + q; }
          break;
        }
        case Prim::fill: {  // ( dst n b -- )
          u64 b = a0.w, nn = a1.w, dst = a2.w;
          if (!m.inb(dst, nn)) return fail(Err::bounds);
          for (u64 i = 0; i < nn; ++i) { m.memw[dst + i] = static_cast<std::uint8_t>(b); m.memp[dst + i] = q; }
          break;
        }
        case Prim::mcmp: {  // ( x y n -- f )
          u64 nn = a0.w, y = a1.w, x = a2.w;
          if (!(m.inb(x, nn) && m.inb(y, nn))) return fail(Err::bounds);
          bool eq = true; u64 pm = 0;
          for (u64 i = 0; i < nn; ++i) { eq = eq && m.memw[x + i] == m.memw[y + i]; pm += m.memp[x + i] + m.memp[y + i]; }
          res = {Val{eq, q + pm}}; break;
        }
        case Prim::alloc:
          if (!m.inb(m.hp, a0.w)) return fail(Err::bounds);
          res = {mk(m.hp)}; m.hp += a0.w; break;
        case Prim::mark: res = {mk(m.hp)}; break;
        case Prim::release: if (a0.w > m.hp) return fail(Err::bounds); m.hp = a0.w; break;
        case Prim::streq: {  // ( a1 l1 a2 l2 -- f )
          u64 l2 = a0.w, ad2 = a1.w, l1 = a2.w, ad1 = a3.w;
          if (!(m.inb(ad1, l1) && m.inb(ad2, l2))) return fail(Err::bounds);
          bool eq = l1 == l2; u64 pm = 0;
          if (l1 == l2) for (u64 i = 0; i < l1; ++i) { eq = eq && m.memw[ad1 + i] == m.memw[ad2 + i]; pm += m.memp[ad1 + i] + m.memp[ad2 + i]; }
          res = {Val{eq, q + pm}}; break;
        }
        case Prim::strhash: {  // ( a l -- h )
          u64 l = a0.w, ad = a1.w;
          if (!m.inb(ad, l)) return fail(Err::bounds);
          u64 h = 14695981039346656037ull, pm = 0;
          for (u64 i = 0; i < l; ++i) { h = (h ^ m.memw[ad + i]) * 1099511628211ull; pm += m.memp[ad + i]; }
          res = {Val{h, q + pm}}; break;
        }
        case Prim::tnew: res = {mk(m.tabs.size())}; m.tabs.emplace_back(); break;
        case Prim::tput: {  // ( t k v -- )
          Val v = a0; u64 k = a1.w, tb = a2.w;
          if (tb >= m.tabs.size()) return fail(Err::badTable);
          m.tabs[tb].push_back({k, Val{v.w, v.p + q}}); break;
        }
        case Prim::tget: {  // ( t k -- v found )
          u64 k = a0.w, tb = a1.w;
          if (tb >= m.tabs.size()) return fail(Err::badTable);
          auto& t = m.tabs[tb]; bool found = false;
          for (auto it = t.rbegin(); it != t.rend(); ++it)
            if (it->first == k) { res = {mk(1), Val{it->second.w, it->second.p + q}}; found = true; break; }
          if (!found) res = {mk(0), mk(0)};
          break;
        }
        case Prim::getb:
          if (m.ipos >= m.inp.size()) res = {mk(~0ull)};
          else res = {Val{m.inp[m.ipos++], m.pc + 1}};
          break;
        case Prim::putb: m.out.push_back(static_cast<std::uint8_t>(a0.w)); break;
        case Prim::putn: for (int i = 0; i < 8; ++i) m.out.push_back(static_cast<std::uint8_t>(a0.w >> (8 * i))); break;
        case Prim::exit_: return {Out::halt, static_cast<int>(a0.w % 256)};
        case Prim::clock: case Prim::core: res = {mk(0)}; break;
        case Prim::sensor: res = {mk(0)}; break;
        case Prim::provget: res = {mk(a0.p), tag(a0)}; break;
        case Prim::confget: res = {mk(0), tag(a0)}; break;
        case Prim::demand:  // ( a th -- a ): confidence is 0 under noPhys
          if (0 < a0.w) return fail(Err::demand);
          res = {tag(a1)}; break;
        default: return fail(Err::underflow);
      }
      ds.resize(n - e.in);
      for (auto it = res.rbegin(); it != res.rend(); ++it) ds.push_back(*it);
      f.pc += 1;
      return {Out::next};
    }
    case Op::lit: ds.push_back({in.n, m.pc + 1}); f.pc += 1; return {Out::next};
    case Op::str: {
      if (in.n >= m.pr.data.size()) return {Out::fail, 0, Err::bounds};
      u64 addr = 0; for (u64 i = 0; i < in.n; ++i) addr += m.pr.data[i].size();
      ds.push_back({addr, m.pc + 1}); ds.push_back({m.pr.data[in.n].size(), m.pc + 1});
      f.pc += 1; return {Out::next};
    }
    case Op::pick:
      if (in.n >= ds.size()) return {Out::fail, 0, Err::underflow};
      ds.push_back(tag(ds[ds.size() - 1 - in.n])); f.pc += 1; return {Out::next};
    case Op::tor:
      if (ds.empty()) return {Out::fail, 0, Err::underflow};
      m.aux.push_back(tag(ds.back())); ds.pop_back(); f.pc += 1; return {Out::next};
    case Op::fromr:
      if (m.aux.empty()) return {Out::fail, 0, Err::auxUnderflow};
      ds.push_back(tag(m.aux.back())); m.aux.pop_back(); f.pc += 1; return {Out::next};
    case Op::rfetch:
      if (m.aux.empty()) return {Out::fail, 0, Err::auxUnderflow};
      ds.push_back(tag(m.aux.back())); f.pc += 1; return {Out::next};
    case Op::locget:
      if (in.n >= f.locs.size()) return {Out::fail, 0, Err::badLocal};
      ds.push_back(tag(f.locs[in.n])); f.pc += 1; return {Out::next};
    case Op::locset: {
      if (ds.empty()) return {Out::fail, 0, Err::underflow};
      if (in.n >= f.locs.size()) return {Out::fail, 0, Err::badLocal};
      f.locs[in.n] = tag(ds.back()); ds.pop_back(); f.pc += 1; return {Out::next};
    }
    case Op::jmp: f.pc = in.n; return {Out::next};
    case Op::jz: case Op::jnz: {
      if (ds.empty()) return {Out::fail, 0, Err::underflow};
      Val c = ds.back(); ds.pop_back();
      m.pc = m.pc + c.p + 1;
      bool zero = c.w == 0;
      f.pc = (in.op == Op::jz) == zero ? in.n : f.pc + 1;
      return {Out::next};
    }
    case Op::call: {
      if (in.n >= m.pr.words.size()) return {Out::fail, 0, Err::badWord};
      const Word& c = m.pr.words[in.n];
      if (ds.size() < c.ins) return {Out::fail, 0, Err::underflow};
      f.pc += 1;
      Frame callee{in.n, 0, std::vector<Val>(c.locals, Val{0, m.pc + 1})};
      m.rs.push_back(std::move(callee));
      return {Out::next};
    }
    case Op::ret:
      if (m.rs.size() <= 1) return {Out::halt, 0};
      m.rs.pop_back(); return {Out::next};
  }
  return {Out::fail, 0, Err::underflow};
}

std::vector<std::uint8_t> readAll(std::FILE* fp) {
  std::vector<std::uint8_t> v; std::uint8_t buf[1 << 16]; std::size_t k;
  while ((k = std::fread(buf, 1, sizeof buf, fp)) > 0) v.insert(v.end(), buf, buf + k);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fputs("usage: sporos_fast PROGRAM.spb [MEMORY-BYTES] < input > output\n", stderr); return 1; }
  std::FILE* fp = std::fopen(argv[1], "rb");
  if (!fp) { std::fprintf(stderr, "sporos_fast: %s: cannot open\n", argv[1]); return 1; }
  auto bytes = readAll(fp); std::fclose(fp);
  u64 msize = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : (1ull << 24);
  auto pr = decode(bytes);
  if (!pr) { std::fprintf(stderr, "sporos_fast: %s: not a Sporos bytecode file\n", argv[1]); return 1; }
  if (!checkProg(*pr)) { std::fprintf(stderr, "sporos_fast: %s: refused by the checker\n", argv[1]); return 1; }
  Machine m(*pr, msize, readAll(stdin));
  for (;;) {
    StepRes r = step(m);
    if (r.k == Out::next) continue;
    if (r.k == Out::halt) {
      std::fwrite(m.out.data(), 1, m.out.size(), stdout); std::fflush(stdout);
      return r.code;
    }
    std::fprintf(stderr, "sporos: run failed: %s\n", errName(r.e));
    return 1;
  }
}
