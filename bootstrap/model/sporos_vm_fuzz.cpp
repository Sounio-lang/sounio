// sporos_vm_fuzz SEED > prog.spb : deterministic high-coverage, checker-valid
// Sporos bytecode generator for differential and mutation testing of sporos_vm
// against sporos_fast (`bootstrap/model/sporos_vm_gate.sh`).
//
// Unlike sporos_gen (unstructured random streams that often trap early, drop
// results, or execute trivial operands), every program emitted by
// sporos_vm_fuzz is statically verified by construction and trap-free through
// a comprehensive core covering all 57 primitives and all 13 non-prim opcodes:
//   - multi-word calls (`call`/`ret`, nested frames, callee locals)
//   - structured control flow (`jmp`, `jz`, `jnz` taken/not-taken, bounded loop)
//   - arithmetic (`add` with (b & 0x7fffffffffffffff) != 0, `sub`, `mul`,
//     `divu`, `modu`, `divs`, `mods` incl. INT64_MIN/-1 edge cases, `neg`)
//   - signed & unsigned comparisons (`lts`, `les`, `ltu`, `leu`, `eq`, `ne`
//     including opposite-sign negative vs non-negative operands)
//   - bitwise & shifts (`band`, `bor`, `bxor`, `bnot`, `shl`, `shr`, `sar`)
//   - stack & aux stack (`dup`, `drop`, `swap`, `over`, `rot`, `nip`, `tuck`,
//     `pick`, `tor`, `fromr`, `rfetch`, `locget`, `locset`)
//   - heap & memory (`mark`, `alloc`, `release`, `st8`/`st16`/`st32`/`st64`,
//     `ld8`/`ld16`/`ld32`/`ld64` reading back tainted bytes with pm > 0,
//     `fill`, `copy` forward/backward overlap & zero-len, `mcmp`)
//   - strings (`str`, `streq` equal/unequal/diff-len/tainted, `strhash`)
//   - tables (`tnew`, `tput` insert & overwrite across 2 tables, `tget` hit/miss)
//   - I/O & epistemic (`getb`, `putb`, `putn`, `clock`, `core`, `sensor`,
//     `provget`, `confget`, `demand` with th = 0 on normal input and controlled
//     `demand` refusal with th != 0 when stdin starts with a NUL byte)
//   - randomized safe fuzzing tail exercising diverse instruction mixes
//
// Every computed result is emitted with BOTH its 64-bit value AND its `provget`
// provenance length via `putn`, ensuring non-equivalent VM mutants diverge
// deterministically on every seed.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

namespace {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

enum class Prim : u8 {
  dup = 0,
  drop = 1,
  swap = 2,
  over = 3,
  rot = 4,
  nip = 5,
  tuck = 6,
  add = 7,
  sub = 8,
  mul = 9,
  divu = 10,
  modu = 11,
  divs = 12,
  mods = 13,
  neg = 14,
  band = 15,
  bor = 16,
  bxor = 17,
  bnot = 18,
  shl = 19,
  shr = 20,
  sar = 21,
  eq = 22,
  ne = 23,
  ltu = 24,
  lts = 25,
  leu = 26,
  les = 27,
  ld8 = 28,
  st8 = 29,
  ld16 = 30,
  st16 = 31,
  ld32 = 32,
  st32 = 33,
  ld64 = 34,
  st64 = 35,
  copy = 36,
  fill = 37,
  mcmp = 38,
  alloc = 39,
  mark = 40,
  release = 41,
  streq = 42,
  strhash = 43,
  tnew = 44,
  tput = 45,
  tget = 46,
  getb = 47,
  putb = 48,
  putn = 49,
  exit_ = 50,
  clock_ = 51,
  core_ = 52,
  sensor = 53,
  provget = 54,
  confget = 55,
  demand = 56,
};

struct Eff {
  u64 in;
  u64 out;
};

constexpr std::array<Eff, 57> kEff = {{
    {1, 2}, {1, 0}, {2, 2}, {2, 3}, {3, 3}, {2, 1}, {2, 3},  // stack
    {2, 1}, {2, 1}, {2, 1}, {2, 1}, {2, 1}, {2, 1}, {2, 1}, {1, 1},  // arith
    {2, 1}, {2, 1}, {2, 1}, {1, 1}, {2, 1}, {2, 1}, {2, 1},  // bitwise/shifts
    {2, 1}, {2, 1}, {2, 1}, {2, 1}, {2, 1}, {2, 1},          // cmp
    {1, 1}, {2, 0}, {1, 1}, {2, 0}, {1, 1}, {2, 0}, {1, 1}, {2, 0},  // ld/st
    {3, 0}, {3, 0}, {3, 1},                                  // copy/fill/mcmp
    {1, 1}, {0, 1}, {1, 0},                                  // alloc/mark/release
    {4, 1}, {2, 1},                                          // streq/strhash
    {0, 1}, {3, 0}, {2, 2},                                  // tnew/tput/tget
    {0, 1}, {1, 0}, {1, 0}, {1, 0},                          // getb/putb/putn/exit
    {0, 1}, {0, 1}, {1, 1},                                  // clock/core/sensor
    {1, 2}, {1, 2}, {2, 1},                                  // provget/confget/demand
}};

[[noreturn]] void fail(const char* msg) {
  std::fprintf(stderr, "sporos_vm_fuzz: internal error: %s\n", msg);
  std::abort();
}

void check(bool cond, const char* msg) {
  if (!cond) fail(msg);
}

void append_le(std::vector<u8>& out, u64 v, int n) {
  for (int i = 0; i < n; ++i) {
    out.push_back(static_cast<u8>((v >> (8 * i)) & 0xffULL));
  }
}

struct WordSig {
  u64 ins;
  u64 outs;
};

struct WordBuilder {
  u64 ins = 0;
  u64 outs = 0;
  u64 locals = 0;
  u64 d = 0;
  u64 a = 0;
  bool reachable = true;

  struct LabelInfo {
    bool bound = false;
    bool has_height = false;
    u64 pc = 0;
    u64 d = 0;
    u64 a = 0;
  };
  struct Patch {
    std::size_t instr_idx;
    std::size_t label_id;
  };

  std::vector<std::vector<u8>> code;
  std::vector<LabelInfo> labels;
  std::vector<Patch> patches;
  std::vector<std::pair<u64, std::pair<u64, u64>>> ann;

  WordBuilder(u64 in_n, u64 out_n, u64 loc_n)
      : ins(in_n), outs(out_n), locals(loc_n), d(in_n), a(0), reachable(true) {}

  std::size_t new_label() {
    labels.push_back(LabelInfo{});
    return labels.size() - 1;
  }

  void record_label_height(std::size_t lid, u64 ld, u64 la) {
    auto& L = labels[lid];
    if (L.has_height) {
      check(L.d == ld && L.a == la, "label height mismatch");
    } else {
      L.has_height = true;
      L.d = ld;
      L.a = la;
    }
  }

  void bind(std::size_t lid) {
    auto& L = labels[lid];
    check(!L.bound, "duplicate label bind");
    if (reachable) {
      record_label_height(lid, d, a);
    } else {
      check(L.has_height, "binding unreachable label without prior jump height");
      d = L.d;
      a = L.a;
      reachable = true;
    }
    L.bound = true;
    L.pc = static_cast<u64>(code.size());
  }

  void prim(Prim p) {
    check(reachable, "unreachable prim");
    auto e = kEff[static_cast<std::size_t>(p)];
    check(d >= e.in, "prim stack underflow");
    code.push_back({static_cast<u8>(p)});
    if (p == Prim::exit_) {
      reachable = false;
    } else {
      d = d - e.in + e.out;
    }
  }

  void lit(u64 v) {
    check(reachable, "unreachable lit");
    std::vector<u8> b{0x80};
    append_le(b, v, 8);
    code.push_back(std::move(b));
    ++d;
  }

  void str(u32 idx) {
    check(reachable, "unreachable str");
    std::vector<u8> b{0x81};
    append_le(b, idx, 4);
    code.push_back(std::move(b));
    d += 2;
  }

  void pick(u32 idx) {
    check(reachable, "unreachable pick");
    check(idx < d, "pick out of bounds");
    std::vector<u8> b{0x82};
    append_le(b, idx, 4);
    code.push_back(std::move(b));
    ++d;
  }

  void tor() {
    check(reachable, "unreachable tor");
    check(d >= 1, "tor underflow");
    code.push_back({0x83});
    --d;
    ++a;
  }

  void fromr() {
    check(reachable, "unreachable fromr");
    check(a >= 1, "fromr underflow");
    code.push_back({0x84});
    ++d;
    --a;
  }

  void rfetch() {
    check(reachable, "unreachable rfetch");
    check(a >= 1, "rfetch underflow");
    code.push_back({0x85});
    ++d;
  }

  void locget(u32 idx) {
    check(reachable, "unreachable locget");
    check(idx < locals, "locget out of range");
    std::vector<u8> b{0x86};
    append_le(b, idx, 4);
    code.push_back(std::move(b));
    ++d;
  }

  void locset(u32 idx) {
    check(reachable, "unreachable locset");
    check(d >= 1 && idx < locals, "locset underflow or out of range");
    std::vector<u8> b{0x87};
    append_le(b, idx, 4);
    code.push_back(std::move(b));
    --d;
  }

  void jmp(std::size_t lid) {
    check(reachable, "unreachable jmp");
    record_label_height(lid, d, a);
    std::size_t idx = code.size();
    code.push_back({0x88, 0, 0, 0, 0});
    patches.push_back({idx, lid});
    reachable = false;
  }

  void jz(std::size_t lid) {
    check(reachable, "unreachable jz");
    check(d >= 1, "jz underflow");
    --d;
    record_label_height(lid, d, a);
    std::size_t idx = code.size();
    code.push_back({0x89, 0, 0, 0, 0});
    patches.push_back({idx, lid});
  }

  void jnz(std::size_t lid) {
    check(reachable, "unreachable jnz");
    check(d >= 1, "jnz underflow");
    --d;
    record_label_height(lid, d, a);
    std::size_t idx = code.size();
    code.push_back({0x8a, 0, 0, 0, 0});
    patches.push_back({idx, lid});
  }

  void call(u32 word_idx, const std::vector<WordSig>& sigs) {
    check(reachable, "unreachable call");
    check(word_idx < sigs.size(), "call bad word");
    const auto& s = sigs[word_idx];
    check(d >= s.ins, "call underflow");
    std::vector<u8> b{0x8b};
    append_le(b, word_idx, 4);
    code.push_back(std::move(b));
    d = d - s.ins + s.outs;
  }

  void ret() {
    check(reachable, "unreachable ret");
    check(d == outs && a == 0, "ret height mismatch");
    code.push_back({0x8c});
    reachable = false;
  }

  // Emit provenance length and 64-bit value of top-of-stack via putn.
  void obs() {
    prim(Prim::provget);
    prim(Prim::putn);
    prim(Prim::putn);
  }

  void finalize() {
    check(!reachable, "word falls off end without ret/exit/jmp");
    for (const auto& pt : patches) {
      const auto& L = labels[pt.label_id];
      check(L.bound && L.pc < code.size(), "unbound or out-of-range label");
      auto& ins = code[pt.instr_idx];
      for (int i = 0; i < 4; ++i) {
        ins[1 + static_cast<std::size_t>(i)] = static_cast<u8>((L.pc >> (8 * i)) & 0xffULL);
      }
    }
    ann.clear();
    for (const auto& L : labels) {
      if (!L.bound) continue;
      bool dup_pc = false;
      for (const auto& ex : ann) {
        if (ex.first == L.pc) {
          check(ex.second.first == L.d && ex.second.second == L.a, "conflicting ann at same pc");
          dup_pc = true;
          break;
        }
      }
      if (!dup_pc) {
        ann.push_back({L.pc, {L.d, L.a}});
      }
    }
  }

  void serialize(std::vector<u8>& out) const {
    append_le(out, ins, 4);
    append_le(out, outs, 4);
    append_le(out, locals, 4);
    append_le(out, static_cast<u64>(code.size()), 4);
    append_le(out, static_cast<u64>(ann.size()), 4);
    for (const auto& c : code) {
      out.insert(out.end(), c.begin(), c.end());
    }
    for (const auto& kv : ann) {
      append_le(out, kv.first, 4);
      append_le(out, kv.second.first, 4);
      append_le(out, kv.second.second, 4);
    }
  }
};

// Random operand helpers (standard-portable rng() % n, zero libc/libstdc++ drift).
u64 pick_u64(std::mt19937_64& rng, u64 n) {
  return rng() % n;
}

u64 rand_val(std::mt19937_64& rng) {
  static constexpr std::array<u64, 16> kEdges = {
      0ULL,
      1ULL,
      2ULL,
      3ULL,
      7ULL,
      63ULL,
      64ULL,
      255ULL,
      256ULL,
      0x7fffffffULL,
      0x80000000ULL,
      0x7fffffffffffffffULL,
      0x8000000000000000ULL,
      0xfffffffffffffffeULL,
      0xffffffffffffffffULL,
      0x0123456789abcdefULL,
  };
  u64 k = pick_u64(rng, 5);
  if (k == 0) return kEdges[static_cast<std::size_t>(pick_u64(rng, kEdges.size()))];
  if (k == 1) return pick_u64(rng, 1024);
  return rng();
}

// Guaranteed non-zero and != 2^63, so (a + b) != (a - b) mod 2^64 for every a.
u64 rand_add_rhs(std::mt19937_64& rng) {
  u64 v = rand_val(rng);
  if ((v & 0x7fffffffffffffffULL) == 0ULL) {
    v |= 1ULL + pick_u64(rng, 255);
  }
  return v;
}

u64 rand_nonzero(std::mt19937_64& rng) {
  u64 v = rand_val(rng);
  if (v == 0ULL) v = 1ULL + pick_u64(rng, 255);
  return v;
}

u64 rand_neg(std::mt19937_64& rng) {
  return rand_val(rng) | 0x8000000000000000ULL;
}

u64 rand_nonneg(std::mt19937_64& rng) {
  return rand_val(rng) & 0x7fffffffffffffffULL;
}

// Word 2: ( x y -- out ), exercises all comparisons (including opposite-sign
// lts/les/ltu/leu) and all bitwise/shift primitives with live observable output.
WordBuilder build_word_cmp_bits(std::mt19937_64& rng) {
  WordBuilder w(2, 1, 2);
  w.locset(1);  // y
  w.locset(0);  // x

  // Opposite-sign comparisons (x and y always have opposite MSB):
  // guarantees signed `lts`/`les` disagree with unsigned `ltu`/`leu`.
  w.locget(0); w.locget(1); w.prim(Prim::lts); w.obs();
  w.locget(1); w.locget(0); w.prim(Prim::lts); w.obs();
  w.locget(0); w.locget(1); w.prim(Prim::les); w.obs();
  w.locget(1); w.locget(0); w.prim(Prim::les); w.obs();
  w.locget(0); w.locget(1); w.prim(Prim::ltu); w.obs();
  w.locget(1); w.locget(0); w.prim(Prim::ltu); w.obs();
  w.locget(0); w.locget(1); w.prim(Prim::leu); w.obs();
  w.locget(1); w.locget(0); w.prim(Prim::leu); w.obs();
  w.locget(0); w.locget(1); w.prim(Prim::eq);  w.obs();
  w.locget(0); w.locget(1); w.prim(Prim::ne);  w.obs();

  // Equal-operand & same-sign boundary comparisons.
  w.locget(0); w.prim(Prim::dup); w.prim(Prim::eq);  w.obs();
  w.locget(0); w.prim(Prim::dup); w.prim(Prim::ne);  w.obs();
  w.locget(0); w.prim(Prim::dup); w.prim(Prim::les); w.obs();
  w.locget(0); w.prim(Prim::dup); w.prim(Prim::leu); w.obs();
  w.locget(0); w.prim(Prim::dup); w.lit(1); w.prim(Prim::bor);
  w.prim(Prim::lts); w.obs();

  // Bitwise & shifts (including shift >= 64 to cover % 64 hardware masking,
  // and `sar` on a guaranteed-negative value to exercise sign extension).
  u64 sh1 = 1ULL + pick_u64(rng, 63);
  u64 sh2 = 64ULL + pick_u64(rng, 64);
  w.locget(0); w.locget(1); w.prim(Prim::band); w.obs();
  w.locget(0); w.locget(1); w.prim(Prim::bor);  w.obs();
  w.locget(0); w.locget(1); w.prim(Prim::bxor); w.obs();
  w.locget(0); w.prim(Prim::bnot); w.obs();
  w.locget(0); w.lit(sh1); w.prim(Prim::shl); w.obs();
  w.locget(1); w.lit(sh2); w.prim(Prim::shr); w.obs();
  w.locget(0); w.lit(0x8000000000000000ULL); w.prim(Prim::bor);
  w.lit(sh1); w.prim(Prim::sar); w.obs();

  // Return combined live value.
  w.locget(0);
  w.locget(1);
  w.prim(Prim::bxor);
  w.ret();
  w.finalize();
  return w;
}

// Word 1: ( a b -- sum_out nested_out ), exercises all arithmetic ops, stack &
// aux stack ops, structured control flow (jz/jnz/jmp + loop), and nested call.
WordBuilder build_word_arith_ctrl(std::mt19937_64& rng, u64 seed, const std::vector<WordSig>& sigs) {
  WordBuilder w(2, 2, 4);
  w.locset(1);  // b: guaranteed (b & 0x7fffffffffffffff) != 0
  w.locset(0);  // a

  // Arithmetic suite: add (with b != 0 and b != 2^63), sub, mul, divu, modu, divs, mods, neg.
  w.locget(0); w.locget(1); w.prim(Prim::add);
  w.prim(Prim::dup); w.locset(2);  // save live sum in local 2
  w.obs();

  w.locget(0); w.locget(1); w.prim(Prim::sub);  w.obs();
  w.locget(0); w.locget(1); w.prim(Prim::mul);  w.obs();
  w.locget(0); w.locget(1); w.prim(Prim::divu); w.obs();
  w.locget(0); w.locget(1); w.prim(Prim::modu); w.obs();
  w.locget(0); w.locget(1); w.prim(Prim::divs); w.obs();
  w.locget(0); w.locget(1); w.prim(Prim::mods); w.obs();
  w.locget(0); w.prim(Prim::neg); w.obs();

  // Signed division edge cases: INT64_MIN / -1 wraps to INT64_MIN, x % -1 == 0.
  w.lit(0x8000000000000000ULL); w.lit(0xffffffffffffffffULL); w.prim(Prim::divs); w.obs();
  w.lit(0x8000000000000000ULL); w.lit(0xffffffffffffffffULL); w.prim(Prim::mods); w.obs();
  w.lit(rand_neg(rng)); w.lit(1ULL + pick_u64(rng, 17)); w.prim(Prim::divs); w.obs();
  w.lit(rand_neg(rng)); w.lit(1ULL + pick_u64(rng, 17)); w.prim(Prim::mods); w.obs();

  // Stack & auxiliary stack operations (dup, drop, swap, over, rot, nip, tuck, pick, tor, rfetch, fromr).
  w.locget(0);
  w.locget(1);
  w.lit(rand_add_rhs(rng));
  w.prim(Prim::rot);   // ( b c a )
  w.prim(Prim::over);  // ( b c a c )
  w.prim(Prim::tuck);  // ( b c c a c )
  w.prim(Prim::swap);  // ( b c c c a )
  w.prim(Prim::nip);   // ( b c c a )
  w.pick(2);           // ( b c c a c )
  w.tor();             // aux: [c], ds: ( b c c a )
  w.tor();             // aux: [c, a], ds: ( b c c )
  w.rfetch();          // ds: ( b c c a )
  w.obs();             // ds: ( b c c )
  w.fromr();           // aux: [c], ds: ( b c c a )
  w.fromr();           // aux: [], ds: ( b c c a c )
  w.prim(Prim::drop);  // ds: ( b c c a )
  w.obs();
  w.obs();
  w.obs();
  w.obs();

  // Control flow 1: jz + jmp diamond (exercises both branches across seeds and grows pc label).
  auto l_else = w.new_label();
  auto l_join1 = w.new_label();
  w.lit(seed & 1ULL);
  w.jz(l_else);
  w.locget(2); w.lit(rand_add_rhs(rng)); w.prim(Prim::add);
  w.jmp(l_join1);
  w.bind(l_else);
  w.locget(2); w.lit(rand_add_rhs(rng)); w.prim(Prim::sub);
  w.bind(l_join1);
  w.obs();

  // Control flow 2: jnz diamond (opposite polarity).
  auto l_nz = w.new_label();
  auto l_join2 = w.new_label();
  w.lit((seed >> 1) & 1ULL);
  w.jnz(l_nz);
  w.locget(0); w.lit(3); w.prim(Prim::mul);
  w.jmp(l_join2);
  w.bind(l_nz);
  w.locget(1); w.lit(5); w.prim(Prim::add);
  w.bind(l_join2);
  w.obs();

  // Control flow 3: bounded countdown loop (2..5 iterations) growing context label & accumulator.
  u64 iters = 2ULL + pick_u64(rng, 4);
  u64 step_c = rand_add_rhs(rng);
  w.lit(iters);
  w.locset(3);
  auto l_loop = w.new_label();
  w.bind(l_loop);
  w.locget(2); w.lit(step_c); w.prim(Prim::add); w.locset(2);
  w.locget(3); w.lit(1); w.prim(Prim::sub);
  w.prim(Prim::dup); w.locset(3);
  w.jnz(l_loop);

  // Nested call to word 2 with opposite-sign operands.
  u64 neg_v = rand_neg(rng);
  u64 pos_v = rand_nonneg(rng);
  if ((seed & 1ULL) == 0ULL) {
    w.lit(neg_v);
    w.lit(pos_v);
  } else {
    w.lit(pos_v);
    w.lit(neg_v);
  }
  w.call(2, sigs);  // returns 1 value

  // Return (sum_acc, nested_out).
  w.locget(2);
  w.prim(Prim::swap);
  w.ret();
  w.finalize();
  return w;
}

// Word 0 (main): ( -- )
WordBuilder build_word_main(std::mt19937_64& rng, u64 seed, const std::vector<WordSig>& sigs) {
  // Locals:
  //   0: first stdin byte (0 triggers controlled demand refusal at end)
  //   1: initial heap mark
  //   2: 128-byte heap buffer pointer
  //   3: secondary heap buffer pointer (after release + realloc)
  //   4: table 0 id
  //   5: table 1 id
  WordBuilder w(0, 0, 6);

  // 1. Stdin mode byte + additional getb reads & putb output.
  w.prim(Prim::getb);
  w.locset(0);  // mode byte: 0 => demand-refusal path at Step 9

  u64 extra_reads = 2ULL + pick_u64(rng, 3);
  for (u64 i = 0; i < extra_reads; ++i) {
    w.prim(Prim::getb);
    if (i == 0) {
      w.prim(Prim::dup);
      w.prim(Prim::putb);
    }
    w.obs();
  }

  // 2. Call word 1 (arithmetic, stack/aux, control flow, nested call to word 2).
  w.lit(rand_val(rng));
  w.lit(rand_add_rhs(rng));
  w.call(1, sigs);  // pushes (sum_acc, nested_out)
  w.obs();
  w.obs();

  // 3. Heap allocation + memory st8/st16/st32/st64 + ld8/ld16/ld32/ld64 (pm > 0).
  w.prim(Prim::mark);
  w.locset(1);

  w.lit(128);
  w.prim(Prim::alloc);
  w.locset(2);
  w.locget(2);
  w.obs();

  // Store tainted values at distinct offsets inside [buf, buf+16).
  w.lit(0x55ULL ^ (rng() & 0xffULL)); w.locget(2); w.prim(Prim::st8);
  w.lit(0x1234ULL ^ (rng() & 0xffffULL)); w.locget(2); w.lit(2); w.prim(Prim::add); w.prim(Prim::st16);
  w.lit(0x9abcdef0ULL ^ (rng() & 0xffffffffULL)); w.locget(2); w.lit(4); w.prim(Prim::add); w.prim(Prim::st32);
  w.lit(rand_nonzero(rng)); w.locget(2); w.lit(8); w.prim(Prim::add); w.prim(Prim::st64);

  // Load back all four widths and emit both value AND provget provenance length (pm > 0).
  w.locget(2); w.prim(Prim::ld8);  w.obs();
  w.locget(2); w.lit(2); w.prim(Prim::add); w.prim(Prim::ld16); w.obs();
  w.locget(2); w.lit(4); w.prim(Prim::add); w.prim(Prim::ld32); w.obs();
  w.locget(2); w.lit(8); w.prim(Prim::add); w.prim(Prim::ld64); w.obs();
  // Unaligned cross-store load exercises multi-byte pm summation.
  w.locget(2); w.lit(3); w.prim(Prim::add); w.prim(Prim::ld64); w.obs();

  // 4. Block memory: fill, copy (non-overlapping, backward overlap, forward overlap, len 0), mcmp.
  u64 fill_b = 0x20ULL + pick_u64(rng, 95);
  // fill: ( dst n b -- )
  w.locget(2); w.lit(16); w.prim(Prim::add);
  w.lit(16);
  w.lit(fill_b);
  w.prim(Prim::fill);

  // copy non-overlapping: ( src dst n -- ) buf+16 -> buf+40, len 16
  w.locget(2); w.lit(16); w.prim(Prim::add);
  w.locget(2); w.lit(40); w.prim(Prim::add);
  w.lit(16);
  w.prim(Prim::copy);

  // mcmp equal: ( x y n -- f )
  w.locget(2); w.lit(16); w.prim(Prim::add);
  w.locget(2); w.lit(40); w.prim(Prim::add);
  w.lit(16);
  w.prim(Prim::mcmp);
  w.obs();

  // Perturb one byte in buf+40 and run mcmp unequal (non-short-circuit pm sum).
  u64 diff_off = pick_u64(rng, 16);
  w.lit(fill_b ^ 0x1fULL);
  w.locget(2); w.lit(40 + diff_off); w.prim(Prim::add);
  w.prim(Prim::st8);
  w.locget(2); w.lit(16); w.prim(Prim::add);
  w.locget(2); w.lit(40); w.prim(Prim::add);
  w.lit(16);
  w.prim(Prim::mcmp);
  w.obs();

  // Backward overlap copy (dst > src): write 16 bytes at buf+64, copy buf+64 -> buf+68, len 12.
  w.lit(rand_nonzero(rng)); w.locget(2); w.lit(64); w.prim(Prim::add); w.prim(Prim::st64);
  w.lit(rand_nonzero(rng)); w.locget(2); w.lit(72); w.prim(Prim::add); w.prim(Prim::st64);
  w.locget(2); w.lit(64); w.prim(Prim::add);
  w.locget(2); w.lit(68); w.prim(Prim::add);
  w.lit(12);
  w.prim(Prim::copy);
  w.locget(2); w.lit(68); w.prim(Prim::add); w.prim(Prim::ld64); w.obs();

  // Forward overlap copy (dst < src): buf+68 -> buf+65, len 10.
  w.locget(2); w.lit(68); w.prim(Prim::add);
  w.locget(2); w.lit(65); w.prim(Prim::add);
  w.lit(10);
  w.prim(Prim::copy);
  w.locget(2); w.lit(65); w.prim(Prim::add); w.prim(Prim::ld64); w.obs();

  // Zero-length copy fast path.
  w.locget(2); w.locget(2); w.lit(0); w.prim(Prim::copy);

  // 5. Strings: str 0..3, streq (equal, equal-len unequal, diff-len, tainted heap), strhash.
  w.str(0); w.str(1); w.prim(Prim::streq); w.obs();
  w.str(0); w.str(2); w.prim(Prim::streq); w.obs();
  w.str(0); w.str(3); w.prim(Prim::streq); w.obs();
  w.str(0); w.prim(Prim::strhash); w.obs();
  w.str(3); w.prim(Prim::strhash); w.obs();

  // Tainted heap streq & strhash (accumulate non-zero memp).
  w.locget(2); w.lit(16); w.prim(Prim::add); w.lit(12);
  w.locget(2); w.lit(40); w.prim(Prim::add); w.lit(12);
  w.prim(Prim::streq); w.obs();
  w.locget(2); w.lit(64); w.prim(Prim::add); w.lit(16);
  w.prim(Prim::strhash); w.obs();

  // 6. Heap release + re-alloc.
  w.locget(1);
  w.prim(Prim::release);
  w.lit(64);
  w.prim(Prim::alloc);
  w.locset(3);
  w.locget(3);
  w.obs();

  // Re-initialize [buf3, buf3+64) so randomized tail can safely read/write it.
  w.locget(3); w.lit(64); w.lit(0x41ULL + pick_u64(rng, 26)); w.prim(Prim::fill);

  // 7. Tables: tnew, tput (insert & overwrite across 2 tables), tget (hit & miss).
  w.prim(Prim::tnew); w.locset(4);
  w.prim(Prim::tnew); w.locset(5);

  u64 k0 = rand_val(rng);
  u64 k1 = k0 ^ 0x101ULL;
  u64 k2 = k0 ^ 0x202ULL;
  u64 k_miss = k0 ^ 0x404ULL;

  // tput: ( tb k v -- )
  w.locget(4); w.lit(k0); w.lit(rand_val(rng)); w.prim(Prim::tput);
  w.locget(4); w.lit(k1); w.lit(rand_val(rng)); w.prim(Prim::tput);
  w.locget(4); w.lit(k2); w.lit(rand_val(rng)); w.prim(Prim::tput);
  w.locget(4); w.lit(k1); w.lit(rand_val(rng)); w.prim(Prim::tput);  // overwrite k1
  w.locget(5); w.lit(k0); w.lit(rand_val(rng)); w.prim(Prim::tput);

  // tget: ( tb k -- v found ), obs() prints found then v.
  w.locget(4); w.lit(k1);     w.prim(Prim::tget); w.obs(); w.obs();
  w.locget(4); w.lit(k0);     w.prim(Prim::tget); w.obs(); w.obs();
  w.locget(4); w.lit(k_miss); w.prim(Prim::tget); w.obs(); w.obs();
  w.locget(5); w.lit(k0);     w.prim(Prim::tget); w.obs(); w.obs();

  // 8. Environment, epistemic, and clean demand(val, 0).
  w.prim(Prim::clock_); w.obs();
  w.prim(Prim::core_);  w.obs();
  w.lit(pick_u64(rng, 16)); w.prim(Prim::sensor); w.obs();
  w.lit(rand_val(rng)); w.prim(Prim::confget); w.obs(); w.obs();
  w.lit(rand_val(rng)); w.lit(0); w.prim(Prim::demand); w.obs();

  // 9. Randomized trap-free fuzzing tail (24..56 steps) for broad instruction diversity.
  // Maintain a small live working stack (depth 2..6) and drain everything via obs().
  w.lit(rand_val(rng));
  w.lit(rand_add_rhs(rng));
  u64 tail_steps = 24ULL + pick_u64(rng, 33);
  for (u64 step = 0; step < tail_steps; ++step) {
    if (w.d < 3) {
      w.lit(rand_val(rng));
      continue;
    }
    if (w.d > 7) {
      w.obs();
      continue;
    }
    u64 kind = pick_u64(rng, 14);
    switch (kind) {
      case 0:  // binary arith (always safe)
        switch (pick_u64(rng, 4)) {
          case 0: w.prim(Prim::add); break;
          case 1: w.prim(Prim::sub); break;
          case 2: w.prim(Prim::mul); break;
          default: w.prim(Prim::neg); break;
        }
        break;
      case 1:  // safe div/mod with guarded non-zero divisor
        w.prim(Prim::drop);
        w.lit(rand_nonzero(rng));
        switch (pick_u64(rng, 4)) {
          case 0: w.prim(Prim::divu); break;
          case 1: w.prim(Prim::modu); break;
          case 2: w.prim(Prim::divs); break;
          default: w.prim(Prim::mods); break;
        }
        break;
      case 2:  // bitwise / shifts
        switch (pick_u64(rng, 7)) {
          case 0: w.prim(Prim::band); break;
          case 1: w.prim(Prim::bor); break;
          case 2: w.prim(Prim::bxor); break;
          case 3: w.prim(Prim::bnot); break;
          case 4: w.prim(Prim::shl); break;
          case 5: w.prim(Prim::shr); break;
          default: w.prim(Prim::sar); break;
        }
        break;
      case 3:  // comparisons
        switch (pick_u64(rng, 6)) {
          case 0: w.prim(Prim::eq); break;
          case 1: w.prim(Prim::ne); break;
          case 2: w.prim(Prim::ltu); break;
          case 3: w.prim(Prim::lts); break;
          case 4: w.prim(Prim::leu); break;
          default: w.prim(Prim::les); break;
        }
        break;
      case 4:  // stack & aux
        switch (pick_u64(rng, 8)) {
          case 0: w.prim(Prim::dup); break;
          case 1: w.prim(Prim::swap); break;
          case 2: w.prim(Prim::over); break;
          case 3: w.prim(Prim::rot); break;
          case 4: w.prim(Prim::nip); break;
          case 5: w.prim(Prim::tuck); break;
          case 6: w.pick(static_cast<u32>(pick_u64(rng, w.d))); break;
          default:
            if (w.a > 0 && pick_u64(rng, 2) == 0) {
              if (pick_u64(rng, 2) == 0) w.fromr();
              else w.rfetch();
            } else {
              w.tor();
            }
            break;
        }
        break;
      case 5: {  // safe memory store + load in [buf3, buf3+56]
        u64 off = pick_u64(rng, 48);
        w.locget(3);
        w.lit(off);
        w.prim(Prim::add);
        switch (pick_u64(rng, 4)) {
          case 0:
            w.prim(Prim::st8);
            w.locget(3); w.lit(off); w.prim(Prim::add); w.prim(Prim::ld8);
            break;
          case 1:
            w.prim(Prim::st16);
            w.locget(3); w.lit(off); w.prim(Prim::add); w.prim(Prim::ld16);
            break;
          case 2:
            w.prim(Prim::st32);
            w.locget(3); w.lit(off); w.prim(Prim::add); w.prim(Prim::ld32);
            break;
          default:
            w.prim(Prim::st64);
            w.locget(3); w.lit(off); w.prim(Prim::add); w.prim(Prim::ld64);
            break;
        }
        break;
      }
      case 6: {  // safe copy/mcmp inside [buf3, buf3+64)
        u64 len = pick_u64(rng, 17);
        u64 off1 = pick_u64(rng, 64 - len);
        u64 off2 = pick_u64(rng, 64 - len);
        w.locget(3); w.lit(off1); w.prim(Prim::add);
        w.locget(3); w.lit(off2); w.prim(Prim::add);
        w.lit(len);
        if (pick_u64(rng, 2) == 0) {
          w.prim(Prim::copy);
        } else {
          w.prim(Prim::mcmp);
        }
        break;
      }
      case 7: {  // safe table put/get on table 0 or 1
        u32 tloc = pick_u64(rng, 2) == 0 ? 4U : 5U;
        u64 k = pick_u64(rng, 6);
        if (pick_u64(rng, 2) == 0) {
          w.locget(tloc); w.lit(k); w.lit(rand_val(rng)); w.prim(Prim::tput);
        } else {
          w.locget(tloc); w.lit(k); w.prim(Prim::tget);
          w.prim(Prim::bxor);
        }
        break;
      }
      case 8: {  // conditional branch to next instruction (grows pc label)
        auto l_next = w.new_label();
        if (pick_u64(rng, 2) == 0) w.jz(l_next);
        else w.jnz(l_next);
        w.bind(l_next);
        break;
      }
      case 9:  // str + strhash on random data segment
        w.str(static_cast<u32>(pick_u64(rng, 4)));
        w.prim(Prim::strhash);
        break;
      case 10:  // demand with threshold 0 (safe provenance tag)
        w.lit(0);
        w.prim(Prim::demand);
        break;
      case 11:  // provget + fold
        w.prim(Prim::provget);
        w.prim(Prim::bxor);
        break;
      case 12:  // putb of low byte (keeps stream varied)
        w.prim(Prim::dup);
        w.prim(Prim::putb);
        break;
      default:  // fresh literal
        w.lit(rand_val(rng));
        break;
    }
  }

  // Drain auxiliary and data stacks, emitting value + provenance length for every item.
  while (w.a > 0) {
    w.fromr();
  }
  while (w.d > 0) {
    w.obs();
  }

  // 10. Controlled demand-refusal check (`M5` mutant detector):
  // If byte 0 of stdin (local 0) is NUL (0), execute `demand` with a non-zero
  // threshold (including high-bit-only values on some seeds), which MUST fail
  // with `Err::demand` under `noPhys`. On normal inputs (bytes in 1..255 or
  // empty stdin returning ~0), jump cleanly to `l_done` and halt with flushed
  // stdout.
  auto l_done = w.new_label();
  w.locget(0);
  w.jnz(l_done);
  w.lit(rand_val(rng));
  u64 dem_th = 0;
  switch (seed % 4ULL) {
    case 0: dem_th = 1ULL; break;
    case 1: dem_th = 1ULL + pick_u64(rng, 255); break;
    case 2: dem_th = 0x8000000000000000ULL; break;  // tests 64-bit test (not 32-bit)
    default: dem_th = rand_nonzero(rng); break;
  }
  w.lit(dem_th);
  w.prim(Prim::demand);
  w.obs();  // reached only if demand refusal was bypassed (M5 mutant!)

  w.bind(l_done);
  if ((seed & 1ULL) == 0ULL) {
    w.ret();
  } else {
    w.lit(seed % 4ULL);
    w.prim(Prim::exit_);
  }
  w.finalize();
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  u64 seed = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 1ULL;
  std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ULL);

  // 4 data segments: seg 0 (base), seg 1 (copy of seg 0), seg 2 (same length,
  // 1 byte flipped), seg 3 (different length).
  u64 len0 = 6ULL + pick_u64(rng, 7);
  std::vector<std::vector<u8>> data(4);
  data[0].resize(static_cast<std::size_t>(len0));
  for (std::size_t i = 0; i < data[0].size(); ++i) {
    data[0][i] = static_cast<u8>(pick_u64(rng, 3) ? ('a' + pick_u64(rng, 26)) : pick_u64(rng, 256));
  }
  data[1] = data[0];
  data[2] = data[0];
  std::size_t flip_pos = static_cast<std::size_t>(pick_u64(rng, len0));
  data[2][flip_pos] = static_cast<u8>(data[2][flip_pos] ^ 0x01U);
  data[3].resize(static_cast<std::size_t>(len0 + 3ULL));
  for (std::size_t i = 0; i < data[3].size(); ++i) {
    data[3][i] = static_cast<u8>('A' + pick_u64(rng, 26));
  }

  std::vector<WordSig> sigs = {
      {0, 0},  // word 0: main
      {2, 2},  // word 1: arith + ctrl + nested call
      {2, 1},  // word 2: cmp + bitwise/shifts
  };

  WordBuilder w0 = build_word_main(rng, seed, sigs);
  WordBuilder w1 = build_word_arith_ctrl(rng, seed, sigs);
  WordBuilder w2 = build_word_cmp_bits(rng);

  std::vector<u8> out = {0x53, 0x50, 0x52, 0x31};  // "SPR1"
  append_le(out, 0, 4);                            // main = word 0
  append_le(out, 3, 4);                            // nw = 3
  append_le(out, static_cast<u64>(data.size()), 4);  // nd = 4
  for (const auto& d : data) {
    append_le(out, static_cast<u64>(d.size()), 4);
    out.insert(out.end(), d.begin(), d.end());
  }
  w0.serialize(out);
  w1.serialize(out);
  w2.serialize(out);

  if (std::fwrite(out.data(), 1, out.size(), stdout) != out.size()) {
    return 1;
  }
  return 0;
}
