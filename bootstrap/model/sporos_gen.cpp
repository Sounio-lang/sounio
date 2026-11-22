// sporos_gen SEED > prog.spb : a random, checker-valid Sporos program for the
// differential test of sporos_fast against the Lean reference (sporos_diff.sh).
//
// Straight-line code over every primitive except exit, with literals biased to
// edge values, small addresses so that memory, copy, fill, strings and tables
// are exercised (and sometimes out of bounds), locals, the auxiliary stack,
// pick, and conditional jumps to the next instruction, which add their
// condition to the context label without changing the control flow. At the end
// every value left on the stack is written out with putn, together with its
// provenance length (provget), so that the comparison sees provenance too.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

using u64 = std::uint64_t;

namespace {

std::vector<std::uint8_t> out;
void le(u64 v, int k) { for (int i = 0; i < k; ++i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i))); }

struct Ins { std::vector<std::uint8_t> bytes; };

// effects, in opcode order (formal/Sporos/Bytecode.lean, Prim.all)
const int effIn[] = {1,1,2,2,3,2,2, 2,2,2,2,2,2,2,1, 2,2,2,1,2,2,2, 2,2,2,2,2,2,
                     1,2,1,2,1,2,1,2, 3,3,3, 1,0,1, 4,2, 0,3,2, 0,1,1,1, 0,0,1, 1,1,2};
const int effOut[] = {2,0,2,3,3,1,3, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1, 1,1,1,1,1,1,
                      1,0,1,0,1,0,1,0, 0,0,1, 1,1,0, 1,1, 1,0,2, 1,0,0,0, 1,1,1, 2,2,1};
constexpr int kExit = 50, kNPrim = 57, kPutn = 49, kProvget = 54;

}  // namespace

int main(int argc, char** argv) {
  std::mt19937_64 rng(argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1);
  auto pick = [&](u64 n) { return rng() % n; };
  const u64 edges[] = {0, 1, 2, 3, 7, 8, 16, 63, 64, 255, 256, 0x7fffffffffffffffull, 0x8000000000000000ull,
                       0xffffffffffffffffull, 0xfffffffffffffffeull, 1000, 12};
  std::vector<std::vector<std::uint8_t>> code;
  std::vector<std::pair<u64, std::pair<u64, u64>>> ann;
  u64 d = 0, a = 0;
  const u64 locals = 4;
  int len = 10 + static_cast<int>(pick(80));
  auto lit = [&](u64 v) { std::vector<std::uint8_t> b{0x80}; for (int i = 0; i < 8; ++i) b.push_back(v >> (8 * i)); return b; };
  auto imm4 = [&](std::uint8_t op, u64 v) { std::vector<std::uint8_t> b{op}; for (int i = 0; i < 4; ++i) b.push_back(v >> (8 * i)); return b; };
  code.push_back({44}); code.push_back(imm4(0x87, 3));   // tnew; local 3 := table 0
  for (int k = 0; k < len; ++k) {
    int choice = static_cast<int>(pick(10));
    if (choice <= 2 || d == 0) {
      u64 v = pick(3) == 0 ? rng() : (pick(2) ? edges[pick(sizeof edges / sizeof edges[0])] : pick(40));
      code.push_back(lit(v)); ++d;
    } else if (choice == 3) {
      // a conditional jump to the next instruction: control unchanged, label grows
      u64 t = code.size() + 1;
      code.push_back(imm4(pick(2) ? 0x89 : 0x8a, t)); --d;
      ann.push_back({t, {d, a}});
    } else if (choice == 4) {
      switch (pick(6)) {
        case 0: code.push_back(imm4(0x82, pick(d))); ++d; break;              // pick
        case 1: code.push_back({0x83}); --d; ++a; break;                       // tor
        case 2: if (a) { code.push_back({0x84}); ++d; --a; } break;            // fromr
        case 3: if (a) { code.push_back({0x85}); ++d; } break;                 // rfetch
        case 4: code.push_back(imm4(0x86, pick(locals))); ++d; break;          // locget
        case 5: code.push_back(imm4(0x87, pick(locals))); --d; break;          // locset
      }
    } else if (choice == 5) {
      code.push_back(imm4(0x81, 0)); d += 2;                                   // str 0
    } else {
      int p;
      do { p = static_cast<int>(pick(kNPrim)); } while (p == kExit || static_cast<u64>(effIn[p]) > d);
      // most of the time, give the operands that make the primitive succeed, so
      // that runs get far enough to exercise what follows
      bool guard = pick(8) != 0;
      auto L = [&](u64 v) { code.push_back(lit(v)); ++d; };
      if (guard) {
        switch (p) {
          case 10: case 11: case 12: case 13:            // divu modu divs mods: nonzero divisor
            code.push_back({0x01}); --d; L(1 + pick(9)); break;    // drop the top, push a divisor
          case 28: case 30: case 32: case 34:            // ld*: small address
            code.push_back({0x01}); --d; L(pick(48)); break;
          case 29: case 31: case 33: case 35:            // st*: small address on top
            code.push_back({0x01}); --d; L(pick(48)); break;
          case 36: case 37:                              // copy fill: three small operands
            code.push_back({0x01}); code.push_back({0x01}); code.push_back({0x01}); d -= 3;
            L(pick(20)); L(pick(20)); L(pick(9)); break;
          case 38:                                       // mcmp
            code.push_back({0x01}); code.push_back({0x01}); code.push_back({0x01}); d -= 3;
            L(pick(20)); L(pick(20)); L(pick(9)); break;
          case 39: code.push_back({0x01}); --d; L(pick(16)); break;              // alloc
          case 41: code.push_back({0x01}); --d; code.push_back({0x28}); ++d; break; // release mark
          case 42:                                                               // streq
            for (int i = 0; i < 4; ++i) code.push_back({0x01});
            d -= 4; L(pick(8)); L(pick(5)); L(pick(8)); L(pick(5)); break;
          case 43: code.push_back({0x01}); code.push_back({0x01}); d -= 2; L(pick(8)); L(pick(6)); break;
          case 45:                                                               // tput: table 0
            for (int i = 0; i < 3; ++i) code.push_back({0x01});
            d -= 3; code.push_back(imm4(0x86, 3)); ++d; L(pick(4)); L(rng()); break;
          case 46:                                                               // tget
            code.push_back({0x01}); code.push_back({0x01}); d -= 2;
            code.push_back(imm4(0x86, 3)); ++d; L(pick(4)); break;
          case 56: code.push_back({0x01}); --d; L(0); break;                    // demand: threshold 0
          default: break;
        }
      }
      code.push_back({static_cast<std::uint8_t>(p)});
      d = d - effIn[p] + effOut[p];
    }
  }
  // write every value, with its provenance length
  while (d > 0) {
    code.push_back({static_cast<std::uint8_t>(kProvget)});
    code.push_back({static_cast<std::uint8_t>(kPutn)});
    code.push_back({static_cast<std::uint8_t>(kPutn)});
    --d;
  }
  code.push_back(lit(static_cast<u64>(pick(4))));
  code.push_back({static_cast<std::uint8_t>(kExit)});

  std::vector<std::uint8_t> data;
  int dl = static_cast<int>(pick(12));
  for (int i = 0; i < dl; ++i) data.push_back(static_cast<std::uint8_t>(pick(3) ? 'a' + pick(3) : pick(256)));

  out = {0x53, 0x50, 0x52, 0x31};
  le(0, 4); le(1, 4); le(1, 4);
  le(data.size(), 4); out.insert(out.end(), data.begin(), data.end());
  le(0, 4); le(0, 4); le(locals, 4); le(code.size(), 4); le(ann.size(), 4);
  for (auto& c : code) out.insert(out.end(), c.begin(), c.end());
  for (auto& [i, h] : ann) { le(i, 4); le(h.first, 4); le(h.second, 4); }
  std::fwrite(out.data(), 1, out.size(), stdout);
  return 0;
}
