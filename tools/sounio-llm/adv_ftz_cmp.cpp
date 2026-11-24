// ADR-009 verified_foreign_reference (C++23 bootstrap recomputation comparator)
// oracle_class: verified_foreign_reference
// toolchain: Apple clang version 21.0.0 (-std=c++23)
// authorship: Independently authored C++23 binary comparator for AFTZ adversarial FTZ/DAZ dumps.
// rationale: Replaces unverified Python tooling as the pass/fail gate for 5-target binary32
//   subnormal parity and CUDA FMA_MIN_NORM_HITS > 0 assertions.

#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct OpDump {
  std::string name;
  uint32_t n = 0;
  uint64_t fma_hits = 0;
  std::vector<uint32_t> words;
};

struct AftzDump {
  uint32_t backend_id = 0;
  uint64_t fma_hits_total = 0;
  uint64_t subnormal_outputs_total = 0;
  std::vector<OpDump> ops;
};

template <typename T>
T read_le(const std::vector<uint8_t>& buf, size_t& off) {
  if (off + sizeof(T) > buf.size()) {
    throw std::runtime_error("unexpected EOF in AFTZ binary");
  }
  T val{};
  std::memcpy(&val, buf.data() + off, sizeof(T));
  off += sizeof(T);
  return val;
}

AftzDump load_aftz(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open AFTZ file: " + path);
  const std::streamsize sz = in.tellg();
  if (sz < 32) throw std::runtime_error(path + ": truncated AFTZ header");
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> buf(static_cast<size_t>(sz));
  if (!in.read(reinterpret_cast<char*>(buf.data()), sz)) {
    throw std::runtime_error(path + ": read error");
  }
  if (std::memcmp(buf.data(), "AFTZ", 4) != 0) {
    throw std::runtime_error(path + ": invalid AFTZ magic");
  }
  size_t off = 4;
  const uint32_t ver = read_le<uint32_t>(buf, off);
  if (ver != 1) throw std::runtime_error(path + ": unsupported version");
  AftzDump d;
  d.backend_id = read_le<uint32_t>(buf, off);
  const uint32_t n_ops = read_le<uint32_t>(buf, off);
  d.fma_hits_total = read_le<uint64_t>(buf, off);
  d.subnormal_outputs_total = read_le<uint64_t>(buf, off);
  d.ops.reserve(n_ops);
  for (uint32_t i = 0; i < n_ops; ++i) {
    const uint32_t nlen = read_le<uint32_t>(buf, off);
    if (off + nlen > buf.size()) throw std::runtime_error(path + ": truncated op name");
    OpDump op;
    op.name.assign(reinterpret_cast<const char*>(buf.data() + off), nlen);
    off += nlen;
    op.n = read_le<uint32_t>(buf, off);
    op.fma_hits = read_le<uint64_t>(buf, off);
    op.words.reserve(op.n);
    for (uint32_t j = 0; j < op.n; ++j) {
      op.words.push_back(read_le<uint32_t>(buf, off));
    }
    d.ops.push_back(std::move(op));
  }
  if (off != buf.size()) {
    throw std::runtime_error(path + ": trailing bytes in AFTZ");
  }
  return d;
}

bool is_subnormal(uint32_t u) noexcept {
  const uint32_t mag = u & 0x7fffffffu;
  return mag != 0u && mag < 0x00800000u;
}

std::string fmt_u32(uint32_t u) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0') << u << std::dec << " ("
     << std::setprecision(9) << std::bit_cast<float>(u) << ")";
  return os.str();
}

int64_t ulp_delta(uint32_t a, uint32_t b) noexcept {
  auto to_ordered = [](uint32_t u) -> int64_t {
    if ((u & 0x80000000u) != 0u) return -static_cast<int64_t>(u & 0x7fffffffu);
    return static_cast<int64_t>(u & 0x7fffffffu);
  };
  int64_t d = to_ordered(a) - to_ordered(b);
  return d < 0 ? -d : d;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string ref_path;
    std::vector<std::pair<std::string, std::string>> targets;
    for (int i = 1; i < argc; ++i) {
      std::string_view arg(argv[i]);
      if (arg == "--ref" && i + 1 < argc) {
        ref_path = argv[++i];
      } else {
        const size_t eq = arg.find('=');
        if (eq == std::string_view::npos) {
          targets.emplace_back(std::string(arg), std::string(arg));
        } else {
          targets.emplace_back(std::string(arg.substr(0, eq)), std::string(arg.substr(eq + 1)));
        }
      }
    }
    if (ref_path.empty() || targets.empty()) {
      std::cerr << "usage: adv_ftz_cmp --ref <metal.bin> <label=path>...\n";
      return 2;
    }

    const AftzDump ref = load_aftz(ref_path);
    size_t ref_subs = 0;
    for (const auto& op : ref.ops) {
      for (uint32_t w : op.words) {
        if (is_subnormal(w)) ++ref_subs;
      }
    }
    std::cout << "[ref] " << ref_path << ": ops=" << ref.ops.size()
              << " subnormal_outputs=" << ref_subs
              << " fma_hits_total=" << ref.fma_hits_total << '\n';
    bool all_ok = (ref_subs == 0);

    for (const auto& [label, path] : targets) {
      const AftzDump tgt = load_aftz(path);
      std::cout << "\n=== Target " << label << " (" << path << ") backend_id=" << tgt.backend_id
                << " fma_hits_total=" << tgt.fma_hits_total << " ===\n";
      if (tgt.ops.size() != ref.ops.size()) {
        std::cout << "FAIL: op count mismatch (" << tgt.ops.size() << " != " << ref.ops.size()
                  << ")\n";
        all_ok = false;
        continue;
      }
      size_t tgt_subs = 0;
      size_t tgt_mismatches = 0;
      for (size_t k = 0; k < ref.ops.size(); ++k) {
        const auto& rop = ref.ops[k];
        const auto& top = tgt.ops[k];
        if (rop.name != top.name || rop.n != top.n) {
          std::cout << "FAIL: metadata mismatch (" << top.name << "," << top.n << ") != ("
                    << rop.name << "," << rop.n << ")\n";
          all_ok = false;
          continue;
        }
        size_t subs = 0;
        std::vector<size_t> diffs;
        for (size_t idx = 0; idx < top.words.size(); ++idx) {
          if (is_subnormal(top.words[idx])) ++subs;
          if (rop.words[idx] != top.words[idx]) diffs.push_back(idx);
        }
        tgt_subs += subs;
        tgt_mismatches += diffs.size();
        std::cout << "  " << std::left << std::setw(24) << top.name << std::right
                  << " N=" << std::setw(5) << top.n
                  << " mismatches=" << std::setw(3) << diffs.size() << "/" << std::left
                  << std::setw(5) << top.n << std::right
                  << " subnormals=" << subs << " fma_hits=" << top.fma_hits << '\n';
        for (size_t d = 0; d < diffs.size() && d < 5; ++d) {
          const size_t idx = diffs[d];
          std::cout << "    diff[" << idx << "]: ref=" << fmt_u32(rop.words[idx])
                    << " vs " << label << "=" << fmt_u32(top.words[idx])
                    << " ulp_delta=" << ulp_delta(rop.words[idx], top.words[idx]) << '\n';
        }
      }
      if (tgt.backend_id == 3 && tgt.fma_hits_total == 0) {
        std::cout << "FAIL: " << label << " CUDA target recorded FMA_MIN_NORM_HITS=0 (must be > 0)\n";
        all_ok = false;
      }
      if (tgt_subs != 0 || tgt_mismatches != 0) {
        std::cout << "FAIL: " << label << " mismatches=" << tgt_mismatches
                  << " subnormals=" << tgt_subs << '\n';
        all_ok = false;
      } else {
        std::cout << "PASS: " << label << " 0 mismatches, 0 subnormal outputs across all "
                  << tgt.ops.size() << " ops\n";
      }
    }
    return all_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "ADV_FTZ_ERROR: " << e.what() << '\n';
    return 1;
  }
}
