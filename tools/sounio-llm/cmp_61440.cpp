// ADR-009 verified_foreign_reference (C++23 bootstrap recomputation comparator)
// oracle_class: verified_foreign_reference
// toolchain: Apple clang version 21.0.0 (-std=c++23)
// authorship: Independently authored in C++23 from the Option A bit-identity TSV & provenance
//   contract (no code shared with or transliterated from Sounio stdlib/llm or tests/llm).
// rationale: Sounio single-backend audit runners cannot serve as self-judge of cross-backend
//   bit-identity or label provenance (and must be rejected if they pre-claim CROSS_BACKEND_IDENTICAL).
//   This tool fails closed on malformed headers/labels, recomputes canonical token-column SHA-256
//   using self-contained FIPS 180-4 SHA-256, compares IEEE-754 u32 logits/margins via std::bit_cast,
//   and is the sole emitter of CROSS_BACKEND_IDENTICAL / CROSS_BACKEND_DIVERGENT.

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

constexpr std::string_view kExpectedFloatModel =
    "binary32, round-to-nearest-even, FTZ+DAZ, fixed op order";
constexpr std::string_view kHeaderPrefix = "# run_header ";

struct Sha256 {
  std::array<uint32_t, 8> h{
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
  };
  std::array<uint8_t, 64> block{};
  size_t block_len = 0;
  uint64_t total_bits = 0;

  static constexpr uint32_t rotr(uint32_t x, uint32_t n) noexcept {
    return (x >> n) | (x << (32u - n));
  }

  void compress(const uint8_t* p) noexcept {
    static constexpr std::array<uint32_t, 64> k{
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
        0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
        0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
        0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    std::array<uint32_t, 64> w{};
    for (size_t i = 0; i < 16; ++i) {
      w[i] = (static_cast<uint32_t>(p[i * 4 + 0]) << 24) |
             (static_cast<uint32_t>(p[i * 4 + 1]) << 16) |
             (static_cast<uint32_t>(p[i * 4 + 2]) << 8) |
             static_cast<uint32_t>(p[i * 4 + 3]);
    }
    for (size_t i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (size_t i = 0; i < 64; ++i) {
      const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ ((~e) & g);
      const uint32_t temp1 = hh + s1 + ch + k[i] + w[i];
      const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temp2 = s0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }

  void update(std::string_view sv) noexcept {
    for (unsigned char ch : sv) {
      block[block_len++] = ch;
      total_bits += 8;
      if (block_len == 64) {
        compress(block.data());
        block_len = 0;
      }
    }
  }

  std::string finalize_hex() noexcept {
    block[block_len++] = 0x80u;
    if (block_len > 56) {
      while (block_len < 64) block[block_len++] = 0;
      compress(block.data());
      block_len = 0;
    }
    while (block_len < 56) block[block_len++] = 0;
    for (int i = 7; i >= 0; --i) {
      block[56 + (7 - i)] = static_cast<uint8_t>((total_bits >> (i * 8)) & 0xffu);
    }
    compress(block.data());

    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (uint32_t word : h) {
      for (int sh = 28; sh >= 0; sh -= 4) {
        out.push_back(hex[(word >> sh) & 0xfu]);
      }
    }
    return out;
  }
};

struct RunHeader {
  std::string binary_sha256;
  std::string float_model;
  std::string backend;
  std::string raw;
};

std::string_view backend_family(std::string_view b) {
  if (b.starts_with("cuda")) return "cuda";
  if (b.starts_with("metal")) return "metal";
  return b;
}

std::string_view verdict_label(bool identical, bool same_backend) {
  if (same_backend) {
    return identical ? "KERNEL_REVISION_IDENTICAL" : "KERNEL_REVISION_DIVERGENT";
  }
  return identical ? "CROSS_BACKEND_IDENTICAL" : "CROSS_BACKEND_DIVERGENT";
}

struct TokenTuple {
  int64_t t1 = 0;
  int64_t t2 = 0;
  uint32_t b1 = 0;
  uint32_t b2 = 0;
  uint32_t mb = 0;
  std::string exactness;
  std::string history;
};

struct PromptRow {
  int64_t pidx = 0;
  int64_t n_prompt = 0;
  int64_t n_gen = 0;
  int64_t fma_hits = 0;
  std::vector<TokenTuple> tuples;
};

struct ParsedTsv {
  RunHeader header;
  std::map<int64_t, PromptRow> rows;
  std::string token_cols_sha256;
};

struct FormatError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

bool is_lower_hex64(std::string_view s) {
  if (s.size() != 64) return false;
  for (char c : s) {
    const bool digit = (c >= '0' && c <= '9');
    const bool hex = (c >= 'a' && c <= 'f');
    if (!digit && !hex) return false;
  }
  return true;
}

std::string_view trim_sv(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n')) {
    s.remove_prefix(1);
  }
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) {
    s.remove_suffix(1);
  }
  return s;
}

template <typename IntT>
IntT parse_exact_int(std::string_view sv, std::string_view context) {
  if (sv.empty()) {
    throw FormatError(std::string(context) + ": empty integer field");
  }
  IntT val{};
  const char* begin = sv.data();
  const char* end = sv.data() + sv.size();
  auto [ptr, ec] = std::from_chars(begin, end, val);
  if (ec != std::errc{} || ptr != end) {
    throw FormatError(std::string(context) + ": invalid integer '" + std::string(sv) + "'");
  }
  return val;
}

std::vector<std::string_view> split_char(std::string_view sv, char delim) {
  std::vector<std::string_view> out;
  size_t pos = 0;
  while (true) {
    const size_t next = sv.find(delim, pos);
    if (next == std::string_view::npos) {
      out.push_back(sv.substr(pos));
      break;
    }
    out.push_back(sv.substr(pos, next - pos));
    pos = next + 1;
  }
  return out;
}

std::optional<std::string_view> find_unquoted_kv(std::string_view header, std::string_view key) {
  const std::string needle = std::string(key) + "=";
  size_t pos = 0;
  while ((pos = header.find(needle, pos)) != std::string_view::npos) {
    const bool left_ok = (pos == 0 || header[pos - 1] == ' ' || header[pos - 1] == '\t');
    if (left_ok) {
      const size_t val_start = pos + needle.size();
      if (val_start < header.size() && header[val_start] == '"') {
        const size_t q_end = header.find('"', val_start + 1);
        if (q_end == std::string_view::npos) return std::nullopt;
        return header.substr(val_start + 1, q_end - val_start - 1);
      }
      size_t val_end = val_start;
      while (val_end < header.size() && header[val_end] != ' ' && header[val_end] != '\t') {
        ++val_end;
      }
      return header.substr(val_start, val_end - val_start);
    }
    pos += needle.size();
  }
  return std::nullopt;
}

RunHeader parse_run_header(std::string_view line, std::string_view path) {
  if (!line.starts_with(kHeaderPrefix)) {
    throw FormatError(std::string(path) + ":1: missing or malformed # run_header with binary_sha256 and float_model");
  }
  const auto sha = find_unquoted_kv(line, "binary_sha256");
  const auto fm = find_unquoted_kv(line, "float_model");
  if (!sha || !is_lower_hex64(*sha)) {
    throw FormatError(std::string(path) + ":1: missing or invalid binary_sha256 in # run_header");
  }
  if (!fm) {
    throw FormatError(std::string(path) + ":1: missing float_model in # run_header");
  }
  if (*fm != kExpectedFloatModel) {
    throw FormatError(std::string(path) + ":1: float_model=\"" + std::string(*fm) +
                      "\" != expected \"" + std::string(kExpectedFloatModel) + "\"");
  }
  const auto bk = find_unquoted_kv(line, "backend");
  return RunHeader{std::string(*sha), std::string(*fm), bk ? std::string(*bk) : std::string{},
                   std::string(line)};
}

std::string compute_token_cols_sha256(const std::map<int64_t, PromptRow>& rows) {
  Sha256 hasher;
  for (const auto& [pidx, row] : rows) {
    std::string canonical = std::to_string(pidx) + "\t" + std::to_string(row.n_prompt) + "\t" +
                            std::to_string(row.n_gen) + "\t" + std::to_string(row.fma_hits) + "\t";
    for (size_t i = 0; i < row.tuples.size(); ++i) {
      if (i > 0) canonical.push_back(',');
      const auto& tok = row.tuples[i];
      canonical += std::to_string(tok.t1) + ":" + std::to_string(tok.t2) + ":" +
                   std::to_string(tok.b1) + ":" + std::to_string(tok.b2) + ":" +
                   std::to_string(tok.mb);
    }
    canonical.push_back('\n');
    hasher.update(canonical);
  }
  return hasher.finalize_hex();
}

ParsedTsv parse_tsv(const std::string& path, bool allow_legacy_cross_backend_ident) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw FormatError("unable to open file: " + path);
  }
  std::optional<RunHeader> header;
  std::map<int64_t, PromptRow> rows;
  std::string raw;
  size_t line_no = 0;

  while (std::getline(in, raw)) {
    ++line_no;
    const std::string_view line = trim_sv(raw);
    if (line.empty()) continue;
    if (line_no == 1) {
      header = parse_run_header(line, path);
      continue;
    }
    if (line.front() == '#') {
      throw FormatError(path + ":" + std::to_string(line_no) + ": unexpected comment line after line 1");
    }
    const auto cols = split_char(line, '\t');
    if (cols.size() != 5) {
      throw FormatError(path + ":" + std::to_string(line_no) + ": expected 5 TSV columns, got " +
                        std::to_string(cols.size()));
    }
    const std::string ctx = path + ":" + std::to_string(line_no);
    PromptRow row;
    row.pidx = parse_exact_int<int64_t>(cols[0], ctx);
    row.n_prompt = parse_exact_int<int64_t>(cols[1], ctx);
    row.n_gen = parse_exact_int<int64_t>(cols[2], ctx);
    row.fma_hits = parse_exact_int<int64_t>(cols[3], ctx);
    if (row.pidx < 0 || row.n_prompt <= 0 || row.n_gen <= 0 || row.fma_hits < 0) {
      throw FormatError(ctx + ": invalid negative/zero prompt metadata");
    }
    if (rows.contains(row.pidx)) {
      throw FormatError(ctx + ": duplicate prompt_idx=" + std::to_string(row.pidx));
    }

    const auto items = split_char(cols[4], ',');
    row.tuples.reserve(items.size());
    for (std::string_view item : items) {
      const auto flds = split_char(item, ':');
      if (flds.size() != 7) {
        throw FormatError(ctx + ": expected 7-field tuple with exactness & history, got '" +
                          std::string(item) + "'");
      }
      TokenTuple tok;
      tok.t1 = parse_exact_int<int64_t>(flds[0], ctx);
      tok.t2 = parse_exact_int<int64_t>(flds[1], ctx);
      tok.b1 = parse_exact_int<uint32_t>(flds[2], ctx);
      tok.b2 = parse_exact_int<uint32_t>(flds[3], ctx);
      tok.mb = parse_exact_int<uint32_t>(flds[4], ctx);
      if (!flds[5].starts_with("exactness=") || !flds[6].starts_with("history=")) {
        throw FormatError(ctx + ": missing exactness=/history= fields in tuple '" +
                          std::string(item) + "'");
      }
      tok.exactness = std::string(flds[5].substr(10));
      tok.history = std::string(flds[6].substr(8));
      if (tok.exactness != "VERIFIED_EXACT") {
        throw FormatError(ctx + ": invalid exactness='" + tok.exactness + "' (expected VERIFIED_EXACT)");
      }
      if (allow_legacy_cross_backend_ident) {
        if (tok.history != "UNCOMPARED" && tok.history != "CROSS_BACKEND_IDENTICAL") {
          throw FormatError(ctx + ": unexpected history='" + tok.history + "' during relabel");
        }
      } else {
        if (tok.history != "UNCOMPARED") {
          throw FormatError(ctx + ": single-backend audit input illegally claims history='" +
                            tok.history + "' (must be UNCOMPARED)");
        }
      }
      row.tuples.push_back(std::move(tok));
    }
    if (static_cast<int64_t>(row.tuples.size()) != row.n_gen) {
      throw FormatError(ctx + ": gen_count " + std::to_string(row.tuples.size()) +
                        " != " + std::to_string(row.n_gen));
    }
    rows.emplace(row.pidx, std::move(row));
  }
  if (!header) {
    throw FormatError(path + ": empty file or missing # run_header on line 1");
  }
  if (rows.empty()) {
    throw FormatError(path + ": zero prompt rows");
  }
  std::string tok_sha = compute_token_cols_sha256(rows);
  return ParsedTsv{std::move(*header), std::move(rows), std::move(tok_sha)};
}

struct CompareResult {
  bool identical = false;
  size_t prompts = 0;
  size_t total_tokens = 0;
  size_t token_mismatches = 0;
  size_t runner_up_mismatches = 0;
  size_t top1_bit_mismatches = 0;
  size_t second_bit_mismatches = 0;
  size_t margin_bit_mismatches = 0;
  size_t provenance_mismatches = 0;
  double max_margin_drift = 0.0;
  bool max_margin_non_finite = false;
  uint64_t max_margin_bit_delta = 0;
  int64_t total_fma_hits = 0;
  std::string first_divergent_op;
  std::string ref_sha256;
  std::string cand_sha256;
  std::string ref_token_cols_sha256;
  std::string cand_token_cols_sha256;
  std::string float_model;
  std::string ref_backend;
  std::string cand_backend;
  bool same_backend = false;
};

CompareResult run_compare(const std::string& ref_path, const std::string& cand_path,
                          const std::string& out_path) {
  const ParsedTsv ref = parse_tsv(ref_path, /*allow_legacy_cross_backend_ident=*/false);
  const ParsedTsv cand = parse_tsv(cand_path, /*allow_legacy_cross_backend_ident=*/false);

  if (ref.rows.size() != cand.rows.size()) {
    throw std::runtime_error("PROMPT_SET_MISMATCH metal=" + std::to_string(ref.rows.size()) +
                             " cuda=" + std::to_string(cand.rows.size()));
  }
  for (auto it_r = ref.rows.begin(), it_c = cand.rows.begin(); it_r != ref.rows.end();
       ++it_r, ++it_c) {
    if (it_r->first != it_c->first) {
      throw std::runtime_error("PROMPT_SET_MISMATCH key " + std::to_string(it_r->first) +
                               " != " + std::to_string(it_c->first));
    }
  }

  CompareResult res;
  res.prompts = ref.rows.size();
  res.ref_sha256 = ref.header.binary_sha256;
  res.cand_sha256 = cand.header.binary_sha256;
  res.ref_token_cols_sha256 = ref.token_cols_sha256;
  res.cand_token_cols_sha256 = cand.token_cols_sha256;
  res.float_model = ref.header.float_model;
  res.ref_backend = ref.header.backend;
  res.cand_backend = cand.header.backend;
  const std::string_view rf = backend_family(res.ref_backend);
  const std::string_view cf = backend_family(res.cand_backend);
  res.same_backend = (!rf.empty() && rf == cf);

  struct AnnotatedRow {
    int64_t pidx = 0;
    int64_t n_prompt = 0;
    int64_t n_gen = 0;
    int64_t fma_hits = 0;
    std::vector<TokenTuple> tuples;
    std::vector<std::string_view> histories;
  };
  std::vector<AnnotatedRow> annotated;
  if (!out_path.empty()) annotated.reserve(ref.rows.size());

  std::vector<std::string> first_tok_logs;
  for (const auto& [pidx, rrow] : ref.rows) {
    const auto& crow = cand.rows.at(pidx);
    res.total_fma_hits += crow.fma_hits;
    if (rrow.n_prompt != crow.n_prompt || rrow.n_gen != crow.n_gen) {
      throw std::runtime_error("PROMPT_" + std::to_string(pidx) + "_LEN_MISMATCH: (" +
                               std::to_string(rrow.n_prompt) + "," + std::to_string(rrow.n_gen) +
                               ") vs (" + std::to_string(crow.n_prompt) + "," +
                               std::to_string(crow.n_gen) + ")");
    }
    AnnotatedRow arow;
    if (!out_path.empty()) {
      arow.pidx = pidx;
      arow.n_prompt = crow.n_prompt;
      arow.n_gen = crow.n_gen;
      arow.fma_hits = crow.fma_hits;
      arow.tuples = crow.tuples;
      arow.histories.reserve(crow.tuples.size());
    }
    bool prompt_history_diverged = false;
    for (size_t step = 0; step < rrow.tuples.size(); ++step) {
      ++res.total_tokens;
      const auto& rm = rrow.tuples[step];
      const auto& cc = crow.tuples[step];
      if (rm.exactness != cc.exactness || rm.history != cc.history) {
        ++res.provenance_mismatches;
      }
      const bool d_t1 = (rm.t1 != cc.t1);
      const bool d_t2 = (rm.t2 != cc.t2);
      const bool d_b1 = (rm.b1 != cc.b1);
      const bool d_b2 = (rm.b2 != cc.b2);
      const bool d_mb = (rm.mb != cc.mb);
      if (d_t1) {
        ++res.token_mismatches;
        if (first_tok_logs.size() < 10) {
          first_tok_logs.push_back("TOK_MISMATCH prompt=" + std::to_string(pidx) + " step=" +
                                   std::to_string(step) + ": metal=" + std::to_string(rm.t1) +
                                   " cuda=" + std::to_string(cc.t1));
        }
      }
      if (d_t2) ++res.runner_up_mismatches;
      if (d_b1) ++res.top1_bit_mismatches;
      if (d_b2) ++res.second_bit_mismatches;
      if (d_mb) {
        ++res.margin_bit_mismatches;
        const float fm = std::bit_cast<float>(rm.mb);
        const float fc = std::bit_cast<float>(cc.mb);
        if (!std::isfinite(fm) || !std::isfinite(fc)) {
          res.max_margin_non_finite = true;
        } else {
          const double dm = std::abs(static_cast<double>(fm) - static_cast<double>(fc));
          if (dm > res.max_margin_drift) res.max_margin_drift = dm;
        }
        const uint64_t bd = rm.mb > cc.mb ? static_cast<uint64_t>(rm.mb - cc.mb)
                                          : static_cast<uint64_t>(cc.mb - rm.mb);
        if (bd > res.max_margin_bit_delta) res.max_margin_bit_delta = bd;
      }

      const bool any_step_diff = (d_t1 || d_t2 || d_b1 || d_b2 || d_mb || rm.exactness != cc.exactness);
      if (any_step_diff && res.first_divergent_op.empty()) {
        std::string_view op = "provenance";
        std::string m_val = rm.exactness;
        std::string c_val = cc.exactness;
        if (d_t1) {
          op = "argmax_top1";
          m_val = std::to_string(rm.t1);
          c_val = std::to_string(cc.t1);
        } else if (d_b1) {
          op = "logit_top1_f32_u32";
          m_val = std::to_string(rm.b1);
          c_val = std::to_string(cc.b1);
        } else if (d_t2) {
          op = "argmax_top2";
          m_val = std::to_string(rm.t2);
          c_val = std::to_string(cc.t2);
        } else if (d_b2) {
          op = "logit_top2_f32_u32";
          m_val = std::to_string(rm.b2);
          c_val = std::to_string(cc.b2);
        } else if (d_mb) {
          op = "margin_f32_u32";
          m_val = std::to_string(rm.mb);
          c_val = std::to_string(cc.mb);
        }
        res.first_divergent_op = "prompt:" + std::to_string(pidx) + ",step:" + std::to_string(step) +
                                 ",op:" + std::string(op) + ",metal:" + m_val + ",cuda:" + c_val;
      }
      if (any_step_diff) prompt_history_diverged = true;
      if (!out_path.empty()) {
        arow.histories.push_back(verdict_label(!prompt_history_diverged, res.same_backend));
      }
    }
    if (!out_path.empty()) annotated.push_back(std::move(arow));
  }

  for (const auto& s : first_tok_logs) {
    std::cout << s << '\n';
  }
  res.identical = (res.prompts > 0 && res.token_mismatches == 0 &&
                   res.runner_up_mismatches == 0 && res.margin_bit_mismatches == 0 &&
                   res.top1_bit_mismatches == 0 && res.second_bit_mismatches == 0 &&
                   res.provenance_mismatches == 0 && !res.max_margin_non_finite);

  if (!out_path.empty()) {
    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("unable to write output file: " + out_path);
    }
    out << "# compare_header ref_binary_sha256=" << res.ref_sha256
        << " cand_binary_sha256=" << res.cand_sha256
        << " ref_token_cols_sha256=" << res.ref_token_cols_sha256
        << " cand_token_cols_sha256=" << res.cand_token_cols_sha256
        << " float_model=\"" << res.float_model << "\""
        << " verdict=" << verdict_label(res.identical, res.same_backend);
    if (!res.first_divergent_op.empty()) {
      out << " first_divergent_op=" << res.first_divergent_op;
    }
    out << '\n';
    for (const auto& arow : annotated) {
      out << arow.pidx << '\t' << arow.n_prompt << '\t' << arow.n_gen << '\t' << arow.fma_hits
          << '\t';
      for (size_t i = 0; i < arow.tuples.size(); ++i) {
        if (i > 0) out << ',';
        const auto& t = arow.tuples[i];
        out << t.t1 << ':' << t.t2 << ':' << t.b1 << ':' << t.b2 << ':' << t.mb
            << ":exactness=" << t.exactness << ":history=" << arow.histories[i];
      }
      out << '\n';
    }
    out.flush();
    if (!out) {
      throw std::runtime_error("flush failed for: " + out_path);
    }
  }
  return res;
}

void print_compare_result(const CompareResult& res) {
  std::cout << "metal_sha256=" << res.ref_sha256 << " cuda_sha256=" << res.cand_sha256
            << " ref_token_cols_sha256=" << res.ref_token_cols_sha256
            << " cand_token_cols_sha256=" << res.cand_token_cols_sha256
            << " float_model=\"" << res.float_model << "\"\n";
  std::cout << std::setprecision(17)
            << "prompts=" << res.prompts
            << " total_tokens=" << res.total_tokens
            << " token_mismatches=" << res.token_mismatches
            << " runner_up_mismatches=" << res.runner_up_mismatches
            << " top1_bit_mismatches=" << res.top1_bit_mismatches
            << " second_bit_mismatches=" << res.second_bit_mismatches
            << " margin_bit_mismatches=" << res.margin_bit_mismatches
            << " provenance_mismatches=" << res.provenance_mismatches
            << " max_margin_drift=" << (res.max_margin_non_finite ? -1.0 : res.max_margin_drift)
            << " max_margin_bit_delta=" << res.max_margin_bit_delta
            << " total_cuda_fma_min_norm_hits=" << res.total_fma_hits
            << " verdict=" << verdict_label(res.identical, res.same_backend);
  if (!res.first_divergent_op.empty()) {
    std::cout << " first_divergent_op=" << res.first_divergent_op;
  }
  std::cout << '\n';
}

std::pair<std::string, std::string> relabel_to_uncompared(const std::string& in_path,
                                                          const std::string& out_path) {
  const ParsedTsv parsed = parse_tsv(in_path, /*allow_legacy_cross_backend_ident=*/true);
  {
    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("unable to write output file: " + out_path);
    }
    out << parsed.header.raw << '\n';
    for (const auto& [pidx, row] : parsed.rows) {
      out << pidx << '\t' << row.n_prompt << '\t' << row.n_gen << '\t' << row.fma_hits << '\t';
      for (size_t i = 0; i < row.tuples.size(); ++i) {
        if (i > 0) out << ',';
        const auto& t = row.tuples[i];
        out << t.t1 << ':' << t.t2 << ':' << t.b1 << ':' << t.b2 << ':' << t.mb
            << ":exactness=" << t.exactness << ":history=UNCOMPARED";
      }
      out << '\n';
    }
    out.flush();
    if (!out) {
      throw std::runtime_error("flush failed writing: " + out_path);
    }
  }
  const ParsedTsv reparsed = parse_tsv(out_path, /*allow_legacy_cross_backend_ident=*/false);
  if (parsed.token_cols_sha256 != reparsed.token_cols_sha256) {
    throw std::runtime_error("token_cols_sha256 changed during relabel: " +
                             parsed.token_cols_sha256 + " != " + reparsed.token_cols_sha256);
  }
  return {parsed.token_cols_sha256, reparsed.token_cols_sha256};
}

bool expect_format_fail(const std::string& label, const std::string& p1, const std::string& p2) {
  try {
    (void)run_compare(p1, p2, "");
    std::cerr << "SELFTEST FAIL (" << label << "): expected FormatError but passed\n";
    return false;
  } catch (const FormatError& e) {
    std::cout << "SELFTEST_NEG [" << label << "]: rejected cleanly (" << e.what() << ")\n";
    return true;
  }
}

int run_selftest() {
  namespace fs = std::filesystem;
  const fs::path tmp = fs::temp_directory_path() / "sounio_cmp61440_selftest";
  fs::create_directories(tmp);
  const auto cleanup = [&]() {
    std::error_code ec;
    fs::remove_all(tmp, ec);
  };

  try {
    const std::string hdr =
        "# run_header binary_sha256=" + std::string(64, 'a') +
        " float_model=\"binary32, round-to-nearest-even, FTZ+DAZ, fixed op order\" "
        "backend=metal prefill=simt\n";
    const std::string hdr_cuda =
        "# run_header binary_sha256=" + std::string(64, 'b') +
        " float_model=\"binary32, round-to-nearest-even, FTZ+DAZ, fixed op order\" "
        "backend=cuda_audit prefill=simt\n";
    const std::string hdr_cuda_q4 =
        "# run_header binary_sha256=" + std::string(64, 'c') +
        " float_model=\"binary32, round-to-nearest-even, FTZ+DAZ, fixed op order\" "
        "backend=cuda_q4 prefill=simt\n";
    const std::string bad_fm_hdr =
        "# run_header binary_sha256=" + std::string(64, 'a') +
        " float_model=\"binary32, fast-math\" backend=metal prefill=simt\n";
    const std::string uncomp_row =
        "0\t8\t1\t0\t10:11:1065353216:1056964608:1056964608:exactness=VERIFIED_EXACT:history=UNCOMPARED\n";
    const std::string illegal_ident_row =
        "0\t8\t1\t0\t10:11:1065353216:1056964608:1056964608:exactness=VERIFIED_EXACT:history=CROSS_BACKEND_IDENTICAL\n";
    const std::string illegal_krev_row =
        "0\t8\t1\t0\t10:11:1065353216:1056964608:1056964608:exactness=VERIFIED_EXACT:history=KERNEL_REVISION_IDENTICAL\n";
    const std::string illegal_vend_row =
        "0\t8\t1\t0\t10:11:1065353216:1056964608:1056964608:exactness=VERIFIED_EXACT:history=VENDOR_PREFILL\n";
    const std::string nolabel_row = "0\t8\t1\t0\t10:11:1065353216:1056964608:1056964608\n";
    const std::string tok_diff_row =
        "0\t8\t1\t5\t99:11:1065353216:1056964608:1056964608:exactness=VERIFIED_EXACT:history=UNCOMPARED\n";
    const std::string margin_diff_row =
        "0\t8\t1\t0\t10:11:1065353216:1056964608:1056964609:exactness=VERIFIED_EXACT:history=UNCOMPARED\n";

    const auto write_f = [&](const std::string& name, const std::string& body) {
      std::string p = (tmp / name).string();
      std::ofstream f(p, std::ios::binary | std::ios::trunc);
      f << body;
      return p;
    };

    const std::string p_good = write_f("good.tsv", hdr + uncomp_row);
    const std::string p_good_cuda = write_f("good_cuda.tsv", hdr_cuda + uncomp_row);
    const std::string p_good_q4_a = write_f("good_q4_a.tsv", hdr_cuda_q4 + uncomp_row);
    const std::string p_good_q4_b = write_f("good_q4_b.tsv", hdr_cuda_q4 + uncomp_row);
    const std::string p_bad_ref_ident = write_f("ref_ident.tsv", hdr + illegal_ident_row);
    const std::string p_bad_krev_ident = write_f("krev_ident.tsv", hdr_cuda_q4 + illegal_krev_row);
    const std::string p_bad_cand_vend = write_f("cand_vend.tsv", hdr + illegal_vend_row);
    const std::string p_bad_nohdr = write_f("nohdr.tsv", uncomp_row);
    const std::string p_bad_fm = write_f("badfm.tsv", bad_fm_hdr + uncomp_row);
    const std::string p_bad_nolbl = write_f("nolbl.tsv", hdr + nolabel_row);
    const std::string p_tok_diff = write_f("tokdiff.tsv", hdr_cuda + tok_diff_row);
    const std::string p_margin_diff = write_f("margindiff.tsv", hdr_cuda + margin_diff_row);

    bool ok = true;
    ok &= expect_format_fail("ref_claims_cross_backend_identical", p_bad_ref_ident, p_good_cuda);
    ok &= expect_format_fail("cand_claims_cross_backend_identical", p_good, p_bad_ref_ident);
    ok &= expect_format_fail("single_claims_kernel_revision_identical", p_bad_krev_ident, p_good_q4_b);
    ok &= expect_format_fail("single_backend_claims_vendor_prefill", p_bad_cand_vend, p_good_cuda);
    ok &= expect_format_fail("missing_header", p_bad_nohdr, p_good_cuda);
    ok &= expect_format_fail("wrong_float_model", p_bad_fm, p_good_cuda);
    ok &= expect_format_fail("missing_labels", p_bad_nolbl, p_good_cuda);
    if (!ok) {
      cleanup();
      return 1;
    }

    const std::string out_match = (tmp / "out_match.tsv").string();
    const auto r_match = run_compare(p_good, p_good_cuda, out_match);
    if (!r_match.identical || r_match.same_backend || r_match.token_mismatches != 0 ||
        r_match.margin_bit_mismatches != 0 ||
        verdict_label(r_match.identical, r_match.same_backend) != "CROSS_BACKEND_IDENTICAL") {
      std::cerr << "SELFTEST FAIL: cross-backend UNCOMPARED match failed\n";
      cleanup();
      return 1;
    }
    std::cout << "SELFTEST_POS [cross_backend]: verdict=CROSS_BACKEND_IDENTICAL sha="
              << r_match.ref_token_cols_sha256 << '\n';

    const auto r_krev = run_compare(p_good_q4_a, p_good_q4_b, "");
    if (!r_krev.identical || !r_krev.same_backend ||
        verdict_label(r_krev.identical, r_krev.same_backend) != "KERNEL_REVISION_IDENTICAL") {
      std::cerr << "SELFTEST FAIL: same-backend KERNEL_REVISION_IDENTICAL match failed\n";
      cleanup();
      return 1;
    }
    std::cout << "SELFTEST_POS [same_backend]: verdict=KERNEL_REVISION_IDENTICAL sha="
              << r_krev.ref_token_cols_sha256 << '\n';

    const std::string out_tok_div = (tmp / "out_tok_div.tsv").string();
    const auto r_tdiv = run_compare(p_good, p_tok_diff, out_tok_div);
    if (r_tdiv.identical || r_tdiv.token_mismatches != 1 ||
        r_tdiv.first_divergent_op.find("op:argmax_top1") == std::string::npos) {
      std::cerr << "SELFTEST FAIL: token divergence diagnostic wrong: "
                << r_tdiv.first_divergent_op << '\n';
      cleanup();
      return 1;
    }
    std::cout << "SELFTEST_DIV [token]: verdict=CROSS_BACKEND_DIVERGENT first_divergent_op="
              << r_tdiv.first_divergent_op << '\n';

    const auto r_mdiv = run_compare(p_good, p_margin_diff, "");
    if (r_mdiv.identical || r_mdiv.margin_bit_mismatches != 1 ||
        r_mdiv.first_divergent_op.find("op:margin_f32_u32") == std::string::npos) {
      std::cerr << "SELFTEST FAIL: margin divergence diagnostic wrong: "
                << r_mdiv.first_divergent_op << '\n';
      cleanup();
      return 1;
    }
    std::cout << "SELFTEST_DIV [margin]: verdict=CROSS_BACKEND_DIVERGENT first_divergent_op="
              << r_mdiv.first_divergent_op << '\n';

    const std::string p_relabeled = (tmp / "relabeled.tsv").string();
    const auto [sha_before, sha_after] = relabel_to_uncompared(p_bad_ref_ident, p_relabeled);
    if (sha_before != sha_after || sha_before != r_match.ref_token_cols_sha256) {
      std::cerr << "SELFTEST FAIL: relabel changed token_cols_sha256\n";
      cleanup();
      return 1;
    }
    std::cout << "SELFTEST_RELABEL: token_cols_sha256=" << sha_after << " status=UNCHANGED\n";
    std::cout << "CMP_61440_SELFTEST_PASS\n";
    cleanup();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "SELFTEST EXCEPTION: " << e.what() << '\n';
    cleanup();
    return 1;
  }
}

void print_usage() {
  std::cerr << "usage:\n"
            << "  cmp_61440 --selftest\n"
            << "  cmp_61440 --token-sha256 <file.tsv>\n"
            << "  cmp_61440 --relabel-uncompared <in.tsv> <out.tsv>\n"
            << "  cmp_61440 <metal.tsv> <cuda.tsv> [--out <compared.tsv>]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--selftest") {
      return run_selftest();
    }
    if (argc == 3 && std::string_view(argv[1]) == "--token-sha256") {
      const ParsedTsv p = parse_tsv(argv[2], /*allow_legacy_cross_backend_ident=*/true);
      std::cout << "token_cols_sha256=" << p.token_cols_sha256 << " prompts=" << p.rows.size()
                << '\n';
      return 0;
    }
    if (argc == 4 && std::string_view(argv[1]) == "--relabel-uncompared") {
      const auto [before_sha, after_sha] = relabel_to_uncompared(argv[2], argv[3]);
      std::cout << "before_token_cols_sha256=" << before_sha
                << " after_token_cols_sha256=" << after_sha << " status=UNCHANGED\n";
      return 0;
    }
    if (argc != 3 && argc != 5) {
      print_usage();
      return 2;
    }
    std::string out_path;
    if (argc == 5) {
      if (std::string_view(argv[3]) != "--out") {
        print_usage();
        return 2;
      }
      out_path = argv[4];
    }
    const CompareResult res = run_compare(argv[1], argv[2], out_path);
    print_compare_result(res);
    return res.identical ? 0 : 1;
  } catch (const FormatError& e) {
    std::cout << "AUDIT_FORMAT_ERROR: " << e.what() << '\n';
    return 1;
  } catch (const std::exception& e) {
    std::cout << "AUDIT_ERROR: " << e.what() << '\n';
    return 1;
  }
}
