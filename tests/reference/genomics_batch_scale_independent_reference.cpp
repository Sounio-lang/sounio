// C++23 binary128 analytic oracle for frozen SIO171 scale_main.sio.
// Central Slurm: g++ -std=c++23 -O2 -Wall -Wextra -Werror this.cpp -lquadmath
// -o reference Default: reference shard0.stdout ... shard19000.stdout (all20
// required). Explicit --partial accepts actual complete 1000-gene shard reports
// only; its marker is deliberately distinct from full20k completion.
// Normalization is PER SHARD, never claimed to reproduce a global binary64
// reduction. Every positive count is base*{1,2}, so all gene log-ratios have
// factors {1/sqrt(2),sqrt(2)} and the normalized mean is base*sqrt(2).
// The saturated Poisson solution has beta0=log(base*sqrt(2)), beta1=0,
// alpha=0. This oracle verifies every emitted fit/ID and actual work counters;
// it does not infer unseen fits, shrinkage, tails, BH/BY or finite-sample
// validity.
#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <quadmath.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace scale_ref {
using Q = __float128;
constexpr int total = 20000, shard = 1000, samples = 512, columns = 2;
void require(bool ok, const std::string &what) {
  if (!ok)
    throw std::runtime_error(what);
}
Q q(const char *s) { return strtoflt128(s, nullptr); }
struct Reader {
  std::ifstream stream;
  std::string path;
  size_t line = 0;
  explicit Reader(std::string p) : stream(p), path(p) {
    require(bool(stream), "open " + p);
  }
  std::string get() {
    std::string s;
    require(bool(std::getline(stream, s)), path + ": unexpected EOF");
    ++line;
    if (!s.empty() && s.back() == '\r')
      s.pop_back();
    return s;
  }
  std::vector<long long> ints(const std::string &prefix, size_t count) {
    auto s = get();
    std::istringstream in(s);
    std::string key;
    require(bool(in >> key) && key == prefix, path + ": expected " + prefix);
    std::vector<long long> out(count);
    for (auto &v : out)
      require(bool(in >> v), "integer field");
    std::string extra;
    require(!(in >> extra), "extra integer field");
    return out;
  }
  void literal(const std::string &expected) {
    auto actual = get();
    require(actual == expected, path + ":" + std::to_string(line) +
                                    ": expected " + expected + " got " +
                                    actual);
  }
  Q number() {
    auto s = get();
    char *end = nullptr;
    Q v = strtoflt128(s.c_str(), &end);
    require(end != s.c_str() && *end == '\0' && !isnanq(v) && !isinfq(v),
            "finite scientific scalar");
    return v;
  }
  void eof() {
    std::string s;
    require(!std::getline(stream, s), "trailing scale report rows");
  }
};
void close(Q actual, Q expected, const char *field, Q relative, Q absolute) {
  if (fabsq(actual - expected) > relative * fabsq(expected) + absolute) {
    char a[100], e[100];
    quadmath_snprintf(a, sizeof(a), "%.24Qg", actual);
    quadmath_snprintf(e, sizeof(e), "%.24Qg", expected);
    throw std::runtime_error(std::string(field) + ": actual=" + a +
                             " expected=" + e);
  }
}
struct Corpus {
  std::array<bool, total> seen{};
  std::array<bool, total / shard> ranges{};
  long long payload = 0, evaluations = 0;
  int reports = 0, rows = 0;
  void read(const std::string &path) {
    Reader r(path);
    auto layout = r.ints("SCALE_LAYOUT", 5);
    require(layout[0] == total && layout[1] == shard && layout[2] == samples &&
                layout[3] == columns,
            "frozen scale dimensions");
    // Metadata sanity, not a claim of measured process RSS or allocator
    // overhead.
    require(layout[4] > 0 && layout[4] <= (1LL << 30),
            "declared payload sanity budget");
    if (reports == 0)
      payload = layout[4];
    else
      require(payload == layout[4], "same workspace layout across shards");
    auto range = r.ints("SCALE_RANGE", 2);
    long long begin = range[0], end = range[1];
    require(begin >= 0 && begin <= total - shard && begin % shard == 0 &&
                end == begin + shard,
            "canonical shard range");
    require(!ranges[begin / shard], "duplicate/overlapping shard range");
    ranges[begin / shard] = true;
    for (int done = 0; done < shard;) {
      done = std::min(done + 16, shard);
      auto progress = r.ints("SCALE_PROGRESS", 1);
      require(progress[0] == done, "actual batch progress sequence");
    }
    long long work = 0;
    for (int local = 0; local < shard; ++local) {
      int global = int(begin) + local;
      auto point = r.ints("SCALE_POINT", 3);
      require(point[0] == global && point[1] == 0,
              "actual successful ordered global point");
      require(point[2] >= 1 && point[2] <= 64,
              "actual per-point evaluation budget");
      work += point[2];
      auto id = r.ints("SCALE_ID", 6);
      require(id[0] == global, "ID row binding");
      const std::array<int, 5> expected{
          71, 65 + global / 17576, 65 + (global / 676) % 26,
          65 + (global / 26) % 26, 65 + global % 26};
      for (size_t j = 0; j < expected.size(); ++j)
        require(id[j + 1] == expected[j], "original five-byte gene identity");
      require(!seen[global], "duplicate gene identity/fit");
      seen[global] = true;
      ++rows;
      Q alpha = r.number(), beta0 = r.number(), beta1 = r.number();
      require(alpha == 0, "saturated Poisson alpha must equal zero");
      Q base = global % 7 + 1;
      Q expected0 = logq(base * sqrtq(Q(2)));
      close(beta0, expected0, "normalized intercept", q("2e-8"), q("2e-8"));
      close(beta1, 0, "normalized group contrast", 0, q("1e-7"));
    }
    auto reported = r.ints("SCALE_EVALUATIONS", 1);
    require(reported[0] == work, "actual evaluation sum");
    auto state = r.ints("SCALE_STATE", 3);
    require(state[0] == shard && state[1] == 2 && state[2] == 0,
            "all points done, phase2, no curvature eligible");
    r.literal("GENOMICS_BATCH_SCALE_PASS");
    r.eof();
    evaluations += work;
    ++reports;
    std::printf("SCALE_INDEPENDENT_SHARD_PASS %lld %lld %lld %s\n", begin, end,
                work, path.c_str());
  }
  void finish(bool partial) {
    require(reports > 0, "no actual shard reports");
    if (!partial) {
      require(reports == total / shard && rows == total,
              "missing full20k actual coverage");
      for (bool v : seen)
        require(v, "missing global gene identity");
      for (bool v : ranges)
        require(v, "missing original shard range");
    }
    std::printf(
        "SCALE_INDEPENDENT_OBSERVED reports=%d genes=%d samples=%d "
        "payload_per_owner=%lld evaluations=%lld normalization=per_shard\n",
        reports, rows, samples, payload, evaluations);
    std::puts(partial ? "GENOMICS_BATCH_SCALE_INDEPENDENT_PARTIAL_PASS"
                      : "GENOMICS_BATCH_SCALE_INDEPENDENT_FULL20K_PASS");
  }
};
} // namespace scale_ref
int main(int argc, char **argv) {
  try {
    bool partial = argc > 1 && std::string(argv[1]) == "--partial";
    int first = partial ? 2 : 1;
    scale_ref::Corpus corpus;
    for (int i = first; i < argc; ++i)
      corpus.read(argv[i]);
    corpus.finish(partial);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "GENOMICS_BATCH_SCALE_ORACLE_REJECT %s\n", e.what());
    return 2;
  }
}
