// C++23 binary128 scalar/availability oracle for exact-key NB2 slot reuse.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <quadmath.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using Q = __float128;
static void need(bool v, const char *w) {
  if (!v)
    throw std::runtime_error(w);
}
static Q expected(long long y, double mu, double a) {
  Q m = mu, alpha = a;
  if (alpha == 0)
    return y * logq(m) - m - lgammaq(Q(y + 1));
  if (y == 0)
    return -log1pq(alpha * m) / alpha;
  Q r = 1 / alpha;
  return lgammaq(r + y) - lgammaq(r) - lgammaq(Q(y + 1)) +
         y * (logq(m) - logq(r)) - (y + r) * log1pq(m / r);
}

struct PerfTime {
  long double cpu = 0, wall = 0, rss = 0;
};
static long double perf_number(std::string text) {
  size_t end = 0;
  long double value = std::stold(text, &end);
  need(end == text.size() && std::isfinite(value) && value >= 0,
       "finite nonnegative timing");
  return value;
}
static PerfTime perf_read(const std::string &path) {
  std::ifstream in(path);
  need(bool(in), "time report open");
  std::string line;
  PerfTime t;
  bool user = false, system = false, wall = false, rss = false;
  while (std::getline(in, line)) {
    auto payload = [&]() {
      auto p = line.rfind(": ");
      need(p != std::string::npos, "time delimiter");
      return line.substr(p + 2);
    };
    if (line.find("User time (seconds):") != std::string::npos) {
      need(!user, "duplicate user");
      user = true;
      t.cpu += perf_number(payload());
    } else if (line.find("System time (seconds):") != std::string::npos) {
      need(!system, "duplicate system");
      system = true;
      t.cpu += perf_number(payload());
    } else if (line.find("Elapsed (wall clock)") != std::string::npos) {
      need(!wall, "duplicate elapsed");
      wall = true;
      std::string s = payload();
      long double value = 0;
      size_t begin = 0;
      while (true) {
        size_t p = s.find(':', begin);
        auto part = s.substr(begin, p == std::string::npos ? p : p - begin);
        value = value * 60 + perf_number(part);
        if (p == std::string::npos)
          break;
        begin = p + 1;
      }
      t.wall = value;
    } else if (line.find("Maximum resident set size (kbytes):") !=
               std::string::npos) {
      need(!rss, "duplicate RSS");
      rss = true;
      t.rss = perf_number(payload());
    }
  }
  need(user && system && wall && rss && t.cpu > 0 && t.wall > 0 && t.rss > 0,
       "complete positive timing");
  return t;
}
static int perf_report(const char *path) {
  std::ifstream in(path);
  need(bool(in), "sample index open");
  struct Block {
    std::array<std::array<PerfTime, 2>, 3> t{};
    std::array<std::array<bool, 2>, 3> seen{};
    int position = 0;
  };
  std::map<std::string, Block> blocks;
  std::string line;
  int count = 0;
  const std::array<char, 6> order{'A', 'B', 'B', 'A', 'A', 'B'};
  while (std::getline(in, line)) {
    std::istringstream s(line);
    std::string engine, scenario, role, time, extra;
    int pair = 0;
    need(bool(s >> engine >> scenario >> pair >> role >> time) && !(s >> extra),
         "sample schema");
    need((engine == "madaros" || engine == "lean_single") &&
             (scenario == "profile_main" || scenario == "control_main"),
         "sample engine/scenario");
    need(pair >= 1 && pair <= 3 && (role == "A" || role == "B"), "pair role");
    auto &b = blocks[engine + "/" + scenario];
    need(b.position < 6 && role[0] == order[b.position] &&
             pair == b.position / 2 + 1,
         "AB/BA/AB actual order");
    ++b.position;
    int r = role == "A" ? 0 : 1;
    need(!b.seen[pair - 1][r], "duplicate sample");
    b.seen[pair - 1][r] = true;
    b.t[pair - 1][r] = perf_read(time);
    ++count;
  }
  need(count == 24 && blocks.size() == 4,
       "all two engines and two scenarios sampled");
  for (auto &[name, b] : blocks) {
    need(b.position == 6, "complete block");
    std::array<long double, 3> cpu{}, wall{};
    long double rssA = 0, rssB = 0;
    for (int i = 0; i < 3; ++i) {
      need(b.seen[i][0] && b.seen[i][1], "complete pair");
      auto a = b.t[i][0], c = b.t[i][1];
      cpu[i] = a.cpu / c.cpu;
      wall[i] = a.wall / c.wall;
      rssA = std::max(rssA, a.rss);
      rssB = std::max(rssB, c.rss);
      std::printf("PAIR %s %d CPU_A %.6Lf CPU_B %.6Lf WALL_A %.6Lf WALL_B "
                  "%.6Lf RSS_A %.0Lf RSS_B %.0Lf CPU_RATIO %.6Lf\n",
                  name.c_str(), i + 1, a.cpu, c.cpu, a.wall, c.wall, a.rss,
                  c.rss, cpu[i]);
    }
    std::sort(cpu.begin(), cpu.end());
    std::sort(wall.begin(), wall.end());
    std::printf("SUMMARY %s CPU_RATIO_MIN %.6Lf MEDIAN %.6Lf MAX %.6Lf "
                "WALL_RATIO_MEDIAN %.6Lf MAXRSS_A %.0Lf MAXRSS_B %.0Lf\n",
                name.c_str(), cpu[0], cpu[1], cpu[2], wall[1], rssA, rssB);
  }
  std::puts(
      "NB2_SLOT_CACHE_PAIRED_TIMING_PASS_NO_STATISTICAL_CONFIDENCE_CLAIM");
  return 0;
}

int main(int argc, char **argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--perf")
      return perf_report(argv[2]);
    need(argc >= 2, "need native output");
    for (int arg = 1; arg < argc; ++arg) {
      std::ifstream in(argv[arg]);
      need(bool(in), "native open");
      const std::array<const char *, 18> fields{
          "INPUT_COUNT",     "INPUT_MEAN",
          "INPUT_ALPHA",     "STATUS",
          "COUNT",           "LOGP",
          "NORMALIZER",      "SATURATED",
          "RESIDUAL_STATUS", "RESIDUAL",
          "DEVIANCE_STATUS", "DEVIANCE",
          "POISSON",         "STIRLING",
          "REMAINDER",       "REMAINDER_AVAILABLE",
          "ASSESSED",        nullptr};
      std::array<std::array<long long, 17>, 22> first{};
      std::array<int, 22> phases{};
      std::string line;
      bool footer = false;
      int records = 0;
      while (std::getline(in, line)) {
        if (line == "NB2_SLOT_CACHE_PASS") {
          footer = true;
          break;
        }
        std::istringstream h(line);
        std::string tag, junk;
        int id = -1, phase = -1;
        need(bool(h >> tag >> id >> phase) && tag == "CASE" && !(h >> junk) &&
                 id >= 0 && id < 22,
             "record header");
        need(phase == phases[id]++, "ordered phase");
        std::array<long long, 17> values{};
        for (size_t j = 0; j < values.size(); ++j) {
          need(bool(std::getline(in, line)), "record truncated");
          std::istringstream s(line);
          need(bool(s >> tag >> values[j]) && tag == fields[j] && !(s >> junk),
               "record scalar shape");
        }
        if (phase == 0)
          first[id] = values;
        else
          need(values == first[id], "repeated/fresh slot bit identity");
        const std::array<int, 22> statuses{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                           2, 3, 4, 2, 2, 3, 3, 5, 0, 5, 0};
        need(values[3] == statuses[id], "main availability status");
        need(values[16] == 0, "assumptions flag remains false");
        need(values[4] == values[0], "count retention");
        if (id == 20) {
          for (size_t j = 0; j < 17; ++j)
            if (j != 1 && j != 2) {
              long long want =
                  (j == 3 ? 5 : (j == 6 || j == 8 || j == 10 ? 2 : 0));
              need(values[j] == want,
                   "explicit reset all availability/payload");
            }
        }
        if (statuses[id] == 0) {
          double mu = std::bit_cast<double>(std::uint64_t(values[1])),
                 alpha = std::bit_cast<double>(std::uint64_t(values[2]));
          double actual = std::bit_cast<double>(std::uint64_t(values[5]));
          need(!isnanq(Q(actual)) && !isinfq(Q(actual)) && actual <= 0,
               "successful finite logP");
          if (id == 9)
            need(std::bit_cast<std::uint64_t>(actual) ==
                     std::bit_cast<std::uint64_t>(-mu),
                 "Poisson zero exact boundary");
          else {
            Q want = expected(values[0], mu, alpha);
            need(fabsq(Q(actual) - want) <=
                     fabsq(want) * strtoflt128("1e-11", nullptr) +
                         strtoflt128("1e-300", nullptr),
                 "independent full logP");
          }
          need(values[12] == (alpha == 0 ? 1 : 0), "Poisson metadata");
        }
        ++records;
      }
      need(footer && !std::getline(in, line), "complete closed report");
      for (int id = 0; id < 22; ++id)
        need(phases[id] == (id == 20 ? 1 : 3), "all cases/phases selected");
      need(records == 64, "nonempty exact record corpus");
      std::printf("NB2_SLOT_CACHE_CPP_PASS %s records=%d\n", argv[arg],
                  records);
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "NB2_SLOT_CACHE_CPP_REJECT %s\n", e.what());
    return 2;
  }
}
