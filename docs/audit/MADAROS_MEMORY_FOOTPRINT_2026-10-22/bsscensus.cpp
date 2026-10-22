// bsscensus: list module-level fixed-size arrays (`var NAME: [T; N]`, optionally `pub`)
// in the .sio files named on stdin, sorted by the bytes they reserve.
//
// Two layouts:
//   natural (default)  i64/u64/f64/usize/isize/string = 8, i32/u32/f32 = 4, i16/u16 = 2,
//                      i8/u8/bool = 1
//   SEED_LAYOUT=1      what lean_single (the seed that compiles Madaros) actually reserves:
//                      i8 = 1 byte per element, every other element type = 8 bytes.
//                      Measured 2026-10-22 by compiling `var A: [T; 1048576]` for
//                      T in {i8,u8,bool,i16,i32,u32,f32,i64,f64}; see README.md §2.
// Any other element type (enums, structs, nested arrays) is counted at 8 bytes and flagged
// '?', so the total is an estimate whose unknown part is printed separately.
//
// Build: g++ -std=c++23 -O2 -o bsscensus bsscensus.cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>

struct Row {
    std::string file, name, elem;
    long long n = 0, bytes = 0;
    bool known = false;
};

int main() {
    const std::regex re(R"(^(?:pub\s+)?var\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\[\s*([^;\]]+?)\s*;\s*([0-9]+)\s*\])");
    const std::map<std::string, int> natural{{"i64", 8}, {"u64", 8}, {"f64", 8}, {"usize", 8}, {"isize", 8},
        {"string", 8}, {"i32", 4}, {"u32", 4}, {"f32", 4}, {"i16", 2}, {"u16", 2}, {"i8", 1}, {"u8", 1},
        {"bool", 1}};
    const bool seed_layout = std::getenv("SEED_LAYOUT") != nullptr;

    std::vector<Row> rows;
    std::string path;
    while (std::getline(std::cin, path)) {
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            std::smatch m;
            if (!std::regex_search(line, m, re)) continue;
            Row r{path, m[1], m[2], std::stoll(m[3])};
            int width = 8;
            if (auto it = natural.find(r.elem); it != natural.end()) {
                r.known = true;
                width = seed_layout ? (r.elem == "i8" ? 1 : 8) : it->second;
            }
            r.bytes = r.n * width;
            rows.push_back(r);
        }
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.bytes > b.bytes; });

    long long total = 0, unknown = 0;
    std::map<std::string, long long> per_file;
    for (const auto& r : rows) {
        total += r.bytes;
        if (!r.known) unknown += r.bytes;
        per_file[r.file] += r.bytes;
    }
    std::printf("layout=%s arrays=%zu total_bytes=%lld unknown_elem_bytes=%lld\n",
                seed_layout ? "seed" : "natural", rows.size(), total, unknown);

    std::vector<std::pair<long long, std::string>> pf;
    for (const auto& [file, bytes] : per_file) pf.emplace_back(bytes, file);
    std::sort(pf.rbegin(), pf.rend());
    std::printf("-- per file (top 15)\n");
    for (size_t i = 0; i < pf.size() && i < 15; ++i) std::printf("%14lld  %s\n", pf[i].first, pf[i].second.c_str());
    std::printf("-- per array (top 25)\n");
    for (size_t i = 0; i < rows.size() && i < 25; ++i)
        std::printf("%14lld  %-34s [%s; %lld]%s  %s\n", rows[i].bytes, rows[i].name.c_str(), rows[i].elem.c_str(),
                    rows[i].n, rows[i].known ? "" : " ?", rows[i].file.c_str());
}
