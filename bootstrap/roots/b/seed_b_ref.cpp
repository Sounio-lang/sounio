// Reference model of seed-b, for the gate only (it is not on the bootstrap
// path). It decodes format B exactly as seed-b does, streaming: bytes are
// emitted as they are decoded, and a refusal stops with status 1 after the
// bytes already emitted.
//
//   seed_b_ref decode < listing > bytes      status 0 or 1, like seed-b
//   seed_b_ref fuzz SEED N DIR               writes DIR/i.in, DIR/i.out,
//                                            DIR/i.rc for i in [0, N)
//   seed_b_ref all256 > listing              listing of bytes 000..377
//
// build: g++ -std=c++23 -O2 -o seed_b_ref seed_b_ref.cpp
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>

namespace {

// Returns the exit status; appends decoded bytes to out.
int decode(const std::string& in, std::string& out) {
    unsigned acc = 0, digits = 0;
    bool comment = false;
    for (unsigned char c : in) {
        if (comment) {
            if (c == '\n') comment = false;
            continue;
        }
        if (c == ';' || c == ' ' || c == '\n' || c == '\t') {
            if (digits != 0) return 1;           // separator inside a group
            if (c == ';') comment = true;
            continue;
        }
        if (c < '0' || c > '7') return 1;        // not an octal digit
        const unsigned v = c - '0';
        if (digits == 0 && v > 3) return 1;      // group above 0377
        acc = acc * 8 + v;
        if (++digits == 3) { out.push_back(static_cast<char>(acc)); acc = 0; digits = 0; }
    }
    return digits == 0 ? 0 : 1;
}

std::string gen(std::mt19937_64& g) {
    static const char edge[] = "89aA\r#!/:@`{\x7f\x80\xff";
    std::string s;
    const int len = static_cast<int>(g() % 120);
    const int mode = static_cast<int>(g() % 4);   // 0,1: mostly valid; 2: comments; 3: anything
    for (int i = 0; i < len; ++i) {
        const unsigned r = static_cast<unsigned>(g() % 1000);
        if (mode == 3 && r < 300) { s.push_back(static_cast<char>(g() & 0xff)); continue; }
        if (r < 6) { s.push_back(edge[g() % (sizeof edge - 1)]); continue; }
        if (r < 12) { s.push_back(static_cast<char>('0' + g() % 8)); continue; }   // breaks grouping
        if (mode == 2 && r < 70) {
            s.push_back(';');
            const int n = static_cast<int>(g() % 20);
            for (int k = 0; k < n; ++k) s.push_back(static_cast<char>(g() & 0xff));
            if (g() % 8) s.push_back('\n');
            continue;
        }
        const unsigned b = static_cast<unsigned>(g() & 0xff);
        char buf[4];
        std::snprintf(buf, sizeof buf, "%03o", b);
        s += buf;
        static const char sep[] = "  \n\t";
        s.push_back(sep[g() % (sizeof sep - 1)]);
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "decode") {
        std::string in, out;
        int c;
        while ((c = std::getchar()) != EOF) in.push_back(static_cast<char>(c));
        const int rc = decode(in, out);
        std::fwrite(out.data(), 1, out.size(), stdout);
        return rc;
    }
    if (mode == "all256") {
        for (int b = 0; b < 256; ++b) std::printf("%03o%c", b, (b % 16 == 15) ? '\n' : ' ');
        return 0;
    }
    if (mode == "fuzz" && argc == 5) {
        std::mt19937_64 g(std::stoull(argv[2]));
        const int n = std::stoi(argv[3]);
        const std::string dir = argv[4];
        for (int i = 0; i < n; ++i) {
            const std::string in = gen(g);
            std::string out;
            const int rc = decode(in, out);
            const std::string base = dir + "/" + std::to_string(i);
            std::ofstream(base + ".in", std::ios::binary) << in;
            std::ofstream(base + ".out", std::ios::binary) << out;
            std::ofstream(base + ".rc") << rc << "\n";
        }
        return 0;
    }
    std::fprintf(stderr, "usage: seed_b_ref decode | all256 | fuzz SEED N DIR\n");
    return 2;
}
