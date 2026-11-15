// Reference model of seed-a, for the gate only (it is not on the bootstrap
// path). It decodes format A exactly as seed-a does, streaming: bytes are
// emitted as they are decoded, and a refusal stops with status 1 after the
// bytes already emitted.
//
//   seed_a_ref decode < listing > bytes      status 0 or 1, like seed-a
//   seed_a_ref fuzz SEED N DIR               writes DIR/i.in, DIR/i.out,
//                                            DIR/i.rc for i in [0, N)
//   seed_a_ref all256 > listing              listing of bytes 00..ff
//
// build: g++ -std=c++23 -O2 -o seed_a_ref seed_a_ref.cpp
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>

namespace {

// Returns the exit status; appends decoded bytes to out.
int decode(const std::string& in, std::string& out) {
    unsigned acc = 1;
    bool comment = false;
    for (unsigned char c : in) {
        if (comment) {
            if (c == '\n') comment = false;
            continue;
        }
        if (c == '#') { comment = true; continue; }
        if (c == ' ' || c == '\n' || c == '\t') continue;
        unsigned v;
        if (c >= '0' && c <= '9') v = c - '0';
        else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
        else return 1;
        acc = acc * 16 + v;
        if (acc >= 0x100) { out.push_back(static_cast<char>(acc & 0xff)); acc = 1; }
    }
    return acc == 1 ? 0 : 1;
}

std::string gen(std::mt19937_64& g) {
    static const char valid[] = "0123456789abcdef  \n\t";
    static const char edge[] = "ABCDEFgG\r#!/:@`{\x7f\x80\xff";
    std::string s;
    const int len = static_cast<int>(g() % 200);
    const int mode = static_cast<int>(g() % 4);   // 0,1: mostly valid; 2: comments; 3: anything
    for (int i = 0; i < len; ++i) {
        const unsigned r = static_cast<unsigned>(g() % 1000);
        if (mode == 3 && r < 300) { s.push_back(static_cast<char>(g() & 0xff)); continue; }
        if (r < 4) { s.push_back(edge[g() % (sizeof edge - 1)]); continue; }
        if (mode == 2 && r < 60) {
            s.push_back('#');
            const int n = static_cast<int>(g() % 20);
            for (int k = 0; k < n; ++k) s.push_back(static_cast<char>(g() & 0xff));
            if (g() % 8) s.push_back('\n');
            continue;
        }
        s.push_back(valid[g() % (sizeof valid - 1)]);
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
        for (int b = 0; b < 256; ++b) std::printf("%02x%c", b, (b % 16 == 15) ? '\n' : ' ');
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
    std::fprintf(stderr, "usage: seed_a_ref decode | all256 | fuzz SEED N DIR\n");
    return 2;
}
