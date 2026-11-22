// Runs the OpenQASM inside a sounio-device document.
//
// v1 (sounio-device 1, or any document with basis z / basis x):
//   two bases, mt19937_64, Bell probabilities, copy mixture, Sounio tape.
// v2 (sounio-device 2):
//   four signed OpenQASM terms. Sampler seeds term k with 20261013+k.
//   Bit-flip noise p uses a second stream seeded 4242+k. p == 0 flips nothing.
//   source file reads four shot lists and does not sample.
//
// Gates: h, cx, ry, rz, measure-at-end. Two qubits. LSB is q[0].
// index = q0 + 2*q1.
//
//   device_run score.sio                 stdin is the document
//   device_run --selftest                no stdin
//   device_run score.sio a.txt b.txt c.txt d.txt

#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kQ = 4;

struct Op {
    int kind;
    int a;
    int b;
    double angle;
};

enum { kH = 1, kRy = 9, kRz = 10, kCx = 11, kMeas = 15 };

using Amp = std::complex<double>;

int wire_at(const std::string& s, std::size_t from) {
    auto b = s.find("q[", from);
    if (b == std::string::npos) return -1;
    auto e = s.find(']', b);
    if (e == std::string::npos) return -1;
    return std::stoi(s.substr(b + 2, e - (b + 2)));
}

bool parse_op(const std::string& line, Op& op) {
    if (line.rfind("h ", 0) == 0) {
        op = Op{kH, wire_at(line, 0), 0, 0.0};
        return op.a >= 0 && op.a < 2;
    }
    if (line.rfind("cx ", 0) == 0) {
        int c = wire_at(line, 0);
        auto comma = line.find(',');
        int t = wire_at(line, comma == std::string::npos ? 0 : comma);
        op = Op{kCx, c, t, 0.0};
        return c >= 0 && t >= 0 && c < 2 && t < 2 && c != t;
    }
    if (line.rfind("ry(", 0) == 0 || line.rfind("rz(", 0) == 0) {
        auto e = line.find(')');
        if (e == std::string::npos) return false;
        double ang = std::stod(line.substr(3, e - 3));
        int w = wire_at(line, e);
        int kind = line[1] == 'y' ? kRy : kRz;
        op = Op{kind, w, 0, ang};
        return w >= 0 && w < 2;
    }
    if (line.find("measure") != std::string::npos) {
        int w = wire_at(line, 0);
        op = Op{kMeas, w, 0, 0.0};
        return w >= 0 && w < 2;
    }
    return false;
}

bool skip_line(const std::string& line) {
    return line.rfind("OPENQASM", 0) == 0 || line.rfind("include ", 0) == 0 ||
           line.rfind("qubit[", 0) == 0 || line.rfind("bit[", 0) == 0;
}

void apply(std::vector<Amp>& a, const Op& op) {
    std::vector<Amp> n(kQ, Amp{0.0, 0.0});
    if (op.kind == kH) {
        const double s = std::sqrt(0.5);
        int q = op.a;
        for (int i = 0; i < kQ; i++) {
            if ((i >> q) & 1) continue;
            int j = i | (1 << q);
            n[i] += s * a[i] + s * a[j];
            n[j] += s * a[i] - s * a[j];
        }
        a.swap(n);
        return;
    }
    if (op.kind == kRy) {
        double c = std::cos(op.angle / 2.0);
        double s = std::sin(op.angle / 2.0);
        int q = op.a;
        for (int i = 0; i < kQ; i++) {
            if ((i >> q) & 1) continue;
            int j = i | (1 << q);
            n[i] += c * a[i] - s * a[j];
            n[j] += s * a[i] + c * a[j];
        }
        a.swap(n);
        return;
    }
    if (op.kind == kRz) {
        // OpenQASM rz: diag(exp(-i*theta/2), exp(+i*theta/2)).
        const Amp m0 = std::exp(Amp{0.0, -0.5 * op.angle});
        const Amp m1 = std::exp(Amp{0.0, 0.5 * op.angle});
        int q = op.a;
        for (int i = 0; i < kQ; i++) {
            if ((i >> q) & 1) a[i] *= m1;
            else a[i] *= m0;
        }
        return;
    }
    if (op.kind == kCx) {
        int ctrl = op.a;
        int tgt = op.b;
        for (int i = 0; i < kQ; i++) {
            if ((i >> ctrl) & 1) {
                int j = i ^ (1 << tgt);
                n[j] += a[i];
            } else {
                n[i] += a[i];
            }
        }
        a.swap(n);
    }
}

struct Run {
    double p[4];
    std::vector<int> shots;
};

Run execute(const std::vector<Op>& ops, int shots, std::uint64_t seed, double noise = 0.0,
            std::uint64_t noise_seed = 0) {
    std::vector<Amp> a(kQ, Amp{0.0, 0.0});
    a[0] = Amp{1.0, 0.0};
    for (const Op& op : ops) {
        if (op.kind != kMeas) apply(a, op);
    }
    Run r;
    double sum = 0.0;
    for (int i = 0; i < kQ; i++) {
        r.p[i] = std::norm(a[i]);
        sum += r.p[i];
    }
    for (int i = 0; i < kQ; i++) r.p[i] /= sum;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    r.shots.reserve(static_cast<std::size_t>(shots));
    // p == 0 keeps the original single-stream sampler, including its draws.
    if (noise == 0.0) {
        for (int n = 0; n < shots; n++) {
            double u = dist(rng);
            double c0 = r.p[0];
            double c1 = c0 + r.p[1];
            double c2 = c1 + r.p[2];
            int k = 3;
            if (u < c0) k = 0;
            else if (u < c1) k = 1;
            else if (u < c2) k = 2;
            r.shots.push_back(k);
        }
        return r;
    }
    std::mt19937_64 noise_rng(noise_seed);
    std::uniform_real_distribution<double> ndist(0.0, 1.0);
    for (int n = 0; n < shots; n++) {
        double u = dist(rng);
        double c0 = r.p[0];
        double c1 = c0 + r.p[1];
        double c2 = c1 + r.p[2];
        int k = 3;
        if (u < c0) k = 0;
        else if (u < c1) k = 1;
        else if (u < c2) k = 2;
        // q0 is the low bit. Flip each bit from the second stream.
        if (ndist(noise_rng) < noise) k ^= 1;
        if (ndist(noise_rng) < noise) k ^= 2;
        r.shots.push_back(k);
    }
    return r;
}

struct Face {
    int n00, n01, n10, n11;
    double corr;
};

Face face_of(const std::vector<int>& shots) {
    Face f{0, 0, 0, 0, 0.0};
    for (int k : shots) {
        if (k == 0) f.n00++;
        else if (k == 1) f.n01++;
        else if (k == 2) f.n10++;
        else f.n11++;
    }
    double n = static_cast<double>(shots.size());
    f.corr = (f.n00 + f.n11 - f.n01 - f.n10) / n;
    return f;
}

void emit_var(std::ostream& o, const char* name, const std::vector<int>& shots) {
    static const char* word[] = {"00", "01", "10", "11"};
    o << "    var " << name << " = \"\"\n";
    const std::size_t chunk = 80;
    for (std::size_t i = 0; i < shots.size(); i += chunk) {
        o << "    " << name << " = str_concat(" << name << ", \"";
        std::size_t end = shots.size() < i + chunk ? shots.size() : i + chunk;
        for (std::size_t k = i; k < end; k++) {
            o << word[shots[k]] << "\\n";
        }
        o << "\")\n";
    }
}

std::string trunc12(double x) {
    double t = std::floor(x * 1e12) / 1e12;
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(12);
    o << t;
    return o.str();
}

std::string program_of(const std::string& body) {
    return std::string("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nbit[2] c;\n") + body +
           "c[0] = measure q[0];\nc[1] = measure q[1];\n";
}

std::vector<Op> must_parse(const std::string& text) {
    std::vector<Op> ops;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || skip_line(line)) continue;
        Op op;
        if (!parse_op(line, op)) {
            std::cerr << "bad qasm line: " << line << "\n";
            std::exit(2);
        }
        ops.push_back(op);
    }
    return ops;
}

void print_run(const char* name, const Run& r) {
    Face f = face_of(r.shots);
    std::cout << name << " probs " << r.p[0] << " " << r.p[1] << " " << r.p[2] << " " << r.p[3]
              << "\n";
    std::cout << name << " counts " << f.n00 << " " << f.n01 << " " << f.n10 << " " << f.n11
              << "\n";
    std::cout << name << " corr " << f.corr << "\n";
}

void print_face(const char* name, const std::vector<int>& shots) {
    Face f = face_of(shots);
    std::cout << name << " counts " << f.n00 << " " << f.n01 << " " << f.n10 << " " << f.n11
              << "\n";
    std::cout << name << " corr " << f.corr << "\n";
}

std::string read_document(std::istream& in) {
    std::string doc;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "end") break;
        doc.push_back('\n');
        doc += line;
    }
    return doc;
}

bool is_v2(const std::string& doc) {
    std::istringstream in(doc);
    std::string line;
    while (std::getline(in, line)) {
        if (line == "sounio-device 2") return true;
    }
    return false;
}

struct Score {
    Face face[4];
    int sign[4];
    double s;
    int keep;
    std::string source;
};

void finish_score(Score& sc) {
    sc.s = 0.0;
    for (int k = 0; k < 4; k++) sc.s += static_cast<double>(sc.sign[k]) * sc.face[k].corr;
    sc.keep = sc.s > 2.0 ? 1 : 0;
}

std::string report_of(const Score& sc) {
    std::ostringstream o;
    for (int k = 0; k < 4; k++) {
        const Face& f = sc.face[k];
        o << "term " << k << " counts " << f.n00 << " " << f.n01 << " " << f.n10 << " " << f.n11
          << " corr " << f.corr << " sign " << (sc.sign[k] > 0 ? "+" : "-") << "\n";
    }
    o << "s " << sc.s << "\n";
    o << "keep " << sc.keep << "\n";
    o << "source " << sc.source << "\n";
    return o.str();
}

bool read_shot_file(const char* path, std::vector<int>& shots) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "cannot read " << path << "\n";
        return false;
    }
    shots.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        int k = -1;
        if (line == "00") k = 0;
        else if (line == "01") k = 1;
        else if (line == "10") k = 2;
        else if (line == "11") k = 3;
        else {
            std::cerr << "bad shot line: " << line << "\n";
            return false;
        }
        shots.push_back(k);
    }
    if (shots.empty()) {
        std::cerr << "empty shot list " << path << "\n";
        return false;
    }
    return true;
}

struct Job2 {
    int shots = 0;
    double noise = 0.0;
    std::string source;
    int n = 0;
    int sign[4] = {0, 0, 0, 0};
    std::vector<Op> ops[4];
};

int parse_sign(const std::string& line, int& sign) {
    if (line.size() < 6 || line.rfind("sign ", 0) != 0) return 2;
    std::string sg = line.substr(5);
    if (sg == "+") sign = 1;
    else if (sg == "-") sign = -1;
    else {
        std::cerr << "bad sign: " << line << "\n";
        return 2;
    }
    return 0;
}

int parse_v2(const std::string& doc, Job2& job) {
    std::istringstream in(doc);
    std::string line;
    int term = -1;
    bool saw_shots = false;
    bool saw_noise = false;
    bool saw_source = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line == "sounio-device 2") continue;
        if (line.rfind("sign ", 0) == 0) {
            if (job.n >= 4) {
                std::cerr << "more than four terms\n";
                return 2;
            }
            if (parse_sign(line, job.sign[job.n]) != 0) return 2;
            term = job.n;
            job.n++;
            continue;
        }
        if (term < 0) {
            if (line.rfind("shots ", 0) == 0) {
                job.shots = std::stoi(line.substr(6));
                saw_shots = true;
                continue;
            }
            if (line.rfind("cut ", 0) == 0) continue;
            if (line.rfind("noise ", 0) == 0) {
                job.noise = std::stod(line.substr(6));
                saw_noise = true;
                continue;
            }
            if (line.rfind("source ", 0) == 0) {
                job.source = line.substr(7);
                saw_source = true;
                continue;
            }
            std::cerr << "bad v2 header: " << line << "\n";
            return 2;
        }
        if (skip_line(line)) continue;
        Op op;
        if (!parse_op(line, op)) {
            std::cerr << "bad qasm line: " << line << "\n";
            return 2;
        }
        job.ops[term].push_back(op);
    }
    if (job.n != 4) {
        std::cerr << "need four terms\n";
        return 2;
    }
    if (!saw_shots || job.shots < 1) {
        std::cerr << "bad shots\n";
        return 2;
    }
    if (!saw_noise || !(job.noise >= 0.0 && job.noise <= 0.5)) {
        std::cerr << "bad noise\n";
        return 2;
    }
    if (!saw_source || (job.source != "sampler" && job.source != "file")) {
        std::cerr << "bad source\n";
        return 2;
    }
    return 0;
}

int run_v2(const std::string& doc, const char* out_path, const char* const* files, int nfiles, Score* score) {
    Job2 job;
    int pr = parse_v2(doc, job);
    if (pr != 0) return pr;
    Score sc;
    sc.source = job.source;
    if (job.source == "file") {
        if (files == nullptr || nfiles < 4) {
            std::cerr << "source file needs four shot lists\n";
            return 2;
        }
        for (int k = 0; k < 4; k++) {
            std::vector<int> shots;
            if (!read_shot_file(files[k], shots)) return 2;
            sc.face[k] = face_of(shots);
            sc.sign[k] = job.sign[k];
        }
    } else {
        for (int k = 0; k < 4; k++) {
            Run r = execute(job.ops[k], job.shots, static_cast<std::uint64_t>(20261013 + k), job.noise,
                            static_cast<std::uint64_t>(4242 + k));
            sc.face[k] = face_of(r.shots);
            sc.sign[k] = job.sign[k];
        }
    }
    finish_score(sc);
    std::string text = report_of(sc);
    std::cout << text;
    if (out_path != nullptr) {
        std::ofstream o(out_path);
        if (!o) {
            std::cerr << "cannot write " << out_path << "\n";
            return 2;
        }
        o << text;
    }
    if (score != nullptr) *score = sc;
    return 0;
}

int run_v1(const std::string& doc, const char* out_path) {
    int shots = 1000;
    std::vector<Op> zops;
    std::vector<Op> xops;
    int basis = 0;
    std::istringstream in(doc);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("shots ", 0) == 0) {
            shots = std::stoi(line.substr(6));
            continue;
        }
        if (line == "basis z") {
            basis = 1;
            continue;
        }
        if (line == "basis x") {
            basis = 2;
            continue;
        }
        if (line.empty() || line.rfind("sounio-device", 0) == 0 || line.rfind("cut ", 0) == 0) {
            continue;
        }
        if (basis == 0 || skip_line(line)) continue;
        Op op;
        if (!parse_op(line, op)) {
            std::cerr << "bad job line: " << line << "\n";
            return 2;
        }
        if (basis == 1) zops.push_back(op);
        else xops.push_back(op);
    }
    if (zops.empty() || xops.empty() || shots < 1) {
        std::cerr << "job missing a basis\n";
        return 2;
    }

    Run z = execute(zops, shots, 20261013);
    Run x = execute(xops, shots, 20261014);
    print_run("bell_z", z);
    print_run("bell_x", x);
    if (z.p[1] > 1e-8 || z.p[2] > 1e-8 || x.p[1] > 1e-8 || x.p[2] > 1e-8) {
        std::cerr << "bell probabilities leaked off the pair\n";
        return 3;
    }

    const double pi = 3.141592653589793;
    const std::string half = trunc12(pi / 2.0);
    const std::string full = trunc12(pi);
    std::string ry_half = "ry(" + half + ") q[0];\nry(" + half + ") q[1];\n";
    std::string ry_full = "ry(" + full + ") q[0];\nry(" + full + ") q[1];\n";
    Run c00 = execute(must_parse(program_of("")), shots, 20261015);
    Run c11 = execute(must_parse(program_of(ry_full)), shots, 20261016);
    Run x00 = execute(must_parse(program_of(ry_half)), shots, 20261017);
    Run x11 = execute(must_parse(program_of(ry_full + ry_half)), shots, 20261018);
    std::vector<int> cz = c00.shots;
    cz.insert(cz.end(), c11.shots.begin(), c11.shots.end());
    std::vector<int> cx = x00.shots;
    cx.insert(cx.end(), x11.shots.begin(), x11.shots.end());
    print_face("copy_z", cz);
    print_face("copy_x", cx);

    std::ofstream o(out_path);
    if (!o) {
        std::cerr << "cannot write " << out_path << "\n";
        return 2;
    }
    o << "use quantum::qasm_tape::{\n"
         "    tape_device_return,\n"
         "    tape_dec,\n"
         "    DeviceRead,\n"
         "}\n"
         "\n"
         "fn main() -> i32 with Mut, Div, Panic, IO {\n";
    emit_var(o, "bz", z.shots);
    emit_var(o, "bx", x.shots);
    emit_var(o, "cz", cz);
    emit_var(o, "cx", cx);
    o << "    let bell: DeviceRead = tape_device_return(bz, bx)\n"
         "    let copy: DeviceRead = tape_device_return(cz, cx)\n"
         "    print(tape_dec(bell.zz))\n"
         "    print(\" \")\n"
         "    print(tape_dec(bell.xx))\n"
         "    print(\" \")\n"
         "    if bell.pair == 1 { print(\"1\") } else { print(\"0\") }\n"
         "    print(\" \")\n"
         "    if bell.oracle == 0 { print(\"0\\n\") } else { print(\"1\\n\") }\n"
         "    print(tape_dec(copy.zz))\n"
         "    print(\" \")\n"
         "    print(tape_dec(copy.xx))\n"
         "    print(\" \")\n"
         "    if copy.pair == 1 { print(\"1\") } else { print(\"0\") }\n"
         "    print(\" \")\n"
         "    if copy.oracle == 0 { print(\"0\\n\") } else { print(\"1\\n\") }\n"
         "    return 0\n"
         "}\n";
    Face bz = face_of(z.shots);
    Face bx = face_of(x.shots);
    Face fz = face_of(cz);
    Face fx = face_of(cx);
    if (bz.corr < 0.5 || bx.corr < 0.5) {
        std::cerr << "bell correlator fell through 1/2\n";
        return 4;
    }
    if (fz.corr < 0.5 || std::fabs(fx.corr) > 0.15) {
        std::cerr << "copy did not separate\n";
        return 4;
    }
    return 0;
}

std::string bell_blocks() {
    const double pi = 3.141592653589793;
    const double A[4] = {0.0, 0.0, pi / 2.0, pi / 2.0};
    const double B[4] = {pi / 4.0, -pi / 4.0, pi / 4.0, -pi / 4.0};
    const char* sg[4] = {"+", "+", "+", "-"};
    std::ostringstream o;
    o.precision(17);
    for (int k = 0; k < 4; k++) {
        o << "sign " << sg[k] << "\n";
        o << "OPENQASM 3.0;\n";
        o << "include \"stdgates.inc\";\n";
        o << "qubit[2] q;\n";
        o << "bit[2] c;\n";
        o << "h q[0];\n";
        o << "cx q[0], q[1];\n";
        o << "ry(" << A[k] << ") q[0];\n";
        o << "ry(" << B[k] << ") q[1];\n";
        o << "c[0] = measure q[0];\n";
        o << "c[1] = measure q[1];\n";
    }
    return o.str();
}

std::string v2_text(const char* shots, const char* noise, const char* source, const std::string& blocks) {
    return std::string("sounio-device 2\nshots ") + shots + "\ncut 2\nnoise " + noise + "\nsource " + source +
           "\n" + blocks + "end\n";
}

int run_text(const std::string& text, const char* const* files, int nfiles, Score& sc) {
    std::istringstream in(text);
    std::string doc = read_document(in);
    return run_v2(doc, nullptr, files, nfiles, &sc);
}

int fail_s(const char* what, const Score& sc) {
    std::ostringstream m;
    m.precision(17);
    m << what << " S " << sc.s << " keep " << sc.keep << " source " << sc.source << " corrs";
    for (int k = 0; k < 4; k++) m << " " << sc.face[k].corr;
    std::cerr << m.str() << "\n";
    return 1;
}

int check_gates() {
    Op parsed;
    if (!parse_op("rz(1.25) q[1];", parsed) || parsed.kind != kRz || parsed.a != 1 || parsed.angle != 1.25) {
        std::cerr << "rz parse\n";
        return 1;
    }
    const double pi = 3.141592653589793;
    {
        std::vector<Op> ops;
        ops.push_back(Op{kH, 0, 0, 0.0});
        Run r = execute(ops, 1, 1);
        if (std::fabs(r.p[0] - 0.5) > 1e-9 || std::fabs(r.p[1] - 0.5) > 1e-9 || r.p[2] > 1e-9 || r.p[3] > 1e-9) {
            std::cerr << "H matrix\n";
            return 1;
        }
    }
    {
        std::vector<Op> ops;
        ops.push_back(Op{kRy, 0, 0, pi});
        ops.push_back(Op{kCx, 0, 1, 0.0});
        Run r = execute(ops, 1, 1);
        if (std::fabs(r.p[3] - 1.0) > 1e-9) {
            std::cerr << "CX matrix\n";
            return 1;
        }
    }
    {
        std::vector<Op> ops;
        ops.push_back(Op{kH, 0, 0, 0.0});
        ops.push_back(Op{kRz, 0, 0, pi});
        ops.push_back(Op{kH, 0, 0, 0.0});
        Run r = execute(ops, 1, 1);
        if (std::fabs(r.p[1] - 1.0) > 1e-8 || r.p[0] > 1e-8 || r.p[2] > 1e-8 || r.p[3] > 1e-8) {
            std::cerr << "rz pi\n";
            return 1;
        }
    }
    {
        const double th = 2.0;
        std::vector<Op> ops;
        ops.push_back(Op{kH, 1, 0, 0.0});
        ops.push_back(Op{kRz, 1, 0, th});
        ops.push_back(Op{kH, 1, 0, 0.0});
        Run r = execute(ops, 1, 1);
        double c2 = std::cos(th / 2.0);
        double s2 = std::sin(th / 2.0);
        if (std::fabs(r.p[0] - c2 * c2) > 1e-8 || std::fabs(r.p[2] - s2 * s2) > 1e-8 || r.p[1] > 1e-8 ||
            r.p[3] > 1e-8) {
            std::cerr << "rz half-angle\n";
            return 1;
        }
    }
    return 0;
}

bool write_text(const char* path, const std::string& body) {
    std::ofstream o(path);
    if (!o) {
        std::cerr << "cannot write " << path << "\n";
        return false;
    }
    o << body;
    return static_cast<bool>(o);
}

int selftest() {
    if (check_gates() != 0) return 1;

    const std::string blocks = bell_blocks();
    {
        Score sc;
        int rc = run_text(v2_text("2000", "0", "sampler", blocks), nullptr, 0, sc);
        if (rc != 0) return rc;
        if (!(std::fabs(sc.s - 2.8284271247461903) <= 0.12 && sc.s > 2.0 && sc.keep == 1 &&
              sc.source == "sampler")) {
            return fail_s("bell", sc);
        }
    }
    {
        const double pi = 3.141592653589793;
        const double A[4] = {0.0, 0.0, pi / 2.0, pi / 2.0};
        const double B[4] = {pi / 4.0, -pi / 4.0, pi / 4.0, -pi / 4.0};
        const int signs[4] = {1, 1, 1, -1};
        Score sc;
        sc.source = "sampler";
        for (int k = 0; k < 4; k++) {
            std::ostringstream body00;
            body00.precision(17);
            body00 << "ry(" << A[k] << ") q[0];\nry(" << B[k] << ") q[1];\n";
            std::ostringstream body11;
            body11.precision(17);
            body11 << "ry(" << pi << ") q[0];\nry(" << pi << ") q[1];\nry(" << A[k] << ") q[0];\nry(" << B[k]
                   << ") q[1];\n";
            Run a = execute(must_parse(program_of(body00.str())), 2000, static_cast<std::uint64_t>(20261013 + k));
            Run b = execute(must_parse(program_of(body11.str())), 2000,
                            static_cast<std::uint64_t>(20261017 + k));
            std::vector<int> both = a.shots;
            both.insert(both.end(), b.shots.begin(), b.shots.end());
            sc.face[k] = face_of(both);
            sc.sign[k] = signs[k];
        }
        finish_score(sc);
        std::cout << report_of(sc);
        int tot = sc.face[0].n00 + sc.face[0].n01 + sc.face[0].n10 + sc.face[0].n11;
        if (tot != 4000 || sc.face[0].n00 < 1 || sc.face[0].n01 < 1 || sc.face[0].n10 < 1 || sc.face[0].n11 < 1) {
            std::cerr << "copy mixture did not add both preparations\n";
            return 1;
        }
        if (!(std::fabs(sc.s - 1.4142135623730951) <= 0.12 && sc.s < 2.0 && sc.keep == 0)) {
            return fail_s("copy", sc);
        }
    }
    {
        Score lo;
        int rc = run_text(v2_text("4000", "0.04", "sampler", blocks), nullptr, 0, lo);
        if (rc != 0) return rc;
        if (!(lo.s > 2.0 && lo.keep == 1)) return fail_s("noise-lo", lo);
        Score hi;
        rc = run_text(v2_text("4000", "0.12", "sampler", blocks), nullptr, 0, hi);
        if (rc != 0) return rc;
        if (!(hi.s < 2.0 && hi.keep == 0)) return fail_s("noise-hi", hi);
    }
    {
        std::string pos;
        for (int i = 0; i < 10; i++) pos += "00\n";
        pos += "\n";
        for (int i = 0; i < 10; i++) pos += "11\n";
        std::string neg;
        for (int i = 0; i < 10; i++) neg += "01\n";
        for (int i = 0; i < 10; i++) neg += "10\n";
        const char* paths[4] = {"/tmp/qgate-st-pos0.txt", "/tmp/qgate-st-pos1.txt", "/tmp/qgate-st-pos2.txt",
                                "/tmp/qgate-st-neg3.txt"};
        if (!write_text(paths[0], pos) || !write_text(paths[1], pos) || !write_text(paths[2], pos) ||
            !write_text(paths[3], neg)) {
            return 1;
        }
        std::ostringstream body;
        body << "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nbit[2] c;\n"
                "h q[0];\nrz(0.3) q[1];\nry(0.1) q[0];\ncx q[1], q[0];\n"
                "c[0] = measure q[0];\nc[1] = measure q[1];\n";
        std::string blocks_file;
        const char* sg[4] = {"+", "+", "+", "-"};
        for (int k = 0; k < 4; k++) {
            blocks_file += "sign ";
            blocks_file += sg[k];
            blocks_file += "\n";
            blocks_file += body.str();
        }
        Score sc;
        int rc = run_text(v2_text("20", "0", "file", blocks_file), paths, 4, sc);
        if (rc != 0) return rc;
        if (sc.source != "file" || sc.keep != 1 || std::fabs(sc.s - 4.0) > 1e-12) return fail_s("file-keep", sc);
        for (int k = 0; k < 3; k++) {
            if (sc.face[k].n00 != 10 || sc.face[k].n11 != 10 || sc.face[k].n01 != 0 || sc.face[k].n10 != 0) {
                return fail_s("file-keep-counts", sc);
            }
        }
        if (sc.face[3].n01 != 10 || sc.face[3].n10 != 10 || sc.face[3].n00 != 0 || sc.face[3].n11 != 0) {
            return fail_s("file-keep-neg", sc);
        }
        if (sc.sign[0] != 1 || sc.sign[1] != 1 || sc.sign[2] != 1 || sc.sign[3] != -1) {
            return fail_s("file-signs", sc);
        }

        // 00,00,11,01 -> (2+1-1-0)/4 = 0.5 exactly. S = 0.5+0.5+0.5-0.5 = 1.
        const std::string half = "00\n00\n11\n01\n";
        const char* halves[4] = {"/tmp/qgate-st-half0.txt", "/tmp/qgate-st-half1.txt", "/tmp/qgate-st-half2.txt",
                                 "/tmp/qgate-st-half3.txt"};
        for (int k = 0; k < 4; k++) {
            if (!write_text(halves[k], half)) return 1;
        }
        Score drop;
        rc = run_text(v2_text("4", "0", "file", blocks_file), halves, 4, drop);
        if (rc != 0) return rc;
        if (drop.source != "file" || drop.keep != 0 || std::fabs(drop.s - 1.0) > 1e-12) return fail_s("file-drop", drop);
        for (int k = 0; k < 4; k++) {
            if (std::fabs(drop.face[k].corr - 0.5) > 1e-12) return fail_s("file-drop-corr", drop);
        }
    }
    {
        const char* v1 =
            "sounio-device 1\n"
            "shots 1000\n"
            "cut 2\n"
            "basis z\n"
            "OPENQASM 3.0;\n"
            "include \"stdgates.inc\";\n"
            "qubit[2] q;\n"
            "bit[2] c;\n"
            "h q[0];\n"
            "cx q[0], q[1];\n"
            "c[0] = measure q[0];\n"
            "c[1] = measure q[1];\n"
            "basis x\n"
            "OPENQASM 3.0;\n"
            "include \"stdgates.inc\";\n"
            "qubit[2] q;\n"
            "bit[2] c;\n"
            "h q[0];\n"
            "cx q[0], q[1];\n"
            "h q[0];\n"
            "h q[1];\n"
            "c[0] = measure q[0];\n"
            "c[1] = measure q[1];\n"
            "end\n";
        std::istringstream in(v1);
        std::string doc = read_document(in);
        const char* out = "/tmp/qgate-st-v1.sio";
        int rc = run_v1(doc, out);
        if (rc != 0) return rc;
        std::ifstream vin(out);
        std::stringstream vs;
        vs << vin.rdbuf();
        if (!vin || vs.str().find("tape_device_return") == std::string::npos ||
            vs.str().find("basis") != std::string::npos) {
            std::cerr << "v1 sio\n";
            return 1;
        }
    }
    std::cout << "SELFTEST_PASS\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--selftest") {
        return selftest();
    }
    if (argc < 2) {
        std::cerr << "usage: device_run score.sio\n";
        return 2;
    }
    std::string doc = read_document(std::cin);
    if (is_v2(doc)) {
        const char* files[4] = {nullptr, nullptr, nullptr, nullptr};
        int nfiles = 0;
        if (argc >= 6) {
            for (int i = 0; i < 4; i++) files[i] = argv[2 + i];
            nfiles = 4;
        }
        return run_v2(doc, argv[1], files, nfiles, nullptr);
    }
    return run_v1(doc, argv[1]);
}
