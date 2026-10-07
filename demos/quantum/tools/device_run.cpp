// Runs the OpenQASM inside a sounio-device document and writes a Sounio
// program that scores the shot lists with tape_device_return.
// Gates: h, cx, ry, measure-at-end. Two qubits. LSB is q[0].
//
//   device_bell.elf | device_run /tmp/device_score.sio

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

enum { kH = 1, kRy = 9, kCx = 11, kMeas = 15 };

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
    if (line.rfind("ry(", 0) == 0) {
        auto e = line.find(')');
        if (e == std::string::npos) return false;
        double ang = std::stod(line.substr(3, e - 3));
        int w = wire_at(line, e);
        op = Op{kRy, w, 0, ang};
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

Run execute(const std::vector<Op>& ops, int shots, std::uint64_t seed) {
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: device_run score.sio\n";
        return 2;
    }
    std::string doc;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "end") break;
        doc.push_back('\n');
        doc += line;
    }

    int shots = 1000;
    std::vector<Op> zops;
    std::vector<Op> xops;
    int basis = 0;
    std::istringstream in(doc);
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

    std::ofstream o(argv[1]);
    if (!o) {
        std::cerr << "cannot write " << argv[1] << "\n";
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
