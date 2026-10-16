// Exercise the production TypeEnv, including its real TypeEntry layout.
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <iostream>
#include <regex>
namespace fs = std::filesystem;
std::string read(const fs::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    return {std::istreambuf_iterator<char>(f), {}};
}
std::string definition(const std::string& s, const std::string& n) {
    auto a = s.find(n), b = s.find("\n}\n", a);
    if (a == std::string::npos || b == std::string::npos) throw std::runtime_error("missing " + n);
    return s.substr(a, b + 3 - a) + "\n";
}
int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("usage: generator ROOT OUTPUT_DIR");
    const fs::path root(argv[1]), out(argv[2]);
    fs::create_directories(out);
    auto ast = read(root / "self-hosted/parser/ast.sio");
    auto types = read(root / "self-hosted/check/types.sio");
    auto env = read(root / "self-hosted/check/env.sio");
    auto constant = [](const std::string& text, const std::string& name) {
        std::smatch match;
        if (!std::regex_search(text, match, std::regex("pub let " + name + ": i64 = ([0-9]+)")))
            throw std::runtime_error("missing numeric capacity " + name);
        return std::stoi(match[1].str());
    };
    const auto capacity = constant(env, "TYPE_ENV_CAPACITY");
    const auto globals = constant(read(root / "self-hosted/ir/ir.sio"), "BSS_MAX_GLOBALS");
    std::ofstream p(out / "probe.sio");
    p << definition(ast, "pub struct Name {") << definition(ast, "pub fn name_eq(")
      << definition(types, "pub enum TypeKind {") << definition(types, "pub struct TypeEntry {")
      << definition(types, "pub struct TypeEntryList {") << definition(types, "pub fn empty_type_entry(")
      << definition(types, "pub fn ty_error(");
    std::istringstream lines(env);
    for (std::string line; std::getline(lines, line);)
        if (!line.starts_with("use ")) p << line << '\n';
    p << R"(
fn probe_name(i: i64) -> Name with Mut, Panic, Div {
    var n = Name { buf: [0; 384], len: 2 }
    n.buf[0] = (i % 256) as i8
    n.buf[1] = (i / 256) as i8
    n
}
fn main() -> i64 with IO, Mut, Panic, Div, Alloc {
    var e = type_env_new()
    var outer_ty = empty_type_entry()
    outer_ty.fn_sig_id = 11
    e = e.bind(probe_name(0), outer_ty, false)
    e = e.push_scope()
    var inner_ty = empty_type_entry()
    inner_ty.fn_sig_id = 22
    e = e.bind(probe_name(0), inner_ty, true)
    if e.lookup(probe_name(0)).fn_sig_id != 22 { return 1 }
    if !e.lookup_is_mutable(probe_name(0)) { return 2 }
    e = e.pop_scope()
    if e.lookup(probe_name(0)).fn_sig_id != 11 {
        print("scope_restore count=")
        print_int(e.count)
        print(" depth=")
        print_int(e.scope_depth)
        print(" value=")
        print_int(e.lookup(probe_name(0)).fn_sig_id)
        print("\n")
        return 3
    }
    if e.lookup_is_mutable(probe_name(0)) { return 4 }
    // Exercise bind(), the path actually used by checker locals. The private
    // type_env_bind_mut helper currently has no production call sites.
    var t = empty_type_entry()
    var i: i64 = 1
    while i < TYPE_ENV_CAPACITY {
        t.fn_sig_id = i
        e = e.bind(probe_name(i), t, false)
        i = i + 1
    }
    if e.count != TYPE_ENV_CAPACITY || e.overflowed { return 5 }
    if e.lookup(probe_name(TYPE_ENV_CAPACITY - 1)).fn_sig_id != TYPE_ENV_CAPACITY - 1 { return 6 }
    e = e.push_scope()
    e = e.bind(probe_name(TYPE_ENV_CAPACITY), t, false)
    if !e.overflowed || e.count != TYPE_ENV_CAPACITY { return 7 }
    e = e.pop_scope()
    if !e.overflowed || e.count != TYPE_ENV_CAPACITY { return 8 }
    e = e.bind(probe_name(TYPE_ENV_CAPACITY), t, false)
    if !e.overflowed || e.count != TYPE_ENV_CAPACITY { return 9 }
    e = type_env_new()
    if e.count != 0 || e.overflowed || e.overflow_reported { return 10 }
    print("TYPE_ENV_CAPACITY_OK\n")
    0
}
)";
    // True compiler front-end controls, not just a container model.
    for (const auto& name : {"pressure", "global-boundary", "overflow"}) {
        std::ofstream f(out / (std::string(name) + ".sio"));
        int count = std::string(name) == "overflow" ? capacity + 1 :
            (std::string(name) == "global-boundary" ? globals : 4090);
        for (int i = 0; i < count; ++i) f << "var g" << i << ": i64 = 7\n";
        if (std::string(name) == "overflow") {
            f << "fn main() -> i64 { 0 }\n";
            continue;
        }
        f << "fn main() -> i64 with IO, Mut, Panic {\n";
        for (int i = 0; i < 16; ++i)
            f << "let local" << i << " = 7\nif local" << i << " != 7 { return 1 }\n";
        f << "print(\"ENV_BINDING_OK\\n\")\n0\n}\n";
    }
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
