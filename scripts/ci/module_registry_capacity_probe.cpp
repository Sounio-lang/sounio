// Extract production Sounio registry functions, never a C++ replacement model.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
namespace fs = std::filesystem;
std::string read(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(in), {}};
}
std::string definition(const std::string& source, const std::string& start) {
    const auto first = source.find(start);
    const auto last = source.find("\n}\n", first);
    if (first == std::string::npos || last == std::string::npos)
        throw std::runtime_error("missing production definition: " + start);
    return source.substr(first, last + 3 - first) + "\n";
}
void globals(std::ostream& out, const std::string& source, const std::string& prefix) {
    std::istringstream lines(source);
    for (std::string line; std::getline(lines, line);)
        if (line.starts_with("var " + prefix) || line.starts_with("let " + prefix))
            out << line << '\n';
}
int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("usage: probe SOURCE_ROOT OUTPUT.sio");
    const fs::path root(argv[1]);
    const auto checker = read(root / "self-hosted/check/check.sio");
    const auto spec = read(root / "self-hosted/check/specializer.sio");
    const auto lower = read(root / "self-hosted/ir/lower.sio");
    const auto ast = read(root / "self-hosted/parser/ast.sio");
    std::ofstream out(argv[2]);
    if (!out) throw std::runtime_error("cannot write fixture");
    auto hash = definition(ast, "pub fn ast_name_hash(");
    // The bootstrap seed requires Mut/Panic on local mutation/indexing;
    // the modular checker does not. Widen effects only, preserving the body.
    const auto effects = hash.find("with Div {");
    if (effects == std::string::npos) throw std::runtime_error("hash effects changed");
    hash.replace(effects, std::string("with Div {").size(), "with Mut, Panic, Div {");
    out << definition(ast, "pub struct Name {") << hash
        << definition(lower, "fn fo_name_hash(");
    globals(out, checker, "CHECKER_USE_MODULE_");
    globals(out, spec, "SPEC_USE_MODULE_");
    globals(out, lower, "LOWER_USE_MODULE_");
    for (const auto* name : {"SPEC_CAP_OVERFLOW", "SPEC_CAP_OVERFLOW_CODE"}) {
        std::istringstream lines(spec);
        for (std::string line; std::getline(lines, line);)
            if (line.starts_with(std::string("var ") + name + ":")) out << line << '\n';
    }
    for (const auto* name : {"IR_LOWER_HARD_ERROR", "IR_LOWER_HARD_ERROR_REASON", "IR_LOWER_HARD_ERROR_LIMIT"}) {
        std::istringstream lines(lower);
        for (std::string line; std::getline(lines, line);)
            if (line.starts_with(std::string("var ") + name + ":")) out << line << '\n';
    }
    out << definition(spec, "fn spec_mark_cap_overflow(");
    for (const auto* name : {"checker_use_module_known", "checker_use_module_note"})
        out << definition(checker, std::string("fn ") + name + "(");
    for (const auto* name : {"spec_use_module_known", "spec_use_module_note"})
        out << definition(spec, std::string("fn ") + name + "(");
    for (const auto* name : {"lower_use_module_known", "lower_use_module_note"})
        out << definition(lower, std::string("fn ") + name + "(");
    out << "var PROBE_TRUE: bool = true\nvar PROBE_FALSE: bool = false\nvar PROBE_GUARD: i64 = 37\n";
    out << "fn main() -> i32 with IO, Mut, Div, Panic {\n";
    out << "if !PROBE_TRUE || PROBE_FALSE || PROBE_GUARD != 37 { return 12 }\n";
    out << "var a = Name { buf: [0; 384], len: 2 }\na.buf[0] = 97\na.buf[1] = 98\n";
    out << "var b = Name { buf: [0; 384], len: 2 }\nb.buf[0] = 98\nb.buf[1] = 65\n";
    out << "if ast_name_hash(a) != ast_name_hash(b) || fo_name_hash(a) != fo_name_hash(b) { return 2 }\n";
    for (const auto* prefix : {"checker", "spec", "lower"}) {
        out << prefix << "_use_module_note(0, a)\n";
        out << "if " << prefix << "_use_module_known(0, b) || " << prefix << "_use_module_known(1, a) { return 3 }\n";
        out << prefix << "_use_module_note(0, b)\n";
        out << "if !" << prefix << "_use_module_known(0, a) || !" << prefix << "_use_module_known(0, b) { return 4 }\n";
    }
    out << "CHECKER_USE_MODULE_COUNT = 0\nSPEC_USE_MODULE_COUNT = 0\nLOWER_USE_MODULE_COUNT = 0\n";
    out << "var n = Name { buf: [0; 384], len: 1 }\nn.buf[0] = 109\nvar i: i64 = 0\nwhile i < 4096 {\n";
    for (const auto* prefix : {"checker", "spec", "lower"}) out << prefix << "_use_module_note(i, n)\n";
    out << "i = i + 1\n}\n";
    out << "if CHECKER_USE_MODULE_COUNT != 4096 || SPEC_USE_MODULE_COUNT != 4096 || LOWER_USE_MODULE_COUNT != 4096 { return 5 }\n";
    out << "if CHECKER_USE_MODULE_OVERFLOWED || SPEC_CAP_OVERFLOW || LOWER_USE_MODULE_OVERFLOWED { return 6 }\n";
    for (const auto* prefix : {"checker", "spec", "lower"}) {
        out << prefix << "_use_module_note(0, n)\n";
        out << "if !" << prefix << "_use_module_known(4095, n) || " << prefix << "_use_module_known(4096, n) { return 7 }\n";
        out << prefix << "_use_module_note(4096, n)\n";
        out << "if " << prefix << "_use_module_known(4096, n) { return 8 }\n";
    }
    out << "if !CHECKER_USE_MODULE_OVERFLOWED || !SPEC_CAP_OVERFLOW || !LOWER_USE_MODULE_OVERFLOWED { return 9 }\n";
    out << "if SPEC_CAP_OVERFLOW_CODE != 7 || IR_LOWER_HARD_ERROR != 1 || IR_LOWER_HARD_ERROR_REASON != 3 || IR_LOWER_HARD_ERROR_LIMIT != 4096 { return 10 }\n";
    out << "if CHECKER_USE_MODULE_COUNT != 4096 || SPEC_USE_MODULE_COUNT != 4096 || LOWER_USE_MODULE_COUNT != 4096 { return 11 }\n";
    out << "PROBE_TRUE = false\nPROBE_FALSE = true\n";
    out << "if PROBE_TRUE || !PROBE_FALSE || PROBE_GUARD != 37 { return 13 }\n";
    out << "println(\"registry_capacity_ok: collision, isolation, duplicate, 4096/4097 refusal\")\nreturn 0\n}\n";
    return out ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
