// Reject references to nonexistent AST expression variants before a self-build.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
namespace fs = std::filesystem;
std::string source(const fs::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    std::string s{std::istreambuf_iterator<char>(f), {}};
    auto original = s;
    for (std::size_t i = 0; i < s.size();) {
        auto begin = i;
        if (original[i] == '"' || original[i] == '\'') {
            char quote = original[i++];
            while (i < s.size()) {
                if (original[i] == '\\') { i += (i + 1 < s.size() ? 2 : 1); continue; }
                if (original[i++] == quote) break;
            }
        } else if (original.compare(i, 2, "//") == 0) {
            while (i < s.size() && original[i] != '\n') ++i;
        } else if (original.compare(i, 2, "/*") == 0) {
            i += 2;
            while (i + 1 < s.size() && original.compare(i, 2, "*/") != 0) ++i;
            i = i + 1 < s.size() ? i + 2 : s.size();
        } else { ++i; continue; }
        for (auto j = begin; j < i; ++j) if (s[j] != '\n') s[j] = ' ';
    }
    return s;
}
int main(int argc, char** argv) try {
    fs::path root = argc == 2 ? argv[1] : ".";
    auto ast = source(root / "self-hosted/parser/ast.sio");
    auto a = ast.find("pub enum ExprKind");
    if (a == std::string::npos) throw std::runtime_error("missing ExprKind enum");
    a = ast.find('{', a);
    auto b = ast.find('}', a);
    if (a == std::string::npos || b == std::string::npos) throw std::runtime_error("invalid ExprKind enum");
    auto body = ast.substr(a, b - a);
    std::regex variant(R"(\bExpr\w+\b)"), reference(R"(\bExprKind\s*::\s*(\w+))");
    std::set<std::string> variants;
    for (auto i = std::sregex_iterator(body.begin(), body.end(), variant); i != std::sregex_iterator(); ++i)
        variants.insert(i->str());
    if (variants.empty()) throw std::runtime_error("empty ExprKind surface");
    bool ok = true;
    std::size_t references = 0;
    const std::regex imports_ast(R"(\buse\s+parser\s*::\s*ast\b)");
    for (const auto& entry : fs::recursive_directory_iterator(root / "self-hosted")) {
        if (!entry.is_regular_file() || entry.path().extension() != ".sio") continue;
        auto s = source(entry.path());
        // Other compiler prototypes own a different enum with the same name.
        // Audit only explicit consumers of parser::ast, not unrelated enums.
        if (!std::regex_search(s, imports_ast) && entry.path() != root / "self-hosted/parser/ast.sio") continue;
        for (auto i = std::sregex_iterator(s.begin(), s.end(), reference); i != std::sregex_iterator(); ++i) {
            ++references;
            if (!variants.contains((*i)[1].str())) {
                std::cerr << entry.path() << ": nonexistent ExprKind::" << (*i)[1] << '\n';
                ok = false;
            }
        }
    }
    if (references == 0) throw std::runtime_error("no explicit parser::ast consumers audited");
    if (!ok) return 1;
    std::cout << "AST_EXPR_KIND_SURFACE_OK\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
