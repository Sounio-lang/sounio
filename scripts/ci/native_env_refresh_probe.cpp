// Extract production readers/refresh functions; instrument reads, not behavior.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
namespace fs = std::filesystem;
std::string read(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(in), {}};
}
std::string function(const std::string& s, const std::string& name) {
    auto a = s.find("fn " + name + "(");
    if (a == std::string::npos) throw std::runtime_error("missing " + name);
    auto b = s.find('{', a);
    int depth = 1;
    auto end = b + 1;
    while (end < s.size() && depth) {
        if (s[end] == '{') ++depth;
        if (s[end] == '}') --depth;
        ++end;
    }
    if (depth) throw std::runtime_error("unclosed " + name);
    return s.substr(a, end - a) + "\n";
}
int main(int argc, char** argv) try {
    if (argc != 4) throw std::runtime_error("ROOT BACKEND_SOURCE OUTPUT");
    const fs::path root = argv[1];
    const auto backend = read(argv[2]);
    std::ofstream out(argv[3]);
    if (!out) throw std::runtime_error("cannot write fixture");
    for (const auto* name : {"NC_RGN_SABOTAGE", "NC_WIDE_MUL_SABOTAGE",
         "NC_IEEE_F64_SABOTAGE", "NC_DYNLINK_THUNK_SABOTAGE",
         "NC_RGN_ENV_CHECKED", "NC_WIDE_ENV_CHECKED", "NC_IEEE_ENV_CHECKED"}) {
        auto a = backend.find(std::string("var ") + name + ":");
        if (a != std::string::npos) out << backend.substr(a, backend.find('\n', a) - a) << '\n';
    }
    auto reader = function(read(root / "self-hosted/io/env.sio"), "read_env");
    reader.replace(reader.find("read_env"), 8, "raw_read_env");
    out << reader << R"SIO(
var ENV_READ_COUNT: i64 = 0
fn read_env(key: string) -> string with IO, Mut, Div, Panic {
    ENV_READ_COUNT = ENV_READ_COUNT + 1
    raw_read_env(key)
}
)SIO";
    // The real setter only assigns its module's shadow state. Keep it separate.
    auto setter = function(read(root / "self-hosted/native/lower_ir.sio"), "lower_ir_set_ieee_f64_sabotage");
    setter.replace(setter.find("NC_IEEE_F64_SABOTAGE"), 20, "LOWER_IEEE_STATE");
    // Seed requires Mut for this assignment; body remains production code.
    setter.insert(setter.find('{'), "with Mut ");
    out << "var LOWER_IEEE_STATE: i64 = 0\n" << setter;
    for (const auto* name : {"nc_refresh_rgn_sabotage", "nc_refresh_wide_mul_sabotage", "nc_refresh_ieee_f64_sabotage"})
        out << function(backend, name);
    out << R"SIO(
fn main() -> i64 with IO, Mut, Panic, Div {
    let on = str_eq(raw_read_env("PROBE_ON"), "1")
    let before = __pool_mark()
    var i: i64 = 0
    while i < 256 {
        nc_refresh_rgn_sabotage()
        nc_refresh_wide_mul_sabotage()
        nc_refresh_ieee_f64_sabotage()
        if on {
            if NC_RGN_SABOTAGE != 4 || NC_WIDE_MUL_SABOTAGE != 1 || NC_IEEE_F64_SABOTAGE != 1 || NC_DYNLINK_THUNK_SABOTAGE != 1 || LOWER_IEEE_STATE != 1 { return 2 }
        } else {
            if NC_RGN_SABOTAGE != 0 || NC_WIDE_MUL_SABOTAGE != 0 || NC_IEEE_F64_SABOTAGE != 0 || NC_DYNLINK_THUNK_SABOTAGE != 0 || LOWER_IEEE_STATE != 0 { return 3 }
        }
        i = i + 1
    }
    print("ENV_REFRESH_READS ")
    print_int(ENV_READ_COUNT)
    print("\nPOOL_BEFORE ")
    print_int(before)
    print("\nPOOL_AFTER ")
    print_int(__pool_mark())
    print("\n")
    if ENV_READ_COUNT != 4 { return 5 }
    println("NATIVE_ENV_REFRESH_OK")
    0
}
)SIO";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
