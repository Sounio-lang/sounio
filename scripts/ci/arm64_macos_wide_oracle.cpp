// C++23 exact arithmetic oracle and mechanical real-IR test generator.
// Generated Sounio is a build artifact, not an independent numerical oracle.
#include <boost/multiprecision/cpp_int.hpp>
#include <bit>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
using boost::multiprecision::cpp_int;
static void load(int reg, std::uint64_t word) {
    const auto value = std::bit_cast<std::int64_t>(word);
    std::cout << "emit(&!f, ir_load_imm(" << reg << ", ";
    if (value == INT64_MIN) std::cout << "-9223372036854775807 - 1";
    else std::cout << value;
    std::cout << "))\n";
}
int main(int argc, char** argv) {
    std::cout << R"(use native::codegen_arm64_macos::{compile_arm64_macos_to_file}
use ir::ir::*
use parser::ast::{make_name, BinaryOp}
fn emit(f: &!IrFunction, instr: IrInstr) with Mut, Panic, Div {
    ir_arena_store(ir_region_slot_w((*f).region, (*f).instr_count), instr)
    (*f).instr_count = (*f).instr_count + 1
}
fn main() with IO, Mut, Panic, Div, Alloc {
    var m = ir_empty_module()
    var f = ir_empty_function()
    f.name = make_name("main")
    f.reg_count = 64
    ir_function_alloc_region(&!f)
    emit(&!f, ir_load_imm(60, 0))
)";
    auto finish = [] {
        std::cout << R"(emit(&!f, ir_return(60))
    ir_module_fn_set(&!m, 0, f)
    m.fn_count = 1
    print_int(compile_arm64_macos_to_file(&m, get_arg(0)) as i64)
    print("\n")
}
)";
    };
    if (argc == 2) {
        const std::string mode = argv[1];
        if (mode != "zero" && mode != "invalid") return 2;
        load(0, 123); load(8, 0);
        if (mode == "zero") std::cout << "emit(&!f, ir_wide_div_full(16, 0, 8, 1))\n";
        else std::cout << "emit(&!f, ir_wide_add(16, 0, 8, 0))\n";
        finish(); return 0;
    }
    if (argc != 1) return 2;
    std::mt19937_64 rng(0xA64CAFE);
    std::size_t cases = 0, words = 0;
    for (int n : {1, 2, 4, 8}) {
        const cpp_int modulus = cpp_int(1) << (64 * n);
        const cpp_int mask = modulus - 1;
        for (int sample = 0; sample < 6; ++sample) {
            cpp_int a = 0, b = 0;
            for (int k = 0; k < n; ++k) {
                a |= cpp_int(rng()) << (64 * k);
                b |= cpp_int(rng()) << (64 * k);
            }
            if (sample == 0) { a = mask; b = 1; }
            if (sample == 1) { a = 0; b = mask; }
            if (sample == 2) { a = cpp_int(1) << (64 * n - 1); b = a + 1; }
            if (sample == 3) b = a;
            auto test = [&](const std::string& ctor, cpp_int expected, int param = -1, bool cmp = false) {
                for (int k = 0; k < n; ++k) {
                    load(k, static_cast<std::uint64_t>((a >> (64*k)) & UINT64_MAX));
                    load(8+k, static_cast<std::uint64_t>((b >> (64*k)) & UINT64_MAX));
                }
                // Alternate disjoint and in-place destination to catch alias bugs.
                const int dst = cases % 2 ? 0 : 16;
                std::cout << "emit(&!f, " << ctor << '(' << dst << ", 0, ";
                if (ctor == "ir_wide_shr" || ctor == "ir_wide_shr_limb") std::cout << n << ", " << param;
                else { std::cout << "8, " << n; if (cmp) std::cout << ", " << param; }
                std::cout << "))\n";
                expected &= mask;
                for (int k = 0; k < (cmp ? 1 : n); ++k) {
                    load(50, static_cast<std::uint64_t>((expected >> (64*k)) & UINT64_MAX));
                    std::cout << "emit(&!f, ir_binop(51, " << dst+k << ", BinaryOp::OpNe, 50))\n"
                                 "emit(&!f, ir_binop(60, 60, BinaryOp::OpBitOr, 51))\n";
                    ++words;
                }
                ++cases;
            };
            test("ir_wide_add", a+b);
            test("ir_wide_sub", a-b);
            test("ir_wide_mul", a*b);
            test("ir_wide_div_full", cpp_int(a/b));
            test("ir_wide_mod_full", cpp_int(a%b));
            const cpp_int low = b & UINT64_MAX;
            if (low != 0) {
                test("ir_wide_div", cpp_int(a/low));
                test("ir_wide_mod", cpp_int(a%low));
            }
            for (int bits : {0, 1, 63, 64, 65, 64*n, 64*n+1}) test("ir_wide_shr", cpp_int(a >> bits), bits);
            for (int limbs : {0, 1, n, n+1}) test("ir_wide_shr_limb", cpp_int(a >> (64*limbs)), limbs);
            const cpp_int sa = (a >> (64*n-1)) != 0 ? a-modulus : a;
            const cpp_int sb = (b >> (64*n-1)) != 0 ? b-modulus : b;
            const bool comparisons[] = {a==b,a!=b,a<b,a<=b,a>b,a>=b,false,false,sa<sb,sa<=sb,sa>sb,sa>=sb};
            for (int op : {0,1,2,3,4,5,8,9,10,11}) test("ir_wide_cmp", cpp_int(comparisons[op]), op, true);
        }
    }
    finish();
    std::cerr << "wide exact-oracle cases=" << cases << " limb comparisons=" << words << '\n';
}
