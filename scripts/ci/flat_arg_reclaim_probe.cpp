#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
std::string read(const std::filesystem::path& p) {
 std::ifstream f(p); if(!f) throw std::runtime_error("missing "+p.string());
 return {std::istreambuf_iterator<char>(f),{}};
}
std::string definition(const std::string& s,const std::string& name) {
 auto a=s.find("fn "+name+"(");
 if(a==std::string::npos) return {};
 auto b=s.find("\n}\n",a);
 if(b==std::string::npos) throw std::runtime_error("unclosed "+name);
 return s.substr(a,b+3-a)+"\n";
}
// Match the stricter seed's effect checker. Widen annotations in the composed
// test only; function bodies and layouts remain production text in both runs.
std::string seed_effects(std::string s) {
 std::regex start(R"((^|\n)[ \t]*(pub(\(crate\))? )?fn [A-Za-z_][A-Za-z_0-9]*\()");
 std::size_t cursor=0; std::smatch match;
 while(cursor<s.size()) {
  auto tail=s.substr(cursor);
  if(!std::regex_search(tail,match,start)) break;
  auto p=cursor+static_cast<std::size_t>(match.position());
  auto brace=s.find('{',p);
  if(brace==std::string::npos) break;
  auto with=s.find(" with ",p);
  if(with==std::string::npos || with>brace) with=brace;
  s.replace(with,brace-with," with IO, Mut, Panic, Div, Alloc ");
  cursor=with+std::string(" with IO, Mut, Panic, Div, Alloc ").size()+1;
 }
 return s;
}
int main(int argc,char** argv) try {
 if(argc!=3 && argc!=4) throw std::runtime_error("usage: probe ROOT OUTPUT [FRONTEND_SOURCE]");
 std::filesystem::path root(argv[1]); std::ofstream out(argv[2]);
 for(auto p:{"lexer/span.sio","check/numeric_format.sio","parser/ast.sio","check/types.sio","ir/ir.sio"}) {
  std::istringstream lines(seed_effects(read(root/"self-hosted"/p))); std::string line;
  while(std::getline(lines,line)) if(!line.starts_with("module ") && !line.starts_with("use ")) out<<line<<'\n';
 }
 auto frontend=read(argc==4 ? std::filesystem::path(argv[3]) : root/"self-hosted/compiler/module_frontend.sio");
 std::istringstream lines(frontend); std::string line;
 while(std::getline(lines,line))
  if(line.starts_with("var MF_AREBOX_") || line.starts_with("let MF_AREBOX_")) out<<line<<'\n';
 for(auto name:{"mf_arebox_reset_scratch","mf_arebox_extract_at","mf_arebox_record_at",
   "mf_arebox_rebuild","mf_flat_args_valid","mf_arebox_scan"}) out<<seed_effects(definition(frontend,name));
 out<<R"SIO(
fn main() -> i64 with IO, Mut, Panic, Div, Alloc {
    ir_instr_arena_init()
    var m = ir_empty_module()
    let warm = Box::new(ir_reg_list_single(31337))
    let mark = __arena_mark()
    var nine: Option<Box<IrRegList>> = None
    var k: i64 = 9
    while k > 0 { nine = Some(Box::new(ir_reg_list_cons(k, nine))); k = k - 1 }
    let one = Some(Box::new(ir_reg_list_single(7)))
    var fi: i64 = 0
    while fi < 3 {
        var f = ir_empty_function()
        if ir_function_alloc_region(&! f) != IR_ARENA_OK { return 1 }
        var ii: i64 = 0
        while ii < 11000 {
            let at = ir_region_slot_w(f.region, ii)
            if fi == 0 && ii == 0 {
                IR_A_ARG_COUNT[at as usize] = 9
                ir_arena_put_args(at, nine, 9)
            } else {
                IR_A_ARG_COUNT[at as usize] = 1
                ir_arena_put_args(at, one, 1)
            }
            ii = ii + 1
        }
        f.instr_count = 11000
        m.fn_count = fi + 1
        ir_fn_set(&! m, fi as usize, f)
        fi = fi + 1
    }
    if !mf_arebox_scan(&m, 0, 3) { return 2 }
    if MF_AREBOX_COUNT != 33000 { return 3 }
    let pool_before = IR_ARG_POOL_LEN
    if __arena_mark() <= mark { return 13 }
    __arena_reset(mark)
    if __arena_mark() != mark { return 14 }
    var churn: i64 = 0
    while churn < 4096 {
        let trash = Box::new(ir_reg_list_single(999))
        if (*trash).head != 999 { return 4 }
        churn = churn + 1
    }
    if (*warm).head != 31337 { return 15 }
    if !mf_arebox_scan(&m, 0, 3) {
        print("FLAT_RECLAIM_FAILURE fn_count="); print_int(m.fn_count)
        print(" arena_violation="); print_int(IR_ARENA_VIOLATION)
        print(" pool_len="); print_int(IR_ARG_POOL_LEN)
        print(" sites="); print_int(MF_AREBOX_COUNT)
        println("")
        var failed_fn: i64 = 0
        while failed_fn < 3 {
            let observed = ir_fn_get(&m, failed_fn)
            print("fn="); print_int(failed_fn)
            print(" instr_count="); print_int(observed.instr_count)
            print(" region_slot="); print_int(observed.region.slot)
            print(" generation="); print_int(observed.region.generation)
            println("")
            failed_fn = failed_fn + 1
        }
        return 5
    }
    if IR_ARG_POOL_LEN != pool_before { return 6 }
    let first = ir_region_slot_r(ir_fn_get(&m, 0).region, 0)
    k = 0
    while k < 9 {
        if ir_arena_arg_at(first, k) != k + 1 { return 7 }
        k = k + 1
    }
    let last = ir_region_slot_r(ir_fn_get(&m, 2).region, 10999)
    if ir_arena_arg_at(last, 0) != 7 { return 8 }
    let base = IR_A_ARG_BASE[last as usize]
    IR_A_ARG_BASE[last as usize] = IR_ARG_POOL_LEN
    if mf_arebox_scan(&m, 0, 3) { return 9 }
    IR_A_ARG_BASE[last as usize] = base
    IR_A_ARG_COUNT[last as usize] = 65
    if mf_arebox_scan(&m, 0, 3) { return 10 }
    IR_A_ARG_COUNT[last as usize] = 1
    if mf_arebox_scan(&m, -1, 3) { return 11 }
    if !mf_arebox_scan(&m, 0, 3) { return 12 }
    println("FLAT_ARG_RECLAIM_OK")
    0
}
)SIO";
 if(!out) throw std::runtime_error("write failed");
 return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
