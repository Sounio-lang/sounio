// Executes extracted production Sounio functions, not a C++ replacement model.
// State shells contain only fields actually accessed; this is a component probe,
// not qualification of full Lowerer layout, SRET, imports or compiler integration.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
namespace fs=std::filesystem;
std::string read(const fs::path& p){std::ifstream f(p);if(!f)throw std::runtime_error(p.string());return {std::istreambuf_iterator<char>(f),{}};}
std::string def(const std::string& s,const std::string& anchor,bool method=false){auto a=s.find(anchor);if(a==s.npos)throw std::runtime_error(anchor);std::string end=method?"\n    }\n":"\n}\n";auto b=s.find(end,a);if(b==s.npos)throw std::runtime_error("end "+anchor);return s.substr(a,b+end.size()-a)+"\n";}
void line(std::ostream& o,const std::string& s,const std::string& anchor){auto a=s.find(anchor);if(a==s.npos)throw std::runtime_error(anchor);auto b=s.find('\n',a);o<<s.substr(a,b-a)<<'\n';}
int main(int argc,char** argv)try{
 if(argc!=3)throw std::runtime_error("probe SOURCE_ROOT OUTPUT.sio");
 fs::path r(argv[1]);auto lo=read(r/"self-hosted/ir/lower.sio"),ir=read(r/"self-hosted/ir/ir.sio"),ast=read(r/"self-hosted/parser/ast.sio");std::ofstream o(argv[2]);
 o<<def(ast,"pub struct Name {");line(o,ir,"pub let IR_MAX_VREGS:");
 for(auto n:{"IR_LOWER_HARD_ERROR","IR_LOWER_HARD_ERROR_REASON","IR_LOWER_HARD_ERROR_LIMIT","IR_LOWER_HARD_ERROR_NEEDED","IR_LOWER_HARD_ERROR_FN_BUF","IR_LOWER_HARD_ERROR_FN_LEN","IR_LOWER_ERROR_FIRST_FN_BUF","IR_LOWER_ERROR_FIRST_FN_LEN","IR_LOWER_ERROR_TOTAL","IR_LOWER_COMPILE_ERROR_LATCH","IR_LOWER_COMPILE_ERROR_SCOPE_ACTIVE","IR_LOWER_ERR_SITES"})line(o,lo,std::string("var ")+n+":");
 o<<R"SIO(
struct IrModule { fn_count: i64 }
struct IrFunction { name: Name, reg_count: i64 }
struct Lowerer {
 module: Box<IrModule>, current_fn: i64, current_func: Box<IrFunction>,
 error_count: i64, had_error: bool, probe_mode: bool,
}
)SIO";
 o<<def(ir,"pub fn ir_reg_count_can_grow(")<<def(lo,"fn ir_copy_name_to_flat(")<<def(lo,"pub fn lower_note_error(")<<def(lo,"fn lowerer_mark_error(")<<def(lo,"fn lowerer_mark_hard_error(")<<def(lo,"fn lowerer_fresh_reg_mut(");
 o<<"impl Lowerer {\n"<<def(lo,"    fn report_error_at(",true)<<def(lo,"    fn fresh_reg(",true)<<"}\n";
 o<<R"SIO(
struct Observation {
 result: i64, reg_count: i64, local_errors: i64, had_error: bool,
 total: i64, site: i64, latch: bool, hard: i64, reason: i64,
 limit: i64, needed: i64, name_len: i64, name_byte: i8,
 first_len: i64, first_byte: i8,
}
fn probe_reset(prior: bool) with Mut {
 IR_LOWER_HARD_ERROR = 0
 IR_LOWER_HARD_ERROR_REASON = 0
 IR_LOWER_HARD_ERROR_LIMIT = 0
 IR_LOWER_HARD_ERROR_NEEDED = 0
 IR_LOWER_HARD_ERROR_FN_LEN = 0
 IR_LOWER_HARD_ERROR_FN_BUF[0] = 0
 IR_LOWER_ERROR_TOTAL = 0
 IR_LOWER_ERR_SITES[0] = 0
 IR_LOWER_ERROR_FIRST_FN_LEN = 0
 IR_LOWER_ERROR_FIRST_FN_BUF[0] = 0
 IR_LOWER_COMPILE_ERROR_SCOPE_ACTIVE = true
 IR_LOWER_COMPILE_ERROR_LATCH = false
 if prior {
  IR_LOWER_HARD_ERROR = 1
  IR_LOWER_HARD_ERROR_REASON = 2
  IR_LOWER_HARD_ERROR_LIMIT = 42
  IR_LOWER_HARD_ERROR_NEEDED = 43
  IR_LOWER_HARD_ERROR_FN_LEN = 1
  IR_LOWER_HARD_ERROR_FN_BUF[0] = 80
 }
}
fn probe_state(reg: i64, cur: i64, probe: bool) -> Lowerer with Alloc, Mut {
 var n = Name { buf: [0; 384], len: 1 }
 n.buf[0] = 67
 Lowerer {
  module: Box::new(IrModule { fn_count: 1 }), current_fn: cur,
  current_func: Box::new(IrFunction { name: n, reg_count: reg }),
  error_count: 0, had_error: false, probe_mode: probe,
 }
}
fn observe(lo: &Lowerer, result: i64) -> Observation with Mut, Panic, Div {
 Observation { result: result, reg_count: (*(*lo).current_func).reg_count,
  local_errors: (*lo).error_count, had_error: (*lo).had_error,
  total: IR_LOWER_ERROR_TOTAL, site: IR_LOWER_ERR_SITES[0],
  latch: IR_LOWER_COMPILE_ERROR_LATCH, hard: IR_LOWER_HARD_ERROR,
  reason: IR_LOWER_HARD_ERROR_REASON, limit: IR_LOWER_HARD_ERROR_LIMIT,
  needed: IR_LOWER_HARD_ERROR_NEEDED, name_len: IR_LOWER_HARD_ERROR_FN_LEN,
  name_byte: IR_LOWER_HARD_ERROR_FN_BUF[0], first_len: IR_LOWER_ERROR_FIRST_FN_LEN,
  first_byte: IR_LOWER_ERROR_FIRST_FN_BUF[0],
 }
}
fn same(a: &Observation, b: &Observation) -> bool {
 (*a).result == (*b).result && (*a).reg_count == (*b).reg_count &&
 (*a).local_errors == (*b).local_errors && (*a).had_error == (*b).had_error &&
 (*a).total == (*b).total && (*a).site == (*b).site && (*a).latch == (*b).latch &&
 (*a).hard == (*b).hard && (*a).reason == (*b).reason &&
 (*a).limit == (*b).limit && (*a).needed == (*b).needed &&
 (*a).name_len == (*b).name_len && (*a).name_byte == (*b).name_byte &&
 (*a).first_len == (*b).first_len && (*a).first_byte == (*b).first_byte
}
fn contract(id: i64, a: &Observation) -> bool with Mut, Panic, Div {
 if id == 3 || id == 4 {
  return (*a).result == 0 && (*a).reg_count == 7 && (*a).local_errors == 1 &&
   (*a).had_error && (*a).total == 1 && (*a).site == 8959 && (*a).latch &&
   (*a).hard == 0 && (*a).reason == 0 && (*a).limit == 0 && (*a).needed == 0 &&
   (*a).name_len == 0 && (*a).name_byte == 0 && (*a).first_len == 1 && (*a).first_byte == 67
 }
 if (*a).local_errors != 0 || (*a).had_error || (*a).total != 0 || (*a).site != 0 ||
  (*a).latch || (*a).first_len != 0 || (*a).first_byte != 0 { return false }
 if id == 1 || id == 2 {
  var r: i64 = 7
  if id == 2 { r = IR_MAX_VREGS - 1 }
  return (*a).result == r && (*a).reg_count == r + 1 && (*a).hard == 0 &&
   (*a).reason == 0 && (*a).limit == 0 && (*a).needed == 0 && (*a).name_len == 0 && (*a).name_byte == 0
 }
 if (*a).result != 0 || (*a).hard != 1 { return false }
 if id == 6 {
  return (*a).reg_count == IR_MAX_VREGS && (*a).reason == 2 &&
   (*a).limit == 42 && (*a).needed == 43 && (*a).name_len == 1 && (*a).name_byte == 80
 }
 var count: i64 = IR_MAX_VREGS
 if id == 7 { count = 0 - 1 }
 (*a).reg_count == count && (*a).reason == 5 && (*a).limit == IR_MAX_VREGS &&
  (*a).needed == count + 1 && (*a).name_len == 1 && (*a).name_byte == 67
}
fn probe_case(id: i64, reg: i64, cur: i64, probe: bool, prior: bool) -> i64 with IO, Alloc, Mut, Panic, Div {
 probe_reset(prior)
 var a = probe_state(reg, cur, probe)
 let pair = a.fresh_reg()
 a = pair.0
 let expected = observe(&a, pair.1)
 probe_reset(prior)
 var b = probe_state(reg, cur, probe)
 let actual_result = lowerer_fresh_reg_mut(&!b)
 let actual = observe(&b, actual_result)
 print("fresh_reg_case ")
 print_int(id)
 if same(&expected, &actual) && contract(id, &expected) && contract(id, &actual) { println(" PASS") return 0 }
 print(" FAIL site expected/actual=")
 print_int(expected.site)
 print("/")
 print_int(actual.site)
 print(" errors=")
 print_int(expected.local_errors)
 print("/")
 print_int(actual.local_errors)
 print(" reason=")
 print_int(expected.reason)
 print("/")
 print_int(actual.reason)
 println("")
 1
}
fn main() -> i32 with IO, Alloc, Mut, Panic, Div {
 var failures: i64 = 0
 failures = failures + probe_case(1, 7, 0, false, false)
 failures = failures + probe_case(2, IR_MAX_VREGS - 1, 0, false, false)
 failures = failures + probe_case(3, 7, 0 - 1, true, false)
 failures = failures + probe_case(4, 7, 1, true, false)
 failures = failures + probe_case(5, IR_MAX_VREGS, 0, false, false)
 failures = failures + probe_case(6, IR_MAX_VREGS, 0, false, true)
 failures = failures + probe_case(7, 0 - 1, 0, false, false)
 if failures != 0 { return 1 }
 println("fresh_reg_production_parity PASS")
 0
}
)SIO";
 return o?0:1;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
