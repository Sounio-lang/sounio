#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
namespace fs=std::filesystem;
struct Case { std::string name; bool reject; std::string expected; };
static std::string read(const fs::path& p) { std::ifstream f(p,std::ios::binary); if(!f) throw std::runtime_error("missing "+p.string()); return {std::istreambuf_iterator<char>(f),{}}; }
static int status(const fs::path& p) { auto s=read(p); std::istringstream f(s); int n; std::string rest; if(!(f>>n)||(f>>rest)) throw std::runtime_error("invalid status "+p.string()); return n; }
static std::vector<Case> cases() {
 std::vector<Case> c;
 for(auto k:{"0","1"}) {
  for(auto n:{"iflet_packet_","match_packet_","match_live_packet_"}) c.push_back({std::string(n)+k,false,"FLOW 11 22 33\n"});
  for(auto n:{"f64_if_","f64_iflet_","f64_match_"}) c.push_back({std::string(n)+k,false,"FLOW "+std::to_string(std::bit_cast<std::int64_t>(1.25))+"\n"});
  for(auto n:{"both_return_","unreachable_"}) c.push_back({std::string(n)+k,false,"FLOW 11\n"});
 }
 for(auto form:{"if","iflet","match"}) for(auto k:{"0","1"}) c.push_back({std::string("shadow_")+form+"_"+k,false,"SHADOW "+std::to_string(std::bit_cast<std::int64_t>(2.5))+" "+std::to_string(std::bit_cast<std::int64_t>(1.0))+"\n"});
 c.push_back({"nested_packet",false,"FLOW 11 22 33\n"});
 for(auto n:{"return_if_allreturn","both_return_dead","closure_return_dead","array_for_continue","while_let_break","or_match","discard_mixed_if","discard_mixed_iflet","discard_mixed_match","while_break","while_continue","for_continue","closure_both_return","closure_outer_live","partial_match","panic_branch","shortcircuit"}) c.push_back({n,false,"FLOW 11\n"});
 for(auto n:{"unit_value_if","unit_value_iflet","unit_value_match","missing_else","missing_loop","duplicate_tags","guard_not_coverage","unmatched_value","unit_value_nested"}) c.push_back({n,true,""});
 for(auto shape:{"small","array"}) for(auto form:{"plain","implicit","explicit"}) for(auto k:{"0","1","2"}) c.push_back({std::string(shape)+"-"+form+"-"+k,false,"PACKET_RETURN 11 22 33\n"});
 for(auto n:{"nominal_0","nominal_1","nominal_or","nominal_nested","bool_true","bool_false","bool_or"}) c.push_back({n,false,"FLOW 11\n"});
 for(auto n:{"nominal_duplicate","nominal_wrong_owner","nominal_homonym","nominal_struct_conflict","nominal_alias_scalar","nominal_payload_identifiers","nominal_payload_nested","nominal_guard","bool_integer","bool_duplicate","result_hash_zero"}) c.push_back({n,true,""});
 for(auto form:{"if","iflet","match"}) for(auto k:{"0","1"}) {
  double value=1.0,variance=0.0;
  for(unsigned i=2;i<32;++i) value+=1.0;
  for(unsigned i=0;i<32;++i) variance+=(0.5*0.5);
  c.push_back({std::string("shadow32_")+form+"_"+k,false,"SHADOW32 "+std::to_string(std::bit_cast<std::int64_t>(value))+" "+std::to_string(std::bit_cast<std::int64_t>(variance))+" "+std::to_string(std::bit_cast<std::int64_t>(1.0))+" "+std::to_string(std::bit_cast<std::int64_t>(1.0))+"\n"});
 }
 c.push_back({"shadow_cap32",false,"CAP32 "+std::to_string(std::bit_cast<std::int64_t>(0.5*0.5))+" 0\n"});
 c.push_back({"hash_collision",true,""});
 for(auto n:{"mixed_explicit_0","mixed_explicit_1"}) c.push_back({n,false,"F64 "+std::to_string(std::bit_cast<std::int64_t>(1.0))+"\n"});
 c.push_back({"pure_float",false,"F64 "+std::to_string(std::bit_cast<std::int64_t>(1.25))+"\n"});
 c.push_back({"mixed_direct",true,""});
 for(auto form:{"if","iflet","match"}) c.push_back({std::string("shadow_cap32_unique_")+form,false,"CAP32 "+std::to_string(std::bit_cast<std::int64_t>(0.5*0.5))+" 0\n"});
 c.push_back({"nominal_depth_budget",true,""});
 c.push_back({"or_budget",true,""});
 c.push_back({"variant_hash_collision",true,""});
 c.push_back({"flow_component",false,"FLOW 11\n"});
 return c;
}
int main(int argc,char**argv) {
 if(argc!=2) return 2;
 int bad=0;
 try {
  auto c=cases();
  for(const auto& q:c) {
   fs::path p=fs::path(argv[1])/q.name;
   int cs=status(p.string()+".compile.status");
   bool ok;
   if(q.reject) ok=cs==1 && !fs::exists(p.string()+".elf");
   else ok=cs==0 && fs::is_regular_file(p.string()+".elf") && status(p.string()+".native.status")==0 && read(p.string()+".native.stdout")==q.expected;
   std::cout << "CASE "<<q.name<<" expected="<<(q.reject?"reject":"native")<<" compile="<<cs<<" ok="<<ok<<"\n";
   bad+=!ok;
  }
  std::cout<<"FLOW_ORACLE cases="<<c.size()<<" failed="<<bad<<"\n";
 } catch(const std::exception&e) {std::cerr<<e.what()<<"\n";return 3;}
 return bad?4:0;
}
