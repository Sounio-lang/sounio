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
