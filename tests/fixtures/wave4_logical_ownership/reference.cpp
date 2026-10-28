#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
namespace fs=std::filesystem;
struct Case {const char* name;const char* before;const char* after;const char* code;};
constexpr std::array cases{
 Case{"linear_and_rhs","0","1","E040"},Case{"linear_or_rhs","0","1","E040"},
 Case{"literal_false_and_rhs","0","1","E040"},Case{"literal_true_or_rhs","0","1","E040"},
 Case{"linear_lhs_and","0","0","-"},Case{"linear_lhs_or","0","0","-"},
 Case{"linear_nested_12","0","1","E040"},Case{"linear_nested_24","0","1","E040"},
 Case{"pure_nested_12","0","0","-"},Case{"pure_nested_24","0","0","-"},
 Case{"pure_bool","0","0","-"},Case{"counter","0","0","-"},
 Case{"affine_rhs_reuse","1","1","E039"},Case{"affine_rhs_discard","0","0","-"},
 Case{"owner_call_reborrow","0","0","-"},
 Case{"conditional_borrow_left","1","1","E038"},Case{"conditional_borrow_right","1","1","E038"},
 Case{"rhs_type_error","1","1","E004"},Case{"namespace_join","0","0","NS_JOIN"}
};
struct Record {std::string status,log;};
using Table=std::map<std::string,Record>;
std::string read(const fs::path&p){
 std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("missing "+p.string());
 return {std::istreambuf_iterator<char>(f),{}};
}
Table load(const fs::path&p){
 Table out;std::istringstream in(read(p/"FOCUSED.tsv"));std::string line;
 while(std::getline(in,line)){
  const auto t=line.find('\t');
  if(t==std::string::npos||t==0||line.find('\t',t+1)!=std::string::npos)throw std::runtime_error("blank/malformed focused producer");
  const auto name=line.substr(0,t),status=line.substr(t+1);
  if(status!="0"&&status!="1")throw std::runtime_error("nonsemantic status "+name);
  if(!out.emplace(name,Record{status,read(p/(name+".stdout"))}).second)throw std::runtime_error("duplicate producer "+name);
 }
 return out;
}
bool valid(const Table&t,bool after,std::string&why){
 if(t.size()!=cases.size()){why="wrong nonempty selection";return false;}
 const std::regex diagnostic(R"(error\[(E[0-9]+)\])");
 for(const auto&c:cases){
  const auto it=t.find(c.name);if(it==t.end()){why="missing "+std::string(c.name);return false;}
  const auto&r=it->second;const std::string expected=after?c.after:c.before;
  if(r.status!=expected){why="wrong status "+std::string(c.name);return false;}
  const auto marker="ACTUAL_CHECKER_VERDICT "+expected+"\n";
  const auto start=r.log.find(marker);
  if(start==std::string::npos||r.log.find("ACTUAL_CHECKER_VERDICT ",start+marker.size())!=std::string::npos){why="missing/duplicate verdict "+std::string(c.name);return false;}
  if(r.log.find("PARSE_REFUSED")!=std::string::npos||r.log.find("parse error")!=std::string::npos){why="parse refusal "+std::string(c.name);return false;}
  if(expected=="0"){
   if(r.log.find("error[")!=std::string::npos){why="positive diagnostic "+std::string(c.name);return false;}
  }else{
   if(r.log.find("error["+std::string(c.code)+"]")==std::string::npos){why="wrong diagnostic "+std::string(c.name);return false;}
   for(auto e=std::sregex_iterator(r.log.begin(),r.log.end(),diagnostic);e!=std::sregex_iterator();++e){
    const auto code=(*e)[1].str();
    if(code!=c.code&&!(std::string(c.name)=="rhs_type_error"&&code=="E051")){why="unrelated "+code+" "+c.name;return false;}
   }
  }
  if(after&&std::string(c.code)=="NS_JOIN"){
   if(r.log.find("NS_TRACE env_join=-1\n")==std::string::npos){why="namespace join absent";return false;}
   const auto last=r.log.rfind("NS_TRACE union=");
   if(last==std::string::npos||r.log.compare(last,18,"NS_TRACE union=-1\n")!=0){why="namespace value did not stay top";return false;}
  }
 }
 return true;
}
int main(int argc,char**argv){
 try{
  if(argc!=4)throw std::runtime_error("usage reference --verify|--self-test BEFORE AFTER");
  const auto before=load(argv[2]),after=load(argv[3]);std::string why;
  if(!valid(before,false,why)||!valid(after,true,why))throw std::runtime_error(why);
  if(std::string(argv[1])=="--self-test"){
   unsigned rejected=0;
   auto wrong=after;wrong.erase(wrong.begin());if(!valid(wrong,true,why))++rejected;
   wrong=after;wrong.emplace("unexpected",Record{"0","ACTUAL_CHECKER_VERDICT 0\n"});if(!valid(wrong,true,why))++rejected;
   wrong=after;wrong["linear_and_rhs"]={"0","ACTUAL_CHECKER_VERDICT 0\n"};if(!valid(wrong,true,why))++rejected;
   wrong=after;wrong["linear_and_rhs"]={"1","error[E137]: undeclared API\nACTUAL_CHECKER_VERDICT 1\n"};if(!valid(wrong,true,why))++rejected;
   wrong=after;wrong["namespace_join"].log="NS_TRACE env_join=-1\nNS_TRACE union=0\nACTUAL_CHECKER_VERDICT 0\n";if(!valid(wrong,true,why))++rejected;
   wrong=after;wrong["pure_bool"].log+="ACTUAL_CHECKER_VERDICT 0\n";if(!valid(wrong,true,why))++rejected;
   wrong=after;wrong["pure_bool"].log+="error[E397]: storage failure\n";if(!valid(wrong,true,why))++rejected;
   if(rejected!=7)throw std::runtime_error("wrong-output control accepted");
   std::cout<<"LOGICAL_WRONG_OUTPUT_REJECTED controls=7\n";
  }else if(std::string(argv[1])!="--verify")throw std::runtime_error("unknown mode");
  std::cout<<"LOGICAL_FOCUSED_OK cases="<<cases.size()<<" before_and_after=1\n";
 }catch(const std::exception&e){std::cerr<<"LOGICAL_FOCUSED_REFUSED "<<e.what()<<'\n';return 1;}
}
