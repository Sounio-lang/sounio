#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
namespace fs=std::filesystem;
struct Case {std::string_view name,exit,code,marker;};
constexpr std::array cases{
 Case{"returned_loan_alias_mutation","1","E038","-"},
 Case{"returned_loan_block_escape","1","E091","-"},
 Case{"returned_loan_branch_future","1","E038","-"},
 Case{"returned_loan_closure_future","1","E038","-"},
 Case{"returned_loan_escape","1","E091","-"},
 Case{"returned_loan_last_use","0","-","RETURNED_LOAN_LAST_USE_OK"},
 Case{"returned_loan_mut_alias","1","E039","-"},
 Case{"returned_loan_mut_branch","1","E039","-"},
 Case{"returned_loan_mut_move_scope","0","-","AFFINE_MUTABLE_REFERENCE_MOVED_ONCE_OK"},
 Case{"returned_loan_mut_unused","0","-","AFFINE_MUTABLE_REFERENCE_DISCARD_OK"},
 Case{"returned_loan_mutation","1","E038","-"},
 Case{"returned_loan_owner_close","1","E038","-"},
 Case{"returned_loan_pattern_escape","1","E091","-"},
 Case{"returned_loan_pattern_forward","0","-","RETURNED_LOAN_PATTERN_FORWARD_OK"},
 Case{"returned_loan_pattern_mutation","1","E038","-"},
 Case{"returned_loan_pattern_scope","0","-","RETURNED_LOAN_PATTERN_SCOPE_OK"},
 Case{"returned_loan_reassign_mutation","1","E038","-"},
 Case{"returned_loan_reassign_scope","0","-","RETURNED_LOAN_REASSIGN_SCOPE_OK"},
 Case{"returned_loan_reborrow_conflict","1","E038","-"},
 Case{"returned_loan_reborrow_forward","0","-","AFFINE_REBORROW_FORWARD_OK"},
 Case{"returned_loan_reborrow_scope","0","-","AFFINE_REBORROW_SCOPE_OK"},
 Case{"returned_loan_scope","0","-","RETURNED_LOAN_SCOPE_OK"},
 Case{"returned_loan_shadow_scope","0","-","RETURNED_LOAN_SHADOW_SCOPE_OK"},
 Case{"returned_loan_temporary_conflict","1","E038","-"},
 Case{"returned_loan_temporary_consumed","0","-","RETURNED_LOAN_TEMPORARY_CONSUMED_OK"},
 Case{"returned_loan_tuple_mutation","1","E038","-"}
};
struct Record {std::string exit,log;};
using Table=std::map<std::string,Record>;
std::string read(const fs::path& p) {
 std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("missing input: "+p.string());
 return {std::istreambuf_iterator<char>(f),{}};
}
Table parse(const std::string& matrix,const fs::path& dir) {
 std::istringstream in(matrix);std::string line;
 if(!std::getline(in,line)||line!="case\texit_status")throw std::runtime_error("wrong matrix header");
 Table table;
 while(std::getline(in,line)) {
  const auto tab=line.find('\t');
  if(tab==std::string::npos||line.find('\t',tab+1)!=std::string::npos)throw std::runtime_error("malformed matrix row");
  auto name=line.substr(0,tab),exit=line.substr(tab+1);
  if(exit!="0"&&exit!="1")throw std::runtime_error("nonsemantic or invalid exit: "+exit);
  if(table.contains(name))throw std::runtime_error("duplicate case: "+name);
  table.emplace(name,Record{exit,read(dir/(name+".stdout"))});
 }
 return table;
}
bool validate(const Table& table,bool kernel,std::string& why) {
 if(table.size()!=cases.size()){why="selection cardinality";return false;}
 const std::regex errors(R"(error\[(E[0-9]{3})\])");
 for(const auto& test:cases) {
  const auto it=table.find(std::string(test.name));
  if(it==table.end()){why="missing case: "+std::string(test.name);return false;}
  const auto& record=it->second;const auto& log=record.log;
  if(record.exit!=test.exit){why="wrong verdict: "+std::string(test.name);return false;}
  if(log.find("parse error")!=std::string::npos||log.find("error[E264]")!=std::string::npos
   ||log.find("error[E137]")!=std::string::npos) {why="parser/unsupported/undefined-API refusal";return false;}
  if(kernel) {
   const std::string marker="ACTUAL_CHECKER_VERDICT "+record.exit+"\n";
   auto p=log.find(marker);
   if(p==std::string::npos||log.find("ACTUAL_CHECKER_VERDICT ",p+marker.size())!=std::string::npos) {
    why="missing or duplicated actual-checker verdict";return false;
   }
  } else if(test.exit=="0"&&log.find("check: OK")==std::string::npos) {
   why="missing raw compiler check verdict";return false;
  }
  if(test.exit=="0") {
   if(log.find("error")!=std::string::npos){why="positive contains an error";return false;}
  } else {
   const std::string required="error["+std::string(test.code)+"]";
   if(log.find(required)==std::string::npos){why="wrong diagnostic: "+std::string(test.name);return false;}
   for(auto e=std::sregex_iterator(log.begin(),log.end(),errors);e!=std::sregex_iterator();++e) {
    const auto code=(*e)[1].str();
    if(code!="E038"&&code!="E039"&&code!="E040"&&code!="E091") {why="unrelated diagnostic: "+code;return false;}
   }
  }
 }
 return true;
}
int main(int argc,char** argv) {
 try {
  if(argc==2&&std::string_view(argv[1])=="--manifest") {
   for(const auto& c:cases)std::cout<<c.name<<'\t'<<c.exit<<'\t'<<c.code<<'\t'<<c.marker<<'\n';
   return 0;
  }
  if(argc!=4)throw std::runtime_error("usage: reference --kernel|--raw --verify|--self-test RESULTS");
  bool kernel=std::string_view(argv[1])=="--kernel";
  if(!kernel&&std::string_view(argv[1])!="--raw")throw std::runtime_error("unknown engine");
  const fs::path dir=argv[3];const auto matrix=read(dir/"VERDICTS.tsv");
  const auto table=parse(matrix,dir);std::string why;
  if(!validate(table,kernel,why))throw std::runtime_error(why);
  if(std::string_view(argv[2])=="--self-test") {
   unsigned rejected=0;
   auto wrong=table;wrong.begin()->second.exit="0";
   if(!validate(wrong,kernel,why))++rejected;
   wrong=table;wrong.erase(wrong.begin());
   if(!validate(wrong,kernel,why))++rejected;
   wrong=table;wrong.emplace("unexpected",Record{"0",""});
   if(!validate(wrong,kernel,why))++rejected;
   wrong=table;wrong.begin()->second.log="error[E264]: unsupported\nACTUAL_CHECKER_VERDICT 1\n";
   if(!validate(wrong,kernel,why))++rejected;
   wrong=table;wrong["returned_loan_scope"].log.clear();
   if(!validate(wrong,kernel,why))++rejected;
   wrong=table;wrong.begin()->second.log+="parse error\n";
   if(!validate(wrong,kernel,why))++rejected;
   bool duplicate=false;
   try {auto p=matrix.find('\n');auto e=matrix.find('\n',p+1);(void)parse(matrix+matrix.substr(p+1,e-p),dir);}
   catch(const std::runtime_error&){duplicate=true;}
   if(duplicate)++rejected;
   if(rejected!=7)throw std::runtime_error("wrong-output control was accepted");
   std::cout<<"RETURNED_LOAN_MATRIX_WRONG_OUTPUT_REJECTED controls="<<rejected<<'\n';
  } else if(std::string_view(argv[2])!="--verify")throw std::runtime_error("unknown operation");
  std::size_t positive=0;for(const auto& c:cases)if(c.exit=="0")++positive;
  std::cout<<"RETURNED_LOAN_MATRIX_OK engine="<<(kernel?"kernel":"raw")<<" cases="<<cases.size()
   <<" positive="<<positive<<" negative="<<cases.size()-positive<<'\n';
  return 0;
 }catch(const std::exception& e){std::cerr<<"RETURNED_LOAN_MATRIX_REFUSED "<<e.what()<<'\n';return 1;}
}
