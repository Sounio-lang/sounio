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
struct Row{std::string status,log;};
using Rows=std::map<std::string,Row>;
constexpr std::array<std::pair<std::string_view,std::string_view>,10> expected{{
 {"good_i64","0"},{"good_f64","0"},{"wrong_root_abi","1"},{"capture","1"},{"local_alias","1"},
 {"forged_cast","1"},{"raw_extract","1"},{"declared_alloc","1"},{"wrong_cap_abi","1"},{"forged_struct","1"}
}};
std::string read(const fs::path&p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("missing "+p.string());return {std::istreambuf_iterator<char>(f),{}};}
Rows parse(const std::string&input,const fs::path&dir){
 std::istringstream in(input);std::string line;
 if(!std::getline(in,line)||line!="case\texit_status")throw std::runtime_error("header");
 Rows rows;
 while(std::getline(in,line)){
  const auto tab=line.find('\t');
  if(tab==std::string::npos||line.find('\t',tab+1)!=std::string::npos)throw std::runtime_error("row");
  const auto name=line.substr(0,tab),status=line.substr(tab+1);
  if(status!="0"&&status!="1")throw std::runtime_error("exit");
  if(!rows.emplace(name,Row{status,read(dir/(name+".stdout"))}).second)throw std::runtime_error("duplicate");
 }
 return rows;
}
bool valid(const Rows&rows){
 if(rows.size()!=expected.size())return false;
 const std::regex unrelated("error\\[(E137|E264)\\]|parse error|PARSE_REFUSED");
 for(const auto&[name,status]:expected){
  const auto it=rows.find(std::string(name));
  if(it==rows.end()||it->second.status!=status)return false;
  const auto&log=it->second.log;
  const std::string marker="ACTUAL_CHECKER_VERDICT "+std::string(status)+"\n";
  const auto p=log.find(marker);
  if(p==std::string::npos||log.find(marker,p+marker.size())!=std::string::npos||std::regex_search(log,unrelated))return false;
  if(status=="0"&&log.find("error")!=std::string::npos)return false;
  if(status=="1"&&log.find("WorkerFn")==std::string::npos)return false;
 }
 return true;
}
int main(int argc,char**argv){
 try{
  if(argc!=2)throw std::runtime_error("usage: reference RESULTS");
  const fs::path dir=argv[1];const auto input=read(dir/"WORKER_VERDICTS.tsv");
  const auto rows=parse(input,dir);if(!valid(rows))throw std::runtime_error("semantic matrix");
  unsigned rejected=0;
  auto wrong=rows;wrong.erase(wrong.begin());if(!valid(wrong))++rejected;
  wrong=rows;wrong["unexpected"]={"0",""};if(!valid(wrong))++rejected;
  wrong=rows;wrong["capture"].status="0";if(!valid(wrong))++rejected;
  wrong=rows;wrong["raw_extract"].log="error[E137]: unknown API\nACTUAL_CHECKER_VERDICT 1\n";if(!valid(wrong))++rejected;
  wrong=rows;wrong["forged_cast"].log="error[E264]: unsupported\nACTUAL_CHECKER_VERDICT 1\n";if(!valid(wrong))++rejected;
  wrong=rows;wrong["good_i64"].log.clear();if(!valid(wrong))++rejected;
  try{(void)parse(input+"capture\t1\n",dir);}catch(const std::runtime_error&){++rejected;}
  if(rejected!=7)throw std::runtime_error("wrong-output control accepted");
  std::cout<<"WORKER_FRONTEND_ADMISSION_OK cases=10 positive=2 negative=8 wrong_controls=7 native_certificate=false\n";
 }catch(const std::exception&e){std::cerr<<"WORKER_FRONTEND_ADMISSION_REFUSED "<<e.what()<<'\n';return 1;}
}
