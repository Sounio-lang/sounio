#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
namespace fs=std::filesystem;
std::string read(const fs::path&p){std::ifstream f(p);if(!f)throw std::runtime_error("missing "+p.string());return {std::istreambuf_iterator<char>(f),{}};}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("usage probe_reference PROBES");
 const fs::path root=argv[1];unsigned checked=0,rejected=0;
 for(const std::string kind:{"real","cleanup-mutant"}){
  std::map<std::string,std::string> table;std::istringstream in(read(root/kind/"VERDICTS.tsv"));std::string line;
  while(std::getline(in,line)){
   const auto a=line.find('\t'),b=line.find('\t',a+1);
   if(a==std::string::npos||b==std::string::npos||line.find('\t',b+1)!=std::string::npos)throw std::runtime_error("malformed probe producer");
   if(!table.emplace(line.substr(0,b),line.substr(b+1)).second)throw std::runtime_error("duplicate probe producer");
  }
  if(table.size()!=6)throw std::runtime_error("probe selection must contain6 records");
  for(const std::string spine:{"inplace","byvalue"})for(const std::string mode:{"good","type","fault"}){
   const auto key=spine+"\t"+mode;const bool mutant=kind=="cleanup-mutant"&&mode=="fault";
   const auto expected=mutant?"91":"0";const auto it=table.find(key);
   if(it==table.end()||it->second!=expected)throw std::runtime_error("wrong status "+kind+" "+key);
   const auto log=read(root/kind/(spine+"-"+mode+".stdout"));
   if(log.find("LOGICAL_FRAME_STATUS "+std::string(expected)+"\n")==std::string::npos)throw std::runtime_error("missing frame status");
   const auto marker=log.find("LOGICAL_FRAME_PROBE_OK\n");
   if(mutant){if(marker!=std::string::npos)throw std::runtime_error("cleanup mutant accepted");++rejected;}
   else{if(marker==std::string::npos)throw std::runtime_error("real probe marker missing");++checked;}
   if(mode=="fault"){
    if(log.find("error[E397]")==std::string::npos||log.find("storage or binding identity overflow")==std::string::npos)throw std::runtime_error("missing exact resource reason");
   }else if(mode=="type"){
    if(log.find("error[E004]")==std::string::npos||log.find("error[E397]")!=std::string::npos)throw std::runtime_error("type error reclassified");
   }else if(log.find("error[")!=std::string::npos)throw std::runtime_error("good probe diagnostic");
   if(log.find("E137")!=std::string::npos||log.find("parse error")!=std::string::npos)throw std::runtime_error("unsupported refusal is not a cleanup result");
  }
 }
 if(checked!=10||rejected!=2)throw std::runtime_error("wrong probe count");
 std::cout<<"LOGICAL_FRAME_CLEANUP_OK real=6 nonfault_mutant=4 omitted_cleanup_refused=2\n";
}catch(const std::exception&e){std::cerr<<"LOGICAL_FRAME_CLEANUP_REFUSED "<<e.what()<<'\n';return 1;}}
