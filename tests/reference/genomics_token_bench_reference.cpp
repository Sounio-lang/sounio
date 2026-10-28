// Materialize the unchanged token_bench body and verify its integer contract.
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
using I=std::int64_t;
constexpr I rows=100000;
std::string load(const char* p){std::ifstream f(p);if(!f)throw std::runtime_error("missing input");std::ostringstream s;s<<f.rdbuf();return s.str();}
void replace(std::string& s,const std::string& from,const std::string& to,std::size_t expected){
 std::size_t found=0,pos=0;while((pos=s.find(from,pos))!=std::string::npos){s.replace(pos,from.size(),to);pos+=to.size();++found;}
 if(found!=expected)throw std::runtime_error("template occurrence disagreement");
}
int main(int argc,char**argv){try{
 if(argc==4&&std::string(argv[1])=="--generate"){
  std::string mode=argv[3];if(mode!="slot"&&mode!="snapshot")return 2;
  auto s=load(argv[2]);
  replace(s," TOKEN_SLOT_DECL",mode=="slot"?" var slot=count_tsv_feed_slot()":"",1);
  replace(s,"TOKEN_COUNT",std::to_string(rows),1);
  replace(s,"  TOKEN_FEED",mode=="slot"?"  let ignored:i64=count_tsv_feed_into(&!state,&bytes,1,&!slot)":"  let snapshot=count_tsv_feed(&!state,&bytes,1)",2);
  replace(s," TOKEN_FINISH",mode=="slot"?" let finished_code:i64=count_tsv_finish_into(&!state,&!slot)":" let eof=count_tsv_finish(&!state)",1);
  std::cout<<"use genomics::io::count_tsv::{count_tsv_new,count_tsv_feed_slot,count_tsv_feed_into,count_tsv_finish_into,count_tsv_feed,count_tsv_finish,count_tsv_ack_row}\n"<<s;return 0;
 }
 if(argc==3&&std::string(argv[1])=="--check"){
  std::istringstream in(load(argv[2]));std::string line;int records=0,markers=0;
  while(std::getline(in,line)){
   if(line=="TOKEN_BENCH_PASS"){++markers;continue;}
   if(line.rfind("TOKEN_BENCH ",0)!=0){if(!line.empty())return 8;continue;}
   std::istringstream row(line);std::string tag,trailing;std::array<I,6> got{};row>>tag;
   for(auto& value:got)if(!(row>>value))return 8;
   if(row>>trailing)return 8;
   const std::array<I,6> want{rows,rows,2*rows,rows,1,0};
   if(got!=want)return 8;
   ++records;
  }
  if(records!=1||markers!=1)return 9;
  std::cout<<"TOKEN_BENCH_CPP23_PASS\n";return 0;
 }
 return 2;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 3;}}
