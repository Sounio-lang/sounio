// ORCHESTRATOR CONTROL ONLY. No Unicode algorithm or acceptance claim.
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
namespace fs=std::filesystem;
void need(bool v,const char* what){if(!v)throw std::runtime_error(what);}
void word(std::ostream& out,std::uint32_t n){for(int i=0;i<4;++i)out.put(static_cast<char>((n>>(8*i))&255));}
void prepare(const fs::path& root) {
 need(!fs::exists(root),"fresh control root");fs::create_directories(root);
 for(int family=1;family<=3;++family) {
  const std::string name=family==1?"unicode":family==2?"locale":"metadata";
  const std::string prefix=family==1?"norm":family==2?"locale":"locale-meta";
  const int kind=family==1?1:family==2?5:6;
  const auto dir=root/name;fs::create_directories(dir);
  std::ofstream manifest(dir/"MANIFEST.tsv");manifest<<"filename\tkind\trows\tpayload_bytes\n";
  std::ofstream scope(dir/"SCOPE.txt");scope<<"CXX_ARGV_CONTROL_ONLY\n";
  for(int row=0;row<2;++row) {
   const auto filename=prefix+"-00000"+std::to_string(row)+".bin";
   const auto count=static_cast<std::uint32_t>(row+2);
   std::ofstream out(dir/filename,std::ios::binary);
   out<<"SOUNUTF1";word(out,kind);word(out,count);word(out,count*4);word(out,0x00120000);
   for(std::uint32_t i=0;i<count;++i)word(out,0);
   out.close();need(bool(out),"write fixture");
   manifest<<filename<<'\t'<<kind<<'\t'<<count<<'\t'<<count*4<<'\n';
  }
  manifest.close();scope.close();need(bool(manifest)&&bool(scope),"write manifest");
 }
}
int main(int argc,char**argv) {
 try {
  if(argc==3&&std::string(argv[1])=="--prepare"){prepare(argv[2]);return 0;}
  const std::string kind=std::getenv("CONTROL_KIND")?std::getenv("CONTROL_KIND"):"";
  if(kind.empty())need(argc==1,"legacy must receive no user args");
  else need(argc==2&&argv[1]==kind,"actual exec argv binding");
  const auto path=kind=="2"?"tests/stdlib/wave3-core/locale-data/current.bin":
                  kind=="3"?"tests/stdlib/wave3-core/locale-meta-data/current.bin":
                            "tests/stdlib/wave3-core/data/current.bin";
  std::ifstream in(path,std::ios::binary);std::array<unsigned char,24> header{};
  in.read(reinterpret_cast<char*>(header.data()),header.size());need(bool(in),"control fixture");
  std::uint32_t rows=0;for(int i=0;i<4;++i)rows|=std::uint32_t(header[12+i])<<(8*i);
  const std::string mode=std::getenv("CONTROL_MODE")?std::getenv("CONTROL_MODE"):"good";
  std::cout<<"CXX_COHERENT_RUNNER_CONTROL_ONLY\nACTUAL_USER_ARGC "<<argc-1<<'\n';
  if(!kind.empty())std::cout<<"ACTUAL_KIND "<<argv[1]<<'\n';
  std::cout<<rows+(mode=="wrong_rows"?1:0)<<'\n';
  if(mode!="missing_marker")std::cout<<"UNICODE_CONFORMANCE_OK\n";
  return mode=="nonzero_exit"?3:0;
 } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 65;}
}
