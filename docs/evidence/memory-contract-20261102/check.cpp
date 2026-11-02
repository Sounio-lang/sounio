#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <sys/mman.h>
using Bytes=std::vector<unsigned char>;
int main(int argc,char**argv){
 if(argc!=4)return 2;
 std::ifstream oracle(argv[1]), emitted(argv[2]);std::string line;std::vector<Bytes> refs;
 const std::regex hex("0x([0-9a-fA-F]{2})");
 while(std::getline(oracle,line))if(line.find("encoding:")!=std::string::npos){Bytes b;for(auto i=std::sregex_iterator(line.begin(),line.end(),hex);i!=std::sregex_iterator();++i)b.push_back(std::stoul((*i)[1],nullptr,16));refs.push_back(b);}
 std::size_t cases=0,bad=0,native=0;bool execute=std::string(argv[3])=="native";
 if(execute&&(!__builtin_cpu_supports("avx512f"))){std::cerr<<"AVX512F unavailable\n";return 3;}
 while(std::getline(emitted,line)){
  if(!line.starts_with("CASE "))continue;
  std::istringstream in(line);std::string tag,op;int base=0,disp=0,len=0,value=0;
  if(!(in>>tag>>base>>disp>>op>>len)||cases>=36)return 6;
  constexpr std::array bases{2,3,4,5,12,13};
  constexpr std::array displacements{-256,1,256};
  if(base!=bases[cases/6]||disp!=displacements[(cases/2)%3]||op!=(cases%2?"store":"load"))return 6;
  Bytes b;while(in>>value){if(value<0||value>255)return 6;b.push_back(static_cast<unsigned char>(value));}
  if(!in.eof())return 6;
  if(cases>=refs.size()||b.size()!=static_cast<std::size_t>(len)||b!=refs[cases]){std::cout<<"ENCODING_MISMATCH base="<<base<<" disp="<<disp<<" op="<<op<<" bytes="<<b.size()<<"\n";++bad;++cases;continue;}++cases;
  if(!execute)continue;
  alignas(64) std::array<unsigned char,8192> src{},dst{},expected{};
  src.fill(0x6b);dst.fill(0xa5);expected=dst;
  const auto offset=2048+disp;
  for(int i=0;i<64;++i)src[(op=="load"?offset:2048)+i]=static_cast<unsigned char>(i*3+1);
  if(op=="load")std::memcpy(expected.data()+2048,src.data()+offset,64);else std::memcpy(expected.data()+offset,src.data()+2048,64);
  Bytes fn{0x53,0x55,0x41,0x54,0x41,0x55,0x49,0x89,0xe3};
  if(op=="store")fn.insert(fn.end(),{0x62,0xf1,0xfd,0x48,0x10,0x07});
  fn.push_back(static_cast<unsigned char>(0x48|(base>=8?1:0)));fn.push_back(0x89);fn.push_back(static_cast<unsigned char>(0xc0|((op=="load"?7:6)<<3)|(base&7)));
  fn.insert(fn.end(),b.begin(),b.end());fn.insert(fn.end(),{0x4c,0x89,0xdc});
  if(op=="load")fn.insert(fn.end(),{0x62,0xf1,0xfd,0x48,0x11,0x06});
  fn.insert(fn.end(),{0x41,0x5d,0x41,0x5c,0x5d,0x5b,0xc5,0xf8,0x77,0xc3});
  void*mem=mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(mem==MAP_FAILED)return 4;
  std::memcpy(mem,fn.data(),fn.size());if(mprotect(mem,4096,PROT_READ|PROT_EXEC))return 4;
  reinterpret_cast<void(*)(void*,void*)>(mem)(src.data()+2048,dst.data()+2048);munmap(mem,4096);
  if(dst!=expected){std::cerr<<"NATIVE_ADDRESS_OR_GUARD_FAIL base="<<base<<" disp="<<disp<<" op="<<op<<"\n";return 5;}++native;
 }
 std::cout<<"CASES="<<cases<<" ENCODING_FAILURES="<<bad<<" NATIVE_CASES="<<native<<"\n";
 if(cases!=36||refs.size()!=36)return 6;
 return bad?1:0;
}
