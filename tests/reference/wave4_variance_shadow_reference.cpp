// ADR009: independent C++23 exact integer reference.
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
using I=std::int64_t;
using R=std::array<I,4>;
static constexpr std::array<I,7> seeds{0,1,5,-1,-5,17,1024};
static R expected(I seed) {
 I s=seed;if(s==0)s=1;
 const I core=s*3+1,aliases=(s+2)*3-1;
 for(int i=0;i<3;++i)s=s*3+1;
 return {seed,core,aliases,s};
}
static void verify(const char*path) {
 std::ifstream in(path);if(!in)throw std::runtime_error("missing native stdout");
 for(I seed:seeds) {
  std::string line;if(!std::getline(in,line))throw std::runtime_error("truncated rows");
  std::istringstream row(line);R got{};
  for(I&x:got)if(!(row>>x))throw std::runtime_error("invalid integer");
  std::string extra;if(row>>extra)throw std::runtime_error("extra field");
  if(got!=expected(seed))throw std::runtime_error("result/order mismatch");
 }
 std::string extra;if(std::getline(in,extra))throw std::runtime_error("extra row");
}
int main(int argc,char**argv)try {
 static_assert(sizeof(I)==8);
 if(argc==2&&std::string_view(argv[1])=="--emit") {
  for(I seed:seeds){const auto r=expected(seed);std::cout<<r[0]<<' '<<r[1]<<' '<<r[2]<<' '<<r[3]<<'\n';}
 }else if(argc==3&&std::string_view(argv[1])=="--check")verify(argv[2]);
 else return 64;
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
