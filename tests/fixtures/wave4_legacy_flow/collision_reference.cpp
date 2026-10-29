#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
static std::uint64_t hash(const std::string&s) {std::uint64_t h=2166136261ULL;for(unsigned char c:s)h=((h^c)*16777619ULL)&0x1FFFFFFFFFULL;return h;}
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 std::unordered_map<std::uint64_t,std::string> seen;
 std::uint64_t state=0x6a09e667f3bcc909ULL;
 for(std::uint64_t i=0;i<2000000ULL;++i) {
  state=state*6364136223846793005ULL+1442695040888963407ULL;
  std::string n="Collision"+std::to_string(state);auto h=hash(n);
  auto [it,inserted]=seen.emplace(h,n);
  if(!inserted&&it->second!=n) {
   const auto&a=it->second; std::ofstream f(argv[1]);
   f<<"enum "<<a<<" { Up, Down }\nenum "<<n<<" { Up, Down }\nfn choose(a: "<<n<<") -> i64 { match a { "<<n<<"::Up => { return 11 }, "<<n<<"::Down => { return 11 } } }\nfn main() -> i32 { 0 }\n";
   std::cout<<"COLLISION "<<a<<" "<<n<<" hash="<<h<<" attempts="<<i<<"\n";
   return f?0:3;
  }
 }
 return 4;
}
