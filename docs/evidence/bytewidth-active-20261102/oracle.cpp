#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
int main(int argc,char**argv){
 if(argc!=2)return 2;
 const std::uint64_t word=0x123456ff;
 const auto low=static_cast<std::uint8_t>(word&255);
 const auto signed_low=low<128?static_cast<int>(low):static_cast<int>(low)-256;
 const std::string expected=std::to_string(signed_low)+"\n"+std::to_string(low)+"\n"+std::to_string(word)+"\nBYTEWIDTH_ACTIVE_PASS\n";
 std::ifstream f(argv[1]);const std::string got((std::istreambuf_iterator<char>(f)),{});
 if(got!=expected){std::cerr<<"BYTEWIDTH_ORACLE_REJECTED\n";return 1;}
 std::cout<<"BYTEWIDTH_CPP23_ORACLE_PASS\n";return 0;
}
