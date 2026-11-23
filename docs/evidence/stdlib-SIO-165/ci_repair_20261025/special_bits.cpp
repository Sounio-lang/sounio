#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
int main(int argc,char**argv){if(argc!=2)return 2;std::ifstream f(argv[1]);int count=0;for(std::string line;std::getline(f,line);){std::istringstream s(line);std::string tag;s>>tag;if(tag!="SPECIAL_BITS")continue;std::int64_t a,b;if(!(s>>a>>b))return 3;auto n=std::uint64_t(a),i=std::uint64_t(b);if((n&0x7ff0000000000000ull)!=0x7ff0000000000000ull||(n&0x000fffffffffffffull)==0||i!=0x7ff0000000000000ull)return 4;std::cout<<"SPECIAL_CLASS_CPP23_PASS nan_bits=0x"<<std::hex<<n<<" positive_inf_bits=0x"<<i<<'\n';++count;}return count==1?0:5;}
