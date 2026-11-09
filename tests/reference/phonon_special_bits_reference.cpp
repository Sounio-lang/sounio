#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("stdout file required");
 std::ifstream in(argv[1]);if(!in)throw std::runtime_error("missing stdout");
 bool nan=false,inf=false,marker=false;
 for(std::string line;std::getline(in,line);){
  if(line=="PHONON_SPECIAL_BITS_PASS"){if(marker)throw std::runtime_error("duplicate marker");marker=true;continue;}
  std::istringstream row(line);std::string tag,name,extra;std::int64_t signed_bits;
  if(!(row>>tag>>name>>signed_bits)||(row>>extra)||tag!="SPECIAL_BITS")throw std::runtime_error("malformed record");
  auto bits=static_cast<std::uint64_t>(signed_bits);auto x=std::bit_cast<double>(bits);
  if(name=="nan"&&!nan){if(((bits>>52)&2047)!=2047||(bits&0xfffffffffffffull)==0||!std::isnan(x))throw std::runtime_error("NaN class lost");nan=true;}
  else if(name=="inf"&&!inf){if(bits!=0x7ff0000000000000ull||!std::isinf(x)||std::signbit(x))throw std::runtime_error("+Inf class lost");inf=true;}
  else throw std::runtime_error("unknown/duplicate value");
 }
 if(!nan||!inf||!marker)throw std::runtime_error("incomplete classes");
 std::cout<<"PHONON_SPECIAL_CPP23_PASS nan_fraction_nonzero=verified positive_infinity=verified payload_identity=not_claimed\n";return 0;
}catch(const std::exception&e){std::cerr<<"PHONON_SPECIAL_CPP23_FAIL "<<e.what()<<'\n';return 1;}}
