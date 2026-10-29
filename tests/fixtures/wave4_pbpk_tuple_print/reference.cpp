#include <bit>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
using I=std::int64_t;
int main(int argc,char**argv){try{
 if(argc!=3){return 64;}
 const std::string kind=argv[1];std::map<std::string,I> ints;std::map<std::string,double> floats;
 if(kind=="scalars"){ints={{"I",-7000000001LL}};floats={{"F",.5}};}
 else if(kind=="float_pair")floats={{"F0",.5},{"F1",1.25}};
 else if(kind=="signed_pair")ints={{"I0",-7},{"I1",-7000000001LL}};
 else if(kind=="mixed_pair"){ints={{"I",-7}};floats={{"F",.5}};}
 else if(kind=="nested"||kind=="nested_typed"||kind=="nested_declared_inner"||kind=="nested_scalar_only"){ints={{"I",-7}};floats={{"F",.5},{"V",1.0}};}
 else{ints={{"I",-7}};floats={{"V",1.0},{"U",.5}};}
 std::ifstream f(argv[2]);if(!f){throw std::runtime_error("missing output");}
 std::map<std::string,bool> seen;std::string line;
 while(std::getline(f,line)){auto eq=line.find('=');if(eq==std::string::npos){throw std::runtime_error("bad line");}const auto key=line.substr(0,eq),value=line.substr(eq+1);if(seen[key]){throw std::runtime_error("duplicate");}seen[key]=true;
  if(auto it=ints.find(key);it!=ints.end()){I got=0;auto p=std::from_chars(value.data(),value.data()+value.size(),got);if(p.ec!=std::errc{}||p.ptr!=value.data()+value.size()||got!=it->second)throw std::runtime_error("signed scalar mismatch");}
  else if(auto jt=floats.find(key);jt!=floats.end()){double got=0;auto p=std::from_chars(value.data(),value.data()+value.size(),got);if(p.ec!=std::errc{}||p.ptr!=value.data()+value.size()||std::bit_cast<std::uint64_t>(got)!=std::bit_cast<std::uint64_t>(jt->second))throw std::runtime_error("float scalar mismatch");}
  else throw std::runtime_error("extra label");
 }
 if(seen.size()!=ints.size()+floats.size()){throw std::runtime_error("missing label");}
 std::cout<<"TUPLE_SCALAR_CPP23_OK "<<kind<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
