#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
int main(int argc,char**argv){if(argc!=2)return 2;std::ifstream f(argv[1]);int n=0,bad=0,markers=0;for(std::string line;std::getline(f,line);){if(line=="PAIR_VALUES_DONE"){++markers;continue;}std::istringstream s(line);std::string tag;int i;std::int64_t bits;if(!(s>>tag>>i>>bits)||tag!="PAIR_VALUE"||i!=n)return 3;double actual=std::bit_cast<double>(std::uint64_t(bits)),expected=double(i+1);if(actual!=expected){++bad;std::cout<<"PAIR_CPP23_MISMATCH index="<<i<<" actual="<<actual<<" expected="<<expected<<'\n';}++n;}if(n!=4||markers!=1)return 4;std::cout<<"PAIR_CPP23_RECORDS values="<<n<<" mismatches="<<bad<<'\n';return bad?1:0;}
