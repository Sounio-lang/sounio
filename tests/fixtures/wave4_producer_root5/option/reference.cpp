#include <bit>
#include <cstdint>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
int main(int ac,char**av){try{if(ac!=2)throw std::runtime_error("argc");using I=std::int64_t;std::array<std::array<I,2>,16> e{{{37,1},{-9,0},{38,1},{-9,0},{std::bit_cast<I>(39.5),1},{std::bit_cast<I>(-9.0),0},{1,1},{-9,0},{40,1},{-9,0},{41,1},{-9,0},{42,1},{-9,0},{7,8},{0,0}}};for(I seed=100;seed<=106;++seed)for(I i=0;i<512;++i)e[15][0]+=seed+i;std::ifstream f(av[1]);std::string line;std::size_t n=0;while(std::getline(f,line)){std::istringstream s(line);I id,a,b;std::string extra;if(!(s>>id>>a>>b)||(s>>extra)||n>=e.size()||id!=I(n)||a!=e[n][0]||b!=e[n][1])throw std::runtime_error("row mismatch "+std::to_string(n));++n;}if(n!=e.size())throw std::runtime_error("missing rows");std::cout<<"PASS rows16 exact tag/payload/f64bits/recursive_box/wide_box/nested/sret/clobber\n";return 0;}catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

