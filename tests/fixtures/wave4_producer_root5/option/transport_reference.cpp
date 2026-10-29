#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <stdexcept>
using I=std::int64_t;
int main(int ac,char**av){try{if(ac!=2)throw std::runtime_error("argc");std::array<std::array<I,2>,18>e{{{17,1},{18,2},{-9,0},{17,1},{-9,0},{-9,0},{20,3},{17,4},{-9,0},{17,5},{18,6},{-9,0},{17,7},{18,8},{-9,0},{17,9},{18,10},{-9,0}}};std::ifstream f(av[1]);std::string l;std::size_t n=0;while(std::getline(f,l)){I id,a,b,c;std::string x;std::istringstream s(l);I sum=0;for(I j=0;j<512;++j)sum+=830+I(n)+j;if(!(s>>id>>a>>b>>c)||(s>>x)||n>=e.size()||id!=I(n)||a!=e[n][0]||b!=e[n][1]||c!=sum)throw std::runtime_error("row "+std::to_string(n));++n;}if(n!=e.size())throw std::runtime_error("census");std::cout<<"PASS transport18 sameSome/sameNone/iflet/match/solelive/deadreturn/clobber\n";return 0;}catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

