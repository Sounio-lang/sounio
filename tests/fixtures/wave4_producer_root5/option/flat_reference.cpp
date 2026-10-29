#include <bit>
#include <cstdint>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
int main(int ac,char**av){try{if(ac<2||ac>3)throw std::runtime_error("argc");using I=std::int64_t;bool cells=ac==3&&std::string(av[2])=="--cells";if(ac==3&&!cells)throw std::runtime_error("mode");std::ifstream f(av[1]);std::string l;std::size_t n=0;if(cells){std::array<std::array<I,2>,4>e{{{1,77},{0,0},{1,88},{0,0}}};while(std::getline(f,l)){std::istringstream s(l);I a,b;std::string extra;if(!(s>>a>>b)||(s>>extra)||n>=e.size()||a!=e[n][0]||b!=e[n][1])throw std::runtime_error("cells "+std::to_string(n));++n;}if(n!=4)throw std::runtime_error("cells census");std::cout<<"PASS actual cells4 None both-zero Some tag/payload\n";return 0;}std::vector<std::array<I,3>>e{{0,37,1},{1,-9,0},{2,38,1},{3,-9,0},{4,std::bit_cast<I>(39.5),1},{5,std::bit_cast<I>(-9.0),0},{6,1,1},{7,-9,0},{8,40,1},{9,-9,0},{10,41,1},{11,-9,0},{15,0,0}};for(I seed=100;seed<=105;++seed)for(I i=0;i<512;++i)e.back()[1]+=seed+i;while(std::getline(f,l)){std::istringstream s(l);I id,a,b;std::string extra;if(!(s>>id>>a>>b)||(s>>extra)||n>=e.size()||id!=e[n][0]||a!=e[n][1]||b!=e[n][2])throw std::runtime_error("flat row "+std::to_string(n));++n;}if(n!=e.size())throw std::runtime_error("flat census");std::cout<<"PASS flat rows13 exact f64bits/recursive_Box/wide_Box/tags/SRet/clobber\n";return 0;}catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

