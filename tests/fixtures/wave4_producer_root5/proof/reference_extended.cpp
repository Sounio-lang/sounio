#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <cstdint>
using I=std::int64_t;
std::vector<I> bytes(int which){std::vector<I>v;auto add=[&](std::initializer_list<I>x){v.insert(v.end(),x);};auto imm=[&](I n){add({72,184});for(int i=0;i<8;++i){v.push_back(n&255);n>>=8;}};auto store=[&](int disp){add({72,137,133,disp&255,255,255,255});};if(which==0){add({72,49,192});store(-8);store(-16);}else{imm(7);store(-8);imm(1);store(-16);}add({72,141,133,240,255,255,255});return v;}
int main(int ac,char**av){try{if(ac!=2)throw std::runtime_error("argc");std::ifstream f(av[1]);std::string l;std::array<I,48>x{};std::array<I,48>y{};x[0]=1;x[1]=2;x[2]=x[3]=y[2]=y[3]=630000000200002003LL;for(int i:{6,7,8,9,11})x[i]=y[i]=-1;x[25]=y[25]=1;x[27]=y[27]=1;x[32]=y[32]=3;x[33]=y[33]=4;x[35]=y[35]=3;x[36]=y[36]=4;x[41]=y[41]=1;x[42]=y[42]=2;int rows=0,codes=0;while(std::getline(f,l)){std::istringstream s(l);std::string first,extra;s>>first;if(first=="CODE"){I kind,n;if(!(s>>kind>>n)||kind!=codes||codes>=2)throw std::runtime_error("code envelope");auto expected=bytes(codes);if(n!=I(expected.size()))throw std::runtime_error("code length");for(auto b:expected){I got;if(!(s>>got)||got!=b)throw std::runtime_error("actual emitter bytes");}if(s>>extra)throw std::runtime_error("extra code bytes");++codes;continue;}I id,a,b;std::istringstream z(l);if(!(z>>id>>a>>b)||(z>>extra)||rows>=48||id!=rows||a!=x[rows]||b!=y[rows])throw std::runtime_error("metadata row "+std::to_string(rows));++rows;}if(rows!=48||codes!=2)throw std::runtime_error("census");std::cout<<"PASS actual_emitters2 None_tag_payload_zero Some_tag_payload7 proof_controls48\n";return 0;}catch(std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

