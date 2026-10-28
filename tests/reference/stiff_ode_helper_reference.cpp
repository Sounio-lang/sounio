// C++23 reference for the unchanged deterministic van der Pol Euler helper.
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
struct Case{double mu,x,y,dt;std::int64_t steps;};
constexpr std::array<Case,5> cases{{{5,.5,0,.02,80},{5.001,.5,0,.02,80},{4.999,.5,0,.02,80},{5,.5,2,.02,0},{1.2,.25,-.5,.005,200}}};
double solve(Case c){for(std::int64_t i=0;i<c.steps;++i){const double dx=c.y;const double dy=c.mu*(1.0-c.x*c.x)*c.y-c.x;c.x=c.x+c.dt*dx;c.y=c.y+c.dt*dy;}return c.y;}
std::string load(const char*p){std::ifstream f(p);if(!f)throw std::runtime_error("missing input");std::ostringstream s;s<<f.rdbuf();return s.str();}
int main(int argc,char**argv){try{
 if(argc==3&&std::string(argv[1])=="--fixture"){
  const auto s=load(argv[2]);const auto begin=s.find("fn vdp_final_y("),end=s.find("\n}\n",begin);
  if(begin==std::string::npos||end==std::string::npos)return 3;
  std::cout<<s.substr(begin,end+3-begin)<<"\nfn main()->i32 with IO,Mut,Div,Panic {\n"<<std::setprecision(17)<<std::showpoint;
  for(std::size_t i=0;i<cases.size();++i){
   const auto c=cases[i];
   std::cout<<"let value"<<i<<":f64=vdp_final_y("<<c.mu<<","<<c.x<<","<<c.y<<","<<c.dt<<","<<c.steps<<")\n";
   std::cout<<"let observed_bits"<<i<<":i64=f64_to_bits(value"<<i<<")\n";
   std::cout<<"println(\"VDP_BITS "<<i<<" {}\",observed_bits"<<i<<")\n";
  }
  std::cout<<"println(\"VDP_HELPER_PASS\");0\n}\n";return 0;
 }
 if(argc==3&&std::string(argv[1])=="--check"){
  std::istringstream in(load(argv[2]));std::array<bool,cases.size()> seen{};std::string line;int markers=0;
  while(std::getline(in,line)){
   if(line=="VDP_HELPER_PASS"){++markers;continue;}
   std::istringstream row(line);std::string tag,extra;std::size_t id=0;std::int64_t bits=0;
   if(!(row>>tag>>id>>bits)||tag!="VDP_BITS"||id>=cases.size()||seen[id]||(row>>extra))return 9;
   if(bits!=std::bit_cast<std::int64_t>(solve(cases[id])))return 8;
   seen[id]=true;
  }
  if(markers!=1)return 9;
  for(const bool yes:seen)if(!yes)return 9;
  std::cout<<"VDP_HELPER_CPP23_PASS\n";return 0;
 }
 return 2;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 3;}}
