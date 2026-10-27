// C++23 independent analytic boundary + augmented RK4 reference.
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using Real=long double;
using State=std::array<Real,4>;
static Real boundary(Real t){const Real pi=std::acos(-1.L),x=2*pi*1.2L*t;return .75L*1e-5L*(1+.05L*std::sin(x))+.25L*8e-6L*(1+.30L*std::sin(x+.785L));}
static State rhs(Real t,const State& y){
 const Real diffusion=(1e-9L/3.9e-6L)/3.9e-6L,advection=3e-4L/3.9e-6L,loss=.1L+.003L*.075L;
 const Real a=-advection-2*diffusion-loss,d=-advection-diffusion-loss;
 return {advection*(boundary(t)-y[0])+diffusion*(boundary(t)-2*y[0]+y[1])+.1L*(2e-6L/5-y[0])-.003L*.075L*y[0],
 (advection+diffusion)*(y[0]-y[1])+.1L*(2e-6L/5-y[1])-.003L*.075L*y[1],
 a*y[2]+diffusion*y[3],(advection+diffusion)*y[2]+d*y[3]};
}
static State rk4(){
 State y={1e-5L,2e-5L,2e-7L,3e-7L};const Real h=1e-4L;
 const auto a=rhs(0,y);State t{};
 for(int i=0;i<4;++i)t[i]=y[i]+h*a[i]/2;
 const auto b=rhs(h/2,t);
 for(int i=0;i<4;++i)t[i]=y[i]+h*b[i]/2;
 const auto c=rhs(h/2,t);
 for(int i=0;i<4;++i)t[i]=y[i]+h*c[i];
 const auto d=rhs(h,t);
 for(int i=0;i<4;++i)y[i]+=h*(a[i]+2*b[i]+2*c[i]+d[i])/6;
 return y;
}
static std::map<std::string,Real> expected(){
 std::map<std::string,Real> e;
 e["shared.initial.cov"]=2e-7L*3e-7L;e["shared.initial.mean_u"]=(2e-7L+3e-7L)/2;
 e["shared.negative.cov"]=-2e-7L*3e-7L;
 const auto y=rk4();e["shared.step.x0"]=y[0];e["shared.step.x1"]=y[1];e["shared.step.f0"]=y[2];e["shared.step.f1"]=y[3];e["shared.step.cov"]=y[2]*y[3];
 const Real pi=std::acos(-1.L),t=.125L,x=2*pi*1.2L*t,z=x+.785L;
 const std::array<Real,6> g={.75L*(1+.05L*std::sin(x)),.25L*(1+.30L*std::sin(z)),2*pi*t*(.75L*1e-5L*.05L*std::cos(x)+.25L*8e-6L*.30L*std::cos(z)),.75L*1e-5L*std::sin(x),.25L*8e-6L*std::sin(z),.25L*8e-6L*.30L*std::cos(z)};
 const std::array<Real,6> u={5e-7L,4e-7L,.05L,.01L,.02L,.01L};
 Real sum=0;for(int i=0;i<6;++i){e["gradient."+std::to_string(i)]=g[i];sum+=(g[i]*u[i])*(g[i]*u[i]);}
 e["boundary.value"]=boundary(t);e["boundary.u"]=std::sqrt(sum);return e;
}
static int check(const char* path){
 auto e=expected();std::ifstream in(path);if(!in)throw std::runtime_error("missing native output");
 std::map<std::string,Real> got;std::map<std::string,int> statuses;std::string line;int marker=0;
 while(std::getline(in,line)){
  std::istringstream r(line);std::string tag,name,extra;std::int64_t bits=0;
  r>>tag;
  if(tag=="QROW"||tag=="QGRAD"){if(!(r>>name>>bits)||(r>>extra))throw std::runtime_error("invalid row");if(tag=="QGRAD")name="gradient."+name;double value=std::bit_cast<double>(bits);if(!std::isfinite(value)||!got.emplace(name,value).second)throw std::runtime_error("nonfinite or duplicate row");}
  else if(tag=="QSTATUS"){int v;if(!(r>>name>>v)||(r>>extra)||!statuses.emplace(name,v).second)throw std::runtime_error("invalid status");}
  else if(line=="PHONON_QUANTITY_CHECKED_OK")++marker;
 }
 if(marker!=1||got.size()!=e.size())throw std::runtime_error("missing numerical rows/marker");
 for(const auto& [name,want]:e){auto it=got.find(name);if(it==got.end())throw std::runtime_error("missing expected row");const auto value=it->second,scale=std::max(std::abs(want),1e-300L);if(std::abs(value-want)>1e-11L*scale){std::cerr<<name<<" got="<<static_cast<double>(value)<<" expected="<<static_cast<double>(want)<<"\n";throw std::runtime_error("phonon Quantity numerical mismatch");}}
 const std::array<std::string,15> fields={"d_eff","dx","k_release","cl_site","fu","kp","c_portal","c_ha","f_heart","a_portal","a_ha","phi_ha","c_tissue","v_blood","zonation"};
 if(statuses.size()!=20)throw std::runtime_error("missing negative controls");
 for(const auto& f:fields)if(statuses.at("dim."+f)!=5)throw std::runtime_error("dimension mismatch accepted");
 for(const auto& [name,v]:std::map<std::string,int>{{"negative_u",6},{"nan",1},{"domain",2},{"budget",7},{"marginal_mismatch",9}})if(statuses.at(name)!=v)throw std::runtime_error("wrong negative status");
 std::cout<<"PHONON_QUANTITY_CPP_CONCORDANCE_OK rows="<<got.size()<<" negatives="<<statuses.size()<<" relative=1e-11\n";return 0;
}
int main(int argc,char** argv){try{
 if(argc==4&&std::string(argv[1])=="--wrong-diagonal"){
  std::ifstream in(argv[2]);std::ofstream out(argv[3]);if(!in||!out)throw std::runtime_error("mutation IO");std::string l;int n=0;
  while(std::getline(in,l)){if(l.starts_with("QROW shared.initial.cov ")){l="QROW shared.initial.cov 0";++n;}out<<l<<"\n";}if(n!=1)throw std::runtime_error("ambiguous mutation");return 0;
 }
 if(argc!=2)throw std::runtime_error("usage: checker native-output");
 return check(argv[1]);
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
