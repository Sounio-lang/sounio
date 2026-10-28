// C++23 independent fixed-case van der Pol/LCG/PBox/GUM/CARA oracle.
// No stdlib or Sounio implementation is included. IEEE double operation order
// for MC, Euler and the retained Newton helper is explicit. Disable FMA.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
constexpr double mu=5.0,mu_var=.25,x0=.5,x0_var=.01,initial_y=0,dt=.02;
constexpr std::int64_t steps=80,samples=5000;
constexpr std::uint64_t initial_seed=12345;
constexpr long double ce_abs_tolerance=2e-12L,efe_abs_tolerance=2e-12L;
[[noreturn]] void fail(std::string s){throw std::runtime_error(s);}
double newton(double x,int n){if(x<=0)return 0;double y=x;for(int i=0;i<n;++i)y=.5*(y+x/y);return y;}
double solve(double parameter){double x=x0,y=initial_y;for(std::int64_t i=0;i<steps;++i){const double dx=y;const double dy=parameter*(1.0-x*x)*y-x;x=x+dt*dx;y=y+dt*dy;}return y;}
struct MC{double mean,variance;};
MC mc(){
 std::uint64_t state=initial_seed;double sum=0,sumsq=0;
 const double sd=newton(mu_var,8);
 for(std::int64_t i=0;i<samples;++i){
  double u_sum=0;
  for(int j=0;j<12;++j){
   state=state*UINT64_C(6364136223846793005)+UINT64_C(1442695040888963407);
   const auto signed_state=std::bit_cast<std::int64_t>(state);
   if(signed_state==std::numeric_limits<std::int64_t>::min())fail("INT64_MIN uniform not covered by this fixed-case reference");
   const std::uint64_t magnitude=signed_state<0?std::uint64_t(0)-state:state;
   const auto bucket=magnitude%UINT64_C(1073741824);
   const double u=static_cast<double>(bucket)/1073741824.0;
   u_sum=u_sum+u;
  }
  const double z=u_sum-6.0;
  const double sample=mu+z*sd;
  const double result=solve(sample);
  sum=sum+result;sumsq=sumsq+result*result;
 }
 const double mean=sum/static_cast<double>(samples);
 const double variance=sumsq/static_cast<double>(samples)-mean*mean;
 return {mean,std::max(variance,0.0)};
}
struct Box{double lo,hi,var;std::int64_t conf;};
Box lift(double value,double var,std::int64_t conf){return{value,value,var,conf};}
Box add(Box a,Box b){return{a.lo+b.lo,a.hi+b.hi,a.var+b.var,std::min(a.conf,b.conf)*99/100};}
Box sub(Box a,Box b){return{a.lo-b.hi,a.hi-b.lo,a.var+b.var,std::min(a.conf,b.conf)*99/100};}
Box mul(Box a,Box b){
 const std::array<double,4> c{a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
 const double am=.5*(a.lo+a.hi),bm=.5*(b.lo+b.hi);
 return{*std::min_element(c.begin(),c.end()),*std::max_element(c.begin(),c.end()),bm*bm*a.var+am*am*b.var+a.var*b.var,std::min(a.conf,b.conf)*99/100};
}
std::pair<Box,Box> box_step(Box x,Box y,Box parameter){
 double low=1e30,high=-1e30;
 for(int i=0;i<7;++i){const double frac=static_cast<double>(i)/6.0;const double t=x.lo+frac*(x.hi-x.lo);const double value=t*t;low=std::min(low,value);high=std::max(high,value);}
 const Box x2{low,high,0,x.conf},one=lift(1,0,1000),dtbox=lift(dt,0,1000);
 const Box term=mul(sub(one,x2),y);
 const Box dy=sub(mul(parameter,term),x);
 return{add(x,mul(dtbox,y)),add(y,mul(dtbox,dy))};
}
struct Expected{std::map<std::string,double> exact;std::map<std::string,long double> approx;bool old_mean_in,old_wider,old_margin;double final_pbox_variance;};
Expected expected(){
 const MC result=mc();
 Box x=lift(x0,x0_var,900),y=lift(initial_y,0,1000);const Box p=lift(mu,mu_var,850);
 for(std::int64_t i=0;i<steps;++i){const auto next=box_step(x,y,p);x=next.first;y=next.second;}
 const double f0=solve(mu),fp=solve(mu+.001),df=(fp-f0)/.001;
 const double delta_var=df*df*mu_var,delta_width=2.0*newton(delta_var,12);
 const double mc_std=newton(result.variance,12),width=y.hi-y.lo;
 const double ratio=delta_width>1e-9?width/delta_width:999.0;
 const double tightness=mc_std>1e-9?width/(2.0*mc_std):999.0;
 const double nominal=result.mean*.04,lo=result.mean*.01,hi=result.mean*.08;
 const double support_lo=std::min(lo,hi),support_hi=std::max(lo,hi);
 const double clamped_nominal=std::clamp(nominal,support_lo,support_hi);
 const long double precise=.75L*static_cast<long double>(clamped_nominal)+.25L*static_cast<long double>(support_lo);
 const long double minimum=std::min({static_cast<long double>(clamped_nominal),static_cast<long double>(support_lo),static_cast<long double>(support_hi)});
 const long double shifted=.75L*std::exp(-2.L*(static_cast<long double>(clamped_nominal)-minimum))+.25L*std::exp(-2.L*(static_cast<long double>(support_lo)-minimum));
 const long double smooth=minimum-.5L*std::log(shifted);
 const long double epistemic=std::max(.5L*std::log(static_cast<long double>(result.variance)/.15L),0.L);
 Expected e;e.final_pbox_variance=y.var;
 e.exact={{"MC_MEAN",result.mean},{"MC_VAR",result.variance},{"MC_STD",mc_std},{"POINT_Y",f0},{"K_LO",y.lo},{"K_HI",y.hi},{"K_WIDTH",width},{"K_VARIANCE",y.var},{"DELTA_VAR",delta_var},{"DELTA_W",delta_width},{"RATIO",ratio},{"TIGHTNESS",tightness},{"READOUT_LO",1.05*y.lo},{"READOUT_HI",1.05*y.hi}};
 e.approx={{"CE_KLIB",smooth},{"CE_PRECISE",precise},{"PRAG_KLIB",smooth},{"PRAG_PRECISE",precise},{"EFE_EPI_K",epistemic},{"EFE_EPI_P",epistemic},{"EFE_TOTAL_K",0.L-epistemic-smooth},{"EFE_TOTAL_P",0.L-epistemic-precise}};
 e.old_mean_in=y.lo<=result.mean&&result.mean<=y.hi;e.old_wider=width>delta_width*1.01;e.old_margin=smooth<precise-.001L;
 if(!(result.variance>0&&y.lo==y.hi&&width==0&&delta_width>0&&smooth<precise&&precise-smooth>100*ce_abs_tolerance))fail("independent selected-case expectations not established");
 return e;
}
std::vector<std::string> read_lines(const std::string& path){std::ifstream in(path);if(!in)fail("missing log");std::vector<std::string> out;for(std::string s;std::getline(in,s);)out.push_back(s);return out;}
void write_lines(const std::string& path,const std::vector<std::string>& rows){std::ofstream out(path);for(const auto&s:rows)out<<s<<'\n';if(!out)fail("control write");}
void verify(const std::string& path){
 const auto e=expected();const auto rows=read_lines(path);std::map<std::string,double> observed;
 bool flags=false;int pass=0;
 for(const auto& row:rows){
  if(row=="STIFF_ODE_KLIB_BRIDGE_PASS"){++pass;continue;}
  std::istringstream in(row);std::string tag,name,extra;
  if(!(in>>tag))fail("empty row");
  if(tag=="STIFF_BITS"){
   std::int64_t bits=0;if(!(in>>name>>bits)||(in>>extra)||observed.contains(name))fail("malformed/duplicate bits");
   const double value=std::bit_cast<double>(bits);if(!std::isfinite(value))fail("nonfinite "+name);
   if(const auto it=e.exact.find(name);it!=e.exact.end()){
    if(bits!=std::bit_cast<std::int64_t>(it->second))fail("bit mismatch "+name);
   }else if(const auto it=e.approx.find(name);it!=e.approx.end()){
    const long double tolerance=name.starts_with("EFE_")?efe_abs_tolerance:ce_abs_tolerance;
    if(std::abs(static_cast<long double>(value)-it->second)>tolerance)fail("reference mismatch "+name);
   }else fail("unknown metric");
   observed.emplace(name,value);
  }else if(tag=="STIFF_ORIGINAL_PREDICATES"){
   int a=-1,b=-1,c=-1;if(flags||!(in>>a>>b>>c)||(in>>extra)||a!=int(e.old_mean_in)||b!=int(e.old_wider)||c!=int(e.old_margin))fail("counterexample flags");flags=true;
  }else fail("unexpected row");
 }
 if(pass!=1||!flags||observed.size()!=e.exact.size()+e.approx.size())fail("missing rows or pass marker");
 if(!(observed.at("CE_KLIB")<observed.at("CE_PRECISE")&&observed.at("PRAG_KLIB")<observed.at("PRAG_PRECISE")))fail("strict selected-case ordering");
 if(observed.at("PRAG_KLIB")!=observed.at("CE_KLIB")||observed.at("PRAG_PRECISE")!=observed.at("CE_PRECISE"))fail("bridge identity");
 std::cout<<"STIFF_ODE_CPP23_REAL_MC_PBOX_GUM_CARA_OK\n";
}
void mutate(std::string_view kind,const std::string& input,const std::string& output){
 auto rows=read_lines(input);
 auto row=std::find_if(rows.begin(),rows.end(),[](const auto&s){return s.starts_with("STIFF_BITS MC_MEAN ");});
 if(row==rows.end())fail("no MC metric");
 if(kind=="value"||kind=="zero-mc"||kind=="nan"){
  const double value=kind=="value"?1.0:kind=="zero-mc"?0.0:std::numeric_limits<double>::quiet_NaN();
  *row="STIFF_BITS MC_MEAN "+std::to_string(std::bit_cast<std::int64_t>(value));
 }else if(kind=="missing")rows.erase(row);
 else if(kind=="duplicate"){const auto copy=*row;rows.insert(row,copy);}
 else fail("unknown mutation");
 write_lines(output,rows);
}
int main(int argc,char**argv){try{
 if(argc==2&&std::string_view(argv[1])=="--report"){
  const auto e=expected();std::cout<<std::setprecision(21);
  for(const auto&[name,value]:e.exact)std::cout<<name<<" "<<value<<" bits="<<std::bit_cast<std::int64_t>(value)<<'\n';
  for(const auto&[name,value]:e.approx)std::cout<<name<<" "<<value<<'\n';
  std::cout<<"PBOX_VARIANCE_DIAGNOSTIC "<<e.final_pbox_variance<<'\n';
  std::cout<<"ORIGINAL_PREDICATES "<<e.old_mean_in<<" "<<e.old_wider<<" "<<e.old_margin<<'\n';
  std::cout<<"CE_ABS_TOLERANCE "<<ce_abs_tolerance<<"\nEFE_ABS_TOLERANCE "<<efe_abs_tolerance<<'\n';return 0;
 }
 if(argc==3&&std::string_view(argv[1])=="--check"){verify(argv[2]);return 0;}
 if(argc==5&&std::string_view(argv[1])=="--mutate"){mutate(argv[2],argv[3],argv[4]);return 0;}
 return 64;
}catch(const std::exception&e){std::cerr<<"STIFF_REFERENCE_ERROR "<<e.what()<<'\n';return 1;}}
