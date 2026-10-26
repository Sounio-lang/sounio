// C++23 Boost113-bit oracle for canonical Gamma prefixes and Poisson-mode weights.
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using Q=boost::multiprecision::number<boost::multiprecision::cpp_bin_float<113,boost::multiprecision::digit_base_2,void,std::int32_t,-100000000,100000000>>;
struct C{int kind;double a,x;};
std::vector<C> cases(){std::vector<C> v;for(double a:{.5,1.,7.999999,8.,8.000001,128.125,8192.,999999.5})for(double r:{.01,.874999,.875001,1.,1.124999,1.125001,100.})v.push_back({0,a,a*r});for(double l:{1e-8,.1,.49,.5,.99,1.01,8.1,127.123,8191.5123,8192.,999999.5}){v.push_back({1,std::floor(l),l});v.push_back({2,std::floor(l),l});}return v;}
std::string lit(double x){std::ostringstream s;s<<std::scientific<<std::setprecision(17)<<x;return s.str();}
int main(int argc,char**argv){auto v=cases();if(argc==2&&std::string(argv[1])=="--generate"){std::cout<<R"SIO(use special::checked_gamma_beta::{gamma_log_prefix_checked}
use math::pure::{ln_checked}
fn row(id:i64,kind:i64,a:f64,x:f64) with IO,Mut,Div,Panic {
 var status:i64=0
 var value:f64=0.0
 if kind==0 {let r=gamma_log_prefix_checked(a,x);status=r.status;value=r.value}
 else if kind==1 {
  if a==0.0 {value= -x}
  else {let r=gamma_log_prefix_checked(a,x);let l=ln_checked(a);status=r.status;value=r.value-l.value}
 }else{let r=gamma_log_prefix_checked(a+0.5,x);let l=ln_checked(a+0.5);status=r.status;value=r.value-l.value}
 let bits:i64=f64_to_bits(value)
 print("ROW ");print(id);print(" ");print(status);print(" ");println(bits)
}
fn main()->i32 with IO,Mut,Div,Panic {
 let inf=1.0e308*10.0
 assert(gamma_log_prefix_checked(0.0,1.0).status==1)
 assert(gamma_log_prefix_checked(1.0,0.0).status==1)
 assert(gamma_log_prefix_checked(1000001.0,1.0).status==5)
 assert(gamma_log_prefix_checked(1.0,inf).status==2)
)SIO";for(size_t i=0;i<v.size();i++)std::cout<<" row("<<i<<","<<v[i].kind<<","<<lit(v[i].a)<<","<<lit(v[i].x)<<")\n";std::cout<<" println(\"GAMMA_PREFIX_COMPLETE "<<v.size()<<"\")\n 0\n}\n";return 0;}
if(argc!=2)return 2;std::ifstream f(argv[1]);std::string line;size_t next=0;int errors=0;bool done=false;double maxmode=0,maxprefix=0;
while(std::getline(f,line)){if(line.rfind("GAMMA_PREFIX_COMPLETE ",0)==0){done=line=="GAMMA_PREFIX_COMPLETE "+std::to_string(v.size());continue;}std::istringstream s(line);std::string tag;size_t id;int status;std::int64_t bits;s>>tag>>id>>status>>bits;if(!s||tag!="ROW"||id!=next||next>=v.size()){errors++;continue;}auto c=v[next++];Q a(c.a),x(c.x),expected;
if(c.kind==0)expected=a*log(x)-x-boost::math::lgamma(a);
else if(c.kind==1)expected=-x+a*log(x)-boost::math::lgamma(a+1);
else expected=-x+(a+Q(.5))*log(x)-boost::math::lgamma(a+Q(1.5));
double actual=std::bit_cast<double>(bits),error=abs(Q(actual)-expected).convert_to<double>();double limit=c.kind?5e-12:std::max(5e-12,abs(expected).convert_to<double>()*3e-14);
if(c.kind)maxmode=std::max(maxmode,error);else maxprefix=std::max(maxprefix,error);
if(status!=0||!std::isfinite(actual)||error>limit){std::cerr<<std::setprecision(17)<<"FAIL "<<id<<" kind "<<c.kind<<" a "<<c.a<<" x "<<c.x<<" status "<<status<<" error "<<error<<" limit "<<limit<<"\n";errors++;}}
if(!done||next!=v.size())errors++;std::cout<<std::setprecision(17)<<"ROWS "<<next<<" MAX_MODE_LOG_ERROR "<<maxmode<<" MAX_PREFIX_ABS_ERROR "<<maxprefix<<" ERRORS "<<errors<<"\n";return errors?1:0;}
