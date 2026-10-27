// Independent C++23 Boost113-bit actual-output NCT reference.
#include <boost/math/distributions/non_central_t.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
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

Q opposite_reference(double t,double df,double d){
 using R=boost::multiprecision::cpp_bin_float_quad;
 R a=R(df)/2,tt(t),dd(d),degrees(df);
 R aa=2+tt*tt/a,bb=dd*tt/sqrt(a);
 R root=4*a/(sqrt(bb*bb+8*a*aa)+bb);
 R scale_s=root/sqrt(a);
 R normalizer=log(R(2))+a*log(a)-boost::math::lgamma(a);
 boost::math::normal_distribution<R> normal;
 auto sf=[&](R z){return cdf(complement(normal,z));};
 // s=scale_s*z^2 removes the chi endpoint singularity for df>=0.5.
 R base_log=normalizer+degrees*log(scale_s)+log(R(2));
 R logscale=base_log-a*scale_s*scale_s+log(sf(tt*scale_s+dd));
 auto integrand=[&](R z)->R{
  if(z==0){if(degrees==R(.5))return exp(base_log+log(sf(dd))-logscale);return 0;}
  R ss=scale_s*z*z;if(ss==0)return 0;R n=sf(tt*ss+dd);
  if(n==0)return 0;
  return exp(base_log+(2*degrees-1)*log(z)-a*ss*ss+log(n)-logscale);
 };
 auto integrate=[&](R endpoint,R tolerance,R& error)->R{
  R sum=0;error=0;R left=0,right=R(.25);
  for(int panel=0;panel<32&&left<endpoint;panel++){
   if(right>endpoint)right=endpoint;
   R e=0,l1=0;
   sum+=boost::math::quadrature::gauss_kronrod<R,61>::integrate(integrand,left,right,15,tolerance,&e,&l1);
   error+=e;left=right;right*=2;
  }
  return sum;
 };
 R answer=0,upper=8;
 bool qualified=false;
 for(int round=0;round<8;round++){
  R e1,e2;
  R first=integrate(upper,R("1e-20"),e1);
  R second=integrate(upper,R("1e-24"),e2);
  if(second<=0||!boost::multiprecision::isfinite(second)||abs(first-second)>abs(second)*R("1e-18")||e2>abs(second)*R("1e-18")){
   std::ostringstream detail;detail<<std::setprecision(36)<<"positive oracle refinement t="<<t<<" df="<<df<<" d="<<d<<" upper="<<upper<<" first="<<first<<" second="<<second<<" e2="<<e2<<" relative="<<e2/second;throw std::runtime_error(detail.str());
  }
  answer=exp(logscale)*second;
  R upper_s=scale_s*upper*upper;
  R truncation=boost::math::gamma_q(a,a*upper_s*upper_s)*sf(tt*upper_s+dd);
  if(answer>sf(dd))throw std::runtime_error("positive oracle normal bound failed");
  if(truncation<=answer*R("1e-20")){qualified=true;break;}
  upper*=sqrt(R(2));
 }
 if(!qualified)throw std::runtime_error("positive oracle truncation budget exhausted");
 return Q(answer);
}

double val(std::int64_t b){return std::bit_cast<double>(b);}
struct Case{double t,df,delta;int budget=2048,status=0;};
std::vector<Case> cases(){std::vector<Case> c;for(double df:{1.,5.,30.,120.})for(double delta:{0.,.1,1.,3.,8.,30.})for(double t:{.1,1.,3.,10.,40.}){c.push_back({t,df,delta});c.push_back({-t,df,-delta});}c.push_back({1,0,1,2048,1});c.push_back({1,5,129,2048,5});c.push_back({1,5,3,1,3});for(double df:{.5,1.,5.,30.,120.})for(double delta:{.1,1.,3.,8.})for(double t:{.1,1.,3.,10.,40.}){c.push_back({t,df,-delta});c.push_back({-t,df,delta});}
for(double df:{1.,5.,30.})for(double t:{1.,10.}){c.push_back({t,df,-20});c.push_back({-t,df,20});c.push_back({t,df,-30});c.push_back({-t,df,30});}
c.push_back({1,5,-3,1,3});c.push_back({1e8,5,-1,2048,5});c.push_back({1,1,-128,8192,0});c.push_back({-1,1,128,8192,0});c.push_back({0,5,128,2048,0});c.push_back({0,5,-128,2048,0});c.push_back({1e308,1,0,2048,0});c.push_back({1e308,1,1,2048,5});return c;}
std::string lit(double x){std::ostringstream s;s<<std::scientific<<std::setprecision(17)<<x;return s.str();}
int main(int argc,char**argv){auto c=cases();if(argc==2&&std::string(argv[1])=="--generate"){std::cout<<R"SIO(use stats::noncentral_t::{noncentral_t_tails_checked}
fn flag(v:bool)->i64 {if v {1}else{0}}
fn bits(v:f64)->i64 {f64_to_bits(v)}
fn row(id:i64,t:f64,df:f64,delta:f64,budget:i64) with IO,Mut,Div,Panic {
 let r=noncentral_t_tails_checked(t,df,delta,budget,1.0e-11)
 print("ROW ");print(id);print(" ");print(r.status);print(" ")
 print(flag(r.cdf_available));print(" ");print(flag(r.sf_available));print(" ")
 print(flag(r.log_cdf_available));print(" ");print(flag(r.log_sf_available));print(" ")
 print(bits(r.cdf));print(" ");print(bits(r.sf));print(" ");print(bits(r.log_cdf));print(" ");print(bits(r.log_sf));print(" ")
 print(r.method);print(" ");print(r.mixture_terms);print(" ");print(r.kernel_iterations);print(" ")
 print(r.quadrature_evaluations);print(" ");print(r.truncation_evaluations);print(" ")
 print(flag(r.estimates_available));print(" ");print(bits(r.quadrature_relative_error_estimate));print(" ");println(bits(r.truncation_log_estimate))
}
fn main()->i32 with IO,Mut,Div,Panic {
)SIO";for(size_t i=0;i<c.size();i++)std::cout<<" row("<<i<<","<<lit(c[i].t)<<","<<lit(c[i].df)<<","<<lit(c[i].delta)<<","<<c[i].budget<<")\n";std::cout<<" println(\"NCT_ROWS_COMPLETE "<<c.size()<<"\")\n 0\n}\n";return 0;}
if(argc!=2)return 2;std::ifstream f(argv[1]);std::string line;size_t next=0;int errors=0;bool done=false;double maxlog=0,maxrel=0;
auto fail=[&](const std::string&s){std::cerr<<"FAIL row "<<next<<" "<<s<<"\n";errors++;};
while(std::getline(f,line)){if(errors)return 1;if(line.rfind("NCT_ROWS_COMPLETE ",0)==0){done=line=="NCT_ROWS_COMPLETE "+std::to_string(c.size());continue;}std::istringstream s(line);std::string tag;size_t id;int status;s>>tag>>id>>status;if(!s||tag!="ROW"||id!=next||next>=c.size()){fail("malformed/reordered");continue;}auto v=c[next];std::vector<std::int64_t> p;std::int64_t k;while(s>>k)p.push_back(k);if(p.size()!=16){fail("field count");next++;continue;}if(status!=v.status)fail("status "+std::to_string(status)+" expected "+std::to_string(v.status));if(p[9]<0||p[9]>v.budget||p[10]<0||p[10]>2LL*v.budget*v.budget+v.budget||p[11]<0||p[12]<0||p[11]+p[12]>v.budget)fail("actual work bound");if(status!=0){for(int i=0;i<4;i++)if(p[i])fail("failure exposes availability");next++;continue;}
Q lo,hi;
if(v.t*v.delta<0){
 Q small=opposite_reference(std::abs(v.t),v.df,std::abs(v.delta));
 if(v.t>0){hi=small;lo=1-small;}else{lo=small;hi=1-small;}
}else if(v.t==0){boost::math::normal_distribution<Q>d;lo=cdf(d,-Q(v.delta));hi=cdf(complement(d,-Q(v.delta)));}else if(v.delta==0&&v.df==1){Q tail=atan(Q(1)/abs(Q(v.t)))/boost::math::constants::pi<Q>();if(v.t>0){hi=tail;lo=1-tail;}else{lo=tail;hi=1-tail;}}else{boost::math::non_central_t_distribution<Q>d(Q(v.df),Q(v.delta));lo=cdf(d,Q(v.t));hi=cdf(complement(d,Q(v.t)));}
if(lo<=0||hi<=0||!boost::multiprecision::isfinite(lo)||!boost::multiprecision::isfinite(hi)){fail("invalid oracle probability");next++;continue;}
Q expected[4]={lo,hi,lo<Q(.5)?log(lo):boost::math::log1p(-hi),hi<Q(.5)?log(hi):boost::math::log1p(-lo)};
for(int i=0;i<4;i++){double actual=std::bit_cast<double>(p[i+4]);Q least=Q(std::numeric_limits<double>::denorm_min())/2;if(!p[i]){if((i<2?expected[i]:abs(expected[i]))>least)fail("representable probability/log unavailable");continue;}if((i<2?expected[i]:abs(expected[i]))<least)fail("underflow falsely marked available");if(i<2&&actual==0&&expected[i]>0)fail("positive probability falsely available as zero");if(i>=2&&actual==0&&abs(expected[i])>least)fail("representable log tail erased to zero");if(!std::isfinite(actual)){fail("nonfinite");continue;}Q error=abs(Q(actual)-expected[i]);double scaled=(i>=2?error:Q(error/std::max(Q(1e-300),abs(expected[i])))).convert_to<double>();if(i>=2)maxlog=std::max(maxlog,scaled);else maxrel=std::max(maxrel,scaled);if(scaled>2e-8){std::ostringstream o;o<<std::setprecision(17)<<"column "<<i<<" error "<<scaled<<" t "<<v.t<<" df "<<v.df<<" delta "<<v.delta<<" actual "<<actual<<" expected "<<expected[i];fail(o.str());}}
if(p[8]!=(v.delta==0?1:v.t==0?4:v.t*v.delta>0?2:3))fail("method metadata");if(p[8]==3){Q small_log=v.t>0?expected[3]:expected[2];if(!p[13]||val(p[14])<0||val(p[14])>1.25e-12||Q(val(p[15]))-small_log>log(Q(1.25e-12))+Q(2e-8))fail("quadrature/truncation estimates");}if(p[11]<0||p[12]<0||p[11]+p[12]>v.budget)fail("integral work");next++;}
if(!done||next!=c.size()){fail("incomplete");}std::cout<<std::setprecision(17)<<"ROWS "<<next<<" MAX_ABS_LOG_ERROR "<<maxlog<<" MAX_RELATIVE_VALUE_ERROR "<<maxrel<<" ERRORS "<<errors<<"\n";return errors?1:0;}
