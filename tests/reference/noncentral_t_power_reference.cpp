// C++23 independent genuine power, NCT-quantile and minimum-integer-n checks.
#include <boost/math/distributions/non_central_t.hpp>
#include <boost/math/distributions/students_t.hpp>
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
using Q=boost::multiprecision::cpp_bin_float_quad;
using R=boost::multiprecision::number<boost::multiprecision::cpp_bin_float<512,boost::multiprecision::digit_base_2,void,std::int32_t,-100000000,100000000>>;
struct P{double d;int n;double alpha;bool two,per;};
struct S{double d,target,alpha;bool two,per;int cap=1000,budget=4096,status=0;};
struct T{double df,delta,p;bool upper=false;};
std::vector<P> powers(){return {{.5,2,.05,true,false},{.8,2,.05,true,false},{.5,34,.05,true,false},{.5,64,.05,true,true},{-.5,34,.05,true,false},{.5,10,.05,false,false},{-.5,10,.05,false,false},{0,50,.05,true,false},{1,60,.05,true,true},{.2,100,.05,false,false},{-.2,100,.05,false,false},{.1,3,.05,true,false}};}
Q power(double dd,int n,double alpha,bool two,bool per){
 if(dd==0)return Q(alpha);
 R nf(n),df=per?2*nf-2:nf-1,d(dd),a(alpha);
 R delta=d*sqrt(per?nf/2:nf);
 boost::math::students_t_distribution<R> central(df);
 R critical=quantile(complement(central,two?a/2:a));
 boost::math::non_central_t_distribution<R> noncentral(df,delta);
 R result=cdf(complement(noncentral,critical));
 if(two)result+=cdf(noncentral,-critical);
 if(result<=0||result>=1)throw std::runtime_error("oracle power outside open support");
 return Q(result);
}
std::vector<S> sizes(){return {{.5,.8,.05,true,false},{.5,.8,.05,true,true},{.8,.9,.05,true,false},{.8,.9,.05,true,true},{0,.8,.05,true,false,1000,4096,7},{0,.01,.05,true,false,1000,1,0},{-.5,.8,.05,false,false,1000,4096,7},{.001,.8,.05,true,false,20,4096,6},{.5,.8,.05,true,false,1000,1,3},{-.5,.04,.05,false,false,1000,4096,5},{.5,power(.5,34,.05,true,false).convert_to<double>(),.05,true,false,1000,4096,5}};}
std::vector<T> quantiles(){return {{1,0,1e-300,true},{1,0,.025,false},{5,0,.025,true},{30,0,.975,false},{1000,0,.025,true},{5,1,.025,false},{5,1,.975,false},{5,-1,.025,false},{5,-1,.975,false},{5,1,.5,false},{5,-1,.5,true}};}
std::string lit(double x){std::ostringstream s;s<<std::scientific<<std::setprecision(17)<<x;return s.str();}
double val(std::int64_t b){return std::bit_cast<double>(b);}
void generate(bool legacy){
 auto pv=powers();
 if(legacy){std::cout<<"use stats::power::{power_ttest_onesample,power_ttest_twosample}\nfn main()->i32 with IO,Mut,Div,Panic {\n";for(size_t i=0;i<pv.size();i++){auto c=pv[i];std::cout<<" let p"<<i<<"="<<(c.per?"power_ttest_twosample":"power_ttest_onesample")<<"("<<lit(c.d)<<","<<c.n<<","<<lit(c.alpha)<<","<<(c.two?"true":"false")<<")\n let b"<<i<<":i64=f64_to_bits(p"<<i<<")\n print(\"LROW "<<i<<" \");println(b"<<i<<")\n";}std::cout<<" println(\"LEGACY_POWER_COMPLETE "<<pv.size()<<"\")\n 0\n}\n";return;}
 std::cout<<R"SIO(use stats::checked_power::{power_onesample_checked,power_twosample_checked,n_for_power_onesample_checked,n_for_power_twosample_checked}
use stats::noncentral_t::{noncentral_t_quantile_checked}
fn flag(v:bool)->i64 {if v {1}else{0}}
fn bits(v:f64)->i64 {f64_to_bits(v)}
fn prow(id:i64,d:f64,n:i64,alpha:f64,two:bool,per:bool) with IO,Mut,Div,Panic {
 let r=if per {power_twosample_checked(d,n,alpha,two,4096,1.0e-11)}else{power_onesample_checked(d,n,alpha,two,4096,1.0e-11)}
 print("PROW ");print(id);print(" ");print(r.status);print(" ")
 print(flag(r.value_available));print(" ");print(flag(r.log_available));print(" ");print(bits(r.value));print(" ");print(bits(r.log_value));print(" ")
 print(flag(r.critical_available));print(" ");print(bits(r.critical));print(" ");print(r.critical_evaluations);print(" ");print(r.cdf_evaluations);print(" ");print(r.mixture_terms);print(" ");print(r.quadrature_evaluations);print(" ");println(r.truncation_evaluations)
}
fn srow(id:i64,d:f64,target:f64,alpha:f64,two:bool,per:bool,cap:i64,budget:i64) with IO,Mut,Div,Panic {
 let r=if per {n_for_power_twosample_checked(d,target,alpha,two,cap,budget,1.0e-11)}else{n_for_power_onesample_checked(d,target,alpha,two,cap,budget,1.0e-11)}
 print("SROW ");print(id);print(" ");print(r.status);print(" ")
 print(r.n);print(" ");print(flag(r.available));print(" ");print(r.candidate_n);print(" ");print(bits(r.previous_power));print(" ");print(bits(r.power));print(" ")
 print(flag(r.previous_available));print(" ");print(flag(r.power_available));print(" ");print(flag(r.at_domain_boundary));print(" ");print(flag(r.verified));print(" ")
 print(bits(r.decision_margin));print(" ");print(r.evaluations);print(" ");println(r.cap)
}
fn qrow(id:i64,df:f64,delta:f64,p:f64,upper:bool) with IO,Mut,Div,Panic {
 let r=noncentral_t_quantile_checked(df,delta,p,upper,4096,1.0e-11)
 print("QROW ");print(id);print(" ");print(r.status);print(" ")
 print(flag(r.available));print(" ");print(flag(r.forward_log_available));print(" ")
 print(bits(r.value));print(" ");print(bits(r.lower));print(" ");print(bits(r.upper));print(" ");print(bits(r.forward_log_tail));print(" ");println(r.evaluations)
}
fn main()->i32 with IO,Mut,Div,Panic {
 let inf=1.0e308*10.0
 assert(power_onesample_checked(inf,20,0.05,true,4096,1.0e-11).status==2)
 assert(power_onesample_checked(0.5,1,0.05,true,4096,1.0e-11).status==1)
 assert(power_onesample_checked(0.5,20,0.0,true,4096,1.0e-11).status==1)
 assert(n_for_power_onesample_checked(0.5,0.8,0.05,true,1,4096,1.0e-11).status==1)
)SIO";
 for(size_t i=0;i<pv.size();i++){auto c=pv[i];std::cout<<" prow("<<i<<","<<lit(c.d)<<","<<c.n<<","<<lit(c.alpha)<<","<<(c.two?"true":"false")<<","<<(c.per?"true":"false")<<")\n";}
 auto sv=sizes();for(size_t i=0;i<sv.size();i++){auto c=sv[i];std::cout<<" srow("<<i<<","<<lit(c.d)<<","<<lit(c.target)<<","<<lit(c.alpha)<<","<<(c.two?"true":"false")<<","<<(c.per?"true":"false")<<","<<c.cap<<","<<c.budget<<")\n";}
 auto tv=quantiles();for(size_t i=0;i<tv.size();i++){auto c=tv[i];std::cout<<" qrow("<<i<<","<<lit(c.df)<<","<<lit(c.delta)<<","<<lit(c.p)<<","<<(c.upper?"true":"false")<<")\n";}
 std::cout<<" println(\"POWER_COMPLETE "<<pv.size()<<" "<<sv.size()<<" "<<tv.size()<<"\")\n 0\n}\n";
}
int main(int argc,char**argv){
 if(argc==2&&std::string(argv[1])=="--generate"){generate(false);return 0;}
 if(argc==2&&std::string(argv[1])=="--generate-legacy"){generate(true);return 0;}
 bool legacy=argc==3&&std::string(argv[1])=="--legacy";if(argc!=2&&!legacy)return 2;
 std::ifstream file(argv[legacy?2:1]);std::string line;auto pv=powers();auto sv=sizes();auto tv=quantiles();size_t pi=0,si=0,qi=0;bool done=false;int errors=0;double maxpower=0,maxquantile=0;
 auto fail=[&](std::string x){std::cerr<<"FAIL "<<x<<"\n";errors++;};
 while(std::getline(file,line)){
  if(line.rfind("LEGACY_POWER_COMPLETE ",0)==0){done=line=="LEGACY_POWER_COMPLETE "+std::to_string(pv.size());continue;}
  if(line.rfind("POWER_COMPLETE ",0)==0){done=line=="POWER_COMPLETE "+std::to_string(pv.size())+" "+std::to_string(sv.size())+" "+std::to_string(tv.size());continue;}
  std::istringstream in(line);std::string tag;size_t id;int status=0;in>>tag>>id;if(!legacy)in>>status;std::vector<std::int64_t>b;std::int64_t bit;while(in>>bit)b.push_back(bit);
  if(legacy||tag=="PROW"){
   if((legacy&&tag!="LROW")||id!=pi||pi>=pv.size()||b.size()!=(legacy?1:11)){fail("power row shape");continue;}auto c=pv[pi++];if(status!=0){fail("power status "+std::to_string(status)+" id "+std::to_string(id));continue;}Q expected=power(c.d,c.n,c.alpha,c.two,c.per);double actual=val(b[legacy?0:2]);double error=abs((Q(actual)-expected)/expected).convert_to<double>();maxpower=std::max(maxpower,error);if(error>2e-8){std::ostringstream s;s<<std::setprecision(17)<<"power "<<id<<" actual "<<actual<<" expected "<<expected<<" relative "<<error;fail(s.str());}
   if(!legacy){if(!b[0]||!b[1])fail("power availability");if(abs(Q(val(b[3]))-log(expected))>Q(2e-8))fail("power log");if(c.d<0&&!c.two&&!(actual<c.alpha))fail("negative directional power");}
  }else if(tag=="SROW"){
   if(id!=si||si>=sv.size()||b.size()!=12){fail("size row shape");continue;}auto c=sv[si++];if(status!=c.status){fail("size status "+std::to_string(id)+" got "+std::to_string(status)+" expected "+std::to_string(c.status));continue;}
   if(b[10]<0||b[10]>c.budget)fail("size work");
   if(status!=0){if(b[0]!=0||b[1]||b[8])fail("invented minimum");if(status==6){if(b[2]!=c.cap||!b[6]||!(power(c.d,c.cap,c.alpha,c.two,c.per)<Q(c.target)))fail("cap proof");}continue;}
   int n=static_cast<int>(b[0]);if(!b[1]||!b[6]||!b[8]||n<2||n>c.cap)fail("minimum fields");
   Q at=power(c.d,n,c.alpha,c.two,c.per);if(at<Q(c.target))fail("actual power(n) below target");
   if(abs(Q(val(b[4]))-at)>Q(2e-8))fail("reported power(n)");
   if(n==2){if(!b[7]||b[5])fail("domain-boundary proof");}
   else{Q previous=power(c.d,n-1,c.alpha,c.two,c.per);if(!(previous<Q(c.target))||!b[5])fail("actual power(n-1) reaches target");if(abs(Q(val(b[3]))-previous)>Q(2e-8))fail("reported power(n-1)");}
  }else if(tag=="QROW"){
   if(id!=qi||qi>=tv.size()||b.size()!=7){fail("quantile shape");continue;}auto c=tv[qi++];if(status!=0){fail("quantile status "+std::to_string(id)+" got "+std::to_string(status));continue;}
   boost::math::non_central_t_distribution<R>d(R(c.df),R(c.delta));R expected=c.upper?quantile(complement(d,R(c.p))):quantile(d,R(c.p));double actual=val(b[2]);R scale=abs(expected);if(scale<1)scale=1;double error=(abs(R(actual)-expected)/scale).convert_to<double>();maxquantile=std::max(maxquantile,error);
   if(!b[0]||!b[1]||error>2e-8)fail("quantile value/witness");if(!(val(b[3])<=actual&&actual<=val(b[4])))fail("quantile bracket");if(std::abs(val(b[5])-std::log(c.p))>2e-11)fail("quantile target residual");if(b[6]<0||b[6]>4096)fail("quantile work");
  }else fail("unexpected output");
 }
 if(!done||pi!=pv.size()||(!legacy&&(si!=sv.size()||qi!=tv.size())))fail("incomplete");
 std::cout<<std::setprecision(17)<<"POWER_ROWS "<<pi<<" SIZE_ROWS "<<si<<" QUANTILE_ROWS "<<qi<<" MAX_POWER_RELATIVE_ERROR "<<maxpower<<" MAX_QUANTILE_SCALED_ERROR "<<maxquantile<<" ERRORS "<<errors<<"\n";return errors?1:0;
}
