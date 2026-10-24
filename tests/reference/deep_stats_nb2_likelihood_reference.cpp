// C++23 canonical independent full NB2 likelihood reference.
// Central Slurm:
// g++ -std=c++23 -O2 tests/reference/deep_stats_nb2_likelihood_reference.cpp -lquadmath -o deep_stats_nb2_likelihood_reference
// Direct binary128 lgammaq is checked where cancellation is tolerable.
// Authoritative extreme-r reference: direct lgamma formula at 400 decimal digits,
// NOT production's nine-term saturated-normalizer/deviance decomposition.
// Optional argument: actual native fixture stdout for 216-case concordance.
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/constants/constants.hpp>
#include <quadmath.h>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <stdexcept>
using Big=boost::multiprecision::number<boost::multiprecision::cpp_dec_float<400>>;
using Q=__float128;
static void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
static Big exact(double value){
 uint64_t bits=std::bit_cast<uint64_t>(value),mantissa=bits&((uint64_t(1)<<52)-1);
 unsigned raw=unsigned((bits>>52)&2047);
 int exponent=-1074;
 if(raw){mantissa|=uint64_t(1)<<52;exponent=int(raw)-1023-52;}
 Big x=Big(mantissa),power=1;
 for(int i=0;i<(exponent<0?-exponent:exponent);++i)power*=2;
 if(exponent<0)x/=power;else x*=power;
 if(bits>>63)x=-x;
 return x;
}
static Big direct(long long count,const Big& mean,const Big& alpha){
 Big y=count,yplus=y+1;
 if(alpha==0){
  Big value=-mean-boost::math::lgamma(yplus);
  if(count)value+=y*log(mean);
  return value;
 }
 Big r=1/alpha,sum=r+y;
 Big logarithm=boost::math::log1p(mean/r);
 Big result=boost::math::lgamma(sum)-boost::math::lgamma(r)-boost::math::lgamma(yplus);
 result-=r*logarithm;
 if(count)result+=y*(log(mean)-log(r)-logarithm);
 return result;
}
static Q direct_quad(long long count,double mean,double alpha){
 Q y=Q(count),mu=Q(mean),a=Q(alpha);
 if(alpha==0)return (count?y*logq(mu):Q(0))-mu-lgammaq(y+1);
 Q r=1/a,z=log1pq(mu/r);
 return lgammaq(r+y)-lgammaq(r)-lgammaq(y+1)-r*z+(count?y*(logq(mu)-logq(r)-z):Q(0));
}
static Big from_quad(Q x){
 char text[160];quadmath_snprintf(text,sizeof(text),"%.40Qg",x);return Big(text);
}
static Big ab(const Big& x){return x<0?-x:x;}
static void concord(const Big& actual,const Big& expected,const char* why){
 require(actual==actual,why);
 if(expected==0){require(actual==0,why);return;}
 const Big quantum=exact(0x0.0000000000001p-1022);
 require(ab(actual-expected)<=ab(expected)*Big("1e-11")+2*quantum,why);
}
static void stirling_checks(){
 const std::array<long long,9> numerator{1,-1,1,-1,1,-691,1,-3617,43867};
 const std::array<long long,9> denominator{12,360,1260,1680,1188,360360,156,122400,244188};
 for(unsigned z0:{8u,16u,1000000u}){
  Big z=z0,u=1/z,power=u,correction=0;
  for(unsigned i=0;i<9;++i){correction+=Big(numerator[i])/Big(denominator[i])*power;power*=u*u;}
  Big omitted=Big(174611)/Big(125400)*power;
  Big log2pi=log(2*boost::math::constants::pi<Big>());
  Big core=(z-Big("0.5"))*log(z)-z+log2pi/2;
  Big exact_correction=boost::math::lgamma(z)-core;
  Big error=correction-exact_correction;
  require(error>=0&&error<=omitted,"real Stirling next-term check");
 }
 // At z~1e300 direct lgamma subtraction cannot resolve Corr at 400 digits.
 // No numerical remainder certificate is claimed in that region.
}
struct Case {long long count;double mean,alpha;Big logp,saturated;};
int main(int argc,char** argv){
 try{
  stirling_checks();
  const std::array<long long,6> counts{0,1,7,8,1000000,9007199254740992LL};
  const std::array<double,9> alphas{0,1e308,1,0.125,1e-6,1e-300,2,0.12500000000000003,0.12499999999999999};
  std::vector<Case> cases;
  std::cout<<std::setprecision(40)<<"case\tcount\talpha\tmean\tlogp\tsaturated\n";
  for(long long count:counts)for(double alpha:alphas)for(unsigned mode=0;mode<4;++mode){
   double mean=1;
   if(mode==0&&count>0)mean=double(count);
   if(mode==1)mean=count>0?double(count)*(1.0+2.2204460492503131e-16):1e-308;
   if(mode==2)mean=1e308;
   if(mode==3)mean=1e-308;
   Big a=exact(alpha),mu=exact(mean);
   Big logp=direct(count,mu,a);
   Big saturated=count?direct(count,Big(count),a):Big(0);
   require(logp<=0&&saturated<=0,"valid log PMF sign");
   if(alpha==1){
    Big y=count;
    Big geometric=y*log(mu)-(y+1)*boost::math::log1p(mu);
    require(ab(logp-geometric)<Big("1e-70")*(1+ab(logp)),"analytic r1 geometric");
   }
   if(alpha==0&&count==0)require(logp==-mu,"analytic zero Poisson");
   // Quad is an independent diagnostic where cancellation retains enough digits.
   // Its large-r failure is not normalized into acceptance or used as a reference.
   if(alpha==0 || alpha>=1e-6){
    Q value=direct_quad(count,mean,alpha);
    concord(from_quad(value),logp,"moderate direct lgammaq concordance");
   }
   cases.push_back({count,mean,alpha,logp,saturated});
   std::cout<<cases.size()-1<<'\t'<<count<<'\t'<<alpha<<'\t'<<mean<<'\t'<<logp<<'\t'<<saturated<<'\n';
  }
  if(argc==2){
   std::ifstream input(argv[1]);require(bool(input),"native report open");
   std::string line;size_t id=cases.size(),completed=0;bool got_log=false,got_sat=false,got_status=false,got_normalizer=false;
   while(std::getline(input,line)){
    if(line.starts_with("likelihood case: ")){
     if(id<cases.size())require(got_log&&got_sat&&got_status&&got_normalizer,"previous native case");
     id=std::stoull(line.substr(17));require(id==completed&&id<cases.size(),"native case sequence");
     got_log=false;got_sat=false;got_status=false;got_normalizer=false;
    }else if(line.starts_with("likelihood status: ")){
     require(id<cases.size()&&!got_status,"native likelihood status field");
     require(std::stoi(line.substr(19))==0,"native likelihood status");got_status=true;
    }else if(line.starts_with("normalizer status: ")){
     require(id<cases.size()&&!got_normalizer,"native normalizer status field");
     require(std::stoi(line.substr(19))==0,"native normalizer status");got_normalizer=true;
    }else if(line.starts_with("log probability: ")){
     require(id<cases.size()&&!got_log,"native log before case");
     try {concord(Big(line.substr(17)),cases[id].logp,"native full log PMF");}
     catch(...) {std::cerr<<"FAILING_CASE "<<id<<" count="<<cases[id].count<<" alpha="<<cases[id].alpha<<" mean="<<cases[id].mean<<" native="<<line.substr(17)<<" expected="<<std::setprecision(50)<<cases[id].logp<<"\n";throw;}
     got_log=true;
    }else if(line.starts_with("saturated log probability: ")){
     require(id<cases.size()&&!got_sat,"native saturated before case");
     try {concord(Big(line.substr(27)),cases[id].saturated,"native saturated normalizer");}
     catch(...) {std::cerr<<"FAILING_SAT_CASE "<<id<<" native="<<line.substr(27)<<" expected="<<std::setprecision(50)<<cases[id].saturated<<"\n";throw;}
     got_sat=true;++completed;
    }
   }
   require(completed==cases.size()&&got_log&&got_sat&&got_status&&got_normalizer,"complete native likelihood report");
  }
  std::cout<<"DEEP_NB2_LIKELIHOOD_CPP_REFERENCE_PASS\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
