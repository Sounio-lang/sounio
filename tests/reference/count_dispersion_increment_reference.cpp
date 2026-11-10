// Independent C++23 audit of a local alpha-increment identity.
// Compile/run only in Slurm: g++ -std=c++23 -O2 this.cpp -o alpha_increment_reference
// Comparator: independent FULL finite-product PMFs at400decimal digits;
// also DIRECT lgamma on ordinary-alpha cases. Tiny-alpha gamma cancellation
// is deliberately not used as an expected-zero oracle. A Big1000 direct-gamma
// exploratory job is preserved separately; this grid uses no increment identity
// in its authoritative full-PMF comparator.
// This checks a finite case grid/algebra, not a universal floating-point bound.
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <string>
using B=boost::multiprecision::number<boost::multiprecision::cpp_dec_float<400>>;
static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
static B exact(double x){uint64_t u=std::bit_cast<uint64_t>(x),m=u&((uint64_t(1)<<52)-1);int r=int((u>>52)&2047),e=-1074;if(r){m|=uint64_t(1)<<52;e=r-1075;}B v=m;B p=pow(B(2),e<0?-e:e);if(e<0)v/=p;else v*=p;if(u>>63)v=-v;return v;}
static B gamma_pmf(int y,const B& mu,const B& a){B yy=y;if(a==0)return (y?B(yy*log(mu)):B(0))-mu-boost::math::lgamma(B(yy+1));B r=1/a,z=boost::math::log1p(B(mu/r));B ans=boost::math::lgamma(B(r+yy))-boost::math::lgamma(r)-boost::math::lgamma(B(yy+1))-r*z;if(y)ans+=yy*(log(mu)-log(r)-z);return ans;}
// Independent full PMF via the integer-count finite product, not a difference
// identity. This avoids a huge-r lgamma subtraction in the near-Poisson grid.
static B pmf(int y,const B& mu,const B& a){
 if(a==0)return gamma_pmf(y,mu,a);
 B total=0;for(int k=1;k<y;k++)total+=boost::math::log1p(B(k*a));
 B yy=y,z=boost::math::log1p(B(a*mu));
 B ans=total-boost::math::lgamma(B(yy+1))-(yy+1/a)*z;
 if(y)ans+=yy*log(mu);
 return ans;
}

static B log1p(const B& x){return boost::math::log1p(x);}
static B identity(int y,const B& mu,const B& a0,const B& a1){
 if(a0==a1)return B(0);
 if(a1==0)return -identity(y,mu,a1,a0);
 B sum=0;
 if(a0==0){for(int k=1;k<y;k++)sum+=log1p(B(k*a1));B z=a1*mu;return sum-y*log1p(z)-(log1p(z)-z)/a1;}
 B da=a1-a0,u=da/a0,z=a0*mu,v=z*u/(1+z);
 for(int k=1;k<y;k++)sum+=log1p(B(da/(a0+B(1)/k)));
 B phi=(1+z)*log1p(z)-z;
 B df=((log1p(v)-v)-u*phi/(1+z))/a1;
 return sum-y*log1p(v)-df;
}
// At y1,mu2 the first dispersion derivative vanishes. Its tiny-alpha changes
// need >600digits, even with the full finite-product PMF. Promote precisely
// those retained grid cases, with no lgamma at huge r or Bernoulli startup.
using E=boost::multiprecision::number<boost::multiprecision::cpp_dec_float<1000>>;
static E exact_extra(double x){uint64_t u=std::bit_cast<uint64_t>(x),m=u&((uint64_t(1)<<52)-1);int r=int((u>>52)&2047),e=-1074;if(r){m|=uint64_t(1)<<52;e=r-1075;}E v=m,p=pow(E(2),e<0?-e:e);if(e<0)v/=p;else v*=p;if(u>>63)v=-v;return v;}
static E elog(const E& x){return boost::math::log1p(x);}
static E full_y1(const E& mu,const E& a){if(a==0)return E(log(mu)-mu);return E(log(mu)-(1+1/a)*elog(E(a*mu)));}
static E identity_y1(const E& mu,const E& a0,const E& a1){if(a0==a1)return E(0);if(a1==0)return -identity_y1(mu,a1,a0);if(a0==0){E z=a1*mu;return E(-elog(z)-(elog(z)-z)/a1);}E u=(a1-a0)/a0,z=a0*mu,v=z*u/(1+z),phi=(1+z)*elog(z)-z;return E(-elog(v)-((elog(v)-v)-u*phi/(1+z))/a1);}
static bool promote(int y,double mu,double a0,double a1){return y==1&&mu==2.&&a0<1e-200&&a1<1e-200;}
static B promoted_truth(double mu,double a0,double a1){E m=exact_extra(mu),a=exact_extra(a0),b=exact_extra(a1);E result=full_y1(m,b)-full_y1(m,a);return B(result.str(410,std::ios_base::scientific));}
static B promoted_identity(double mu,double a0,double a1){E result=identity_y1(exact_extra(mu),exact_extra(a0),exact_extra(a1));return B(result.str(410,std::ios_base::scientific));}
static B full_difference(int y,double mu,double a0,double a1){if(promote(y,mu,a0,a1))return promoted_truth(mu,a0,a1);return B(pmf(y,exact(mu),exact(a1))-pmf(y,exact(mu),exact(a0)));}

struct C{int y;double mu,a0,a1;const char* label;};
int main(int argc,char** argv){try{
 bool mutant=argc==2&&std::string(argv[1])=="--mutant-sign";need(argc==1||mutant,"arguments");
 std::vector<C> cases;
 for(int y:{0,1,7,8,64,4096}){
  cases.push_back({y,3.25,.25,std::nextafter(.25,1.),"adjacent"});
  cases.push_back({y,3.25,.25,.28125,"local"});
  cases.push_back({y,3.25,.28125,.25,"reverse"});
  cases.push_back({y,3.25,0.,1e-20,"poisson-near"});
 }
 for(int y:{0,1,8,4096}){
  cases.push_back({y,2.,1e-300,std::nextafter(1e-300,1.),"tiny-adjacent"});
  cases.push_back({y,2.,0.,1e-300,"poisson-extreme"});
  cases.push_back({y,2.,1e-300,0.,"reverse-poisson-extreme"});
 }
 cases.push_back({1,1e308,1e-300,1.125e-300,"large-mean"});
 cases.push_back({8,1e-300,.25,.28125,"tiny-mean"});
 unsigned index=0;
 for(const C& c:cases){B mu=exact(c.mu),a0=exact(c.a0),a1=exact(c.a1);B truth=full_difference(c.y,c.mu,c.a0,c.a1);B got=promote(c.y,c.mu,c.a0,c.a1)?promoted_identity(c.mu,c.a0,c.a1):identity(c.y,mu,a0,a1);if(mutant)got=-got;if(c.a0>=1e-10&&c.a1>=1e-10){B gamma_truth=gamma_pmf(c.y,mu,a1)-gamma_pmf(c.y,mu,a0);need(abs(gamma_truth-truth)<=abs(truth)*B("1e-70")+B("1e-380"),"gamma vs full finite-product PMF");}B error=abs(got-truth);B tolerance=abs(truth)*B("1e-70");
  std::cout<<index++<<'\t'<<c.label<<'\t'<<c.y<<'\t'<<std::setprecision(60)<<truth<<'\t'<<got<<'\t'<<error<<'\n';
  need(error<=tolerance,"FULL PMF vs alpha increment identity");
  B backward=promote(c.y,c.mu,c.a0,c.a1)?promoted_identity(c.mu,c.a1,c.a0):identity(c.y,mu,a1,a0);B forward=promote(c.y,c.mu,c.a0,c.a1)?promoted_identity(c.mu,c.a0,c.a1):identity(c.y,mu,a0,a1);need(abs(backward+forward)<=tolerance,"identity orientation");
 }
 need(identity(8,B(3),B(".25"),B(".25"))==0,"equal alpha exact zero");
 std::cout<<"ALPHA_INCREMENT_DIRECT_GAMMA_COMPLETE\t"<<cases.size()<<"\n";
 std::cout<<"LIMITS\tfinite grid only; unsupported overflow regions require unavailable; no binary64 relative/sign certification near stationary scores\n";
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
