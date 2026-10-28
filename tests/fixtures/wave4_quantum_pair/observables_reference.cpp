#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <string_view>
using F=long double;using C=std::complex<F>;using V=std::array<C,16>;using M=std::map<std::string,F>;
static_assert(std::numeric_limits<double>::is_iec559&&sizeof(double)==8);
static_assert(std::numeric_limits<F>::digits>=64);
static V init(unsigned basis=0){V s{};s[basis]=1;return s;}
static void one(V&s,unsigned q,C a,C b,C c,C d,unsigned n=2){
 for(unsigned i=0;i<(1u<<n);++i)if((i&(1u<<q))==0){auto j=i|(1u<<q);auto x=s[i],y=s[j];s[i]=a*x+b*y;s[j]=c*x+d*y;}
}
static void ry(V&s,unsigned q,F x,unsigned n=2){F c=std::cos(x/2),v=std::sin(x/2);one(s,q,c,-v,v,c,n);}
static void had(V&s,unsigned q,unsigned n=2){F r=1/std::sqrt(F(2));one(s,q,r,r,r,-r,n);}
static void rz(V&s,unsigned q,F x){one(s,q,std::exp(C(0,-x/2)),0,0,std::exp(C(0,x/2)));}
static void cx(V&s,unsigned c,unsigned t,unsigned n=2){
 for(unsigned i=0;i<(1u<<n);++i)if((i&(1u<<c))!=0&&(i&(1u<<t))==0)std::swap(s[i],s[i|(1u<<t)]);
}
static void cz(V&s){s[3]=-s[3];}
static F parity(const V&s,unsigned n=2){F z=0;for(unsigned i=0;i<(1u<<n);++i)z+=(std::popcount(i)%2?-1:1)*std::norm(s[i]);return z;}
static F bit0(const V&s,unsigned n=2){F z=0;for(unsigned i=1;i<(1u<<n);i+=2)z+=std::norm(s[i]);return z;}
static F corr(V s,F a,F b){ry(s,0,a);ry(s,1,b);return parity(s);}
static F chsh(V s,F a0,F a1,F b0,F b1){return corr(s,a0,b0)+corr(s,a0,b1)+corr(s,a1,b0)-corr(s,a1,b1);}
static V bell(){auto s=init();had(s,0);cx(s,0,1);return s;}
static V partial(F p){auto s=init();ry(s,0,p);cx(s,0,1);return s;}
static void joint(M&m,std::string tag,const V&s){
 F p0=std::norm(s[1])+std::norm(s[3]),p1=std::norm(s[2])+std::norm(s[3]),p2=std::norm(s[3]),cov=p2-p0*p1;
 m[tag+".mu0"]=p0;m[tag+".mu1"]=p1;m[tag+".second"]=p2;m[tag+".cov"]=cov;m[tag+".given_angle"]=cov;m[tag+".across_angles"]=0;
}
static void probs(M&m,std::string tag,const V&s){const char* labels[]={"p00","p01","p10","p11"};for(unsigned i=0;i<4;++i)m[tag+"."+labels[i]]=std::norm(s[i]);}
static M expected(){
 const F pi=static_cast<F>(3.141592653589793);M m;
 auto a=init();ry(a,0,F(0.37));ry(a,1,F(0.37));auto b=a;cx(a,0,1);cz(b);joint(m,"cn",a);joint(m,"cz",b);
 auto g=init();had(g,0,4);cx(g,0,1,4);cx(g,1,2,4);cx(g,2,3,4);m["gh.parity"]=parity(g,4);m["gh.marginal"]=bit0(g,4);
 auto e=bell();m["ch.zz_copy"]=(corr(init(0),0,0)+corr(init(3),0,0))/2;m["ch.xx_copy"]=(corr(init(0),pi/2,pi/2)+corr(init(3),pi/2,pi/2))/2;
 m["ch.zz_bell"]=corr(e,0,0);m["ch.xx_bell"]=corr(e,pi/2,pi/2);m["ch.chsh_bell"]=chsh(e,0,pi/2,pi/4,-pi/4);
 const char* labels[]={"chsh_00","chsh_01","chsh_10","chsh_11"};for(unsigned i=0;i<4;++i)m["ch."+std::string(labels[i])]=chsh(init(i),0,pi/2,pi/4,-pi/4);
 auto p=partial(F(0.7));m["pt.zz"]=corr(p,0,0);m["pt.xx"]=corr(p,pi/2,pi/2);m["pt.fixed"]=chsh(p,0,pi/2,pi/4,-pi/4);m["pt.tilted"]=chsh(p,0,pi/2,F(0.41),-F(0.41));
 for(unsigned rev=0;rev<2;++rev){auto s=init();ry(s,0,F(0.31));ry(s,1,F(1.13));cx(s,rev,1-rev);rz(s,0,-F(0.43));had(s,1);probs(m,rev?"rev":"fwd",s);}
 probs(m,"bell",e);return m;
}
static bool verify(std::istream&in,bool verbose){
 const auto exp=expected();M seen;std::string line;bool ended=false;
 while(std::getline(in,line)){
  if(line=="QUANTUM_OBSERVABLES_END"){if(ended)return false;ended=true;continue;}
  if(ended){return false;}
  auto eq=line.find('=');if(eq==std::string::npos)return false;
  auto key=line.substr(0,eq);if(!exp.contains(key)||seen.contains(key))return false;
  std::int64_t bits=0;auto p=line.data()+eq+1;auto r=std::from_chars(p,line.data()+line.size(),bits);if(r.ec!=std::errc{}||r.ptr!=line.data()+line.size())return false;
  double value=std::bit_cast<double>(bits);if(!std::isfinite(value))return false;seen[key]=static_cast<F>(value);
 }
 if(!ended||seen.size()!=exp.size()||exp.size()!=39)return false;
 F maxerr=0;for(const auto&[key,want]:exp){F err=std::abs(seen.at(key)-want);maxerr=std::max(maxerr,err);if(err>F(1e-10)){if(verbose)std::cerr<<key<<" got "<<std::setprecision(20)<<seen.at(key)<<" expected "<<want<<" abs "<<err<<"\n";return false;}}
 if(verbose)std::cout<<"QUANTUM_CPP_OBSERVABLES_OK rows="<<seen.size()<<" max_abs="<<std::setprecision(20)<<maxerr<<" tolerance=1e-10\n";
 return true;
}
static std::string fixture(M e){std::ostringstream out;for(const auto&[k,v]:e)out<<k<<'='<<std::bit_cast<std::int64_t>(static_cast<double>(v))<<'\n';out<<"QUANTUM_OBSERVABLES_END\n";return out.str();}
int main(int argc,char**argv){
 if(argc==2&&std::string_view(argv[1])=="--self-test"){
  auto e=expected();std::istringstream a(fixture(e));if(!verify(a,false))return 1;
  auto bad=e;bad.at("cn.mu0")+=F(0.0001);std::istringstream b(fixture(bad));if(verify(b,false))return 1;
  bad=e;std::swap(bad.at("cn.mu0"),bad.at("cn.mu1"));std::istringstream c(fixture(bad));if(verify(c,false))return 1;
  bad=e;bad.erase("gh.parity");std::istringstream d(fixture(bad));if(verify(d,false))return 1;
  bad=e;bad.at("gh.parity")=std::numeric_limits<F>::quiet_NaN();std::istringstream f(fixture(bad));if(verify(f,false))return 1;
  std::cout<<"QUANTUM_ORACLE_ADVERSARIAL_OK\n";return 0;
 }
 if(argc!=2){return 64;}
 std::ifstream in(argv[1]);if(!in)return 2;return verify(in,true)?0:1;
}
