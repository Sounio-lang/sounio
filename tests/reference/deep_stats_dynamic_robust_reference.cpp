// C++23/binary128 analytic one-hot-design HC3 and clustered CR0/CR1S oracle.
// No production QR, normal-equation solver or robust implementation reused.
// Slurm: g++ -std=c++23 -O2 tests/reference/deep_stats_dynamic_robust_reference.cpp -lquadmath -o dynamic_robust_reference
#include <quadmath.h>
#include <array>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
using Q=__float128;
constexpr int DRN=516,DRP=12,DRM=43;
static void drneed(bool b,const char* w){if(!b)throw std::runtime_error(w);}
struct DRReference{std::array<Q,DRP> beta{},se{};std::array<Q,DRP*DRP> cov{};Q factor=1;int groups=0;};
static DRReference drreference(int kind){
 DRReference out;std::array<Q,DRP> sums{};std::array<Q,DRN> y{};
 for(int i=0;i<DRN;i++){int g=i/DRM,t=i%DRM;y[i]=kind==7?Q(0):Q((g+1)*10+t%5-2)+Q(t%3)/4;sums[g]+=y[i];}
 auto index=[&](int g){return kind==4?DRP-1-g:g;};
 auto scale=[&](int g)->Q{return kind==3?(g%2==0?Q(double(1e75)):Q(double(1e-75))):Q(1);};
 for(int g=0;g<DRP;g++)out.beta[index(g)]=(sums[g]/DRM)/scale(g);
 bool hc3=kind==0||kind==3||kind==7;
 std::map<long long,std::array<Q,DRP>> influences;
 for(int i=0;i<DRN;i++){
  int g=i/DRM,t=i%DRM;Q residual=y[i]-sums[g]/DRM;
  Q influence=residual/(DRM*scale(g));if(hc3)influence/=1-Q(1)/DRM;
  long long label=hc3||kind==6?i:static_cast<long long>(t%7)*1000000007LL-3000000021LL;
  influences[label][index(g)]+=influence;
 }
 out.groups=hc3?0:int(influences.size());
 if(kind==2||kind==4)out.factor=(Q(out.groups)/(out.groups-1))*(Q(DRN-1)/(DRN-DRP));
 for(const auto& [label,v]:influences){(void)label;for(int j=0;j<DRP;j++)for(int k=0;k<DRP;k++)out.cov[j*DRP+k]+=out.factor*v[j]*v[k];}
 for(int j=0;j<DRP;j++){drneed(out.cov[j*DRP+j]>=0,"nonnegative analytic diagonal");out.se[j]=sqrtq(out.cov[j*DRP+j]);}
 return out;
}
static void drrow(const char* w,Q x){char b[160];quadmath_snprintf(b,sizeof b,"%.35Qg",x);std::cout<<w<<": "<<b<<'\n';}
static void drclose(Q got,Q expected,Q absolute,Q relative,const char* why){
 drneed(!isnanq(got)&&!isinfq(got),why);
 if(fabsq(got-expected)>absolute+relative*fabsq(expected)){drrow("actual",got);drrow("expected",expected);throw std::runtime_error(why);}
}
int main(int argc,char** argv){try{
 std::array<DRReference,9> refs;for(int kind=0;kind<9;kind++){refs[kind]=drreference(kind);std::cout<<"reference case: "<<kind<<'\n';drrow("beta0",refs[kind].beta[0]);drrow("se0",refs[kind].se[0]);drrow("cov01",refs[kind].cov[1]);}
 // Independent discriminators: grouping creates cross-coefficient covariance,
 // CR1S changes numerical covariance, HC3 differs from singleton HC0.
 drneed(refs[1].cov[1]!=0,"grouped meat offdiagonal discriminator");
 drneed(refs[2].cov[0]>refs[1].cov[0],"CR1S correction discriminator");
 drneed(refs[0].cov[0]>refs[6].cov[0],"HC3 leverage discriminator");
 if(argc==2){std::ifstream in(argv[1]);drneed(bool(in),"native report open");std::string line;int kind=-1,cases=0,beta=0,se=0,cov=0,metadata=0;bool complete=false;
  auto finished=[&](){drneed(beta==DRP&&se==DRP&&cov==DRP*DRP&&metadata==3,"complete native case");};
  const std::string prefix="dynamic robust ";
  while(std::getline(in,line)){
   if(line=="DEEP DYNAMIC ROBUST PASS")complete=true;
   if(!line.starts_with(prefix))continue;auto sep=line.find(": ");drneed(sep!=std::string::npos,"native key syntax");auto key=line.substr(prefix.size(),sep-prefix.size()),value=line.substr(sep+2);
   if(key.starts_with("trace"))continue;
   if(key=="case"){if(kind>=0)finished();kind=std::stoi(value);drneed(kind==cases&&kind<9,"native case order");cases++;beta=se=cov=metadata=0;continue;}
   drneed(kind>=0,"native case context");Q got=strtoflt128(value.c_str(),nullptr);const auto& r=refs[kind];
   if(key=="groups"){drneed(std::stoi(value)==r.groups,"native groups");metadata|=1;}
   else if(key=="factor"){drclose(got,r.factor,0,Q(2e-14),"native CR correction");metadata|=2;}
   else if(key=="beta"){drneed(beta<DRP,"beta count");drclose(got,r.beta[beta++],0,Q(2e-11),"native analytic coefficients");}
   else if(key=="se"){drneed(se<DRP,"SE count");drclose(got,r.se[se++],0,Q(2e-9),"native analytic robust SE");}
   else if(key=="covariance"){drneed(cov<DRP*DRP,"covariance count");Q expected=r.cov[cov],absolute=expected==0?r.se[cov/DRP]*r.se[cov%DRP]*Q(2e-11):Q(0);drclose(got,expected,absolute,Q(4e-9),"native analytic covariance");cov++;}
  }
  finished();drneed(cases==9&&complete,"full native completion marker");std::cout<<"ACTUAL DYNAMIC ROBUST CONCORDANCE PASS\n";
 }
 std::cout<<"DEEP DYNAMIC ROBUST CPP PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
