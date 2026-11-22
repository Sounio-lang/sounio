// Independent C++23/binary128 conditional-Fisher group-model reference.
// Central Slurm: g++ -std=c++23 -O2 tests/reference/deep_stats_count_fisher_reference.cpp -lquadmath -o fisher_reference
// Optional native report. Closed-form independent group means and covariance;
// no production QR, IRLS or covariance implementation reused.
#include <quadmath.h>
#include <array>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstdio>
using Q=__float128;
static void need(bool x,const char* why){if(!x)throw std::runtime_error(why);}
static void row(const char* key,Q x){char b[150];quadmath_snprintf(b,sizeof(b),"%.35Qg",x);std::printf("%s\t%s\n",key,b);}
static void close(Q x,Q y,const char* why){
 need(!isnanq(x)&&!isinfq(x),why);
 if(y==0){need(fabsq(x)<=Q(1e-12),why);return;}
 if(fabsq(x-y)>Q(2e-8)*fabsq(y)){row("actual",x);row("expected",y);throw std::runtime_error(why);}
}
struct Ref {unsigned p;std::array<Q,12> beta{},se{},contrast{};std::array<Q,144> covariance{};Q estimate=0,variance=0;};
int main(int argc,char** argv){try{
 Q quantum=ldexpq(Q(1),-1074);
 need(Q(double(quantum/Q(2)))==0,"binary64 half-quantum underflow witness");
 need(quantum/Q(2)+quantum/Q(2)==quantum,"true contrast remains one quantum");
 row("underflow_witness_true_estimate",quantum);
 std::array<Ref,6> refs{};
 for(unsigned model=0;model<6;++model){
  auto& ref=refs[model];Q alpha=model%2?Q(0.25):Q(0);
  if(model<4){
   ref.p=2;Q m0=6,m1=9,v0=(1+alpha*m0)/(8*m0),v1=(1+alpha*m1)/(8*m1);
   Q scale=model==2?Q(double(1e150)):model==3?Q(double(1e-150)):Q(1);
   ref.beta[0]=logq(m0);ref.beta[1]=(logq(m1)-logq(m0))/scale;
   ref.covariance[0]=v0;ref.covariance[1]=-v0/scale;
   ref.covariance[2]=ref.covariance[1];ref.covariance[3]=(v0+v1)/scale/scale;
   ref.se[0]=sqrtq(v0);ref.se[1]=sqrtq(v0+v1)/fabsq(scale);
   ref.contrast[1]=scale;ref.estimate=logq(m1)-logq(m0);ref.variance=v0+v1;
  }else{
   ref.p=12;std::array<unsigned,12> sums{},sizes{};
   for(unsigned i=0;i<512;++i){unsigned group=i%12;int d=(i/12)%2==0?1:-1;sums[group]+=unsigned(int(group)+2+d);++sizes[group];}
   for(unsigned g=0;g<12;++g){Q mu=Q(sums[g])/sizes[g];ref.beta[g]=logq(mu);
    ref.covariance[g*12+g]=(1+alpha*mu)/(Q(sizes[g])*mu);
    ref.se[g]=sqrtq(ref.covariance[g*12+g]);}
   ref.contrast[0]=-1;ref.contrast[11]=1;
   ref.estimate=ref.beta[11]-ref.beta[0];
   ref.variance=ref.covariance[0]+ref.covariance[143];
  }
  std::printf("reference fisher: %u\n",model);
  for(unsigned j=0;j<ref.p;++j){row("beta",ref.beta[j]);row("standard_error",ref.se[j]);}
  for(unsigned j=0;j<ref.p*ref.p;++j)row("covariance",ref.covariance[j]);
  row("contrast_estimate",ref.estimate);row("contrast_standard_error",sqrtq(ref.variance));row("contrast_variance",ref.variance);
 }
 if(argc==2){
  std::ifstream input(argv[1]);need(bool(input),"native report open");std::string line;
  int model=-1;unsigned cases=0,b=0,s=0,c=0,seen=0;bool complete=false;
  while(std::getline(input,line)){
   if(line=="DEEP COUNT FISHER PASS")complete=true;
   if(line.starts_with("fisher case: ")){
    if(model>=0)need(b==refs[model].p&&s==refs[model].p&&c==refs[model].p*refs[model].p&&seen==15,"previous fisher complete");
    model=std::stoi(line.substr(13));need(model==int(cases)&&model<6,"native model order");++cases;b=s=c=seen=0;
   }else if(line.starts_with("fisher status: ")){need(model>=0,"status context");need(std::stoi(line.substr(15))==0,"native covariance status");seen|=1;}
   else if(line.starts_with("fisher beta: ")){need(model>=0&&b<refs[model].p,"beta row");close(strtoflt128(line.c_str()+13,nullptr),refs[model].beta[b++],"native conditional beta");}
   else if(line.starts_with("fisher se: ")){need(model>=0&&s<refs[model].p,"se row");close(strtoflt128(line.c_str()+11,nullptr),refs[model].se[s++],"native conditional SE");}
   else if(line.starts_with("fisher covariance: ")){need(model>=0&&c<refs[model].p*refs[model].p,"cov row");
    Q expected=refs[model].covariance[c];Q actual=strtoflt128(line.c_str()+19,nullptr);
    if(expected==0){unsigned i=c/refs[model].p,j=c%refs[model].p;need(fabsq(actual)<=Q(1e-12)*refs[model].se[i]*refs[model].se[j],"native zero offdiagonal scale");}
    else close(actual,expected,"native covariance");++c;}
   else if(line.starts_with("fisher contrast estimate: ")){need(model>=0,"contrast context");close(strtoflt128(line.c_str()+26,nullptr),refs[model].estimate,"native contrast estimate");seen|=2;}
   else if(line.starts_with("fisher contrast se: ")){need(model>=0,"contrast se context");close(strtoflt128(line.c_str()+20,nullptr),sqrtq(refs[model].variance),"native contrast SE");seen|=4;}
   else if(line.starts_with("fisher contrast variance: ")){need(model>=0,"variance context");close(strtoflt128(line.c_str()+26,nullptr),refs[model].variance,"native contrast variance");seen|=8;}
  }
  need(cases==6&&b==refs[5].p&&s==refs[5].p&&c==refs[5].p*refs[5].p&&seen==15&&complete,"complete native Fisher fixture");
  std::puts("actual native closed-form Fisher concordance PASS");
 }
 std::puts("DEEP COUNT FISHER CPP PASS");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
