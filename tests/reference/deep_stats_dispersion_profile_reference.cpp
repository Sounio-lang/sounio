// Independent C++23/binary128 grouped NB2 profile reference.
// Central Slurm only: g++ -std=c++23 -O2 tests/reference/deep_stats_dispersion_profile_reference.cpp -lquadmath -o profile_reference
// Optional actual Sounio stdout argument. Analytic group means, finite harmonic
// digamma differences and score bisection; no production optimizer/QR reused.
#include <quadmath.h>
#include <array>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdio>
using Q=__float128;
static constexpr int n=16,p=2;
static constexpr std::array<int,n> ys={0,0,1,2,5,8,13,19,1,1,2,3,7,12,18,28};
static void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static Q mean(int group){Q s=0;for(int i=group*8;i<(group+1)*8;++i)s+=ys[i];return s/8;}
static Q loglik(Q t){
 Q r=expq(-t),sum=0;
 for(int i=0;i<n;++i){
  Q mu=mean(i/8),y=ys[i];
  sum+=lgammaq(y+r)-lgammaq(r)-lgammaq(y+1)
      +y*logq(mu/r)-(y+r)*log1pq(mu/r);
 }
 return sum;
}
static Q logdet(Q t){Q a=expq(t),s=0;for(int g=0;g<p;++g){Q m=mean(g);s+=logq(8*m/(1+a*m));}return s;}
static Q objective(Q t,int method,bool prior){
 Q s=loglik(t);if(method==1)s-=logdet(t)/2;
 if(prior){Q d=t-Q(-1);s-=d*d/(2*Q(double(0.7)));}
 return s/n;
}
static Q score(Q t,int method,bool prior){
 Q r=expq(-t),a=expq(t),s=0;
 for(int i=0;i<n;++i){
  Q m=mean(i/8),y=ys[i],harmonic=0;
  for(int k=0;k<ys[i];++k)harmonic+=1/(r+k);
  s+=-r*(harmonic-log1pq(m/r)+(m-y)/(r+m));
 }
 if(method==1)for(int g=0;g<p;++g){Q m=mean(g);s+=Q(0.5)*a*m/(1+a*m);}
 if(prior)s-=(t+1)/Q(double(0.7));
 return s;
}
static Q root(int method,bool prior,Q lo,Q hi){
 Q left=score(lo,method,prior),right=score(hi,method,prior);
 need(left>0&&right<0,"independent score bracket");
 for(int i=0;i<160;++i){Q mid=(lo+hi)/2;if(mid==lo||mid==hi)break;
  if(score(mid,method,prior)>0)lo=mid;else hi=mid;}
 return (lo+hi)/2;
}
static void row(const char* label,Q x){char out[160];quadmath_snprintf(out,sizeof(out),"%.35Qg",x);std::printf("%s\t%s\n",label,out);}
static void close(Q actual,Q expected,Q relative,Q absolute,const char* what){
 need(!isnanq(actual)&&!isinfq(actual),what);
 if(fabsq(actual-expected)>absolute+relative*fabsq(expected)){
  row("actual",actual);row("expected",expected);throw std::runtime_error(what);}
}
int main(int argc,char** argv){try{
 std::array<Q,4> roots{},criteria{},variances{};
 for(int j=0;j<4;++j){
  int method=j%2;bool prior=j>=2;Q t=root(method,prior,-5,3);
  roots[j]=t;criteria[j]=objective(t,method,prior);
  Q h=Q(double(0.0001220703125));
  Q curvature=-(score(t+h,method,prior)-score(t-h,method,prior))/(2*h);
  need(curvature>0,"positive local total curvature");variances[j]=1/curvature;
  std::printf("reference profile: %d\n",j);row("log_alpha",t);row("alpha",expq(t));
  row("objective_mean",criteria[j]);row("full_log_likelihood_mean",loglik(t)/n);
  row("cox_reid_logdet",method==1?logdet(t):Q(0));row("local_variance",variances[j]);
  for(int g=0;g<2;++g)row("beta",logq(mean(g)));
  close(score(t,method,prior),0,0,Q(1e-25),"root score");
 }
 // Exact Poisson boundary dataset: every observation equals its group mean.
 // Every positive-alpha PMF is smaller than the Poisson value at this dataset;
 // this statement is an analytic property, not a full model-existence test.
 need(score(Q(2),0,false)<0&&score(Q(3),0,false)<0,"left-censored bracket signs");
 need(score(Q(-5),0,false)>0&&score(Q(-4),0,false)>0,"right-censored bracket signs");
 if(argc==2){
  std::ifstream in(argv[1]);need(bool(in),"native report open");std::string line;
  int current=-1;unsigned cases=0,beta=0,seen=0;bool native_complete=false;
  while(std::getline(in,line)){
   if(line=="DEEP DISPERSION PROFILE PASS")native_complete=true;
   if(line.starts_with("profile case: ")){
    if(current>=0)need(beta==2&&seen==31,"previous native row fields");
    current=std::stoi(line.substr(14));need(current==int(cases)&&current<4,"profile order");
    ++cases;beta=0;seen=0;
   }else if(line.starts_with("profile status: ")){need(current>=0,"status context");need(std::stoi(line.substr(16))==0,"native profile status");seen|=1;}
   else if(line.starts_with("profile log alpha: ")){need(current>=0,"alpha context");Q a=strtoflt128(line.c_str()+19,nullptr);close(a,roots[current],0,Q(3e-5),"native profile log alpha");seen|=2;}
   else if(line.starts_with("profile criterion: ")){need(current>=0,"criterion context");Q a=strtoflt128(line.c_str()+19,nullptr);close(a,criteria[current],Q(2e-10),Q(1e-12),"native profile criterion");seen|=4;}
   else if(line.starts_with("profile variance: ")){need(current>=0,"variance context");Q a=strtoflt128(line.c_str()+18,nullptr);close(a,variances[current],Q(0.003),0,"native local variance");seen|=8;}
   else if(line.starts_with("profile curvature available: ")){need(current>=0,"curvature context");need(std::stoi(line.substr(29))==1,"native curvature availability");seen|=16;}
   else if(line.starts_with("profile beta: ")){need(current>=0&&beta<2,"beta context");Q a=strtoflt128(line.c_str()+14,nullptr);close(a,logq(mean(int(beta))),Q(1e-9),Q(1e-10),"native grouped beta");++beta;}
  }
  need(cases==4&&beta==2&&seen==31,"complete four native profile cases");
  need(native_complete,"native full fixture completion marker");
  std::puts("actual native score-bisection concordance PASS");
 }
 std::puts("DEEP DISPERSION PROFILE CPP PASS");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
