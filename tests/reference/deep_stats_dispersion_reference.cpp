// Independent C++23 binary128 reference: declared empirical log-dispersion model.
// Central Slurm: g++ -std=c++23 -O2 tests/reference/deep_stats_dispersion_reference.cpp -lquadmath -o deep_stats_dispersion_reference
// Run without args for canonical rows; optionally pass native fixture stdout
// for concordance with the eight-gene report. No production QR/logarithm reused.
#include <quadmath.h>
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <stdexcept>
using Q=__float128;
struct Point {Q t,j0,j1;};
static Point point(Q t0,Q t1,Q mean){
 Q a=expq(t0),b=expq(t1)/mean,sum=a+b;
 return {logq(sum),a/sum,b/sum};
}
struct Fit {
 Q a0,a1,prior,ms,corr;
 bool boundary;
 std::vector<Q> mapped,variance;
 std::vector<int> flags;
};
static void require(bool yes,const char* what){if(!yes)throw std::runtime_error(what);}
static Q objective(Q t0,Q t1,const std::vector<Q>& means,const std::vector<Q>& estimates){
 Q sum=0;
 for(size_t i=0;i<means.size();++i){Q r=logq(estimates[i])-point(t0,t1,means[i]).t;sum+=r*r;}
 return sum;
}
static Fit fit(const std::vector<Q>& means,const std::vector<Q>& estimates,
 const std::vector<Q>& variances,Q initial0,Q initial1,Q cutoff){
 const size_t n=means.size();
 require(n>2 && n==estimates.size() && n==variances.size(),"shape");
 Q t0=logq(initial0),t1=logq(initial1);
 bool converged=false;
 for(int iteration=0;iteration<600;++iteration){
  Q g00=0,g01=0,g11=0,s0=0,s1=0;
  for(size_t i=0;i<n;++i){
   auto z=point(t0,t1,means[i]);Q r=logq(estimates[i])-z.t;
   g00+=z.j0*z.j0;g01+=z.j0*z.j1;g11+=z.j1*z.j1;s0+=z.j0*r;s1+=z.j1*r;
  }
  Q det=g00*g11-g01*g01;require(det>0,"rank");
  // Deliberately independent high-precision direct 2x2 solve.
  Q d0=(g11*s0-g01*s1)/det,d1=(g00*s1-g01*s0)/det;
  if(fabsq(d0)<Q(1e-24) && fabsq(d1)<Q(1e-24)){converged=true;break;}
  Q old=objective(t0,t1,means,estimates),fraction=1;
  bool accepted=false;
  for(int k=0;k<100;++k){
   if(objective(t0+fraction*d0,t1+fraction*d1,means,estimates)<=old){
    t0+=fraction*d0;t1+=fraction*d1;accepted=true;break;
   }
   fraction/=2;
  }
  require(accepted,"reference line search");
  // At binary128 objective resolution a tiny accepted step is stationary.
  if(fabsq(fraction*d0)<Q(1e-25) && fabsq(fraction*d1)<Q(1e-25)){converged=true;break;}
 }
 require(converged,"reference convergence");
 Q g00=0,g01=0,g11=0,sse=0;
 for(size_t i=0;i<n;++i){
  auto z=point(t0,t1,means[i]);Q r=logq(estimates[i])-z.t;
  g00+=z.j0*z.j0;g01+=z.j0*z.j1;g11+=z.j1*z.j1;sse+=r*r;
 }
 Q det=g00*g11-g01*g01;require(det>0,"final rank");
 Q correction=0;
 for(size_t i=0;i<n;++i){
  auto z=point(t0,t1,means[i]);
  // Independent inverse-Gram leverage, not centered-column production identity.
  Q h=(z.j0*z.j0*g11-2*z.j0*z.j1*g01+z.j1*z.j1*g00)/det;
  require(h>=0 && h<=1,"leverage");
  correction+=(1-h)*variances[i];
 }
 Q raw=(sse-correction)/Q(n-2),prior=raw>0?raw:Q(0);
 Fit out{expq(t0),expq(t1),prior,sse/Q(n),correction/Q(n),raw<=0,{},{},{}};
 for(size_t i=0;i<n;++i){
  Q estimate=logq(estimates[i]),trend=point(t0,t1,means[i]).t;
  bool high=estimate-trend>cutoff*sqrtq(prior+variances[i]);
  out.flags.push_back(high?1:0);
  if(high){out.mapped.push_back(estimate);out.variance.push_back(variances[i]);}
  else if(prior==0){out.mapped.push_back(trend);out.variance.push_back(0);}
  else {
   out.mapped.push_back((prior*estimate+variances[i]*trend)/(prior+variances[i]));
   out.variance.push_back(prior*variances[i]/(prior+variances[i]));
  }
 }
 return out;
}
static void row(const char* name,Q v){
 char b[160];quadmath_snprintf(b,sizeof(b),"%.35Qg",v);std::printf("%s\t%s\n",name,b);
}
static void concord(Q actual,Q expected,const char* field){
 require(!isnanq(actual) && !isinfq(actual),field);
 require(fabsq(actual-expected)<=Q(1e-6)*fabsq(expected)+Q(1e-12),field);
}
int main(int argc,char** argv){
 try{
  std::vector<Q> means,estimates,variances;
  const std::array<double,8> observed{80,70,15,20,4,5,1,4};
  for(size_t i=0;i<observed.size();++i){means.push_back(Q(1ULL<<i));estimates.push_back(Q(observed[i]));variances.push_back(Q(double(0.01)));}
  auto result=fit(means,estimates,variances,Q(double(0.1)),100,1);
  row("a0",result.a0);row("a1",result.a1);row("prior_variance",result.prior);
  row("residual_mean_square",result.ms);row("sampling_correction",result.corr);
  int flags=0;
  for(size_t i=0;i<result.mapped.size();++i){
   std::string name="mapped_"+std::to_string(i);row(name.c_str(),result.mapped[i]);
   name="variance_"+std::to_string(i);row(name.c_str(),result.variance[i]);
   flags+=result.flags[i];
  }
  require(result.prior>0 && flags>0,"nontrivial shrink/outlier reference");
  std::printf("outliers\t%d\n",flags);
  means.clear();estimates.clear();variances.clear();
  for(int i=1;i<=513;++i){
   double mu=double(i),estimate=2.0+3.0/mu;
   means.push_back(Q(mu));estimates.push_back(Q(estimate));variances.push_back(Q(double(0.01)));
  }
  auto large=fit(means,estimates,variances,2,3,2);
  require(large.boundary && large.prior==0,"large boundary");
  concord(large.a0,2,"large a0");concord(large.a1,3,"large a1");
  row("large_a0",large.a0);row("large_a1",large.a1);
  if(argc==2){
   std::ifstream input(argv[1]);require(bool(input),"native report open");
   std::string line;size_t maps=0;bool got0=false,got1=false,gotprior=false;
   while(std::getline(input,line)){
    auto read=[&](const char* prefix)->Q{
     char* end=nullptr;Q v=strtoflt128(line.c_str()+std::string(prefix).size(),&end);
     require(end && (*end=='\0'||*end=='\r'),"native report number");return v;
    };
    if(line.starts_with("a0: ")){concord(read("a0: "),result.a0,"native a0");got0=true;}
    else if(line.starts_with("a1: ")){concord(read("a1: "),result.a1,"native a1");got1=true;}
    else if(line.starts_with("prior variance: ")){concord(read("prior variance: "),result.prior,"native prior");gotprior=true;}
    else if(line.starts_with("mapped log dispersion: ")){
     require(maps<result.mapped.size(),"extra native map");
     concord(read("mapped log dispersion: "),result.mapped[maps++],"native map");
    }
   }
   require(got0&&got1&&gotprior&&maps==8,"complete native report");
  }
  std::puts("DEEP_DISPERSION_CPP_REFERENCE_PASS");
  return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}
}
