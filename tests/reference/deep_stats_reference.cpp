// Independent C++23 binary128 references for the bounded checked stats layer.
// Compile on the central Slurm worker:
// g++ -std=c++23 -O2 tests/reference/deep_stats_reference.cpp -lquadmath -o deep_stats_reference
// Execute ./deep_stats_reference; status 0 requires all independent checks.
// No production sigmoid, QR, IRLS, gamma tail or Sounio numerical helper is reused.
#include <quadmath.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
using Q=__float128;
static Q q(double x){return static_cast<Q>(x);}
static void row(const char* name,Q value) {
 char out[160];quadmath_snprintf(out,sizeof(out),"%.35Qg",value);
 std::printf("%s\t%s\n",name,out);
}
static void require(bool yes,const char* what){if(!yes)throw std::runtime_error(what);}
static Q median6(std::array<Q,6> x) {
 // Independent insertion ordering; no production heapsort/MAD implementation.
 for(unsigned i=1;i<x.size();++i)for(unsigned j=i;j && x[j]<x[j-1];--j){
  Q t=x[j];x[j]=x[j-1];x[j-1]=t;
 }
 return (x[2]+x[3])/2;
}
static Q tukey_score(Q beta,const std::array<Q,6>& y,Q scale,Q tuning) {
 Q sum=0;
 for(Q v:y){
  Q u=(v-beta)/(scale*tuning);
  if(fabsq(u)<1){Q a=1-u*u;sum+=(v-beta)*a*a;}
 }
 return sum;
}
int main() {
 try {
  std::puts("quantity\tbinary128_reference");
  // Intercept Bernoulli MLE from sufficient counts; no IRLS.
  Q beta=logq(Q(4)/Q(2));
  Q dev=2*(Q(6)*logq(3)-Q(4)*logq(2));
  row("logistic_intercept_4of6",beta);
  row("logistic_intercept_4of6_deviance",dev);
  require(fabsq(beta-q(0.6931471805599453094))<Q(1e-15),"intercept fixture literal");
  // erfcq is independent of production's regularized incomplete gamma tail.
  Q p1=erfcq(1/sqrtq(2)),p2=erfcq(2/sqrtq(2));
  row("normal_two_tail_z1",p1);row("normal_two_tail_z2",p2);
  require(fabsq(p1-q(0.3173105078629141028))<Q(1e-15),"normal tail1 fixture");
  require(fabsq(p2-q(0.0455002638963584144))<Q(1e-15),"normal tail2 fixture");
  // A stable mathematical logistic pair without 1-p cancellation.
  Q tiny=expq(-40)/(1+expq(-40));
  row("logistic_probability_minus40",tiny);
  require(tiny>0,"nonzero logistic tail");
  // Independent symmetric slope: x=(-2,-1,0,0,1,2), y=(0,1,0,1,0,1).
  // beta0=0 by paired complementary outcomes. Solve scalar score by bisection.
  const std::array<int,6> xs{-2,-1,0,0,1,2},ys{0,1,0,1,0,1};
  Q lo=-10,hi=10;
  for(int it=0;it<180;++it){
   Q b=(lo+hi)/2,score=0;
   for(unsigned i=0;i<xs.size();++i)score+=Q(xs[i])*(Q(ys[i])-1/(1+expq(-b*Q(xs[i]))));
   if(score>0)lo=b;else hi=b;
  }
  row("symmetric_logistic_slope",(lo+hi)/2);
  // Intercept-only robust fit: scale is translation invariant. Declared
  // normal-consistency multiplier is a binary64 input, not estimated here.
  std::array<Q,6> y{-2,-1,0,1,2,20},absdev{};
  Q center=median6(y);
  for(unsigned i=0;i<y.size();++i)absdev[i]=fabsq(y[i]-center);
  Q scale=median6(absdev)*q(1.482602218505602),k=q(1.345),c=q(4.685);
  row("robust_scale",scale);
  // At the root the first five residuals are quadratic and final one saturated.
  Q huber=k*scale/5;
  for(unsigned i=0;i<5;++i)require(fabsq(y[i]-huber)<k*scale,"Huber branch condition");
  require(y[5]-huber>k*scale,"Huber outlier condition");
  row("huber_intercept",huber);
  Q score=0;
  for(unsigned i=0;i<y.size();++i){
   Q r=(y[i]-huber)/scale,w=fabsq(r)<=k?Q(1):k/fabsq(r);
   score+=r*w;
   char name[64];std::snprintf(name,sizeof(name),"huber_weight_%u",i);row(name,w);
  }
  require(fabsq(score)<Q(1e-29),"Huber independent score");
  // Tukey's local root in [-0.5,0.5], containing symmetric inliers and a
  // genuinely zero-weight outlier. This is not a global-optimum certificate.
  lo=-Q(0.5);hi=Q(0.5);
  require(tukey_score(lo,y,scale,c)>0 && tukey_score(hi,y,scale,c)<0,"Tukey bracket");
  for(int it=0;it<180;++it){Q mid=(lo+hi)/2;if(tukey_score(mid,y,scale,c)>0)lo=mid;else hi=mid;}
  Q tukey=(lo+hi)/2;
  require(fabsq(tukey)<Q(1e-29),"symmetric Tukey local root");
  row("tukey_local_intercept",tukey);
  for(unsigned i=0;i<y.size();++i){
   Q u=(y[i]-tukey)/(scale*c),w=0;
   if(fabsq(u)<1){Q v=1-u*u;w=v*v;}
   char name[64];std::snprintf(name,sizeof(name),"tukey_weight_%u",i);row(name,w);
   if(i==5)require(w==0,"Tukey outlier zero weight");
  }
  std::puts("DEEP_STATS_CPP_REFERENCE_PASS");
  return 0;
 } catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}
}
