// C++23 concordance oracle: long-double raw centering, std::erfc normal tails,
// sorted reverse-paired summation; no Sounio gamma kernel or scaled algorithm.
// The documented Stephens calibration coefficients are intentionally shared.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <limits>
struct Ref {long double a,aa,p; int status;};
Ref calc(std::vector<double> x){
 const int n=x.size(); std::sort(x.begin(),x.end());
 long double mean=0;for(auto v:x)mean+=(long double)v;mean/=n;
 long double ss=0;for(auto v:x){auto d=(long double)v-mean;ss+=d*d;}
 auto sd=std::sqrt(ss/(n-1));
 long double s=0;
 for(int i=0;i<n;i++){
  auto z=((long double)x[i]-mean)/sd;
  auto zr=((long double)x[n-1-i]-mean)/sd;
  s+=(2*i+1)*(std::log(std::erfc(-z/std::sqrt(2.L))/2)+std::log(std::erfc(zr/std::sqrt(2.L))/2));
 }
 auto a=-n-s/n, aa=a*(1+.75L/n+2.25L/(n*(long double)n));
 long double p=0;int status=0;
 if(n<8 || aa>=10)status=5;
 else if(aa<.2L)p=1-std::exp(-13.436L+101.14L*aa-223.73L*aa*aa);
 else if(aa<.34L)p=1-std::exp(-8.318L+42.796L*aa-59.938L*aa*aa);
 else if(aa<.6L)p=std::exp(.9177L-4.279L*aa-1.38L*aa*aa);
 else p=std::exp(1.2937L-5.709L*aa+.0186L*aa*aa);
 return {a,aa,p,status};
}
int main(){
 std::printf("//@ run-pass\n//@ expect-stdout: STATS_AD_REFERENCE_PASS\nuse stats::anderson_darling::{anderson_darling}\nfn abs(x:f64)->f64 {if x<0.0{return -x} return x}\nfn main() with IO,Mut,Div,Panic {\n var x:[f64;256]=[0.0;256]\n");
 int cases=0;
 auto emit=[&](std::vector<double> x){
  auto r=calc(x);++cases;
  std::printf(" // C++23 case %d\n",cases);
  for(int i=0;i<(int)x.size();i++)std::printf(" x[%d]=%.17e\n",i,x[i]);
  std::printf(" let r%d=anderson_darling(&x,%zu)\n assert(r%d.status==%d)\n assert(abs(r%d.a2-%.17e)<5.0e-10)\n assert(abs(r%d.a2_star-%.17e)<5.0e-10)\n assert(abs(r%d.p-%.17e)<5.0e-9*%.17e+5.0e-324)\n",cases,x.size(),cases,r.status,cases,(double)r.a,cases,(double)r.aa,cases,(double)r.p,(double)r.p);
  std::fprintf(stderr,"case=%d n=%zu A2=%.18Lg Astar=%.18Lg p=%.18Lg status=%d\n",cases,x.size(),r.a,r.aa,r.p,r.status);
 };
 for(int n:{3,7,8,12,20,32,64,100,256}){
  std::vector<double> x;for(int i=0;i<n;i++)x.push_back(i);
  emit(x);
 }
 for(int n:{8,20,64,256}){
  std::vector<double>x(n,0.);x[n-1]=1.;emit(x);
 }
 for(int mode=0;mode<5;mode++){
  std::vector<double>x;for(int i=0;i<20;i++){double v=(i-9.5);if(mode==0)v=v*v*v;if(mode==1)v=1e300+v*1e285;if(mode==2)v*=1e-310;if(mode==3)v*=1e307;if(mode==4)v=(i%2==0?-1.:1.)*(i+1);x.push_back(v);}emit(x);
 }
 std::printf(" println(\"STATS_AD_REFERENCE_PASS\")\n 0\n}\n");
 std::fprintf(stderr,"CPP23_REFERENCE_CASES=%d\n",cases);
}
