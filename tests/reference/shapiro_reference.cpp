#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>
template<size_t N> long double poly(const std::array<long double,N>&c,long double x){
 long double sum=0,power=1;for(auto v:c){sum+=v*power;power*=x;}return sum;
}
long double quantile(long double p){
 long double lo=-8,hi=0;
 for(int i=0;i<100;i++){auto mid=(lo+hi)/2;auto c=std::erfc(-mid/std::sqrt(2.L))/2;if(c<p)lo=mid;else hi=mid;}
 return (lo+hi)/2;
}
std::vector<long double> weights(int n){
 std::vector<long double>a(n,0);int h=n/2;
 if(n==3){a[0]=-std::sqrt(.5L);a[2]=-a[0];return a;}
 std::vector<long double>q;for(int i=1;i<=h;i++)q.push_back(quantile((i-.375L)/(n+.25L)));
 long double norm=2*std::inner_product(q.begin(),q.end(),q.begin(),0.L);
 auto t=1/std::sqrt((long double)n);
 auto first=poly(std::array{0.L,.221157L,-.147981L,-2.07119L,4.434685L,-2.706056L},t)-q[0]/std::sqrt(norm);
 auto removed=2*q[0]*q[0],remaining=1-2*first*first;int begin=1;
 long double second=0;
 if(n>5){second=poly(std::array{0.L,.042981L,-.293762L,-1.752461L,5.682633L,-3.582633L},t)-q[1]/std::sqrt(norm);removed+=2*q[1]*q[1];remaining-=2*second*second;begin=2;}
 auto factor=std::sqrt((norm-removed)/remaining);
 std::vector<long double>b(h);b[0]=first;if(n>5)b[1]=second;for(int i=begin;i<h;i++)b[i]=-q[i]/factor;
 for(int i=0;i<h;i++){a[i]=-b[i];a[n-1-i]=b[i];}return a;
}
struct Result{long double w,w1,p;};
Result reference(std::vector<double>raw){
 std::sort(raw.begin(),raw.end());
 auto a=weights((int)raw.size());std::vector<long double>x(raw.begin(),raw.end());
 auto mean=std::accumulate(x.begin(),x.end(),0.L)/x.size();
 for(auto&v:x)v-=mean;
 auto xx=std::inner_product(x.begin(),x.end(),x.begin(),0.L);
 auto aa=std::inner_product(a.begin(),a.end(),a.begin(),0.L);
 auto dot=std::inner_product(a.begin(),a.end(),x.begin(),0.L);
 auto w=dot*dot/(aa*xx);
 long double sse=0;for(size_t i=0;i<x.size();i++){auto e=x[i]-dot/aa*a[i];sse+=e*e;}
 auto w1=sse/xx;
 if(std::abs((1-w1)-w)>2e-17L){std::cerr<<"independent W identity failed\n";std::exit(2);}
 int n=(int)x.size();long double p;
 if(n==3){p=1-6/std::acos(-1.L)*std::atan(std::sqrt(w1/(1-w1)));p=std::clamp(p,0.L,1.L);}
 else if(w1==0)p=1;
 else{
  auto y=std::log(w1);long double m,s;
  if(n<=11){auto gamma=-2.273L+.459L*n;y=-std::log(gamma-y);m=poly(std::array{.544L,-.39978L,.025054L,-.0006714L},(long double)n);s=std::exp(poly(std::array{1.3822L,-.77857L,.062767L,-.0020322L},(long double)n));}
  else{auto z=std::log((long double)n);m=poly(std::array{-1.5861L,-.31082L,-.083751L,.0038915L},z);s=std::exp(poly(std::array{-.4803L,-.082676L,.0030302L},z));}
  p=std::erfc((y-m)/(s*std::sqrt(2.L)))/2;
 }
 return {1-w1,w1,p};
}
int main(){
 static_assert(std::numeric_limits<long double>::max_exponent>=16384);
 std::vector<std::vector<double>>cases{{0,1,2},{0,0,1},{0,1,4},{1,2,3,4},{-2,-1,0,1,2},{2,4,4,4,5,5,7,9}};
 auto base=cases.back();
 for(double scale:{1e-200,1e200,-1e-200,-1e200}){auto z=base;for(auto&v:z)v*=scale;cases.push_back(z);}
 auto offset=base;for(auto&v:offset)v+=1e15;cases.push_back(offset);
 cases.push_back({-1.7e308,1.7e308,0,1e308,-1e308,5e307,-7e307});
 cases.push_back({1e-320,2e-320,-3e-320,5e-320,8e-320,-2e-320});
 for(int n:{6,10,11,12,20,50,99,100}){
  std::vector<double>x;for(int i=0;i<n;i++)x.push_back((i%7)*2.0+i*.1+(i%3)*.125);
  cases.push_back(x);
 }
 std::vector<double>outlier(100,0);outlier[99]=1;cases.push_back(outlier);
 std::vector<double>uniform;for(int i=0;i<100;i++)uniform.push_back(i);cases.push_back(uniform);
 auto a=weights(20);std::vector<double>nearperfect(a.begin(),a.end());cases.push_back(nearperfect);
 std::cout<<std::scientific<<std::setprecision(17);
 std::cout<<"//@ run-pass\n//@ expect-stdout: STATS_SHAPIRO_REFERENCE_PASS\nuse stats::shapiro_wilk::{shapiro_wilk}\nuse stats::regression::assessments::{assess_simple_linear}\n";
 std::cout<<"fn close(a:f64,b:f64,at:f64,rt:f64)->bool {let d=if a>b{a-b}else{b-a} let v=if b<0.0{-b}else{b} return d<=at+rt*v}\nfn main() with IO,Mut,Div,Panic {\n";
 for(size_t c=0;c<cases.size();c++){
 auto r=reference(cases[c]);
 std::cerr<<std::setprecision(21)<<"case="<<c<<" n="<<cases[c].size()<<" W="<<r.w<<" 1-W="<<r.w1<<" p="<<r.p<<"\n";
 std::cout<<" var x"<<c<<":[f64;100]=[0.0;100]\n";
 for(size_t j=0;j<cases[c].size();j++)if(cases[c][j]!=0)std::cout<<" x"<<c<<"["<<j<<"]="<<cases[c][j]<<"\n";
 std::cout<<" let r"<<c<<"=shapiro_wilk(&x"<<c<<","<<cases[c].size()<<")\n assert(r"<<c<<".status==0)\n";
 std::cout<<" assert(close(r"<<c<<".w,"<<(double)r.w<<",3.0e-12,3.0e-12))\n";
 if(c+1==cases.size())std::cout<<" assert(r"<<c<<".one_minus_w>0.0 && r"<<c<<".one_minus_w<1.0e-24)\n";
 else std::cout<<" assert(close(r"<<c<<".one_minus_w,"<<(double)r.w1<<",3.0e-12,3.0e-10))\n";
 std::cout<<" assert(close(r"<<c<<".p,"<<(double)r.p<<","<<(r.p==0?3e-14:(r.p<1e-8L?1e-300:3e-11))<<",3.0e-8))\n";
 }
 // Existing model fixture; independent long-double fit before residual analysis.
 std::vector<double>x,y,e;for(int i=0;i<20;i++){x.push_back(i);y.push_back(2+3*i+(i%2?-1:1)*(.2+.1*i));}
 long double mx=9.5,my=std::accumulate(y.begin(),y.end(),0.L)/20,xx=0,xy=0;
 for(int i=0;i<20;i++){xx+=(x[i]-mx)*(x[i]-mx);xy+=(x[i]-mx)*(y[i]-my);}
 for(int i=0;i<20;i++)e.push_back((double)((long double)y[i]-my-xy/xx*(x[i]-mx)));
 auto r=reference(e);std::cerr<<"model n=20 W="<<r.w<<" p="<<r.p<<"\n";
 std::cout<<" var mx:[f64;100]=[0.0;100]\n var my:[f64;100]=[0.0;100]\n";
 for(int i=0;i<20;i++)std::cout<<" mx["<<i<<"]="<<x[i]<<"\n my["<<i<<"]="<<y[i]<<"\n";
 std::cout<<" let m=assess_simple_linear(&mx,&my,20)\n assert(m.shapiro.status==0)\n assert(close(m.shapiro.w,"<<(double)r.w<<",3.0e-12,3.0e-12))\n assert(close(m.shapiro.p,"<<(double)r.p<<",3.0e-11,3.0e-8))\n";
 std::cout<<" println(\"STATS_SHAPIRO_REFERENCE_PASS\")\n 0\n}\n";
 std::cerr<<"datasets="<<cases.size()<<" plus model=1; shared Royston coefficients, independent normal quantiles/precision/W identity\n";
}
