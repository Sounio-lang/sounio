// Independent checked KS fixture oracle (C++23).
// std::erfc long-double normal CDF; brute ECDF at pooled supports, no merge.
// P values evaluate the same DECLARED Stephens-corrected limiting null as
// Sounio; agreement is numerical, not finite-sample calibration certification.
// Usage: ks_reference --fixture > stats_ks_reference.sio ; default TSV report.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <vector>
using V=std::vector<double>;
struct Case { std::string name; V x,y; double mu=0,sd=1; bool two=false; };
struct Result { long double d,p,ne; bool ties; };
long double kolmogorov_sf(long double t) {
 if(t==0) return 1;
 // Independent branch boundary; theta positive CDF then complement.
 if(t<1) {
  long double sum=0;
  for(int k=1;k<10000;++k) {
   long double odd=2*k-1;
   long double term=std::exp(-std::numbers::pi_v<long double>*std::numbers::pi_v<long double>*odd*odd/(8*t*t));
   sum+=term;
   if(term==0 || term<std::numeric_limits<long double>::epsilon()*sum) break;
  }
  return 1-std::sqrt(2*std::numbers::pi_v<long double>)/t*sum;
 }
 long double sum=0;
 for(int j=1;j<10000;++j) {
  long double term=std::exp(-2*t*t*j*j);
  sum+=(j%2?term:-term);
  if(term==0 || term<std::numeric_limits<long double>::epsilon()*std::abs(sum)) break;
 }
 return 2*sum;
}
Result eval(const Case& c) {
 long double d=0; bool ties=false;
 V x=c.x,y=c.y;
 std::sort(x.begin(),x.end()); std::sort(y.begin(),y.end());
 for(size_t i=1;i<x.size();++i) ties|=x[i]==x[i-1];
 for(size_t i=1;i<y.size();++i) ties|=y[i]==y[i-1];
 if(c.two) {
  V support=x; support.insert(support.end(),y.begin(),y.end());
  for(double v:support) {
   long double a=std::count_if(x.begin(),x.end(),[&](double z){return z<=v;});
   long double b=std::count_if(y.begin(),y.end(),[&](double z){return z<=v;});
   d=std::max(d,std::abs(a/x.size()-b/y.size()));
  }
 } else {
  for(size_t i=0;i<x.size();++i) {
   long double z=((long double)x[i]-(long double)c.mu)/(long double)c.sd;
   long double f=std::erfc(-z/std::sqrt(2.L))/2;
   d=std::max({d,(i+1.L)/x.size()-f,f-(long double)i/x.size()});
  }
 }
 long double ne=c.two?(long double)x.size()*y.size()/(x.size()+y.size()):(long double)x.size();
 long double sn=std::sqrt(ne),lambda=(sn+.12L+.11L/sn)*d;
 return {d,kolmogorov_sf(lambda),ne,ties};
}
std::vector<Case> cases() {
 std::vector<Case> c;
 auto one=[&](std::string n,V x,double mu=0,double sd=1){c.push_back({n,x,{},mu,sd,false});};
 auto two=[&](std::string n,V x,V y){c.push_back({n,x,y,0,1,true});};
 one("one_center",{0}); one("one_triplet",{-1,0,1});
 one("one_shifted",{1,2,3,4,5,6},3.5,2);
 one("one_tied",{0,0,0,0,1,1});
 one("one_tails",{-40,-9,-3,0,3,9,40});
 one("one_overflow_subtract",{-1.7e308,0,1.7e308},-1.7e308,1.7e308);
 one("one_tiny_sigma",{-1e-300,0,1e-300},0,1e-300);
 one("one_scaled_large",{-1e300,0,1e300},0,1e300);
 two("two_tie_groups",{0,0,1},{0,1,1});
 two("two_tied_reverse",{0,1,1},{0,0,1});
 two("two_tied_unequal",{0,0,0,1,3},{0,2,2});
 two("two_identical",{0,0,1},{0,0,1});
 two("two_tail_completion",{0},{-3,-2,-1,0,1,2,3});
 two("two_extreme",{-1.7e308,-1e-300,0},{1e-300,1.7e308});
 V a(256),b(256),z(256),o(256,1);
 for(int i=0;i<256;++i){a[i]=2*i;b[i]=2*i+1;}
 two("two_256_interleaved_small_lambda",a,b);
 two("two_256_separated_tiny_tail",z,o);
 two("two_256_identical",a,a);
 V c255(255),c256(256); c255.back()=1;c256.back()=1;
 two("two_small_lambda_tied_255_256",c255,c256);
 one("one_256_extreme_tail",V(256,40));
 for(int i=0;i<256;++i)a[i]=((double)i-127.5)/64;
 one("one_256",a);
 two("two_256_v_1",a,{0});
 two("two_singletons_separated",{0},{1});
 return c;
}
void emit_array(std::string name,const V& v) {
 std::cout<<" var "<<name<<":[f64;256]=[0.0;256]\n";
 // Compress exact arithmetic grids (all covered grids use dyadic steps).
 if(v.size()>=16) {
  double step=v[1]-v[0]; bool grid=step!=0;
  for(size_t k=0;k<v.size();++k)grid&=v[k]==v[0]+k*step;
  if(grid){std::cout<<" var j"<<name<<"=0\n while j"<<name<<"<"<<v.size()<<" { "<<name<<"[j"<<name<<" as usize]="<<v[0]<<"+(j"<<name<<" as f64)*"<<step<<" j"<<name<<"=j"<<name<<"+1 }\n";return;}
 }
 // Compress equal-value runs; singleton assignments remain explicit binary64.
 size_t i=0;
 while(i<v.size()) {
  size_t e=i+1;while(e<v.size()&&v[e]==v[i])++e;
  if(e-i>=4) {
   std::cout<<" var i"<<name<<i<<"="<<i<<"\n while i"<<name<<i<<"<"<<e<<" { "<<name<<"[i"<<name<<i<<" as usize]="<<v[i]<<" i"<<name<<i<<"=i"<<name<<i<<"+1 }\n";
  } else for(size_t k=i;k<e;++k) if(v[k]!=0)std::cout<<" "<<name<<"["<<k<<"]="<<v[k]<<"\n";
  i=e;
 }
}
int main(int argc,char** argv) {
 bool fixture=argc>1&&std::string(argv[1])=="--fixture";
 auto all=cases();std::cout<<std::scientific<<std::setprecision(17);
 // Analytic / series identity checks independent from generated fixture.
 if(std::abs(kolmogorov_sf(0.L)-1)>1e-18L)return 2;
 long double at=kolmogorov_sf(.8L);
 long double alternate=0;
 for(int j=1;j<40;++j)alternate+=(j%2?2.L:-2.L)*std::exp(-2*.8L*.8L*j*j);
 if(std::abs(at-alternate)>1e-18L)return 3;
 if(fixture)std::cout<<"//@ run-pass\n//@ expect-stdout: STATS_KS_REFERENCE_PASS\n// Generated by tests/reference/ks_reference.cpp; asymptotic null approximation.\nuse stats::ks_test::{ks_one_normal,ks_two}\nfn close(a:f64,b:f64,at:f64,rt:f64)->bool { let d=if a>b{a-b}else{b-a} let v=if b<0.0{-b}else{b} return d<=at+rt*v }\n";
 for(size_t k=0;k<all.size();++k) {
  const auto& c=all[k];auto r=eval(c);
  if((k==8||k==9)&&std::abs(r.d-1.L/3)>1e-18L)return 4;
  if(k==11&&r.d!=0)return 5;
  if(k==17&&std::abs(r.d-1.L/(255*256))>1e-18L)return 6;
  if(!fixture) {std::cout<<k<<"\t"<<c.name<<"\t"<<(double)r.d<<"\t"<<(double)r.p<<"\t"<<(double)r.ne<<"\n";continue;}
  std::cout<<"// "<<c.name<<"\nfn case_"<<k<<"() with Mut,Div,Panic {\n";emit_array("x",c.x);
  if(c.two)emit_array("y",c.y);
  std::cout<<" let r="<<(c.two?"ks_two(&x,":"ks_one_normal(&x,")<<c.x.size();
  if(c.two)std::cout<<",&y,"<<c.y.size();
  else std::cout<<","<<c.mu<<","<<c.sd;
  std::cout<<")\n assert(r.status==0)\n assert(r.asymptotic)\n assert(r.n_x=="<<c.x.size()<<")\n assert(r.n_y=="<<(c.two?c.y.size():0)<<")\n";
  if(r.ties)std::cout<<" assert(r.has_ties)\n";
  std::cout<<" assert(close(r.d,"<<(double)r.d<<",2.0e-12,2.0e-12))\n assert(close(r.ne,"<<(double)r.ne<<",2.0e-12,2.0e-12))\n";
  // Relative p check, no absolute tolerance masking small probabilities.
  std::cout<<" assert(close(r.p,"<<(double)r.p<<",0.0,3.0e-10))\n}\n";
 }
 if(fixture) {std::cout<<"fn main() with IO,Mut,Div,Panic {\n";for(size_t k=0;k<all.size();++k)std::cout<<" case_"<<k<<"()\n";std::cout<<" println(\"STATS_KS_REFERENCE_PASS\")\n}\n";}
}
