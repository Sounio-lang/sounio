// C++23 independent information-theory oracle.
// Build: g++ -std=c++23 -O2 information_reference.cpp -lquadmath
// __float128 arithmetic is used for near-equal KL/JSD; inputs/outputs binary64.
// Entropies in nats; BSC capacity only is in bits. Independent generalized KL.
#include <quadmath.h>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
using Q=__float128;using A=std::array<double,8>;
struct R{int status=0;Q value=0;bool infinite=false;};
struct Case{std::string name;A p,q;int k;};
Q absq(Q a){return a<0?-a:a;}
Q generalized(Q a,Q b) {
 if(a==0)return b;
 Q u=(a-b)/b;
 if(absq(u)<(Q).125) {
  Q power=u*u,sum=0;
  for(int n=2;n<256;++n){Q term=power/(Q)(n*(n-1));if(n%2)term=-term;sum+=term;if(absq(term)<absq(sum)*(Q)1e-32)break;power*=u;}
  return b*sum;
 }
 return a*logq(a/b)-a+b;
}
int validate(const A&a,int k,Q& sum) {
 if(k<1||k>8)return 1;
 sum=0;
 for(int i=0;i<k;++i){if(!std::isfinite(a[i]))return 2;if(a[i]<0)return 3;sum+=(Q)a[i];}
 if(sum<=0||absq(sum-1)>(Q)1e-12)return 4;
 return 0;
}
R evaluate(const Case&c,int which) {
 R r;Q ps,qs;int a=validate(c.p,c.k,ps);if(a){r.status=a;return r;}
 if(which>=3){int b=validate(c.q,c.k,qs);if(b){r.status=b;return r;}}
 std::array<Q,8> p{},q{};
 int imax=0;for(int i=0;i<c.k;++i){p[i]=(Q)c.p[i]/ps;if(c.p[i]>c.p[imax])imax=i;if(which>=3)q[i]=(Q)c.q[i]/qs;}
 if(which<3) {
  Q tail=0,ratiosq=0;for(int i=0;i<c.k;++i)if(i!=imax){tail+=(Q)c.p[i];Q t=(Q)c.p[i]/(Q)c.p[imax];ratiosq+=t*t;}
  Q logscale=log1pq(tail/(Q)c.p[imax]);
  if(which==2)r.value=logscale;
  if(which==1)r.value=2*logscale-log1pq(ratiosq);
  if(which==0){r.value=logscale;for(int i=0;i<c.k;++i)if(i!=imax&&c.p[i]>0)r.value+=p[i]*logq((Q)c.p[imax]/(Q)c.p[i]);}
 } else if(which==3) {
  for(int i=0;i<c.k;++i) {
   if(p[i]>0&&q[i]==0){r.status=5;r.infinite=true;r.value=0;return r;}
   if(q[i]>0)r.value+=generalized(p[i],q[i]);
  }
 } else {
  for(int i=0;i<c.k;++i){Q m=(p[i]+q[i])/2;if(m>0)r.value+=(generalized(p[i],m)+generalized(q[i],m))/2;}
 }
 if(!std::isfinite((double)r.value)||r.value<0){r.status=6;r.value=0;}
 return r;
}
std::string lit(double d) {
 std::ostringstream o;o<<std::scientific<<std::setprecision(17);
 if(std::fpclassify(d)==FP_SUBNORMAL)o<<"("<<std::ldexp(d,52)<<" / 4503599627370496.0)";else o<<d;
 return o.str();
}
std::vector<Case> cases() {
 double tiny=std::numeric_limits<double>::denorm_min(),d=std::nextafter(.5,1.);
 return {
 {"deterministic",{1},{1},1},
 {"uniform2",{.5,.5},{.5,.5},2},
 {"uniform8",{.125,.125,.125,.125,.125,.125,.125,.125},{.125,.125,.125,.125,.125,.125,.125,.125},8},
 {"zeros",{.5,0,.5,0},{.25,0,.75,0},4},
 {"ordinary",{.2,.3,.5},{.3,.3,.4},3},
 {"ordinary_permuted",{.5,.2,.3},{.4,.3,.3},3},
 {"ordinary_reversed",{.3,.3,.4},{.2,.3,.5},3},
 {"near_equal_ulp",{d,1-d},{.5,.5},2},
 {"near_equal_ulp_reversed",{.5,.5},{d,1-d},2},
 {"near_equal_1e8",{.5+1e-8,.5-1e-8},{.5,.5},2},
 {"near_equal_quarter",{std::nextafter(.25,1.),(.25-(std::nextafter(.25,1.)-.25)),.5},{.25,.25,.5},3},
 {"disjoint",{1,0},{0,1},2},
 {"tiny_concentrated",{1,1e-300},{1,2e-300},2},
 {"minsub_concentrated",{1,tiny},{1,2*tiny},2},
 {"minsub_aggregate_zero_q",{1,tiny,tiny,tiny,tiny,tiny,tiny,tiny},{1,0,0,0,0,0,0,0},8},
 {"minsub_aggregate_double_q",{1,tiny,tiny,tiny,tiny,tiny,tiny,tiny},{1,2*tiny,2*tiny,2*tiny,2*tiny,2*tiny,2*tiny,2*tiny},8},
 {"minsub_aggregate_four_q",{1,tiny,tiny,tiny,tiny,tiny,tiny,tiny},{1,4*tiny,4*tiny,4*tiny,4*tiny,4*tiny,4*tiny,4*tiny},8},
 {"extreme_ratio",{1e-300,1},{.5,.5},2},
 {"extreme_ratio_reversed",{.5,.5},{1e-300,1},2},
 {"tolerance_mass",{.2,.3,.5+4e-13},{.3,.3,.4},3},
 {"tolerance_uniform",{.5+2e-13,.5+2e-13},{.5,.5},2},
 {"invalid_zero_mass",{0,0},{.5,.5},2},
 {"invalid_large_mass",{.6,.5},{.5,.5},2},
 {"invalid_negative",{-.1,1.1},{.5,.5},2},
 {"invalid_size_zero",{1},{1},0},
 {"invalid_size_nine",{1},{1},9},
 {"inactive_tail_ignored",{1,-3,4},{1,-3,4},1}
 };
}
R scalar(int kind,double x) {
 R r;if(!std::isfinite(x)){r.status=2;return r;}
 if((kind==0&&(x<0||x>1))||((kind==1||kind==2)&&x<0)||(kind==3&&x<=0)){r.status=3;return r;}
 Q a=x;
 if(kind==0){Q b=1-a;r.value=0;if(a>0)r.value+=generalized(a,(Q).5);if(b>0)r.value+=generalized(b,(Q).5);if(a==0||b==0)r.value=logq(2);r.value/=logq(2);}
 if(kind==1){r.value=log1pq(a)/2;if(x>0&&(double)r.value<=0){r.status=6;r.value=0;return r;}}
 if(kind==2)r.value=sqrtq(a/2);
 if(kind==3)r.value=1/a;
 if(!std::isfinite((double)r.value)){r.status=6;r.value=0;}
 return r;
}
void assertion(const R&r,std::string variable="r") {
 std::cout<<" assert("<<variable<<".status=="<<r.status<<")\n assert("<<(r.infinite?"":"!")<<variable<<".infinite)\n";
 if(r.value==0||(double)r.value==0)std::cout<<" assert("<<variable<<".value==0.0)\n";
 else std::cout<<" assert("<<variable<<".value>0.0)\n assert(close("<<variable<<".value,"<<lit((double)r.value)<<"))\n";
}
int main(int argc,char**argv) {
 bool f=argc>1&&std::string(argv[1])=="--fixture";auto all=cases();
 const char*fn[]={"information_shannon","information_collision","information_min","information_kl","information_js"};
 const char*sf[]={"bsc_capacity_bits","awgn_capacity_nats","pinsker_bound_nats","cramer_rao_bound"};
 std::cout<<std::scientific<<std::setprecision(17);
 if(f)std::cout<<"//@ run-pass\n//@ expect-stdout: MATH_INFORMATION_REFERENCE_PASS\n// Generated C++23 quadmath oracle; nats except BSC capacity(bits).\nuse math::probability::{information_shannon,information_collision,information_min,information_kl,information_js}\nuse math::channels::{bsc_capacity_bits,awgn_capacity_nats,pinsker_bound_nats,cramer_rao_bound}\nfn close(a:f64,b:f64)->bool {let d=if a>b{a-b}else{b-a} let v=if b<0.0{-b}else{b} return d<=v*1.0e-11+"<<lit(2*std::numeric_limits<double>::denorm_min())<<"}\n";
 for(size_t j=0;j<all.size();++j) {
  const auto&c=all[j];
  if(f){std::cout<<"// "<<c.name<<"\nfn case_"<<j<<"() with Mut,Div,Panic {\n var p:[f64;8]=[0.0;8]\n var q:[f64;8]=[0.0;8]\n";for(int i=0;i<8;++i){if(c.p[i])std::cout<<" p["<<i<<"]="<<lit(c.p[i])<<"\n";if(c.q[i])std::cout<<" q["<<i<<"]="<<lit(c.q[i])<<"\n";}}
  for(int m=0;m<5;++m){R r=evaluate(c,m);if(f){std::cout<<" let r="<<fn[m]<<"(&p,";if(m>=3)std::cout<<"&q,";std::cout<<c.k<<")\n";assertion(r);}
  else std::cout<<j<<"\t"<<c.name<<"\t"<<fn[m]<<"\t"<<r.status<<"\t"<<(double)r.value<<"\t"<<r.infinite<<"\n";}
  if(f)std::cout<<"}\n";
 }
 double tiny=std::numeric_limits<double>::denorm_min();
 std::vector<std::pair<int,double>> scalars;
 for(double x:{0.,tiny,1e-300,.1,std::nextafter(.5,0.),.5,std::nextafter(.5,1.),.9,1.,-.1,1.1})scalars.push_back({0,x});
 for(double x:{0.,tiny,2*tiny,1e-300,1e-16,.1,1.,1e300,-1.})scalars.push_back({1,x});
 for(double x:{0.,tiny,1e-300,1.,2.,8.,1e300,-1.})scalars.push_back({2,x});
 for(double x:{tiny,1e-300,1.,2.,1e300,std::numeric_limits<double>::max(),0.,-1.})scalars.push_back({3,x});
 if(f)std::cout<<"fn scalar_cases() with Mut,Div,Panic {\n";
 for(auto [m,x]:scalars){R r=scalar(m,x);if(f){std::cout<<" let r="<<sf[m]<<"("<<lit(x)<<")\n";assertion(r);}else std::cout<<"scalar\t"<<sf[m]<<"\t"<<x<<"\t"<<r.status<<"\t"<<(double)r.value<<"\n";}
 if(f){std::cout<<"}\nfn main() with IO,Mut,Div,Panic {\n";for(size_t j=0;j<all.size();++j)std::cout<<" case_"<<j<<"()\n";std::cout<<" scalar_cases()\n println(\"MATH_INFORMATION_REFERENCE_PASS\")\n}\n";}
 // Oracle invariances at high precision, no dependence on generated fixture.
 auto h=evaluate(all[4],0),hp=evaluate(all[5],0);
 if(absq(h.value-hp.value)>(Q)1e-30)return 2;
 auto js=evaluate(all[4],4),jr=evaluate(all[6],4);
 if(absq(js.value-jr.value)>(Q)1e-30)return 3;
 if(evaluate(all[7],3).value<=0)return 4;
}
