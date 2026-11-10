// C++23 independent checked correlation oracle.
// Build: g++ -std=c++23 -O2 correlation_reference.cpp -lquadmath
// Raw binary64 inputs are converted exactly to __float128. Raw centered
// moments and explicit least-squares residuals; no production scaling logic.
// Spearman ranks use independent O(n^2) less/equal counts, with averaged ties.
#include <quadmath.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
using Q=__float128;using V=std::vector<double>;
struct Case{std::string name;V x,y;int n;bool poison=false;};
struct R{int status=0;Q coefficient=0,residual=0;};
R eval(const Case&c,bool rank) {
 R r;if(c.n<2||c.n>256){r.status=1;return r;}
 for(int i=0;i<c.n;++i)if(!std::isfinite(c.x[i])||!std::isfinite(c.y[i])){r.status=2;return r;}
 std::vector<Q>x(c.n),y(c.n);
 for(int i=0;i<c.n;++i) {
  if(rank) {
   int xl=0,xe=0,yl=0,ye=0;
   for(int j=0;j<c.n;++j){xl+=c.x[j]<c.x[i];xe+=c.x[j]==c.x[i];yl+=c.y[j]<c.y[i];ye+=c.y[j]==c.y[i];}
   x[i]=(Q)xl+(Q)(xe+1)/2;y[i]=(Q)yl+(Q)(ye+1)/2;
  }else{x[i]=(Q)c.x[i];y[i]=(Q)c.y[i];}
 }
 Q mx=0,my=0;for(int i=0;i<c.n;++i){mx+=x[i];my+=y[i];}mx/=c.n;my/=c.n;
 Q xx=0,yy=0,xy=0;for(int i=0;i<c.n;++i){x[i]-=mx;y[i]-=my;xx+=x[i]*x[i];yy+=y[i]*y[i];xy+=x[i]*y[i];}
 if(xx==0||yy==0){r.status=3;return r;}
 r.coefficient=xy/sqrtq(xx*yy);
 Q slope=xy/xx,res=0;
 for(int i=0;i<c.n;++i){Q e=y[i]-slope*x[i];res+=e*e;}
 r.residual=sqrtq(res/yy);
 return r;
}
std::vector<Case> cases(){
 std::vector<Case>c;
 auto add=[&](std::string n,V x,V y){int size=x.size();c.push_back({n,x,y,size});};
 add("ordinary",{1,2,4,7,11},{2,1,5,4,10});
 add("negative",{1,2,4,7,11},{-2,-1,-5,-4,-10});
 add("zero",{1,-1,1,-1},{1,1,-1,-1});
 add("perfect_positive",{0,1,2,3,4},{3,5,7,9,11});
 add("perfect_negative",{0,1,2,3,4},{11,9,7,5,3});
 add("two",{1,4},{9,2});
 add("ties",{0,0,1,2,2,2},{3,1,1,0,0,4});
 add("ties_reordered",{2,0,2,1,0,2},{0,3,4,1,1,0});
 add("constant_x",{1,1,1},{2,3,4});
 add("constant_y",{1,2,3},{4,4,4});
 add("both_constant",{1,1,1},{4,4,4});
 double mn=std::numeric_limits<double>::denorm_min(),mx=std::numeric_limits<double>::max();
 add("subnormal",{mn,2*mn,4*mn,8*mn},{0,3*mn,2*mn,9*mn});
 add("opposite_max",{-mx,-mx/2,0,mx/2,mx},{-3,1,0,2,4});
 V ox(5),oy(5);double z=1e300;
 for(int i=0;i<5;++i){ox[i]=z;z=std::nextafter(z,std::numeric_limits<double>::infinity());}
 oy={1,3,2,5,4};add("huge_offset_nextafter",ox,oy);
 add("near_perfect",{0,1,2,3,4},{0,1,2+1e-7,3,4});
 add("ulp_perfect",{0,1,2,3,4},{0,1,std::nextafter(2.,3.),3,4});
 add("inactive_poison",{1,2,3},{3,1,4});c.back().poison=true;
 const auto ordinary=c[0];
 V xs=ordinary.x,ys=ordinary.y;for(auto&v:xs)v*=1e-300;for(auto&v:ys)v*=1e-300;add("tiny_scale",xs,ys);
 xs=ordinary.x;ys=ordinary.y;for(auto&v:xs)v*=1e300;for(auto&v:ys)v*=1e300;add("huge_scale",xs,ys);
 xs=ordinary.x;ys=ordinary.y;for(auto&v:xs)v=v*3+1000;for(auto&v:ys)v=v*5-1000;add("affine",xs,ys);
 xs=ordinary.x;ys=ordinary.y;std::reverse(xs.begin(),xs.end());std::reverse(ys.begin(),ys.end());add("permutation",xs,ys);
 V a(256),b(256);for(int i=0;i<256;++i){a[i]=i;b[i]=(i*73)%257;}
 add("max256",a,b);
 for(int i=0;i<256;++i){a[i]=i/7;b[i]=((i*73)%257)/11;}add("max256_tied",a,b);
 for(int n:{-1,0,1,257})c.push_back({"invalid_n_"+std::to_string(n),{1,2,3},{3,1,4},n});
 add("nonfinite_x",{1,std::numeric_limits<double>::infinity(),3},{3,1,4});
 add("nonfinite_y",{1,2,3},{3,std::numeric_limits<double>::quiet_NaN(),4});
 return c;
}
// Inference oracle ONLY for df=3 (n=5) and df=1 (n=3).
// No general Student CDF claim: these are independent closed forms.
Q student_p_closed(Q t,int df) {
 Q pi=acosq((Q)-1),at=fabsq(t);
 if(at==0)return 1;
 if(df==1)return 2/pi*atanq(1/at);
 Q theta=atanq(sqrtq((Q)3)/at);
 return 2/pi*(theta-sinq(theta)*cosq(theta));
}
Q normal_975() {
 Q lo=0,hi=10,target=(Q).025;
 for(int j=0;j<96;++j){Q mid=(lo+hi)/2;Q sf=erfcq(mid/sqrtq((Q)2))/2;if(sf>target)lo=mid;else hi=mid;}
 return (lo+hi)/2;
}
struct Inference {Case data;bool spearman;};
std::vector<Inference> inference_cases(const std::vector<Case>&a) {
 return {{a[0],false},{a[1],false},{{"n5_zero",{1,-1,1,-1,0},{1,1,-1,-1,0},5},false},
 {{"n5_ties",{0,0,1,2,2},{3,1,1,0,0},5},true},
 {a[14],false},{{"n3_cauchy",{0,1,2},{0,2,1},3},false}};
}
std::string lit(double v) {
 if(std::isnan(v))return "(0.0 / 0.0)";
 if(std::isinf(v))return v>0?"(1.0e308 * 2.0)":"(0.0 - (1.0e308 * 2.0))";
 std::ostringstream o;o<<std::scientific<<std::setprecision(17);
 if(std::fpclassify(v)==FP_SUBNORMAL)o<<"("<<std::ldexp(v,52)<<" / 4503599627370496.0)";else o<<v;
 return o.str();
}
void emit(std::string name,const V&a) {
 std::cout<<" var "<<name<<":[f64;256]=[0.0;256]\n";
 for(size_t i=0;i<a.size();++i)if(a[i]!=0)std::cout<<" "<<name<<"["<<i<<"]="<<lit(a[i])<<"\n";
}
int main(int argc,char**argv) {
 bool f=argc>1&&std::string(argv[1])=="--fixture";auto all=cases();
 std::cout<<std::scientific<<std::setprecision(17);
 if(f)std::cout<<"//@ run-pass\n//@ expect-stdout: STATS_CORRELATION_REFERENCE_PASS\n// C++23 quadmath raw-moment and naive-rank reference.\nuse stats::correlation_core::{pearson_checked,spearman_checked}\nuse stats::correlation_inference::{pearson_inference,spearman_inference}\nfn close(a:f64,b:f64,at:f64,rt:f64)->bool {let d=if a>b{a-b}else{b-a} let v=if b<0.0{-b}else{b} return d<=at+rt*v}\n";
 for(size_t j=0;j<all.size();++j){const auto&c=all[j];
  if(f){std::cout<<"// "<<c.name<<"\nfn case_"<<j<<"() with Mut,Div,Panic {\n";emit("x",c.x);emit("y",c.y);if(c.poison)std::cout<<" x[100]=(0.0/0.0)\n y[255]=(1.0e308*2.0)\n";}
  for(int m=0;m<2;++m){R r=eval(c,m);
   if(!f){std::cout<<j<<"\t"<<c.name<<"\t"<<(m?"spearman":"pearson")<<"\t"<<r.status<<"\t"<<(double)r.coefficient<<"\t"<<(double)r.residual<<"\n";continue;}
   std::cout<<" let r="<<(m?"spearman_checked":"pearson_checked")<<"(&x,&y,"<<c.n<<")\n assert(r.status=="<<r.status<<")\n assert(r.n=="<<c.n<<")\n";
   if(r.status){std::cout<<" assert(r.coefficient==0.0)\n assert(r.residual_ratio==0.0)\n";continue;}
   std::cout<<" assert(close(r.coefficient,"<<lit((double)r.coefficient)<<",1.0e-12,0.0))\n";
   if(r.residual<(Q)1e-30)std::cout<<" assert(r.residual_ratio>=0.0 && r.residual_ratio<=1.0e-14)\n";
   else if(r.residual<(Q)1e-12)std::cout<<" assert(r.residual_ratio>0.0)\n assert(r.residual_ratio>="<<lit((double)(r.residual/(Q)10))<<" && r.residual_ratio<="<<lit((double)(r.residual*(Q)10))<<")\n";
   else std::cout<<" assert(close(r.residual_ratio,"<<lit((double)r.residual)<<",0.0,1.0e-9))\n";
  }
  if(f)std::cout<<"}\n";
 }
 auto inf=inference_cases(all);
 for(size_t j=0;j<inf.size();++j) {
  const auto&c=inf[j].data;bool rank=inf[j].spearman;R est=eval(c,rank);
  int df=c.n-2;
  Q t=est.coefficient*sqrtq((Q)df)/est.residual;
  Q p=student_p_closed(t,df);
  Q lower=0,upper=0;
  if(!rank&&c.n>=4){Q z=atanhq(est.coefficient),h=normal_975()/sqrtq((Q)c.n-3);lower=tanhq(z-h);upper=tanhq(z+h);}
  if(!f){std::cout<<"inference\t"<<c.name<<"\t"<<(rank?"spearman_t_approx":"pearson_normal_model")<<"\t"<<df<<"\t"<<(double)t<<"\t"<<(double)p<<"\t"<<(double)lower<<"\t"<<(double)upper<<"\n";continue;}
  std::cout<<"// Inference "<<c.name<<"; closed Student null df="<<df<<"\nfn inference_"<<j<<"() with Mut,Div,Panic {\n";emit("x",c.x);emit("y",c.y);
  std::cout<<" let r="<<(rank?"spearman_inference":"pearson_inference")<<"(&x,&y,"<<c.n<<","<<(rank?"true":"0.95,true")<<")\n assert(r.status==0)\n assert(r.method=="<<(rank?2:1)<<")\n assert(r.n=="<<c.n<<")\n assert(r.df=="<<lit((double)df)<<")\n assert(r.p_status==0)\n assert(r.calibration=="<<(rank?2:1)<<")\n assert(close(r.coefficient,"<<lit((double)est.coefficient)<<",1.0e-12,0.0))\n";
  bool near=c.name=="near_perfect";
  std::cout<<" assert(close(r.t_stat,"<<lit((double)t)<<",0.0,"<<(near?"4.0e-9":"1.0e-10")<<"))\n assert(close(r.p_value,"<<lit((double)p)<<",0.0,"<<(near?"1.0e-8":"1.0e-10")<<"))\n";
  if(rank||c.n<4)std::cout<<" assert(r.ci_status=="<<(rank?1:2)<<")\n assert(r.ci_lower==0.0 && r.ci_upper==0.0)\n";
  else std::cout<<" assert(r.ci_status==0)\n assert(close(r.level,0.95,0.0,1.0e-14))\n assert(close(r.ci_lower,"<<lit((double)lower)<<",3.0e-11,0.0))\n assert(close(r.ci_upper,"<<lit((double)upper)<<",3.0e-11,0.0))\n";
  std::cout<<"}\n";
 }
 if(f){std::cout<<"fn main() with IO,Mut,Div,Panic {\n";for(size_t j=0;j<all.size();++j)std::cout<<" case_"<<j<<"()\n";for(size_t j=0;j<inf.size();++j)std::cout<<" inference_"<<j<<"()\n";std::cout<<" println(\"STATS_CORRELATION_REFERENCE_PASS\")\n}\n";}
 // Exact invariances of integer inputs.
 auto a=eval(all[0],false),b=eval(all[19],false),p=eval(all[20],false);
 if(fabsq(a.coefficient-b.coefficient)>(Q)1e-30||fabsq(a.coefficient-p.coefficient)>(Q)1e-30)return 2;
 if(eval(all[2],false).coefficient!=0)return 3;
}
