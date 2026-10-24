#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <limits>
#include <cassert>
#include <functional>
// Independent positive quadrature, not the candidate incomplete-beta algorithm.
long double integrate(const std::function<long double(long double)>&f,long double a,long double b,long double fa,long double fm,long double fb,long double whole,long double tol,int depth){
 auto m=(a+b)/2,l=(a+m)/2,r=(m+b)/2;
 auto fl=f(l),fr=f(r),left=(m-a)*(fa+4*fl+fm)/6,right=(b-m)*(fm+4*fr+fb)/6;
 auto delta=left+right-whole;
 if(depth==0 || std::abs(delta)<15*tol)return left+right+delta/15;
 return integrate(f,a,m,fa,fl,fm,left,tol/2,depth-1)+integrate(f,m,b,fm,fr,fb,right,tol/2,depth-1);
}
long double survival(long double F,int df){
 if(F==0)return 1;
 auto u=std::atan(std::sqrt((long double)df/F)),su=std::sin(u);
 auto f=[=](long double t){return std::pow(std::sin(u*t)/su,df-1);};
 auto fa=f(0),fm=f(.5L),fb=f(1),initial=(fa+4*fm+fb)/6;
 auto J=integrate(f,0,1,fa,fm,fb,initial,2e-19L,28);
 auto logD=.5L*std::log(std::acos(-1.L))+std::lgamma(df/2.L)-std::lgamma((df+1)/2.L)-std::log(2.L);
 return std::exp((df-1)*std::log(su)+std::log(u)+std::log(J)-logD);
}
long double median(std::vector<long double> v){std::sort(v.begin(),v.end());size_t m=v.size()/2;return v.size()%2?v[m]:(v[m-1]+v[m])/2;}
struct R{long double f,p,ma,mb;};
R reference(const std::vector<double>&a,const std::vector<double>&b){
 std::vector<long double>da(a.begin(),a.end()),db(b.begin(),b.end());
 auto ca=median(da),cb=median(db);
 for(auto&v:da){v=std::abs(v-ca);}
 for(auto&v:db){v=std::abs(v-cb);}
 auto ma=std::accumulate(da.begin(),da.end(),0.L)/a.size(),mb=std::accumulate(db.begin(),db.end(),0.L)/b.size();
 auto grand=(a.size()*ma+b.size()*mb)/(a.size()+b.size());
 long double ssb=a.size()*(ma-grand)*(ma-grand)+b.size()*(mb-grand)*(mb-grand),ssw=0;
 for(auto v:da){ssw+=(v-ma)*(v-ma);}
 for(auto v:db){ssw+=(v-mb)*(v-mb);}
 auto F=ssb*(a.size()+b.size()-2)/ssw;
 return {F,survival(F,(int)(a.size()+b.size()-2)),ma,mb};
}
int main(){
 static_assert(std::numeric_limits<long double>::max_exponent>=16384);
 for(long double f:{1e-12L,.1L,1.L,10.L,1e12L,1e30L}){
  auto exact=(2/(f+2))/(1+std::sqrt(f/(f+2))),numeric=survival(f,2);
  assert(std::abs(numeric/exact-1)<2e-16L);
 }
 std::vector<std::pair<std::vector<double>,std::vector<double>>>cases;
 std::vector<double>a{1,2,4,8,16},b{2,3,4,5,6,8};cases.push_back({a,b});
 for(double s:{1e-200,1e200,-1e-200,-1e200}){auto x=a,y=b;for(auto&v:x)v*=s;for(auto&v:y)v*=s;cases.push_back({x,y});}
 auto x=a,y=b;for(auto&v:x)v+=1e15;for(auto&v:y)v-=1e15;cases.push_back({x,y});
 cases.push_back({{1e15,1e15+.125,1e15+.25,1e15+.375},{0,.5,1,2}});
 cases.push_back({{-1.7e308,1.7e308,0,1e308,-1e308},{-1e308,1e308,0,2e307,-4e307,8e307}});
 cases.push_back({{-1e-320,2e-320,4e-320,-8e-320,16e-320},{2e-320,3e-320,4e-320,5e-320,6e-320,8e-320}});
 cases.push_back({{1,2,3,4,5},{1,2,3,4,5}});
 cases.push_back({{2,2,2,2},{1,2,3,4,8}});
 cases.push_back({{1,3},{1,2,4,8,16}});
 cases.push_back({{1,2,4,8,16},{1,3}});
 std::vector<double>u,v;
 for(int i=0;i<100;i++){u.push_back((i%2?-1:1)*(1+(i%5)*1e-4));v.push_back((i%2?-1:1)*(2+(i%7)*2e-4));}
 cases.push_back({u,v});
 u.clear();v.clear();for(int i=0;i<100;i++){u.push_back(i%11);v.push_back((i%13)*3.0+.25);}cases.push_back({u,v});
 std::cout<<std::scientific<<std::setprecision(17);
 std::cout<<"//@ run-pass\n//@ expect-stdout: STATS_BROWN_FORSYTHE_REFERENCE_PASS\nuse stats::brown_forsythe::{brown_forsythe}\n";
 std::cout<<"fn close(a:f64,b:f64,at:f64,rt:f64)->bool{let d=if a>b{a-b}else{b-a} let v=if b<0.0{-b}else{b} return d<=at+rt*v}\nfn main() with IO,Mut,Div,Panic{\n";
 for(size_t c=0;c<cases.size();c++){
  auto [x,y]=cases[c];auto r=reference(x,y);
  std::cerr<<std::setprecision(21)<<"case="<<c<<" nA="<<x.size()<<" nB="<<y.size()<<" F="<<r.f<<" p="<<r.p<<" MADmeans="<<r.ma<<","<<r.mb<<"\n";
  std::cout<<" var a"<<c<<":[f64;100]=[0.0;100]\n var b"<<c<<":[f64;100]=[0.0;100]\n";
  for(size_t i=0;i<x.size();i++)std::cout<<" a"<<c<<"["<<i<<"]="<<x[i]<<"\n";
  for(size_t i=0;i<y.size();i++)std::cout<<" b"<<c<<"["<<i<<"]="<<y[i]<<"\n";
  std::cout<<" let r"<<c<<"=brown_forsythe(&a"<<c<<","<<x.size()<<",&b"<<c<<","<<y.size()<<")\n assert(r"<<c<<".status==0 && r"<<c<<".df1==1 && r"<<c<<".df2=="<<x.size()+y.size()-2<<")\n";
  std::cout<<" assert(close(r"<<c<<".f_stat,"<<(double)r.f<<",5.0e-11,5.0e-10))\n";
  if((double)r.p==0)std::cout<<" assert(r"<<c<<".p==0.0)\n";
  else std::cout<<" assert(close(r"<<c<<".p,"<<(double)r.p<<",1.0e-300,5.0e-10))\n";
  std::cout<<" assert(close(r"<<c<<".mean_abs_a_scaled*r"<<c<<".deviation_scale,"<<(double)r.ma<<","<<(r.ma!=0 && std::abs(r.ma)<std::numeric_limits<double>::min()?2*std::numeric_limits<double>::denorm_min():0.0)<<",5.0e-10))\n";
  std::cout<<" assert(close(r"<<c<<".mean_abs_b_scaled*r"<<c<<".deviation_scale,"<<(double)r.mb<<","<<(r.mb!=0 && std::abs(r.mb)<std::numeric_limits<double>::min()?2*std::numeric_limits<double>::denorm_min():0.0)<<",5.0e-10))\n";
 }
 std::cout<<" println(\"STATS_BROWN_FORSYTHE_REFERENCE_PASS\")\n 0\n}\n";
 // Independent tail fixture written on stdout after a delimiter for splitting.
 std::cout<<"//TAIL_FIXTURE\n//@ run-pass\n//@ expect-stdout: STATS_BROWN_FORSYTHE_TAIL_PASS\nuse stats::fisher_f::{f_sf}\nuse special::beta::{ibeta}\nfn close(a:f64,b:f64,rt:f64)->bool{let d=if a>b{a-b}else{b-a} let floor=if b>0.0 && b<2.2250738585072014e-308 {1.0e-323} else {0.0} return d<=floor+rt*b}\nfn main() with IO,Mut,Div,Panic{\n";
 for(int df:{2,4,9,10,98,198})for(long double F:{.01L,1.L,10.L,1e4L,3e5L,1e30L}){
  auto p=survival(F,df);std::cerr<<"tail df="<<df<<" F="<<F<<" p="<<p<<"\n";
  if((double)p==0)std::cout<<" assert(f_sf("<<(double)F<<",1.0,"<<df<<".0)==0.0)\n";
  else std::cout<<" assert(close(f_sf("<<(double)F<<",1.0,"<<df<<".0),"<<(double)p<<",5.0e-10))\n";
 }
 std::cout<<" assert(ibeta(100.0,0.5,1.0e-30)==0.0)\n println(\"STATS_BROWN_FORSYTHE_TAIL_PASS\")\n 0\n}\n";
 std::cerr<<"quadrature validated against6 analytic df=2 tails;15 datasets;36 F tails;binary64-underflow beta case\n";
}
