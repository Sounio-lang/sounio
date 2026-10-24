// Independent C++23 tail reference: transformed density quadrature, not incomplete beta.
#include <cmath>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <array>
using R = long double;
R quad(const std::function<R(R)>& f) {
 std::function<R(R,R,R,R,R,R,R,int)> rec;
 rec = [&](R a,R b,R fa,R fm,R fb,R whole,R eps,int depth)->R {
  R m=(a+b)/2, fl=f((a+m)/2), fr=f((m+b)/2);
  R l=(m-a)*(fa+4*fl+fm)/6, r=(b-m)*(fm+4*fr+fb)/6, d=l+r-whole;
  if (std::abs(d)<=15*eps) return l+r+d/15;
  if (!depth) throw std::runtime_error("quadrature convergence");
  return rec(a,m,fa,fl,fm,l,eps/2,depth-1)+rec(m,b,fm,fr,fb,r,eps/2,depth-1);
 };
 R a=f(0),m=f(0.5L),b=f(1);
 return rec(0,1,a,m,b,(a+4*m+b)/6,2e-17L,30);
}
R sf(R t,R df) {
 if(t==0) return 0.5L;
 if(t<0) return 1-sf(-t,df);
 R d=std::atan(std::sqrt(df)/t), sd=std::sin(d);
 auto f=[=](R v)->R { if(df==1) return 1; if(v==0) return 0;
   return std::exp((df-1)*std::log(std::sin(d*v)/sd)); };
 R logk=std::lgamma((df+1)/2)-std::lgamma(df/2)-std::log(std::acos(-1.0L))/2;
 return std::exp(logk+std::log(d)+(df-1)*std::log(sd))*quad(f);
}
int main(int argc,char**argv) {
 if(argc!=2) return 2;
 for(R t: {0.1L,1.0L,10.0L,100000.0L}) {
   R c=std::atan(1/t)/std::acos(-1.0L);
   if(std::abs(sf(t,1)/c-1)>1e-13L) throw std::runtime_error("Cauchy identity");
   R h=std::sqrt(t*t+2), d2=1/(h*(h+t));
   if(std::abs(sf(t,2)/d2-1)>1e-13L) throw std::runtime_error("df2 identity");
 }
 std::ofstream s(argv[1]);
 s<<std::setprecision(18)<<std::showpoint;
 s<<"//@ run-pass\n//@ expect-stdout: STATS_INFERENTIAL_TAILS_PASS\n"
   <<"use stats::student_t::{t_sf, t_two_tail}\nuse stats::inferential::{spearman_test}\n"
   <<"use stats::fisher_f::{f_sf}\n"
   <<"fn close_relative(got: f64, expected: f64) with Mut, Div, Panic {\n"
   <<" assert(got > 0.0)\n assert(got <= 1.0)\n"
   <<" let relative = got / expected - 1.0\n assert(relative < 1.0e-7)\n assert(relative > -1.0e-7)\n}\n"
   <<"fn main() with IO, Mut, Div, Panic {\n";
 std::cout<<std::setprecision(18)<<"df\tt\tupper_tail\n";
 int i=0;
 for(R df: {1.0L,2.0L,5.5L,10.0L,30.0L,100.0L,500.0L})
 for(R t: {0.0L,1.0L,2.0L,5.0L,10.0L,100.0L,100000.0L}) {
   R p=sf(t,df);
   if(p<1e-290L) continue; // current exp helpers do not promise subnormal accuracy.
   std::cout<<df<<'\t'<<t<<'\t'<<p<<'\n';
   s<<" println(\"tail_case_"<<++i<<"\")\n";
   s<<" close_relative(t_sf("<<t<<", "<<df<<"), "<<p<<")\n";
   s<<" close_relative(t_two_tail("<<t<<", "<<df<<"), "<<2*p<<")\n";
 }
 for(R d2: {1.0L,2.0L,5.0L,10.0L,30.0L,100.0L,500.0L})
 for(R x: {1.0L,5.0L,100.0L}) {
   // For d1=2, analytic SF=(d2/(d2+2*x))^(d2/2).
   R f=std::exp(-d2/2*std::log1p(2*x/d2));
   std::cout<<"F_D1_2\t"<<d2<<'\t'<<x<<'\t'<<f<<'\n';
   s<<" println(\"f_tail_case_d2_"<<d2<<"_x_"<<x<<"\")\n";
   s<<" close_relative(f_sf("<<x<<", 2.0, "<<d2<<"), "<<f<<")\n";
 }
 s<<" assert(t_sf(-2.0, 10.0) > 0.96)\n assert(t_sf(-2.0, 10.0) < 0.97)\n"
  <<" assert(t_sf(0.0, 0.0) == 0.5)\n"
  <<" var x: [f64; 100] = [0.0; 100]\n var y: [f64; 100] = [0.0; 100]\n"
  <<" x[0]=1.0; x[1]=2.0; x[2]=3.0; x[3]=4.0; x[4]=5.0\n"
  <<" y[0]=1.0; y[1]=3.0; y[2]=2.0; y[3]=5.0; y[4]=4.0\n"
  <<" let rank = spearman_test(x,y,5)\n";
 R rho=0.8L,t=rho*std::sqrt(3/(1-rho*rho)),p=2*sf(t,3);
 s<<" close_relative(rank.p_value, "<<p<<")\n"
  <<" assert(rank.rho > 0.799999999)\n assert(rank.rho < 0.800000001)\n"
  <<" println(\"STATS_INFERENTIAL_TAILS_PASS\")\n 0\n}\n";
 std::cout<<"SPEARMAN_T_APPROX\t"<<t<<"\t"<<p<<"\nCPP23_TAIL_REFERENCE_PASS\n";
}
