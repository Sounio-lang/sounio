// Numerical counterexample to the original zero-gap "wider than Delta" test.
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
struct P { double lo,hi,var; };
P lift(double mean,double variance){return {mean,mean,variance};}
P add(P a,P b){return {a.lo+b.lo,a.hi+b.hi,a.var+b.var};}
P sub(P a,P b){return {a.lo-b.hi,a.hi-b.lo,a.var+b.var};}
P mul(P a,P b){
 const std::array<double,4> corners{a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};
 const double am=.5*(a.lo+a.hi),bm=.5*(b.lo+b.hi);
 return {*std::min_element(corners.begin(),corners.end()),*std::max_element(corners.begin(),corners.end()),bm*bm*a.var+am*am*b.var+a.var*b.var};
}
double solve(double mu){double x=.5,y=0;for(int i=0;i<80;++i){const double dx=y,dy=mu*(1-x*x)*y-x;x=x+.02*dx;y=y+.02*dy;}return y;}
int main(){
 const P mu=lift(5,.25),dt=lift(.02,0),one=lift(1,0);
 P x=lift(.5,.01),y=lift(0,0);
 for(int step=0;step<80;++step){
  double low=1e30,high=-1e30;
  for(int i=0;i<7;++i){const double t=x.lo+(static_cast<double>(i)/6)*(x.hi-x.lo),v=t*t;low=std::min(low,v);high=std::max(high,v);}
  const P square{low,high,0};
  const P dy=sub(mul(mu,mul(sub(one,square),y)),x);
  const P nx=add(x,mul(dt,y)),ny=add(y,mul(dt,dy));
  x=nx;y=ny;
 }
 const double width=y.hi-y.lo,derivative=(solve(5.001)-solve(5))/.001,delta_width=2*std::sqrt(derivative*derivative*.25);
 std::cout<<std::setprecision(17)
  <<"PBOX_INPUT_MEAN "<<mu.lo<<' '<<mu.hi<<" variance "<<mu.var<<'\n'
  <<"PBOX_FINAL_MEAN "<<y.lo<<' '<<y.hi<<" width "<<width<<'\n'
  <<"DELTA_WIDTH "<<delta_width<<'\n'
  <<"ORIGINAL_WIDER_PREDICATE "<<(width>delta_width)<<'\n';
 if(!std::isfinite(width)||width!=0||!(delta_width>0)||width>delta_width)return 1;
 std::cout<<"STIFF_PBOX_COUNTEREXAMPLE_CPP23_OK\n";
}
