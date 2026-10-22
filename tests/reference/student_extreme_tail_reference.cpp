// C++23 analytic extreme Student two-tail checks; inputs are binary64.
#include <quadmath.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
struct T {int df;double t,expected,tolerance;};
std::string literal(double v) {std::ostringstream s;s<<std::scientific<<std::setprecision(17);if(std::fpclassify(v)==FP_SUBNORMAL)s<<"("<<std::ldexp(v,52)<<" / 4503599627370496.0)";else s<<v;return s.str();}
int main() {
 using Q=__float128;Q pi=acosq((Q)-1);
 std::cout<<std::scientific<<std::setprecision(17);
 for(auto c:std::vector<T>{{1,1e200,6.36619772367581343e-201,2e-13},{1,1e308,6.36619772367581343e-309,2e-13},{1,1,.5,2e-13},{1,.25,.8440417392452613,2e-13},{2,1e155,1e-310,2e-10}}) {
  Q t=(Q)c.t;
  Q exact=c.df==1?2/pi*atanq(1/t):2/(sqrtq(t*t+2)*(sqrtq(t*t+2)+t));
  Q rel=fabsq((Q)c.expected-exact)/exact;
  std::cout<<c.df<<"\t"<<c.t<<"\t"<<(double)exact<<"\t"<<c.expected<<"\t"<<(double)rel<<"\t"<<c.tolerance<<"\t"<<literal((double)exact)<<"\n";
  if(rel>(Q)c.tolerance)return 1;
 }
 return 0;
}
