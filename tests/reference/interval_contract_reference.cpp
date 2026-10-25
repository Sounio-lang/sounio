// C++23 independent structural oracle. No directed arithmetic is claimed here.
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
struct Interval { double lo, hi; int decoration, status; };
constexpr double inf=std::numeric_limits<double>::infinity();
Interval empty(int status) { return {inf,-inf,status?0:1,status}; }
Interval make(double lo,double hi,int d) {
 if(d<1 || d>4) return empty(2);
 if(std::isnan(lo)||std::isnan(hi)) return empty(1);
 if(lo>hi) return empty(0);
 if(lo==inf||hi==-inf) return empty(1);
 if(lo==0) lo=-0.0;
 if(hi==0) hi=0.0;
 if((lo==-inf||hi==inf)&&d==4) d=3;
 return {lo,hi,d,0};
}
int less(Interval a,Interval b) {
 if(a.status||b.status||a.lo>a.hi||b.lo>b.hi) return 3;
 if(a.hi<b.lo) return 2;
 if(a.lo>=b.hi) return 0;
 return 1;
}
Interval hull(Interval a,Interval b) {
 if(a.status||b.status) return empty(1);
 if(a.lo>a.hi) return b;
 if(b.lo>b.hi) return a;
 return make(std::min(a.lo,b.lo),std::max(a.hi,b.hi),std::min(a.decoration,b.decoration));
}
std::int64_t bits(double x) {return std::bit_cast<std::int64_t>(x);}
int main(int argc,char**argv) {
 if(argc!=2) return 2;
 std::ifstream in(argv[1]);if(!in) return 3;
 const std::array<double,12> v={-inf,-std::numeric_limits<double>::max(),-1.0,-std::numeric_limits<double>::denorm_min(),-0.0,0.0,std::numeric_limits<double>::denorm_min(),1.0,std::numeric_limits<double>::max(),inf,std::numeric_limits<double>::quiet_NaN(),2.0};
 std::string line; std::size_t rows=0; bool marker=false;
 while(std::getline(in,line)) {
  if(line=="INTERVAL_CONTRACT_PASS") {if(marker) return 4;marker=true;continue;}
  std::istringstream s(line); std::string tag; int i,j,d,st,dec,cmp,hst,hdec;std::int64_t lo,hi,hlo,hhi;
  if(!(s>>tag>>i>>j>>d>>st>>dec>>lo>>hi>>cmp>>hst>>hdec>>hlo>>hhi)||tag!="IC") return 5;
  std::string extra;if(s>>extra) return 6;
  if(rows>=864||i!=int(rows/72)||j!=int(rows/6%12)||d!=int(rows%6)) return 7;
  auto a=make(v[i],v[j],d),b=make(v[j],v[i],4),h=hull(a,b);
  if(st!=a.status||dec!=a.decoration||lo!=bits(a.lo)||hi!=bits(a.hi)||cmp!=less(a,b)||hst!=h.status||hdec!=h.decoration||hlo!=bits(h.lo)||hhi!=bits(h.hi)) {
   std::cerr<<"mismatch row "<<rows<<'\n';return 8;
  }
  ++rows;
 }
 if(rows!=864||!marker) return 9;
 std::cout<<"INTERVAL_CONTRACT_REFERENCE_PASS rows="<<rows<<"\n";
}
