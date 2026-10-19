// C++23 independent enumeration of decimal candidates, not Sounio normalization.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <array>
#include <stdexcept>
struct C { double lo,hi;int budget; };
int main(int argc,char**argv){
 if(argc!=2)return 2;
 std::ofstream o(argv[1]);o<<std::scientific<<std::setprecision(17);
 o<<"//@ run-pass\n//@ expect-stdout: NICE_AXIS_REFERENCE_PASS\n"
    "use viz::nice_axis::{axis_nice_cover}\n"
    "fn near(a:f64,b:f64,s:f64)->bool with Div {let d=if a>b {a-b}else{b-a} d/s < 2.0e-12}\n"
    "fn main() with IO, Mut, Div, Panic {\n";
 const std::array<C,12> cases{{
 {0,8,4},{.13,.91,5},{1.1,9.1,4},{-3.1,8.7,5},{-.00031,.00079,5},
 {1.3e-299,9.1e-299,5},{1.3e299,9.1e299,5},{-91,-13,5},
 {-.91,-.13,5},{0,1,1},{12.3,98.7,4},{-1,1,8}}};
 int k=0;
 for(const auto& c:cases){
  long double lower=0,upper=0,step=0;int count=0;
  // Globally enumerate increasing decimal 1,2,2.5,5 candidates in long double.
  for(int e=-320;e<=308&&!count;e++){
   long double unit=std::pow(10.0L,e);
   for(long double m:{1.0L,2.0L,2.5L,5.0L}){
    long double s=m*unit;
    if(s<((long double)c.hi-c.lo)/c.budget)continue;
    long double first=std::floor((long double)c.lo/s),last=std::ceil((long double)c.hi/s);
    long double n=last-first;
    if(n>=1&&n<=c.budget){lower=first*s;upper=last*s;step=s;count=(int)n;break;}
   }
  }
  if(!count)throw std::runtime_error("oracle has no plan");
  o<<"    println(\"NICE_CASE_"<<k<<"\")\n    let p"<<k<<"=axis_nice_cover("<<c.lo<<","<<c.hi<<","<<c.budget<<")\n";
  std::string p="p"+std::to_string(k);
  o<<"    assert("<<p<<".status == 0 && "<<p<<".intervals == "<<count<<")\n";
  for(auto pair:{std::pair{".lo",lower},std::pair{".hi",upper},std::pair{".step",step}})
   o<<"    assert(near("<<p<<pair.first<<","<<(double)pair.second<<","<<(double)step<<"))\n";
  o<<"    assert("<<p<<".lo <= "<<c.lo<<" && "<<p<<".hi >= "<<c.hi<<")\n";
  k++;
 }
 o<<"    assert(axis_nice_cover(1.0,1.0,4).status == 1)\n"
    "    assert(axis_nice_cover(1.0,2.0,0).status == 1)\n"
    "    assert(axis_nice_cover(1.0,2.0,9).status == 1)\n"
    "    assert(axis_nice_cover(-1.0,1.0,1).status == 4)\n"
    "    assert(axis_nice_cover(-1.0e308,1.0e308,4).status == 2)\n"
    "    assert(axis_nice_cover(1.0e12,1.0e12+7.0,4).status == 3)\n"
    "    var huge=1.0e308\n    let inf=huge*10.0\n"
    "    assert(axis_nice_cover(0.0,inf,4).status == 1)\n"
    "    assert(axis_nice_cover(inf-inf,1.0,4).status == 1)\n"
    "    println(\"NICE_AXIS_REFERENCE_PASS\")\n    0\n}\n";
 std::cout<<"C++23 long-double enumeration: "<<k<<" nice covering domains generated.\n";
 return !o;
}
