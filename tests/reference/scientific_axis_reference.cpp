#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
int main() {
  const std::array<double,8> values={0.0,-0.0,0.125,-2.5,1e-300,1e300,9.999999,4.9406564584124654e-324};
  std::cout<<"fn reference_labels() with Mut, Div, Panic {\n var smallest_normal = 2.2250738585072014e-308\n";
  for(double x:values) {
    std::ostringstream fmt; fmt<<std::scientific<<std::setprecision(5)<<x;
    std::string expected=fmt.str();
    if(x==0)expected="0";
    else {auto p=expected.find('e'); auto e=std::stoi(expected.substr(p+1));expected=expected.substr(0,p+1)+(e<0?"-":"+")+std::to_string(std::abs(e));}
    std::cout<<" { let label = axis_label(";
    if(std::fpclassify(x)==FP_SUBNORMAL)std::cout<<"smallest_normal / 4503599627370496.0";
    else std::cout<<std::scientific<<std::setprecision(17)<<x;
    std::cout<<")\n assert(label.n == "<<expected.size()<<")\n";
    for(std::size_t i=0;i<expected.size();++i)std::cout<<" assert(label.text["<<i<<"] == "<<int(expected[i])<<")\n";
    std::cout<<" }\n";
  }
  std::cout<<"}\nfn reference_ticks() with Mut, Div, Panic {\n";
  for(auto domain:std::array<std::array<double,2>,4>{{{-1e308,1e308},{1e-300,9e-300},{-7,13},{1e300,1.5e300}}}) {
    std::cout<<" { let plan = axis_ticks("<<domain[0]<<", "<<domain[1]<<", 4)\n assert(plan.status == 0 && plan.n == 5)\n";
    for(int i=0;i<=4;++i){ long double f=(long double)i/4;double ref=(1-f)*(long double)domain[0]+f*(long double)domain[1];std::cout<<" assert(close(plan.values["<<i<<"], "<<ref<<"))\n";}
    std::cout<<" }\n";
  }
  std::cout<<"}\n";
}
