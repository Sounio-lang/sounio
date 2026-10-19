#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

struct Case {std::vector<double>x; int b; double conf; std::int64_t seed;};
int main() {
 std::vector<Case> cases={
 {{1,2,3,4,5},1000,.95,20260715},
 {{1,2,3,4,5},1000,.95,2147483647},
 {{-7,-2,0,4,13},511,.9,-99},
 {{-7,-2,0,4,13},511,.9,std::numeric_limits<std::int64_t>::min()},
 {{1e308,1.1e308,1.2e308,1.3e308},1024,.95,42},
 {{1e-300,2e-300,3e-300,4e-300},300,.8,7},
 {{-1e308,1e308,-1e308,1e308},1024,.95,17},
 {{7,7,7},2,.95,0},
 {{1,9},2,.5,1},
 {std::vector<double>(256),1024,.99,2147483646}};
 for(int i=0;i<256;++i)cases.back().x[i]=(i%17)-8.;
 std::cout<<std::scientific<<std::setprecision(17);
 std::cout<<"//@ run-pass\n//@ expect-stdout: STATS_BOOTSTRAP_REFERENCE_PASS\n"
 "use stats::bootstrap::{bootstrap_mean}\nuse math::pure::{is_finite}\n";
 for(std::size_t id=0;id<cases.size();++id) {
  auto const& c=cases[id]; double scale=0;for(double v:c.x)scale=std::max(scale,std::abs(v));
  std::int64_t st=c.seed%2147483647LL;if(st<=0)st+=2147483646LL;
  std::vector<long double> means;long double point=0;
  for(double v:c.x)point+=(long double)v/scale;
  point/=c.x.size();
  for(int b=0;b<c.b;++b) {
   long double sum=0;
   for(std::size_t j=0;j<c.x.size();++j) {
    std::int64_t draw;
    do {st=(16807LL*st)%2147483647LL;draw=st-1;}
    while(draw>= (2147483646LL/(std::int64_t)c.x.size())*(std::int64_t)c.x.size());
    sum+=(long double)c.x[draw%(std::int64_t)c.x.size()]/scale;
   }
   means.push_back(sum/c.x.size());
  }
  long double center=0;for(auto v:means)center+=v;center/=means.size();
  long double ss=0;for(auto v:means)ss+=(v-center)*(v-center);
  long double se=std::sqrt(ss/(means.size()-1));
  std::sort(means.begin(),means.end());
  auto quant=[&](long double p){auto h=(means.size()-1)*p;auto k=(std::size_t)h;
   return k+1==means.size()?means[k]:means[k]*(1-(h-k))+means[k+1]*(h-k);};
  auto alpha=(1.L-c.conf)/2;
  std::cout<<"fn case_"<<id<<"() with Mut, Div, Panic {\nvar x: [f64; 256] = [0.0; 256]\n";
  for(std::size_t j=0;j<c.x.size();++j)std::cout<<"x["<<j<<"] = "<<c.x[j]<<"\n";
  std::cout<<"let r = bootstrap_mean(&x, "<<c.x.size()<<", "<<c.b<<", "<<c.conf<<", ";
  if(c.seed==std::numeric_limits<std::int64_t>::min())std::cout<<"(0 - 9223372036854775807 - 1)";
  else std::cout<<c.seed;
  std::cout<<")\n";
  std::array<std::pair<char const*,long double>,4> expected={{{"mean",point},{"se",se},{"lo",quant(alpha)},{"hi",quant(1-alpha)}}};
  for(auto const& [name,v]:expected)
   std::cout<<"assert(is_finite(r."<<name<<"))\nassert(abs(r."<<name<<" / "<<scale<<" - "<<(double)v<<") < 3.0e-12)\n";
  std::cout<<"assert(r.b == "<<c.b<<")\nassert(x[0] == "<<c.x[0]<<")\n}\n";
 }
 std::cout<<"fn abs(x: f64) -> f64 { if x < 0.0 { 0.0 - x } else { x } }\nfn main() with IO, Mut, Div, Panic {\n";
 for(std::size_t i=0;i<cases.size();++i)std::cout<<"case_"<<i<<"()\n";
 std::cout<<"println(\"STATS_BOOTSTRAP_REFERENCE_PASS\")\n0\n}\n";
 std::cerr<<"Independent long-double reference: "<<cases.size()<<" datasets; std::sort; two-pass sample variance; integer MINSTD rejection sampling.\n";
}
