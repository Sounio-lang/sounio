#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>
struct Case {std::vector<double>x,y; int b; std::int64_t seed; bool exact;};
int main(){
 std::vector<Case> cases={
 {{1,2,3},{7,8,9},0,42,true},
 {{1,2,3},{7,8,9},2048,42,false},
 {{1e-20,2e-20,3e-20},{7e-20,8e-20,9e-20},2048,42,false},
 {{1,2,3},{1,2,3},0,1,true},
 {{7,7},{7,7,7},100,0,false},
 {{-2,4},{0,1,3},0,1,true},
 {{-2},std::vector<double>(255),0,1,true},
 {{1e308,1.1e308},{1.2e308,1.3e308},0,1,true},
 {{1e15,1e15+1},{1e15+2,1e15+3},0,1,true},
 {{-7,-2,4},{0,1,9},1024,std::numeric_limits<std::int64_t>::min(),false},
 {{-7,-2,4},{0,1,9},1,2147483647,false},
 {{1,2,3},{7,8,9},256,42,false}};
 for(int i=0;i<255;++i)cases[6].y[i]=(i%13)-6.;
 cases.push_back({cases[6].y,cases[6].x,0,1,true});
 cases.push_back({{0,1,2,3,4,5,6,7,8},{9,10,11,12,13,14,15,16,17},0,1,true});
 std::cout<<std::scientific<<std::setprecision(17);
 std::cout<<"//@ run-pass\n//@ expect-stdout: STATS_PERMUTATION_REFERENCE_PASS\nuse stats::permutation::{permutation_means}\n";
 for(std::size_t id=0;id<cases.size();++id){
  auto const& c=cases[id];auto raw=c.x;raw.insert(raw.end(),c.y.begin(),c.y.end());
  int n=raw.size(),nx=c.x.size();double scale=0;for(double v:raw)scale=std::max(scale,std::abs(v));if(scale==0)scale=1;
  std::vector<long double> pool;for(double v:raw)pool.push_back(((long double)v-raw[0])/scale);
  std::vector<int> order(n);std::iota(order.begin(),order.end(),0);
  auto stat=[&](){long double sx=0,sy=0;for(int j=0;j<n;++j){if(j<nx)sx+=pool[order[j]];else sy+=pool[order[j]];}return sx/nx-sy/(n-nx);};
  long double signed_obs=stat(),obs=std::abs(signed_obs),threshold=obs*(1.L-64.L*std::numeric_limits<double>::epsilon());
  std::int64_t state=c.seed%2147483647LL;if(state<=0)state+=2147483646LL;
  auto initial=c.exact?0:state;std::int64_t hits=0,draws=0;
  auto count=[&](){if(std::abs(stat())>=threshold)++hits;++draws;};
  if(c.exact){
   std::vector<int> mask(n,0);std::fill(mask.end()-nx,mask.end(),1);
   do {int a=0,b=nx;for(int i=0;i<n;++i){if(mask[i])order[a++]=i;else order[b++]=i;}count();}
   while(std::next_permutation(mask.begin(),mask.end()));
  }else{
   for(int rep=0;rep<c.b;++rep){std::iota(order.begin(),order.end(),0);
    for(int i=0;i<nx;++i){std::int64_t draw,width=n-i,limit=(2147483646LL/width)*width;
     do{state=(16807LL*state)%2147483647LL;draw=state-1;}while(draw>=limit);
     std::swap(order[i],order[i+draw%width]);}count();
   }
  }
  long double q=(long double)hits/draws,p=c.exact?q:(long double)(hits+1)/(draws+1);
  long double z=1.959963984540054L,denom=1+z*z/draws,center=(q+z*z/(2*draws))/denom;
  long double half=z*std::sqrt(q*(1-q)/draws+z*z/(4.L*draws*draws))/denom;
  std::cout<<"fn case_"<<id<<"() with Mut, Div, Panic {\nvar x:[f64;256]=[0.0;256]\nvar y:[f64;256]=[0.0;256]\n";
  for(std::size_t i=0;i<c.x.size();++i)std::cout<<"x["<<i<<"]="<<c.x[i]<<"\n";
  for(std::size_t i=0;i<c.y.size();++i)std::cout<<"y["<<i<<"]="<<c.y[i]<<"\n";
  std::cout<<"let r=permutation_means(&x,"<<nx<<",&y,"<<n-nx<<","<<c.b<<",";
  if(c.seed==std::numeric_limits<std::int64_t>::min())std::cout<<"(0-9223372036854775807-1)";else std::cout<<c.seed;
  std::cout<<","<<(c.exact?"true":"false")<<")\n";
  std::cout<<"assert(r.status == 0)\nassert(r.method == "<<(c.exact?2:1)<<")\nassert(r.hits == "<<hits<<")\nassert(r.draws == "<<draws<<")\nassert(r.seed_used == "<<initial<<")\n";
  if(signed_obs == 0)std::cout<<"assert(r.observed_diff == 0.0)\n";
  else std::cout<<"assert(abs(r.observed_diff / "<<(double)(signed_obs*scale)<<" - 1.0) < 3.0e-12)\n";
  std::cout<<"assert(abs(r.p_value - "<<(double)p<<") < 3.0e-12)\n";
  std::cout<<"assert(!r.exchangeability_assessed)\nassert(r.mc_available == "<<(c.exact?"false":"true")<<")\n";
  if(!c.exact)std::cout<<"assert(abs(r.mc_se - "<<(double)std::sqrt(q*(1-q)/draws)<<") < 3.0e-12)\nassert(abs(r.mc_lo - "<<(double)std::max(0.L,center-half)<<") < 3.0e-12)\nassert(abs(r.mc_hi - "<<(double)std::min(1.L,center+half)<<") < 3.0e-12)\n";
  std::cout<<"assert(x[0] == "<<c.x[0]<<" && y[0] == "<<c.y[0]<<")\n}\n";
  std::cerr<<"case "<<id<<" method="<<(c.exact?"enumeration":"MC")<<" hits="<<hits<<" draws="<<draws<<" p="<<(double)p<<"\n";
 }
 std::cout<<"fn abs(v:f64)->f64 {if v<0.0 {0.0-v}else{v}}\nfn main() with IO, Mut, Div, Panic {\n";
 for(std::size_t id=0;id<cases.size();++id)std::cout<<"case_"<<id<<"()\n";
 std::cout<<"println(\"STATS_PERMUTATION_REFERENCE_PASS\")\n0\n}\n";
}
