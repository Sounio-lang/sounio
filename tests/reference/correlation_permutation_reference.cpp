// C++23 quadmath correlation permutation oracle.
// Independent raw centered moments and O(n^2) averaged ranks.
// Exact enumeration is over labelled index permutations, retaining ties.
// Pearson tail boundary intentionally matches declared 64-epsilon policy.
// RNG specification is Park-Miller 16807, unbiased rejection, full shuffle reset.
#include <quadmath.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
using Q=__float128;using V=std::vector<double>;using I=std::int64_t;
struct Case {std::string name;V x,y;I n,statistic,draws,seed;bool exact,poison=false;};
struct R {I status=0,method=0,hits=0,draws=0,seed=0;Q coefficient=0,p=0,se=0,lo=0,hi=0;bool mc=false;};
I normalize(I seed) {I s=seed%2147483647;if(s<=0)s+=2147483646;if(s==0)s=2147483646;return s;}
I random_index(I&s,I width) {
 I limit=(2147483646/width)*width;
 for(;;){s=(16807*s)%2147483647;I d=s-1;if(d<limit)return d%width;}
}
R eval(const Case&c,bool omit_rejection=false) {
 R r;
 if(c.statistic!=1&&c.statistic!=2){r.status=7;return r;}
 if(c.n<2||c.n>256){r.status=1;return r;}
 if(c.exact&&c.n>8){r.status=6;return r;}
 if(!c.exact&&(c.draws<1||c.draws>100000)){r.status=5;return r;}
 for(I i=0;i<c.n;++i)if(!std::isfinite(c.x[i])||!std::isfinite(c.y[i])){r.status=2;return r;}
 std::vector<Q>x(c.n),y(c.n);
 for(I i=0;i<c.n;++i) {
  if(c.statistic==2) {
   I xl=0,xe=0,yl=0,ye=0;
   for(I j=0;j<c.n;++j){xl+=c.x[j]<c.x[i];xe+=c.x[j]==c.x[i];yl+=c.y[j]<c.y[i];ye+=c.y[j]==c.y[i];}
   x[i]=(Q)xl+(Q)(xe+1)/2-(Q)(c.n+1)/2;y[i]=(Q)yl+(Q)(ye+1)/2-(Q)(c.n+1)/2;
  }else{x[i]=c.x[i];y[i]=c.y[i];}
 }
 if(c.statistic==1){Q mx=0,my=0;for(I i=0;i<c.n;++i){mx+=x[i];my+=y[i];}mx/=c.n;my/=c.n;for(I i=0;i<c.n;++i){x[i]-=mx;y[i]-=my;}}
 Q xx=0,yy=0,observed=0;
 for(I i=0;i<c.n;++i){xx+=x[i]*x[i];yy+=y[i]*y[i];observed+=x[i]*y[i];}
 if(xx==0||yy==0){r.status=3;return r;}
 r.coefficient=observed/sqrtq(xx*yy);
 Q threshold=fabsq(observed);if(c.statistic==1)threshold*=1-(Q)1.4210854715202004e-14;
 std::vector<I>order(c.n);std::iota(order.begin(),order.end(),0);
 auto hit=[&](){Q dot=0;for(I i=0;i<c.n;++i)dot+=x[i]*y[order[i]];return fabsq(dot)>=threshold;};
 if(c.exact) {
  do{r.hits+=hit();++r.draws;}while(std::next_permutation(order.begin(),order.end()));
  r.method=2;r.p=(Q)r.hits/r.draws;return r;
 }
 r.method=1;r.draws=c.draws;r.seed=normalize(c.seed);I state=r.seed;
 for(I b=0;b<c.draws;++b){
  std::iota(order.begin(),order.end(),0);
  for(I j=c.n-1;j>0;--j){I k=0;if(omit_rejection){state=(16807*state)%2147483647;k=(state-1)%(j+1);}else{k=random_index(state,j+1);}std::swap(order[j],order[k]);}
  r.hits+=hit();
 }
 r.p=(Q)(r.hits+1)/(r.draws+1);r.mc=true;
 Q q=(Q)r.hits/r.draws,B=r.draws,z=(Q)1.959963984540054,z2=z*z;
 r.se=sqrtq(q*(1-q)/B);
 Q denom=1+z2/B,center=(q+z2/(2*B))/denom;
 Q half=z/denom*sqrtq(q*(1-q)/B+z2/(4*B*B));
 r.lo=r.hits==0?0:center-half;r.hi=r.hits==r.draws?1:center+half;
 return r;
}
I rejection_seed() {
 // Modular inverse of 16807; force first state=m-1, rejected at width=256.
 I a=16807,b=2147483647,u=1,v=0;
 while(b){I q=a/b;I t=a-q*b;a=b;b=t;t=u-q*v;u=v;v=t;}
 u%=2147483647;if(u<0)u+=2147483647;
 return 2147483647-u;
}
std::vector<Case> cases(){
 std::vector<Case> c;
 auto exact=[&](std::string name,V x,V y,I stat){I n=x.size();c.push_back({name,x,y,n,stat,-99,std::numeric_limits<I>::min(),true});};
 exact("n2_pearson",{1,2},{3,1},1);exact("n2_spearman",{1,2},{3,1},2);
 exact("n3_perfect",{0,1,2},{0,1,2},1);exact("n3_spearman",{0,1,2},{0,2,1},2);
 exact("n4_tied_pearson",{0,0,1,2},{3,1,1,0},1);exact("n4_tied_spearman",{0,0,1,2},{3,1,1,0},2);
 exact("n4_tied_multiplicity",{0,0,1,1},{0,0,1,1},2);
 exact("n4_tied_swapped",{3,1,1,0},{0,0,1,2},2);
 exact("n8_perfect",{0,1,2,3,4,5,6,7},{7,6,5,4,3,2,1,0},1);
 exact("n8_tied",{0,0,1,1,2,2,3,3},{0,1,0,1,2,3,2,3},2);
 double mx=std::numeric_limits<double>::max(),mn=std::numeric_limits<double>::denorm_min();
 exact("extreme",{ -mx,-mx/2,mx/2,mx},{0,2,1,4},1);
 V offset(4);double z=1e300;for(int j=0;j<4;++j){offset[j]=z;z=std::nextafter(z,std::numeric_limits<double>::infinity());}
 exact("huge_offset",offset,{1,3,2,4},1);
 exact("subnormal",{mn,2*mn,4*mn,8*mn},{0,3*mn,2*mn,9*mn},1);
 exact("inactive_poison",{1,2,3},{3,1,4},2);c.back().poison=true;
 for(I seed:{(I)0,(I)-1,std::numeric_limits<I>::min(),std::numeric_limits<I>::max(),(I)1234,(I)-2147483646,(I)-4294967293})
  c.push_back({"mc_seed_"+std::to_string(seed),{0,0,1,2,2},{3,1,1,0,0},5,2,128,seed,false});
 c.push_back({"mc_one_draw",{0,1,2,3},{2,0,3,1},4,1,1,19,false});
 c.push_back({"mc_max_budget",{1,2},{3,1},2,1,100000,0,false});
 V a(256),b(256);for(int j=0;j<256;++j){a[j]=j;b[j]=(j*73)%257;}
 c.push_back({"mc_n256",a,b,256,2,16,1234,false});
 c.push_back({"mc_forced_rejection",a,b,256,1,32,rejection_seed(),false});
 c.push_back({"mc_zero_hits",{0,1,2,3,4,5,6,7},{0,1,2,3,4,5,6,7},8,1,1,19,false});
 V nine(9);std::iota(nine.begin(),nine.end(),0);
 c.push_back({"exact_n9_rejected",nine,nine,9,1,0,0,true});
 c.push_back({"invalid_stat",{1,2},{2,1},2,3,0,0,true});
 c.push_back({"invalid_budget",{1,2},{2,1},2,1,0,0,false});
 c.push_back({"constant",{1,1},{2,3},2,2,0,0,true});
 c.push_back({"nonfinite",{1,std::numeric_limits<double>::infinity()},{2,3},2,1,0,0,true});
 return c;
}
std::string integer(I v){if(v==std::numeric_limits<I>::min())return "(-9223372036854775807 - 1)";return std::to_string(v);}
std::string lit(double v) {
 if(std::isnan(v))return "(0.0 / 0.0)";
 if(std::isinf(v))return "(1.0e308 * 2.0)";
 std::ostringstream o;o<<std::scientific<<std::setprecision(17);
 if(std::fpclassify(v)==FP_SUBNORMAL)o<<"("<<std::ldexp(v,52)<<" / 4503599627370496.0)";else o<<v;return o.str();
}
void array(std::string name,const V&a) {
 std::cout<<" var "<<name<<":[f64;256]=[0.0;256]\n";
 // Compress exactly representable unit grids used by n=256 cases.
 bool grid=a.size()>=16;
 for(size_t j=0;j<a.size();++j)grid&=a[j]==(double)j;
 if(grid){std::cout<<" var i"<<name<<"=0\n while i"<<name<<"<"<<a.size()<<" { "<<name<<"[i"<<name<<" as usize]=i"<<name<<" as f64 i"<<name<<"=i"<<name<<"+1 }\n";return;}
 for(size_t j=0;j<a.size();++j)if(a[j]!=0)std::cout<<" "<<name<<"["<<j<<"]="<<lit(a[j])<<"\n";
}
int main(int argc,char**argv) {
 bool f=argc>1&&std::string(argv[1])=="--fixture";auto all=cases();
 std::cout<<std::scientific<<std::setprecision(17);
 if(f)std::cout<<"//@ run-pass\n//@ expect-stdout: STATS_CORRELATION_PERMUTATION_REFERENCE_PASS\n// C++23 independent raw moments/ranks and labelled permutation oracle.\nuse stats::correlation_permutation::{correlation_permutation}\nfn close(a:f64,b:f64)->bool {let d=if a>b{a-b}else{b-a} let v=if b<0.0{-b}else{b} return d<=1.0e-13+1.0e-11*v}\n";
 for(size_t j=0;j<all.size();++j){const auto&c=all[j];auto r=eval(c);
  if(!f){std::cout<<j<<"\t"<<c.name<<"\t"<<r.status<<"\t"<<r.hits<<"\t"<<r.draws<<"\t"<<r.seed<<"\t"<<(double)r.coefficient<<"\t"<<(double)r.p<<"\t"<<(double)r.se<<"\t"<<(double)r.lo<<"\t"<<(double)r.hi<<"\n";continue;}
  std::cout<<"// "<<c.name<<"\nfn case_"<<j<<"() with Mut,Div,Panic {\n";array("x",c.x);array("y",c.y);if(c.poison)std::cout<<" x[100]=(0.0/0.0)\n y[255]=(1.0e308*2.0)\n";
  std::cout<<" let r=correlation_permutation(&x,&y,"<<c.n<<","<<c.statistic<<","<<c.draws<<","<<integer(c.seed)<<","<<(c.exact?"true":"false")<<")\n assert(r.status=="<<r.status<<")\n assert(r.statistic=="<<c.statistic<<")\n assert(r.n=="<<c.n<<")\n assert(r.method=="<<r.method<<")\n assert(r.hits=="<<r.hits<<")\n assert(r.draws=="<<r.draws<<")\n assert(r.seed_used=="<<r.seed<<")\n assert("<<(r.mc?"":"!")<<"r.mc_available)\n assert(!r.exchangeability_assessed)\n";
  for(auto [name,val]:{std::pair{"coefficient",r.coefficient},std::pair{"p_value",r.p},std::pair{"mc_se",r.se},std::pair{"mc_lo",r.lo},std::pair{"mc_hi",r.hi}}){
   if(val==0 && (r.status || (!r.mc && std::string(name)!="coefficient") || std::string(name)=="mc_se" || std::string(name)=="p_value" || std::string(name)=="mc_hi"))std::cout<<" assert(r."<<name<<"==0.0)\n";
   else std::cout<<" assert(close(r."<<name<<","<<lit((double)val)<<"))\n";
  }
  std::cout<<"}\n";
 }
 if(f){std::cout<<"fn main() with IO,Mut,Div,Panic {\n";for(size_t j=0;j<all.size();++j)std::cout<<" case_"<<j<<"()\n";std::cout<<" println(\"STATS_CORRELATION_PERMUTATION_REFERENCE_PASS\")\n}\n";}
 auto n2=eval(all[0]),ties=eval(all[6]),n8=eval(all[8]);
 if(n2.hits!=2||n2.draws!=2||ties.hits!=8||ties.draws!=24||n8.hits!=2||n8.draws!=40320)return 2;
 I state=rejection_seed();state=(16807*state)%2147483647;if(state!=2147483646)return 3;
 if(normalize(-2147483646)!=2147483646||normalize(-4294967293)!=2147483646)return 4;
 // The forced first rejection must expose a modulo-only faulty RNG.
 if(eval(all[24]).hits==eval(all[24],true).hits)return 5;
}
