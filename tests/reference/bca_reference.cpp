#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>
long double phi(long double z){return .5L*std::erfc(-z/std::sqrt(2.L));}
long double inv(long double p){long double lo=-8,hi=8;for(int i=0;i<100;++i){auto m=(lo+hi)/2;if(phi(m)<p)lo=m;else hi=m;}return (lo+hi)/2;}
struct Case{std::vector<double>x;int b;double conf;std::int64_t seed;};
int main(){
 std::cout<<std::scientific<<std::setprecision(17);
 std::cout<<"//@ run-pass\n//@ expect-stdout: STATS_BCA_REFERENCE_PASS\nuse stats::bootstrap::{bootstrap_bca_mean}\nuse stats::bca_normal::{bca_normal_cdf,bca_normal_quantile}\nuse math::pure::{is_finite}\n";
 std::vector<Case> cases={{{1,2,3,4,8,10},512,.95,42},
 {{1e-200,2e-200,3e-200,4e-200,8e-200,10e-200},512,.95,42},
 {{1e200,2e200,3e200,4e200,8e200,10e200},512,.95,42},
 {{1e15+1,1e15+2,1e15+3,1e15+4,1e15+8,1e15+10},512,.95,42},
 {{1e308,1.1e308,1.2e308,1.3e308},500,.95,42},
 {{0,0,0,1,1,10},1024,.9,99},
 {{-3,-2,-1,1,2,3},1024,.95,2147483647},
 {std::vector<double>(256),1024,.95,std::numeric_limits<std::int64_t>::min()}};
 for(int i=0;i<256;++i)cases.back().x[i]=(i%17)-8.;
 cases.push_back({{1,2,3,4,8,10},8,.95,42});
 for(std::size_t id=0;id<cases.size();++id){
  const auto& c=cases[id];int n=c.x.size();long double scale=0,anchor=c.x[0],range=0;
  auto [xmin,xmax]=std::minmax_element(c.x.begin(),c.x.end());range=(long double)*xmax-*xmin;
  for(auto v:c.x)scale=std::max(scale,std::abs((long double)v));
  std::vector<long double> x;for(auto v:c.x)x.push_back(((long double)v-anchor)/scale);
  long double point=std::accumulate(x.begin(),x.end(),0.L)/n,spread=0;
  for(auto v:x)spread=std::max(spread,std::abs(v-point));
  // Independent explicit leave-one-out means.
  std::vector<long double> jack;
  for(int omit=0;omit<n;++omit){long double s=0;for(int i=0;i<n;++i)if(i!=omit)s+=x[i];jack.push_back(s/(n-1));}
  long double jm=std::accumulate(jack.begin(),jack.end(),0.L)/n,num=0,den=0;
  for(auto v:jack){auto d=jm-v;num+=d*d*d;den+=d*d;}
  long double accel=num/(6*den*std::sqrt(den));
  std::int64_t st=c.seed%2147483647LL;if(st<=0)st+=2147483646LL;auto initial=st;
  std::vector<long double> boot;
  for(int b=0;b<c.b;++b){long double s=0;for(int j=0;j<n;++j){std::int64_t draw,limit=(2147483646LL/n)*n;
    do{st=(16807LL*st)%2147483647LL;draw=st-1;}while(draw>=limit);s+=x[draw%n];}boot.push_back(s/n);}
  std::int64_t less=0,equal=0;auto tie=64.L*std::numeric_limits<double>::epsilon()*spread;
  for(auto v:boot){auto d=v-point;if(d < -tie)++less;else if(d<=tie)++equal;}
  long double bp=(less+.5L*equal)/c.b,z0=inv(bp),alpha=(1.L-c.conf)/2,zl=inv(alpha),zu=inv(1-alpha);
  long double dl=1-accel*(z0+zl),du=1-accel*(z0+zu);
  auto pl=phi(z0+(z0+zl)/dl),pu=phi(z0+(z0+zu)/du);
  auto bm=std::accumulate(boot.begin(),boot.end(),0.L)/c.b;long double ss=0;
  for(auto v:boot)ss+=(v-bm)*(v-bm);
  auto se=std::sqrt(ss/(c.b-1))*scale;
  std::sort(boot.begin(),boot.end());
  auto quant=[&](long double p){auto h=(c.b-1)*p;auto k=(std::size_t)h;return boot[k]*(1-(h-k))+boot[std::min(k+1,boot.size()-1)]*(h-k);};
  std::cout<<"fn case_"<<id<<"() with Mut,Div,Panic {\nvar x:[f64;256]=[0.0;256]\n";
  for(int i=0;i<n;++i)std::cout<<"x["<<i<<"]="<<c.x[i]<<"\n";
  std::cout<<"let r=bootstrap_bca_mean(&x,"<<n<<","<<c.b<<","<<c.conf<<",";
  if(c.seed==std::numeric_limits<std::int64_t>::min())std::cout<<"(0-9223372036854775807-1)";else std::cout<<c.seed;
  std::cout<<")\nassert(r.status == 0)\nassert(r.b == "<<c.b<<" && r.seed_used == "<<initial<<")\n";
  std::cout<<"assert(r.less_count == "<<less<<" && r.equal_count == "<<equal<<")\nassert(r.tail_resolved == "<<((pl*c.b>=1&&(1-pu)*c.b>=1)?"true":"false")<<")\n";
  for(auto [name,v]:std::vector<std::pair<const char*,long double>>{{"mean",anchor+point*scale},{"lo",anchor+quant(pl)*scale},{"hi",anchor+quant(pu)*scale}}){
   double expected=(double)v;
   auto ulp=std::abs((long double)std::nextafter(expected,std::numeric_limits<double>::infinity())-expected);
   long double tol=std::max(3e-9L*range,2*ulp);
   std::cout<<"assert(is_finite(r."<<name<<"))\nassert(abs(r."<<name<<" - "<<expected<<") <= "<<(double)tol<<")\n";
  }
  std::cout<<"assert(abs(r.se / "<<(double)se<<" - 1.0) < 3.0e-10)\n";
  for(auto [name,v]:std::vector<std::pair<const char*,long double>>{{"bias_p",bp},{"z0",z0},{"acceleration",accel},{"lower_p",pl},{"upper_p",pu}})
   std::cout<<"assert(abs(r."<<name<<" - "<<(double)v<<") < 3.0e-9)\n";
  std::cout<<"assert(x[0] == "<<c.x[0]<<")\n}\n";
  std::cerr<<std::setprecision(17)<<"case "<<id<<" less="<<less<<" ties="<<equal<<" accel="<<(double)accel<<" z0="<<(double)z0<<" pl="<<(double)pl<<" pu="<<(double)pu<<"\n";
 }
 std::cout<<"fn normal_cases() with Mut,Div,Panic {\n";
 for(int i=-12;i<=12;++i){double z=i*.5;std::cout<<"assert(bca_normal_cdf("<<z<<").status == 0)\nassert(abs(bca_normal_cdf("<<z<<").value - "<<(double)phi(z)<<") < 2.0e-14)\n";}
 for(double p:{1e-6,1e-4,.001,.025,.25,.5,.75,.975,.999,.9999,.999999})
  std::cout<<"assert(bca_normal_quantile("<<p<<").status == 0)\nassert(abs(bca_normal_quantile("<<p<<").value - "<<(double)inv(p)<<") < 3.0e-9)\n";
 std::cout<<"}\n";
 int bias_seed=0;
 for(int seed=1;seed<10000&&!bias_seed;++seed){
  std::int64_t st=seed;long double means[2]={};
  long double data[4]={0,0,1,10};
  for(int b=0;b<2;++b){for(int j=0;j<4;++j){st=(16807LL*st)%2147483647LL;means[b]+=data[(st-1)%4];}means[b]/=4;}
  if(means[0]!=means[1] && ((means[0]<2.75L&&means[1]<2.75L)||(means[0]>2.75L&&means[1]>2.75L)))bias_seed=seed;
 }
 if(!bias_seed)return 2;
 std::cerr<<"extreme-bias nondegenerate two-replicate seed="<<bias_seed<<"\n";
 std::cout<<"fn bias_edge() with Mut,Div,Panic {\nvar x:[f64;256]=[0.0;256]\nx[2]=1.0\nx[3]=10.0\nassert(bootstrap_bca_mean(&x,4,2,0.95,"<<bias_seed<<").status == 6)\n}\n";
 std::cout<<"fn abs(v:f64)->f64 {if v<0.0 {0.0-v}else{v}}\nfn main() with IO,Mut,Div,Panic {\nnormal_cases()\nbias_edge()\n";
 for(std::size_t i=0;i<cases.size();++i)std::cout<<"case_"<<i<<"()\n";
 std::cout<<"println(\"STATS_BCA_REFERENCE_PASS\")\n0\n}\n";
}
