// Independent C++23 balanced-design cluster sandwich/SVG reference.
// Raw two-column OLS and cluster scores; no production QR/influence reuse.
// Numerical execution is coordinated remotely by the root agent.
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
using Q=long double;
[[noreturn]] void fail(const std::string&s){throw std::runtime_error(s);}
struct Mark {int x,y,r,g,b;};
int pixel(Q value,Q maximum){
 // A physical-unit fit/covariance budget is propagated before truncation.
 // The selected239-pixel height keeps both bounds in one pixel bin.
 Q budget=1e-9L*(1+std::abs(value));
 Q lo=239*(value-budget)/maximum,hi=239*(value+budget)/maximum;
 int a=int(std::floor(lo)),b=int(std::floor(hi));
 if(a!=b)fail("oracle pixel margin is ambiguous");
 return 389-a;
}
int main(int argc,char**argv){
 try{
  if(argc!=2)fail("usage: cluster_covariance_svg_reference SVG_OR_NATIVE_LOG");
  const std::array<Q,8>x{-1,-1,1,1,-1,-1,1,1};
  const std::array<Q,8>y{8,10,12,14,9,11,13,15};
  Q s0=0,sx=0,sxx=0,sy=0,sxy=0;
  for(int i=0;i<8;i++){s0+=1;sx+=x[i];sxx+=x[i]*x[i];sy+=y[i];sxy+=x[i]*y[i];}
  Q det=s0*sxx-sx*sx;if(det<=0)fail("reference rank");
  std::array<Q,2> beta{(sxx*sy-sx*sxy)/det,(s0*sxy-sx*sy)/det};
  std::map<int,std::array<Q,2>> scores;
  for(int i=0;i<8;i++){
   Q residual=y[i]-beta[0]-beta[1]*x[i];
   auto& score=scores[i/2];score[0]+=residual;score[1]+=x[i]*residual;
  }
  std::array<Q,4> covariance{};
  for(const auto&[id,score]:scores){
   (void)id;
   std::array<Q,2> v{(sxx*score[0]-sx*score[1])/det,(s0*score[1]-sx*score[0])/det};
   for(int j=0;j<2;j++)for(int k=0;k<2;k++)covariance[j*2+k]+=v[j]*v[k];
  }
  Q groups=Q(scores.size()),factor=groups/(groups-1)*(s0-1)/(s0-2);
  if(scores.size()!=4||beta[0]!=11.5L||beta[1]!=2||
     covariance[0]!=.0625L||covariance[3]!=.0625L||covariance[1]!=0||covariance[2]!=0)
      fail("independent balanced reference");
  std::array<Q,2> se0{std::sqrt(covariance[0]),std::sqrt(covariance[3])};
  std::array<Q,2> se1{std::sqrt(factor*covariance[0]),std::sqrt(factor*covariance[3])};
  std::ifstream in(argv[1]);if(!in)fail("cannot open SVG/log");
  std::string svg((std::istreambuf_iterator<char>(in)),{});
  auto start=svg.find("<svg "),end=svg.find("</svg>",start);
  if(start==svg.npos||end==svg.npos)fail("SVG bounds");
  svg=svg.substr(start,end+6-start);
  for(const std::string text:{
   "OLS coefficients and cluster covariance standard errors",
   "Synthetic clustered calibration; n=8, G=4, p=2",
   "Coefficient estimate (mV)","Standard-error magnitude (mV)",
   "No confidence intervals; assumptions unassessed",
   "CR1S factor=G/(G-1)*(n-1)/(n-p); x dimensionless",
   "index: 1=intercept, 2=slope",
   "CR0 standard error","CR1S (Stata-style) standard error"})
    if(svg.find(text)==svg.npos)fail("missing label: "+text);
  if(svg.find("data-interval-kind=")!=svg.npos)fail("unexpected inferential interval metadata");
  std::vector<Mark> expected;
  for(int j=0;j<2;j++)expected.push_back({140+480*(j+1)/3,pixel(beta[j],16),220,135,225});
  for(int j=0;j<2;j++)expected.push_back({800+480*(j+1)/3,pixel(se0[j],1),70,180,235});
  for(int j=0;j<2;j++)expected.push_back({800+480*(j+1)/3,pixel(se1[j],1),245,170,65});
  std::regex circle(R"rx(<circle cx="(-?[0-9]+)" cy="(-?[0-9]+)" r="([0-9]+)" fill="rgb\(([0-9]+),([0-9]+),([0-9]+)\)"/>)rx");
  std::size_t count=0;
  for(auto i=std::sregex_iterator(svg.begin(),svg.end(),circle);i!=std::sregex_iterator();++i){
   if(count>=expected.size())fail("excess circle");
   auto m=*i;const auto&e=expected[count];
   if(std::stoi(m[1])!=e.x||std::stoi(m[2])!=e.y||std::stoi(m[3])!=3||
      std::stoi(m[4])!=e.r||std::stoi(m[5])!=e.g||std::stoi(m[6])!=e.b)
      fail("circle geometry/color/radius "+std::to_string(count));
   ++count;
  }
  if(count!=6)fail("missing circle");
  std::size_t p=0,all=0;while((p=svg.find("<circle",p))!=svg.npos){++all;p+=7;}
  if(all!=6)fail("unexpected circle markup/count");
  p=0;all=0;while((p=svg.find("<!-- scatter n=2 -->",p))!=svg.npos){++all;p+=20;}
  if(all!=3)fail("scatter metadata");
  std::cout<<"CLUSTER_COVARIANCE_SVG_REFERENCE_PASS circles="<<count<<" G="<<scores.size()
           <<" factor="<<factor<<" SE_CR0="<<se0[0]<<" SE_CR1S="<<se1[0]<<"\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<"REJECTED: "<<e.what()<<"\n";return 2;}
}
