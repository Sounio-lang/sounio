// Independent C++23 SVG reference for conditional Gaussian-precision WLS.
// Two-column scalar weighted projections; analytic df=6 Student CDF inverted
// by bisection. No Sounio numerical output supplies expected geometry.
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <regex>
#include <string>
#include <utility>
#include <vector>
long double student6_cdf(long double t){
 const long double u=t/std::sqrt(6+t*t),u2=u*u;
 return .5L+(15.0L/16)*(u-(2.0L/3)*u*u2+(1.0L/5)*u*u2*u2);
}
struct Rect {long long x,y,w,h,r,g,b;};
struct Line {long long x1,y1,x2,y2,r,g,b;};
int main(int argc,char**argv){
 const std::array<double,8> raw{1.2,2.7,5.4,6.6,9.6,10.4,13.5,14.5};
 long double sw=0,sx=0,sy=0,sxx=0,sxy=0,response_ss=0;
 for(unsigned i=0;i<8;++i){
  const long double w=i%2==0?1:4,x=i,y=raw[i];
  sw+=w;sx+=w*x;sy+=w*y;sxx+=w*x*x;sxy+=w*x*y;response_ss+=y*y;
 }
 const long double det=sw*sxx-sx*sx;
 const long double slope=(sw*sxy-sx*sy)/det,intercept=(sy-slope*sx)/sw;
 long double weighted_ss=0;
 for(unsigned i=0;i<8;++i){const long double e=raw[i]-intercept-slope*i;weighted_ss+=(i%2==0?1:4)*e*e;}
 const long double sigma2=weighted_ss/6;
 const std::array<long double,2> beta{intercept,slope};
 const std::array<long double,2> se{std::sqrt(sigma2*sxx/det),std::sqrt(sigma2*sw/det)};
 long double left=0,right=16;
 for(unsigned i=0;i<160;++i){const auto mid=(left+right)/2;if(student6_cdf(mid)<.975L)left=mid;else right=mid;}
 const long double critical=(left+right)/2;
 std::cout<<std::setprecision(21)<<"STUDENT6_CRITICAL "<<critical<<"\n";
 std::array<long double,2> lower{},upper{};
 for(unsigned i=0;i<2;++i){
  lower[i]=beta[i]-critical*se[i];upper[i]=beta[i]+critical*se[i];
  std::cout<<"ROW "<<i<<" beta="<<beta[i]<<" SE="<<se[i]<<" CI=["<<lower[i]<<","<<upper[i]<<"]\n";
 }
 // Response-scaled physical comparison budget, propagated before pixel floor.
 const long double physical_budget=3e-11L*std::sqrt(response_ss);
 const auto pixels=[&](long double value,long double budget){
  const long double projected=(value+1)*150;
  const long double rounding=8*std::numeric_limits<double>::epsilon()*(std::fabs(projected)+600+1);
  const long double half=150*budget+rounding;
  const auto low=180+static_cast<long long>(std::floor(projected-half));
  const auto high=180+static_cast<long long>(std::floor(projected+half));
  if(high-low>1)throw std::runtime_error("comparison budget exceeds adjacent pixels");
  std::cout<<"PIXEL_RANGE value="<<value<<" budget_mV="<<budget<<" x=["<<low<<","<<high<<"]\n";
  return std::pair{low,high};
 };
 std::array<std::pair<long long,long long>,2> center{},lo{},hi{};
 for(unsigned i=0;i<2;++i){
  center[i]=pixels(beta[i],physical_budget);
  // Matches the independent inference fixture's absolute CI budget in mV.
  const long double endpoint_budget=3e-8L*se[i];
  lo[i]=pixels(lower[i],endpoint_budget);hi[i]=pixels(upper[i],endpoint_budget);
 }
 if(argc==1){std::cout<<"WEIGHTED_INFERENCE_SVG_REFERENCE_READY\n";return 0;}
 if(argc!=2){std::cerr<<"Usage: reference [SVG-or-native-log]\n";return 2;}
 std::ifstream file(argv[1]);if(!file){std::cerr<<"Cannot read SVG artifact\n";return 2;}
 const std::string text{std::istreambuf_iterator<char>(file),{}};
 const std::regex rect(R"rx(<rect x="(-?[0-9]+)" y="(-?[0-9]+)" width="([0-9]+)" height="([0-9]+)" fill="rgb\(([0-9]+),([0-9]+),([0-9]+)\)"/>)rx");
 const std::regex line(R"rx(<line x1="(-?[0-9]+)" y1="(-?[0-9]+)" x2="(-?[0-9]+)" y2="(-?[0-9]+)" stroke="rgb\(([0-9]+),([0-9]+),([0-9]+)\)" stroke-width="1.5"/>)rx");
 std::vector<Rect> markers,caps;std::vector<Line> intervals;
 for(std::sregex_iterator i(text.begin(),text.end(),rect),end;i!=end;++i){
  Rect r{std::stoll((*i)[1]),std::stoll((*i)[2]),std::stoll((*i)[3]),std::stoll((*i)[4]),std::stoll((*i)[5]),std::stoll((*i)[6]),std::stoll((*i)[7])};
  if(r.w==7&&r.h==7)markers.push_back(r);
  if(r.w==1&&r.h==5)caps.push_back(r);
 }
 for(std::sregex_iterator i(text.begin(),text.end(),line),end;i!=end;++i){
  Line r{std::stoll((*i)[1]),std::stoll((*i)[2]),std::stoll((*i)[3]),std::stoll((*i)[4]),std::stoll((*i)[5]),std::stoll((*i)[6]),std::stoll((*i)[7])};
  if(r.r==180&&r.g==180&&r.b==220)intervals.push_back(r);
 }
 if(markers.size()!=2||caps.size()!=4||intervals.size()!=2){std::cerr<<"Wrong coefficient marker/cap/interval counts\n";return 1;}
 const auto in=[](long long value,const std::pair<long long,long long>& range){return value>=range.first&&value<=range.second;};
 for(unsigned i=0;i<2;++i){
  const long long y=151+22*i;
  if(markers[i].r!=80||markers[i].g!=180||markers[i].b!=240||
     caps[2*i].r!=180||caps[2*i].g!=180||caps[2*i].b!=220||
     caps[2*i+1].r!=180||caps[2*i+1].g!=180||caps[2*i+1].b!=220||
     !in(markers[i].x+3,center[i])||markers[i].y!=y-3||
     !in(caps[2*i].x,lo[i])||caps[2*i].y!=y-2||
     !in(caps[2*i+1].x,hi[i])||caps[2*i+1].y!=y-2||
     !in(intervals[i].x1,lo[i])||!in(intervals[i].x2,hi[i])||
     intervals[i].y1!=y||intervals[i].y2!=y){
   std::cerr<<"Coefficient interval geometry mismatch at row "<<i<<"\n";return 1;
  }
 }
 if(text.find("data-interval-kind=\"confidence\"")==std::string::npos){std::cerr<<"Missing confidence metadata\n";return 1;}
 const std::regex level(R"rx(data-interval-kind="confidence" data-interval-level="([^"]+)")rx");
 std::smatch level_match;
 if(!std::regex_search(text,level_match,level)||std::fabs(std::stold(level_match[1])-.95L)>1e-12L){std::cerr<<"Wrong interval confidence level\n";return 1;}
 const std::array<std::string,5> labels{"Conditional pointwise 95% Student intervals; df=6",
  "Declared Gaussian precision model; not assessed","coefficient (mV)",
  "Independent Gaussian errors; Var(error_i)=sigma^2 / w_i",
  "Unknown sigma; design columns: intercept, dimensionless signal"};
 for(const auto& label:labels)if(text.find(label)==std::string::npos){std::cerr<<"Missing model label "<<label<<"\n";return 1;}
 const std::regex se_label(R"rx(SE \(mV\)=([^<]+)</text>)rx");
 std::vector<long double> shown;
 for(std::sregex_iterator i(text.begin(),text.end(),se_label),end;i!=end;++i)shown.push_back(std::stold((*i)[1]));
 if(shown.size()!=2){std::cerr<<"Missing standard-error labels\n";return 1;}
 for(unsigned i=0;i<2;++i)if(std::fabs(shown[i]-se[i])>5e-6L*se[i]){std::cerr<<"Incorrect formatted standard error\n";return 1;}
 std::cout<<"WEIGHTED_INFERENCE_SVG_REFERENCE_PASS rows=2 conditional_student_df=6\n";
}
