// Independent C++23 geometry oracle for viz_multivariable_qr.sio.
// Orthogonal raw design gives coefficients by direct projection, independently
// of the Sounio column-pivoted Householder algorithm. No native output is used
// to construct expected coordinates.
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>
struct Circle {long long x,y,r,g,b;};
int main(int argc,char**argv){
 const std::array<long double,8> sensor_a{-1,-1,-1,-1,1,1,1,1};
 const std::array<long double,8> sensor_b{-1.5L,-.5L,.5L,1.5L,-1.5L,-.5L,.5L,1.5L};
 const std::array<long double,8> response{3.7L,6.4L,9.6L,12.3L,7.4L,10.7L,13.3L,16.6L};
 long double sum=0,cross_a=0,cross_b=0,norm_a=0,norm_b=0;
 for(unsigned i=0;i<8;++i){
  sum+=response[i];cross_a+=sensor_a[i]*response[i];cross_b+=sensor_b[i]*response[i];
  norm_a+=sensor_a[i]*sensor_a[i];norm_b+=sensor_b[i]*sensor_b[i];
 }
 const std::array<long double,3> beta{sum/8,cross_a/norm_a,cross_b/norm_b};
 std::array<long double,8> fitted{},residual{};
 long double ss=0;
 for(unsigned i=0;i<8;++i){
  fitted[i]=beta[0]+beta[1]*sensor_a[i]+beta[2]*sensor_b[i];
  residual[i]=response[i]-fitted[i];ss+=residual[i]*residual[i];
 }
 std::cout<<std::setprecision(21)<<"BETA "<<beta[0]<<" "<<beta[1]<<" "<<beta[2]
          <<"\nRESIDUAL_NORM "<<std::sqrt(ss)<<"\n";
 // Match the documented binary64 display transform and truncation into pixels.
 const auto px=[](long long origin,long long extent,double value,double lo,double hi){
  return origin+static_cast<long long>(((value-lo)/(hi-lo))*static_cast<double>(extent));
 };
 const auto py=[](long long origin,long long extent,double value,double lo,double hi){
  return origin+extent-static_cast<long long>(((value-lo)/(hi-lo))*static_cast<double>(extent));
 };
 std::vector<Circle> expected;
 for(unsigned i=0;i<8;++i)
  expected.push_back({px(140,480,static_cast<double>(response[i]),0,20),
                      py(150,240,static_cast<double>(fitted[i]),0,20),70,180,235});
 for(unsigned i=0;i<8;++i)
  expected.push_back({px(800,480,static_cast<double>(i+1),0,8),
                      py(150,240,static_cast<double>(residual[i]),-1,1),245,170,65});
 for(const auto& c:expected)
  std::cout<<"EXPECTED_CIRCLE "<<c.x<<" "<<c.y<<" "<<c.r<<" "<<c.g<<" "<<c.b<<"\n";
 if(argc==1){std::cout<<"LEAST_SQUARES_SVG_REFERENCE_READY\n";return 0;}
 if(argc!=2){std::cerr<<"Usage: reference [SVG-or-native-log]\n";return 2;}
 std::ifstream file(argv[1]);
 if(!file){std::cerr<<"Cannot read SVG artifact\n";return 2;}
 const std::string text{std::istreambuf_iterator<char>(file),{}};
 const std::regex circle(R"rx(<circle cx="(-?[0-9]+)" cy="(-?[0-9]+)" r="3" fill="rgb\(([0-9]+),([0-9]+),([0-9]+)\)"/>)rx");
 const std::regex any_circle(R"rx(<circle\b)rx");
 const auto all_circles=std::distance(std::sregex_iterator(text.begin(),text.end(),any_circle),std::sregex_iterator());
 if(all_circles!=16){std::cerr<<"Wrong total circle element count: "<<all_circles<<"\\n";return 1;}
 std::vector<Circle> actual;
 for(std::sregex_iterator i(text.begin(),text.end(),circle),end;i!=end;++i)
  actual.push_back({std::stoll((*i)[1]),std::stoll((*i)[2]),std::stoll((*i)[3]),std::stoll((*i)[4]),std::stoll((*i)[5])});
 if(actual.size()!=expected.size()){
  std::cerr<<"Wrong circle count: "<<actual.size()<<" expected "<<expected.size()<<"\n";return 1;
 }
 for(unsigned i=0;i<expected.size();++i){
  const auto a=actual[i],e=expected[i];
  if(a.x!=e.x||a.y!=e.y||a.r!=e.r||a.g!=e.g||a.b!=e.b){
   std::cerr<<"Circle mismatch at observation "<<i<<" actual("<<a.x<<","<<a.y
            <<") expected("<<e.x<<","<<e.y<<")\n";return 1;
  }
 }
 std::size_t at=0,count=0;
 while((at=text.find("<!-- scatter n=8 -->",at))!=std::string::npos){++count;++at;}
 if(count!=2){std::cerr<<"Expected two eight-observation scatter metadata records\n";return 1;}
 const std::array<std::string,6> labels{"observed response (mV)","fitted response (mV)",
   "residual (mV)","beta0 (mV)=1.00000e+1","beta1 (mV)=2.00000e+0","beta2 (mV)=2.96000e+0"};
 for(const auto& label:labels)if(text.find(label)==std::string::npos){
  std::cerr<<"Missing label "<<label<<"\n";return 1;
 }
 std::cout<<"LEAST_SQUARES_SVG_REFERENCE_PASS circles=16 panels=2\n";
}
