// C++23 independent integer/rational footprint oracle and actual SVG checker.
// Build on coordinator's Linux execution surface, never as a Sounio compiler.
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
struct Rect {long long x,y,w,h; int part,r,g,b;};
Rect clip(Rect z) {
 const auto endx=std::min(z.x+z.w,41LL);
 const auto endy=std::min(z.y+z.h,41LL);
 z.x=std::max(z.x,20LL);z.y=std::max(z.y,20LL);
 z.w=endx-z.x;z.h=endy-z.y;
 return z;
}
std::vector<Rect> reference() {
 std::vector<Rect> result;
 // Independent endpoint projection: exact integer count coordinates [0,10].
 for(int row=0;row<2;++row) {
  const long long center=23+row*7;
  const long long point=20+(row*10*20)/10;
  result.push_back(clip({point,center,1,1,0,180,180,220}));
  result.push_back(clip({point,center-2,1,5,1,180,180,220}));
  result.push_back(clip({point,center-2,1,5,2,180,180,220}));
  result.push_back(clip({point-3,center-3,7,7,3,70,180,235}));
 }
 return result;
}

std::string legacy_span_marker(int which) {
 const long long x=which==0?10:20,y=which==0?10:20;
 const long long w=which==0?120:100,h=which==0?60:40,n=which==0?1:2;
 const long long denominator=which==0?1:10,lower=0,upper=which==0?1:6;
 const long long step=std::clamp(h/(n+1),7LL,22LL);
 const long long center=y+step/2;
 const long long left=x+lower*w/denominator,right=x+upper*w/denominator;
 const long long clipped_left=std::max(x,left),clipped_right=std::min(x+w+1,right+1);
 std::ostringstream out;
 out<<"<rect data-forest-part=\"0\" x=\""<<std::fixed<<std::setprecision(6)
    <<static_cast<long double>(clipped_left)-0.5L<<"\" y=\""
    <<static_cast<long double>(center)-0.5L<<"\" width=\""<<clipped_right-clipped_left
    <<"\" height=\"1\" fill=\"rgb(180,180,220)\"/>";
 return out.str();
}
std::string legacy_escape_quotes(const std::string& value) {
 std::string out;for(char c:value){if(c=='"')out+='\\';out+=c;}return out;
}

int main(int argc,char**argv) {
 try {
  if(argc==2&&std::string(argv[1])=="--legacy-markers"){
   for(int which=0;which<2;++which)std::cout<<legacy_span_marker(which)<<"\n";
   return 0;
  }
  if(argc==4&&std::string(argv[1])=="--legacy-check"){
   for(int which=0;which<2;++which){
    std::ifstream input(argv[which+2],std::ios::binary);
    if(!input)throw std::runtime_error("legacy stdout unavailable");
    std::string actual((std::istreambuf_iterator<char>(input)),{}),expected=legacy_span_marker(which);
    const bool normal=actual.find(expected)!=std::string::npos;
    const bool escaped=actual.find(legacy_escape_quotes(expected))!=std::string::npos;
    if(!normal&&!escaped)throw std::runtime_error("legacy interval pixel-cell geometry");
    std::cout<<"LEGACY_FOREST_SPAN_PASS case="<<which
       <<" serialization="<<(normal?"literal-quotes":"stage2-escaped-quotes")<<"\n";
   }
   return 0;
  }
  const auto expected=reference();
  for(const auto& z:expected) {
   if(z.x<20||z.y<20||z.x+z.w>41||z.y+z.h>41||z.w<=0||z.h<=0)
    throw std::runtime_error("invalid independent footprint");
  }
  if(argc==1) {
   for(const auto&z:expected)std::cout<<z.part<<'\t'<<z.x<<'\t'<<z.y<<'\t'<<z.w<<'\t'<<z.h<<'\n';
   std::cout<<"DEEP_VIZ_FOREST_REFERENCE_PASS\n";
   return 0;
  }
  if(argc!=2)throw std::runtime_error("usage: deep_viz_forest [actual-native-svg-or-stdout]");
  std::ifstream input(argv[1],std::ios::binary);
  if(!input)throw std::runtime_error("input unavailable");
  std::string text((std::istreambuf_iterator<char>(input)),{});
  auto first=text.find("<svg"),last=text.find("</svg>",first);
  if(first==std::string::npos||last==std::string::npos)throw std::runtime_error("missing actual SVG");
  text=text.substr(first,last+6-first);
  if(text.find("<!-- forest n=2 -->")==std::string::npos||
     text.find("data-interval-kind=\"confidence\"")==std::string::npos)
    throw std::runtime_error("forest metadata missing");
  const std::regex mark(R"re(<rect data-forest-part="([0-3])" x="([^"]+)" y="([^"]+)" width="([0-9]+)" height="([0-9]+)" fill="rgb\(([0-9]+),([0-9]+),([0-9]+)\)"/>)re");
  std::vector<std::array<long double,8>> actual;
  for(std::sregex_iterator it(text.begin(),text.end(),mark),end;it!=end;++it) {
   const auto&m=*it;
   actual.push_back({std::stold(m[1]),std::stold(m[2]),std::stold(m[3]),
    std::stold(m[4]),std::stold(m[5]),std::stold(m[6]),std::stold(m[7]),std::stold(m[8])});
  }
  if(actual.size()!=expected.size())throw std::runtime_error("forest primitive count");
  for(std::size_t i=0;i<expected.size();++i) {
   const auto&z=expected[i];
   const std::array<long double,8> want{static_cast<long double>(z.part),
    static_cast<long double>(z.x)-0.5L,static_cast<long double>(z.y)-0.5L,
    static_cast<long double>(z.w),static_cast<long double>(z.h),
    static_cast<long double>(z.r),static_cast<long double>(z.g),static_cast<long double>(z.b)};
   if(actual[i]!=want)throw std::runtime_error("forest footprint/color "+std::to_string(i));
  }
  std::cout<<"DEEP_VIZ_FOREST_SVG_PASS eight exact clipped primitives\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<"REJECTED: "<<e.what()<<'\n';return 2;}
}
