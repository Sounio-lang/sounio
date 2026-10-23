// C++23 independent integer/rational footprint oracle and actual SVG checker.
// Build on coordinator's Linux execution surface, never as a Sounio compiler.
#include <algorithm>
#include <array>
#include <fstream>
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
int main(int argc,char**argv) {
 try {
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
