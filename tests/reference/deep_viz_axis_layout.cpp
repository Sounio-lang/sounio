// C++23 independent exact 8px-cell geometry oracle for the native fixture.
// Usage: deep_viz_axis_layout actual-native-stdout.txt
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
struct Label {long long x,y,w;std::string text;};
int main(int argc,char**argv) {
 try {
  if(argc!=2)throw std::runtime_error("usage: deep_viz_axis_layout actual-native-stdout");
  std::ifstream in(argv[1],std::ios::binary);
  if(!in)throw std::runtime_error("input unavailable");
  const std::string source((std::istreambuf_iterator<char>(in)),{});
  const std::array<std::string,3> strings{"-1.00000e+0","0","1.00000e+0"};
  std::array<Label,3> expected;
  for(std::size_t i=0;i<strings.size();++i) {
   const long long width=8*static_cast<long long>(strings[i].size());
   const long long tick=100+(static_cast<long long>(i)*160)/2;
   expected[i]={tick-width/2,48,width,strings[i]};
  }
  const std::regex text(R"re(<text data-axis-label="([0-9]+)" x="(-?[0-9]+)" y="(-?[0-9]+)" textLength="([0-9]+)" lengthAdjust="spacingAndGlyphs" font-family="monospace" font-size="8" fill="rgb\(70,180,235\)">([^<]+)</text>)re");
  std::size_t count=0;
  for(std::sregex_iterator it(source.begin(),source.end(),text),end;it!=end;++it) {
   if(count>=expected.size())throw std::runtime_error("extra SVG label");
   const auto&m=*it;const auto&e=expected[count];
   if(std::stoull(m[1])!=count||std::stoll(m[2])!=e.x||
      std::stoll(m[3])!=e.y+7||std::stoll(m[4])!=e.w||m[5].str()!=e.text)
    throw std::runtime_error("SVG label geometry/content "+std::to_string(count));
   ++count;
  }
  if(count!=expected.size())throw std::runtime_error("SVG label count");
  const std::regex pixel(R"(AXIS_PIXEL (-?[0-9]+) (-?[0-9]+) (-?[0-9]+))");
  std::array<bool,3> painted{};
  std::set<std::pair<long long,long long>> seen;
  for(std::sregex_iterator it(source.begin(),source.end(),pixel),end;it!=end;++it) {
   const auto&m=*it;
   const auto x=std::stoll(m[1]),y=std::stoll(m[2]),color=std::stoll(m[3]);
   if(color!=((70LL<<16)|(180LL<<8)|235LL))throw std::runtime_error("Canvas pixel color");
   if(!seen.emplace(x,y).second)throw std::runtime_error("duplicate Canvas pixel");
   bool inside=false;
   for(std::size_t i=0;i<expected.size();++i) {
    const auto&e=expected[i];
    if(x>=e.x&&x<e.x+e.w&&y>=e.y&&y<e.y+8) {inside=true;painted[i]=true;}
   }
   if(!inside)throw std::runtime_error("Canvas paint outside independent label cells");
  }
  for(bool p:painted)if(!p)throw std::runtime_error("empty Canvas label");
  if(source.find("STDLIB_DEEP_VIZ_AXIS_LAYOUT_PASS")==std::string::npos)
   throw std::runtime_error("native completion marker missing");
  std::cout<<"DEEP_VIZ_AXIS_LAYOUT_PASS three exact SVG labels and native Canvas cell containment\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<"REJECTED: "<<e.what()<<'\n';return 2;}
}
