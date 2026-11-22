// C++23 independent public-IR/Canvas/SVG fitted-axis geometry oracle.
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
struct Label {long long x,y,w;std::string text;};
std::string attr(const std::string&s,const std::string&name) {
 std::smatch m;
 if(!std::regex_search(s,m,std::regex("(?:^| )"+name+"=\"([^\"]*)\"")))
  throw std::runtime_error("missing attribute "+name);
 return m[1];
}
int main(int argc,char**argv) {
 try {
  if(argc!=2)throw std::runtime_error("usage: deep_viz_fitted_axis actual-native-stdout");
  std::ifstream in(argv[1],std::ios::binary);
  if(!in)throw std::runtime_error("input unavailable");
  const std::string source((std::istreambuf_iterator<char>(in)),{});
  const auto start=source.find("<svg"),end=source.find("</svg>",start);
  if(start==std::string::npos||end==std::string::npos)throw std::runtime_error("missing SVG");
  const auto svg=source.substr(start,end+6-start);
  if(svg.find("data-axis-layout=\"fitted\"")==std::string::npos||
     svg.find("data-axis-intervals=\"2\"")==std::string::npos)throw std::runtime_error("fitted metadata");
  const std::array<std::string,3> strings{"-1.00000e-8","0","1.00000e-8"};
  std::array<Label,3> expected;
  for(std::size_t i=0;i<strings.size();++i) {
   const auto width=8*static_cast<long long>(strings[i].size());
   const auto tick=100+(static_cast<long long>(i)*160)/2;
   expected[i]={tick-width/2,48,width,strings[i]};
  }
  const std::regex text(R"re(<text data-axis-label="([0-9]+)"([^>]*)>([^<]*)</text>)re");
  std::size_t count=0;
  for(std::sregex_iterator it(svg.begin(),svg.end(),text),last;it!=last;++it) {
   if(count>=expected.size())throw std::runtime_error("extra label");
   const auto&m=*it;const auto&e=expected[count];const auto attrs=m[2].str();
   if(std::stoull(m[1])!=count||std::stoll(attr(attrs,"x"))!=e.x||
      std::stoll(attr(attrs,"y"))!=e.y+7||std::stoll(attr(attrs,"textLength"))!=e.w||
      attr(attrs,"fill")!="rgb(70,180,235)"||attr(attrs,"font-size")!="8"||
      attr(attrs,"lengthAdjust")!="spacingAndGlyphs"||m[3].str()!=e.text)
    throw std::runtime_error("label geometry/content "+std::to_string(count));
   ++count;
  }
  if(count!=3)throw std::runtime_error("label count");
  const std::array<std::array<long long,4>,4> lines{{{100,40,260,40},{100,40,100,44},{180,40,180,44},{260,40,260,44}}};
  const std::regex line(R"(<line ([^>]*)/>)");
  count=0;
  for(std::sregex_iterator it(svg.begin(),svg.end(),line),last;it!=last;++it) {
   if(count>=lines.size())throw std::runtime_error("extra axis line");
   const auto attrs=(*it)[1].str();const auto&w=lines[count];
   if(std::stoll(attr(attrs,"x1"))!=w[0]||std::stoll(attr(attrs,"y1"))!=w[1]||
      std::stoll(attr(attrs,"x2"))!=w[2]||std::stoll(attr(attrs,"y2"))!=w[3]||
      attr(attrs,"stroke")!="rgb(70,180,235)")
    throw std::runtime_error("axis line/tick geometry");
   ++count;
  }
  if(count!=4)throw std::runtime_error("axis/tick count");
  const std::regex pixel(R"(AXIS_PIXEL (-?[0-9]+) (-?[0-9]+) (-?[0-9]+))");
  std::array<bool,3> painted{};
  std::set<std::pair<long long,long long>> seen;
  for(std::sregex_iterator it(source.begin(),source.end(),pixel),last;it!=last;++it) {
   const auto&m=*it;const auto x=std::stoll(m[1]),y=std::stoll(m[2]),color=std::stoll(m[3]);
   if(color!=((70LL<<16)|(180LL<<8)|235LL))throw std::runtime_error("Canvas color");
   if(!seen.emplace(x,y).second)throw std::runtime_error("duplicate Canvas pixel");
   bool inside=(x>=100&&x<=260&&y==40)||
    ((x==100||x==180||x==260)&&y>=40&&y<=44);
   for(std::size_t i=0;i<expected.size();++i) {
    const auto&e=expected[i];
    if(x>=e.x&&x<e.x+e.w&&y>=e.y&&y<e.y+8) {inside=true;painted[i]=true;}
   }
   if(!inside)throw std::runtime_error("Canvas outside independent axis/label cells");
  }
  for(bool p:painted)if(!p)throw std::runtime_error("missing Canvas label");
  const auto resize=source.find("<svg",end+6);
  if(resize==std::string::npos)throw std::runtime_error("missing resized SVG");
  const auto resize_end=source.find("</svg>",resize);
  if(resize_end==std::string::npos)throw std::runtime_error("unterminated resized SVG");
  const auto resized=source.substr(resize,resize_end+6-resize);
  if(resized.find("data-axis-layout-status=\"3\"")==std::string::npos||
     resized.find("data-axis-label")!=std::string::npos||resized.find("<line ")!=std::string::npos)
   throw std::runtime_error("resized axis availability gate");
  if(source.find("STDLIB_DEEP_VIZ_FITTED_AXIS_PASS")==std::string::npos)
   throw std::runtime_error("native completion missing");
  std::cout<<"DEEP_VIZ_FITTED_AXIS_PASS shared public IR Canvas/SVG geometry and resize refusal\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<"REJECTED: "<<e.what()<<'\n';return 2;}
}
