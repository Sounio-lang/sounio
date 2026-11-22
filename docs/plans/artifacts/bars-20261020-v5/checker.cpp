// Independent C++23 raw OLS and signed-bar SVG rectangle reference.
// Does not reuse production BarRect or bar projection/width helpers.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
using Q=long double;
struct Rect {Q x,y;int w,h,r,g,b;};
[[noreturn]] void fail(const std::string&s){throw std::runtime_error(s);}
std::size_t occurrences(const std::string&s,const std::string&t){
 std::size_t n=0,p=0;while((p=s.find(t,p))!=s.npos){++n;p+=t.size();}return n;
}
Q attribute(const std::string&tag,const std::string&name){
 std::regex re(name+"=\"([^\"]+)\"");std::smatch m;
 if(!std::regex_search(tag,m,re))fail("missing rectangle attribute "+name);
 std::size_t used=0;Q v=std::stold(m[1],&used);
 if(used!=static_cast<std::size_t>(m[1].length())||!std::isfinite(v))fail("nonfinite rectangle");
 return v;
}
std::vector<Rect> reference(const std::array<Q,4>&x,const std::array<Q,4>&y,
 int ox,int oy,int width,int height,Q xmin,Q xmax,Q ymin,Q ymax,int r,int g,int b){
 auto project_x=[&](Q value){return ox+int((value-xmin)/(xmax-xmin)*width);};
 auto project_y=[&](Q value){return oy+height-int((value-ymin)/(ymax-ymin)*height);};
 int gap=std::abs(project_x(x[1])-project_x(x[0]));
 int bar_width=std::clamp(gap*8/10,2,40);
 std::vector<Rect> result;
 for(std::size_t i=0;i<x.size();++i){
  int center=project_x(x[i]);
  Q low=std::clamp(Q(0),ymin,ymax),tip=std::clamp(y[i],ymin,ymax);
  int base=project_y(low),end=project_y(tip);
  int left=center-bar_width/2,right=left+bar_width;
  int top=std::min(base,end),bottom=std::max(base,end);
  left=std::max(left,ox);right=std::min(right,ox+width+1);
  top=std::max(top,oy);bottom=std::min(bottom,oy+height+1);
  if(right>left&&bottom>top)result.push_back({Q(left)-.5L,Q(top)-.5L,right-left,bottom-top,r,g,b});
 }
 return result;
}
int main(int argc,char**argv){
 try{
  if(argc!=2)fail("usage: bar_viewport_svg_reference SVG_OR_NATIVE_LOG");
  const std::array<Q,4> response{1,2,3,4},x{0,1,3,4};
  Q mean=0;for(Q v:response)mean+=v;mean/=response.size();
  std::array<Q,4> residual{},shifted{};
  for(std::size_t i=0;i<response.size();++i){residual[i]=response[i]-mean;shifted[i]=residual[i]+8.5L;}
  auto expected=reference(x,residual,164,140,180,149,0,4,-2,2,70,180,235);
  auto second=reference(x,shifted,580,140,180,149,.25L,3.75L,5,10,245,170,65);
  expected.insert(expected.end(),second.begin(),second.end());
  if(expected.size()!=8)fail("reference rectangle count");
  // Explicitly establish the positive-only window target: sample8 maps [5,8].
  if(shifted[1]!=8||std::clamp(shifted[1],Q(5),Q(10))!=8)fail("shifted residual reference");
  if(!(x.front()<.25L&&x.back()>3.75L&&second.size()==4))fail("outside-center footprint retention");
  std::ifstream in(argv[1]);if(!in)fail("cannot open SVG/log");
  std::string svg((std::istreambuf_iterator<char>(in)),{});
  auto begin=svg.find("<svg "),end=svg.find("</svg>",begin);
  if(begin==svg.npos||end==svg.npos)fail("SVG bounds");
  svg=svg.substr(begin,end+6-begin);
  // Check explicit SVG tick extents before any bar-geometry comparison.
  // Non-textLength labels are not assigned an approximate glyph-width promise.
  const auto root_end=svg.find('>');
  const std::string root_tag=svg.substr(0,root_end+1);
  const Q root_width=attribute(root_tag,"width"),root_height=attribute(root_tag,"height");
  if(root_width<=0||root_height<=0)fail("root SVG dimensions");
  std::regex text_tags(R"rx(<text\b[^>]*>)rx");
  std::size_t tick_texts=0;
  for(auto i=std::sregex_iterator(svg.begin(),svg.end(),text_tags);i!=std::sregex_iterator();++i){
   const std::string tag=i->str();
   if(tag.find("textLength=")==tag.npos)continue;
   const Q x=attribute(tag,"x"),y=attribute(tag,"y");
   const Q length=attribute(tag,"textLength"),font=attribute(tag,"font-size");
   if(length<0||font<=0||x<0||x+length>root_width||y-font<0||y>root_height)
    fail("tick text bounds");
   ++tick_texts;
  }
  if(tick_texts==0)fail("missing explicit tick text extents");
  for(const std::string text:{
   "Synthetic OLS residuals at irregular positions; n=4",
   "Signed residual (mV)","Residual + 8.5 mV; crop y=[5,10]",
   "No intervals; cropping preserves original zero baseline",
   "Outside centers retain overlapping bar footprints",
   "position (dimensionless)"})
    if(svg.find(text)==svg.npos)fail("missing scientific label: "+text);
  std::regex tags(R"rx(<rect\b[^>]*>)rx"),color(R"rx(fill="rgb\(([0-9]+),([0-9]+),([0-9]+)\)")rx");
  std::vector<Rect> actual;
  for(auto i=std::sregex_iterator(svg.begin(),svg.end(),tags);i!=std::sregex_iterator();++i){
   const std::string tag=i->str();std::smatch m;if(!std::regex_search(tag,m,color))continue;
   int r=std::stoi(m[1]),g=std::stoi(m[2]),b=std::stoi(m[3]);
   bool blue=r==70&&g==180&&b==235,orange=r==245&&g==170&&b==65;
   if(!blue&&!orange)continue;
   Q width=attribute(tag,"width"),height=attribute(tag,"height");
   if(width<=0||height<=0||width!=std::trunc(width)||height!=std::trunc(height)||width>1000000||height>1000000)fail("invalid bar extent");
   actual.push_back({attribute(tag,"x"),attribute(tag,"y"),int(width),int(height),r,g,b});
  }
  if(actual.size()!=expected.size())fail("colored bar rectangle count");
  for(std::size_t i=0;i<expected.size();++i){
   const auto&a=actual[i];const auto&e=expected[i];
   if(a.x!=e.x||a.y!=e.y||a.w!=e.w||a.h!=e.h||a.r!=e.r||a.g!=e.g||a.b!=e.b)
    fail("bar geometry/color "+std::to_string(i));
  }
  if(occurrences(svg,"<!-- bar n=4 -->")!=2||
     occurrences(svg,"<!-- bar counts visible=4 empty_or_outside=0 invalid=0 -->")!=2)
      fail("bar counters");
  if(svg.find("<clipPath")!=svg.npos)fail("unexpected clipPath: rectangles must be intersected already");
  if(svg.find("data-interval-kind=")!=svg.npos)fail("unexpected inferential interval");
  std::cout<<"BAR_VIEWPORT_SVG_REFERENCE_PASS rectangles="<<actual.size()
           <<" signed=4 shifted_positive_crop=4 outside_centers_retained=2 tick_text_bounds_checked="<<tick_texts<<"\n";
  for(std::size_t i=0;i<expected.size();++i){
   const auto&e=expected[i];std::cout<<"rect\t"<<i<<"\t"<<e.x<<"\t"<<e.y<<"\t"<<e.w<<"\t"<<e.h<<"\n";
  }
  return 0;
 }catch(const std::exception&e){std::cerr<<"REJECTED: "<<e.what()<<"\n";return 2;}
}
