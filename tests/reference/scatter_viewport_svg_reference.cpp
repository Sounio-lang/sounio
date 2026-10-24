// Independent C++23 scientific residual/scatter SVG geometry checker.
// No production projection helper or QR implementation is reused.
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
[[noreturn]] void fail(const std::string&s){throw std::runtime_error(s);}
std::size_t occurrences(const std::string&s,const std::string&token){
 std::size_t n=0,p=0;while((p=s.find(token,p))!=s.npos){++n;p+=token.size();}return n;
}
struct Mark{int x,y,r,g,b;};
std::vector<Mark> marks(const std::string&s){
 std::regex re(R"rx(<circle cx="(-?[0-9]+)" cy="(-?[0-9]+)" r="([0-9]+)" fill="rgb\(([0-9]+),([0-9]+),([0-9]+)\)"/>)rx");
 std::vector<Mark> out;
 for(auto i=std::sregex_iterator(s.begin(),s.end(),re);i!=std::sregex_iterator();++i){
  auto m=*i;if(std::stoi(m[3])!=3)fail("marker radius");
  out.push_back({std::stoi(m[1]),std::stoi(m[2]),std::stoi(m[4]),std::stoi(m[5]),std::stoi(m[6])});
 }
 if(occurrences(s,"<circle")!=out.size())fail("unrecognized circle markup");
 return out;
}
int main(int argc,char**argv){
 try{
  if(argc!=2)fail("usage: scatter_viewport_svg_reference SVG_OR_NATIVE_LOG");
  // Independent intercept-only OLS mean from raw responses; fixed crop is
  // defined by the second and third residuals, as in the scientific consumer.
  const std::array<long double,4> y{1,2,3,4};
  long double mean=0;for(auto value:y)mean+=value;mean/=y.size();
  std::array<long double,4> residual{};
  for(std::size_t i=0;i<y.size();++i)residual[i]=y[i]-mean;
  const long double lo=residual[1],hi=residual[2];
  std::vector<Mark> expected;
  int outside=0;
  for(std::size_t i=0;i<y.size();++i){
   long double x=i+1;
   if(x<2||x>3||residual[i]<lo||residual[i]>hi){++outside;continue;}
   int px=100+int((x-2)*240),py=300-int((residual[i]-lo)/(hi-lo)*160);
   expected.push_back({px,py,70,180,235});
  }
  if(outside!=2||expected.size()!=2)fail("independent crop classification");
  // Equal-domain centers use integer half extents. Magnitude has no role.
  expected.push_back({440+160/2,140+160-160/2,220,135,225});
  expected.push_back({740+160/2,140+160-160/2,245,170,65});
  expected.push_back({440+160/2,420+160-160/2,100,220,170});
  std::ifstream in(argv[1]);if(!in)fail("cannot open SVG/log");
  std::string svg((std::istreambuf_iterator<char>(in)),{});
  // Only the FIRST SVG is checked here; the later boundary-serialization SVG
  // is checked by the native fixture expect-stdout directive, not this oracle.
  auto begin=svg.find("<svg "),end=svg.find("</svg>",begin);
  if(begin==svg.npos||end==svg.npos)fail("SVG bounds");
  svg=svg.substr(begin,end+6-begin);
  for(const std::string text:{
   "Synthetic intercept-only residuals (mV); n=4",
   "Closed crop: observations 2 and 3; no intervals",
   "Legacy singleton zero","Legacy singleton 1e300",
   "Paired-finite bounds; invalid points omitted",
   "View cropping does not alter the fitted model",
   "observation index","residual (mV)"})
    if(svg.find(text)==svg.npos)fail("missing scientific label: "+text);
  auto actual=marks(svg);if(actual.size()!=expected.size())fail("circle count");
  for(std::size_t i=0;i<expected.size();++i){
   const auto&a=actual[i];const auto&e=expected[i];
   if(a.x!=e.x||a.y!=e.y||a.r!=e.r||a.g!=e.g||a.b!=e.b)
     fail("circle geometry/color "+std::to_string(i));
  }
  for(const std::string token:{
   "<!-- scatter n=4 -->","<!-- scatter n=3 -->",
   "<!-- scatter counts visible=2 outside=2 invalid=0 -->",
   "<!-- scatter counts visible=1 outside=0 invalid=2 -->"})
    if(occurrences(svg,token)!=1)fail("scatter counter: "+token);
  if(occurrences(svg,"<!-- scatter n=1 -->")!=2||
     occurrences(svg,"<!-- scatter counts visible=1 outside=0 invalid=0 -->")!=2)
      fail("singleton counters");
  // Verify the sole opt-in clip is a closed-center viewport expanded by half
  // a pixel cell, not an incorrectly shrunken or whole-image clip.
  std::regex clip(R"rx(<clipPath[^>]*id="scatter-clip-0"[^>]*>[\s\S]*?</clipPath>)rx");
  std::smatch c;if(!std::regex_search(svg,c,clip))fail("missing clip definition");
  const std::string definition=c.str();
  if(definition.find("clipPathUnits=\"userSpaceOnUse\"")==definition.npos)
   fail("clip coordinate units");
  std::regex rectangle(R"rx(<rect[^>]*>)rx");std::smatch r;
  if(!std::regex_search(definition,r,rectangle))fail("clip rectangle");
  auto attribute=[&](const std::string&name){
   std::regex value(name+"=\"([^\"]+)\"");std::smatch m;const std::string rect=r.str();
   if(!std::regex_search(rect,m,value))fail("missing clip attribute "+name);
   std::size_t used=0;double result=std::stod(m[1],&used);
   if(used!=static_cast<std::size_t>(m[1].length())||!std::isfinite(result))fail("clip number");
   return result;
  };
  if(attribute("x")!=99.5||attribute("y")!=139.5||
     attribute("width")!=241||attribute("height")!=161)fail("clip geometry");
  if(occurrences(svg,"<clipPath")!=1)fail("unexpected legacy clipping");
  auto group=svg.find("<g clip-path=\"url(#scatter-clip-0)\">");
  if(group==svg.npos)fail("clip group");
  auto close=svg.find("</g>",group);
  if(close==svg.npos||marks(svg.substr(group,close-group)).size()!=2)fail("clip group marks");
  if(svg.find("data-interval-kind=")!=svg.npos)fail("unexpected interval");
  std::cout<<"SCATTER_VIEWPORT_SVG_REFERENCE_PASS circles="<<actual.size()
           <<" visible_crop=2 outside_crop=2 invalid_mixed=2 clip=closed_pixel_cells\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<"REJECTED: "<<e.what()<<"\n";return 2;}
}
