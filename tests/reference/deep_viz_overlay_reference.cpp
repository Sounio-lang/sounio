// Independent C++23 score-root and actual SVG/Canvas geometry concordance.
// NB2 score roots: group means2/4, pooled-intercept mean3, alpha1/4.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
static void need(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
struct Range{long long lo,hi;};
static Range project(long double value,long double error,long double lo,long double hi,long long origin,long long extent,bool vertical){
 const auto lower=std::clamp((value-error-lo)/(hi-lo)*extent,0.L,static_cast<long double>(extent));
 const auto upper=std::clamp((value+error-lo)/(hi-lo)*extent,0.L,static_cast<long double>(extent));
 const auto a=static_cast<long long>(std::trunc(lower)),b=static_cast<long long>(std::trunc(upper));
 Range out=vertical?Range{origin+extent-b,origin+extent-a}:Range{origin+a,origin+b};
 need(out.hi-out.lo<=1,"projection uncertainty too wide");return out;
}
static long double residual(int count,long double mean){
 const long double first=count==0?0:count*std::log(static_cast<long double>(count)/mean);
 const auto d=2*(first-(count+4)*std::log((count+4)/(mean+4)));
 return (count<mean?-1:1)*std::sqrt(d);
}
static long double budget(long double value){return 2e-8L*(1+std::fabs(value));}
static std::string layer(const std::string& svg,const std::string& id){
 const std::string begin="<g id=\""+id+"\">";const auto a=svg.find(begin),b=svg.find("</g>",a);
 need(a!=std::string::npos&&b!=std::string::npos,"missing layer "+id);return svg.substr(a,b-a);
}
static void scatter(const std::string& body,bool secondary,bool right){
 const std::regex pattern(R"re(<rect data-source-index="([0-9]+)" data-source-row="-1" data-source-col="-1" x="([-0-9.]+)" y="([-0-9.]+)" width="([0-9]+)" height="([0-9]+)" fill="rgb\(([0-9]+,[0-9]+,[0-9]+)\)"/>)re");
 auto actual=std::sregex_iterator(body.begin(),body.end(),pattern),end=std::sregex_iterator();
 // Series identity, not panel identity, owns the palette in the opt-in API.
 const auto color=secondary?"235,130,70":"70,180,235";
 const long long ox=right?802:112,oy=100,w=514,h=380;
 for(long long index=0;index<1028;++index){
  if(secondary&&index%11==0)continue;
  need(actual!=end,"missing scatter mark");const auto& m=*actual;
  need(std::stoll(m[1])==index&&m[6]==color,"scatter provenance/RGB");
  const long double mean=secondary?3:(index%2?4:2);
  const int count=static_cast<int>((index%4)*2);
  const auto y=right?residual(count,mean):static_cast<long double>(count);
  const auto cx=project(mean,budget(mean),0,6,ox,w,false);
  const auto cy=project(y,right?budget(y):0,right?-3:0,right?3:6,oy,h,true);
  const long double rawx=std::stold(m[2])+0.5L,rawy=std::stold(m[3])+0.5L;
  need(rawx==std::trunc(rawx)&&rawy==std::trunc(rawy),"not pixelcell origin");
  bool found=false;
  for(auto x=cx.lo;x<=cx.hi;++x)for(auto yy=cy.lo;yy<=cy.hi;++yy){
   const auto left=std::max(ox,x-1),top=std::max(oy,yy-1);
   const auto rw=std::min(ox+w+1,x+2)-left,rh=std::min(oy+h+1,yy+2)-top;
   if(rawx==left&&rawy==top&&std::stoll(m[4])==rw&&std::stoll(m[5])==rh)found=true;
  }
  need(found,"scatter geometry at original index "+std::to_string(index));++actual;
 }
 need(actual==end,"extra scatter marks");
}
static void lines(const std::string& body,bool right){
 const std::regex pattern(R"re(<line data-dynamic-part="line" x1="([0-9]+)" y1="([0-9]+)" x2="([0-9]+)" y2="([0-9]+)" stroke="rgb\(([0-9]+,[0-9]+,[0-9]+)\)" stroke-width="1" stroke-linecap="round"/>)re");
 auto actual=std::sregex_iterator(body.begin(),body.end(),pattern),end=std::sregex_iterator();
 const long long ox=right?802:112;
 const auto cx=project(3,budget(3),0,6,ox,514,false);
 for(int index=1;index<1028;++index){
  if(index%11==0||(index-1)%11==0)continue;
  need(actual!=end,"missing ordered segment");const auto& m=*actual;
  need(m[5]=="235,130,70","line RGB");
  const auto x1=std::stoll(m[1]),x2=std::stoll(m[3]);
  need(x1==x2&&x1>=cx.lo&&x1<=cx.hi,"line constant fitted X geometry");
  for(int side=0;side<2;++side){
   const int count=((index-1+side)%4)*2;
   const long double value=right?residual(count,3):static_cast<long double>(count);
   const auto cy=project(value,right?budget(value):0,right?-3:0,right?3:6,100,380,true);
   const auto ay=std::stoll(m[side==0?2:4]);need(ay>=cy.lo&&ay<=cy.hi,"line endpoint/gap geometry");
  }
  ++actual;
 }
 need(actual==end,"extra ordered segments or bridged gap");
}
static void check(const std::string& raw){
 // Validate the numerical forward budget BEFORE using it for pixel intervals.
 // Geometry alone cannot reveal a numerical error smaller than a pixel.
 std::istringstream scalars(raw);std::string token;int numeric_rows=0;
 while(scalars>>token){
  if(token=="OVERLAY_NUMERIC"){
   int model=-1,index=-1;long double mean=0,value=0;
   need(bool(scalars>>model>>index>>mean>>value),"incomplete numerical witness");
   need(model==numeric_rows/4&&index==numeric_rows%4&&numeric_rows<8,"numerical witness order");
   const long double expected_mean=model==0?(index%2?4:2):3;
   const long double expected_residual=residual(index*2,expected_mean);
   need(std::isfinite(mean)&&std::fabs(mean-expected_mean)<=budget(expected_mean),"independent fitted mean");
   need(std::isfinite(value)&&std::fabs(value-expected_residual)<=budget(expected_residual),"independent signed residual");
   ++numeric_rows;
  }
 }
 need(numeric_rows==8,"missing numerical payloads");
 for(const std::string tag:{"OVERLAY_ZERO","OVERLAY_LABEL_ZERO"}){
  const auto start=raw.find(tag+"_BEGIN\n"),stop=raw.find(tag+"_END",start);
  need(start!=std::string::npos&&stop!=std::string::npos,"missing zero-output witness");
  const auto a=start+tag.size()+7;
  need(raw.substr(a,stop-a).find_first_not_of("\r\n \t")==std::string::npos,"invalid late layer/legend emitted bytes");
 }
 std::size_t at=0;int roots=0;
 while((at=raw.find("<svg ",at))!=std::string::npos){
  const auto finish=raw.find("</svg>",at);need(finish!=std::string::npos,"unclosed SVG");
  const auto svg=raw.substr(at,finish+6-at);need(roots<2,"extra SVG root");
  need(svg.find("width=\"1380\" height=\"620\"")!=std::string::npos,"root dimensions");
  need(svg.find("data-interval-kind=\"none\"")!=std::string::npos&&svg.find("data-scientific-overlay-complete=\"true\"")!=std::string::npos,"false interval/missing completion");
  std::set<std::string> ids;const std::regex id(R"re(\bid="([^"]+)")re");
  for(std::sregex_iterator i(svg.begin(),svg.end(),id),e;i!=e;++i)need(ids.insert((*i)[1]).second,"duplicate id");
  need(ids.size()==static_cast<std::size_t>(roots==0?4:6),"missing layer/clip ids");
  const std::regex text(R"re(<text [^>]*x="(-?[0-9]+)" y="(-?[0-9]+)"[^>]*textLength="([0-9]+)"[^>]*>([^<]*)</text>)re");
  int count_text=0;
  for(std::sregex_iterator i(svg.begin(),svg.end(),text),e;i!=e;++i){
   const auto x=std::stoll((*i)[1]),y=std::stoll((*i)[2]),w=std::stoll((*i)[3]);
   need(x>=0&&x+w<=1380&&y-7>=0&&y+1<=620,"label bounds");
   need(w==static_cast<long long>((*i)[4].length())*8,"text cell width");++count_text;
  }
  need(count_text==28,"lost axis/title/legend text");
  for(const auto& s:{"Synthetic NB2 fits","two groups","pooled: subset","No CI; supplied alpha; assumptions unassessed; coincident rows overlap."})need(svg.find(s)!=std::string::npos,"missing scientific legend/caveat");
  const std::regex legends(R"re(<(rect|line) data-overlay-legend="(scatter|line)"[^>]*(?:fill|stroke)="rgb\(([0-9]+,[0-9]+,[0-9]+)\)"[^>]*/>)re");
  const std::array<std::string,4> colors{"70,180,235","235,130,70","70,180,235","235,130,70"};
  std::size_t legend_count=0;
  for(std::sregex_iterator i(svg.begin(),svg.end(),legends),e;i!=e;++i){
   need(legend_count<4&&(*i)[3]==colors[legend_count],"legend dataset/color binding");
   const bool line=roots==1&&legend_count%2==1;
   need((*i)[2]==(line?"line":"scatter"),"legend glyph/mode binding");++legend_count;
  }
  need(legend_count==4,"missing/extra legend glyphs");
  scatter(layer(svg,"scientific-layer-left-primary"),false,false);
  scatter(layer(svg,"scientific-layer-right-primary"),false,true);
  if(roots==0){scatter(layer(svg,"scientific-layer-left-secondary"),true,false);scatter(layer(svg,"scientific-layer-right-secondary"),true,true);}
  else {lines(layer(svg,"scientific-layer-left-secondary"),false);lines(layer(svg,"scientific-layer-right-secondary"),true);}
  ++roots;at=finish+6;
 }
 need(roots==2,"missing scatter/explicit-line roots");
 const std::regex pixel(R"(OVERLAY_PIXEL ([0-9]+) ([0-9]+) ([0-9]+))");std::smatch m;
 need(std::regex_search(raw,m,pixel),"missing actual Canvas probe");
 const auto px=std::stoll(m[1]),py=std::stoll(m[2]);const auto expected=project(2,budget(2),0,6,112,514,false);
 need(px>=expected.lo&&px<=expected.hi&&py==480&&m[3]=="4633835","actual Canvas geometry/RGB");
 need(raw.find("STDLIB_DEEP_VIZ_OVERLAY_PASS")!=std::string::npos,"missing native pass");
}
int main(int argc,char** argv){
 try{
  const bool controls=argc==3&&std::string(argv[1])=="--negative-selftest";
  need(argc==(controls?3:2),"usage: ORACLE [--negative-selftest] actual.stdout");
  std::ifstream in(argv[controls?2:1]);need(bool(in),"open stdout");const std::string raw{std::istreambuf_iterator<char>(in),{}};check(raw);
  if(controls){
   const auto begin=raw.find("<g id=\"scientific-layer-left-primary\">");
   const auto start=raw.find("<rect data-source-index=",begin),finish=raw.find("/>",start)+2;
   need(start!=std::string::npos&&finish>start,"mutation mark");
   const auto mark=raw.substr(start,finish-start);
   std::array<std::string,3> mutants{raw,raw,raw};
   auto moved=mark;const auto x=moved.find(" x=\""),last=moved.find('"',x+4);moved.replace(x+4,last-(x+4),"999999.500000");
   mutants[0].replace(start,mark.size(),moved);mutants[1].erase(start,mark.size());
   auto color=mark;const auto rgb=color.find("70,180,235");need(rgb!=std::string::npos,"color mutation");color.replace(rgb,10,"1,2,3");mutants[2].replace(start,mark.size(),color);
   for(std::size_t i=0;i<mutants.size();++i){bool rejected=false;try{check(mutants[i]);}catch(const std::exception&){rejected=true;}need(rejected,"undetected negative control");std::cout<<"OVERLAY_MUTANT_REJECTED kind="<<i<<'\n';}
  }
  std::cout<<"VIZ_OVERLAY_REFERENCE_PASS roots=2 rows=1028 models=2 numeric_rows=8 text_bounds=56 zero_bytes=2 actual_canvas_pixel=1\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
