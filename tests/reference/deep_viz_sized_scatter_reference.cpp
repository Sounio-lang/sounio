// C++23 independent integer-footprint/raster oracle for parameterized marks.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using I=std::int64_t;
static void need(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
static std::string read(const char* path){std::ifstream f(path);need(bool(f),"open stdout");return {std::istreambuf_iterator<char>(f),{}};}
struct Rect{I x=0,y=0,w=0,h=0;};
static Rect expected(int radius,int index){
 // The ordinary finite inputs are exact dyadics; extreme inputs are rejected
 // as far outside without ever attempting floating-to-integer conversion.
 const std::array<long double,9> xs{-1.5L,9.5L,4,4,0,8,4,-4.5L,12.5L};
 const std::array<long double,9> ys{4,4,-1.5L,9.5L,0,8,4,4,4};
 if(index>=9)return {};
 const I cx=10+static_cast<I>(std::trunc(xs[static_cast<std::size_t>(index)]));
 const I cy=16-static_cast<I>(std::trunc(ys[static_cast<std::size_t>(index)]));
 const I left=std::max<I>(10,cx-radius),right=std::min<I>(19,cx+radius+1);
 const I top=std::max<I>(8,cy-radius),bottom=std::min<I>(17,cy+radius+1);
 if(right<=left||bottom<=top)return {};
 return {left,top,right-left,bottom-top};
}
static void zero(const std::string& raw,const std::string& prefix){
 const auto begin=raw.find(prefix+"_BEGIN\n"),end=raw.find(prefix+"_END",begin);need(begin!=std::string::npos&&end!=std::string::npos,"missing preflight witness");
 const auto start=begin+prefix.size()+7;need(raw.substr(start,end-start).find_first_not_of("\r\n \t")==std::string::npos,"partial output before invalid radius/payload rejection");
}
static void check_rect(const std::string& body,const Rect& wanted,const std::string& color){
 const std::regex rect(R"re(<rect data-source-index="0" data-source-row="-1" data-source-col="-1" x="([-0-9.]+)" y="([-0-9.]+)" width="([0-9]+)" height="([0-9]+)" fill="rgb\(([0-9]+,[0-9]+,[0-9]+)\)"/>)re");
 auto i=std::sregex_iterator(body.begin(),body.end(),rect),end=std::sregex_iterator();
 if(wanted.w==0){need(i==end,"outside/masked point fabricated");return;}
 need(i!=end,"missing projected footprint");const auto& m=*i;
 need(std::stold(m[1])+0.5L==wanted.x&&std::stold(m[2])+0.5L==wanted.y&&std::stoll(m[3])==wanted.w&&std::stoll(m[4])==wanted.h&&m[5]==color,"square footprint/pixelcell/RGB mismatch");
 ++i;need(i==end,"duplicate footprint");
}
static void footprint(const std::string& raw,bool before){
 const std::regex projection(R"(FOOTPRINT_PROJECT (-?[0-9]+) (-?[0-9]+) ([0-9]+))");std::smatch p;
 need(std::regex_search(raw,p,projection)&&p[1]=="9"&&p[2]=="14"&&p[3]=="0","native integer projection discriminator");
 const std::regex counts(R"(FOOTPRINT_RESULT ([0-9]+) ([0-9]+) ([0-9]+))");std::smatch c;
 need(std::regex_search(raw,c,counts)&&c[1]=="0"&&c[2]==(before?"0":"4")&&c[3]==(before?"4":"0"),"native footprint result discriminator");
 const std::regex pixels(R"(FOOTPRINT_BOUNDARY ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+))");std::smatch m;need(std::regex_search(raw,m,pixels),"boundary pixels missing");
 const I color=before?((7LL<<16)|(8LL<<8)|9):((70LL<<16)|(180LL<<8)|235);
 for(int i=1;i<=4;++i)need(std::stoll(m[i])==color,"actual boundary pixel mismatch");
 if(before){need(raw.find("STDLIB_DEEP_VIZ_SCATTER_FOOTPRINT_PASS")==std::string::npos,"before unexpectedly passed");return;}
 const auto at=raw.find("<svg "),end=raw.find("</svg>",at);need(at!=std::string::npos&&end!=std::string::npos,"boundary SVG missing");
 const auto body=raw.substr(at,end+6-at);
 const std::regex rect(R"re(<rect data-source-index="([0-3])" data-source-row="-1" data-source-col="-1" x="([-0-9.]+)" y="([-0-9.]+)" width="([0-9]+)" height="([0-9]+)" fill="rgb\(70,180,235\)"/>)re");
 const std::array<Rect,4> expect{{{10,13,1,3},{18,13,1,3},{13,18,3,1},{13,10,3,1}}};std::size_t n=0;
 for(std::sregex_iterator i(body.begin(),body.end(),rect),e;i!=e;++i){need(n<4&&std::stoull((*i)[1])==n,"edge original-index ordering");const auto& r=expect[n];need(std::stold((*i)[2])+0.5L==r.x&&std::stold((*i)[3])+0.5L==r.y&&std::stoll((*i)[4])==r.w&&std::stoll((*i)[5])==r.h,"edge SVG integer clip");++n;}
 need(n==4&&raw.find("STDLIB_DEEP_VIZ_SCATTER_FOOTPRINT_PASS")!=std::string::npos,"incomplete corrected footprint witness");
}
static void sized(const std::string& raw){
 zero(raw,"SIZED_INVALID");zero(raw,"SIZED_NAN");zero(raw,"GLOBAL_SIZE_INVALID");
 const I bg=(7LL<<16)|(8LL<<8)|9,fg=(70LL<<16)|(180LL<<8)|235;
 std::array<std::array<std::vector<I>,12>,4> rasters;std::array<std::array<I,12>,4> next{};std::array<std::array<bool,12>,4> ended{},svg_seen{};
 for(auto& radius:rasters)for(auto& image:radius)image.assign(32*24,-1);
 std::istringstream in(raw);std::string token;
 while(in>>token){
  if(token=="SIZED_RUN"){
   int radius=0,index=0;I start=0,length=0,color=0;need(bool(in>>radius>>index>>start>>length>>color)&&radius>=1&&radius<=4&&index>=0&&index<12,"RLE header");
   auto& n=next[static_cast<std::size_t>(radius-1)][static_cast<std::size_t>(index)];need(!ended[static_cast<std::size_t>(radius-1)][static_cast<std::size_t>(index)]&&start==n&&length>0&&length<=32*24-n,"RLE coverage/range");
   auto& pixels=rasters[static_cast<std::size_t>(radius-1)][static_cast<std::size_t>(index)];std::fill(pixels.begin()+n,pixels.begin()+n+length,color);n+=length;
  }else if(token=="SIZED_RASTER_END"){
   int radius=0,index=0;need(bool(in>>radius>>index)&&radius>=1&&radius<=4&&index>=0&&index<12,"RLE terminator");auto& done=ended[static_cast<std::size_t>(radius-1)][static_cast<std::size_t>(index)];need(!done&&next[static_cast<std::size_t>(radius-1)][static_cast<std::size_t>(index)]==32*24,"incomplete RLE");done=true;
  }
 }
 for(int radius=1;radius<=4;++radius)for(int index=0;index<12;++index){
  need(ended[static_cast<std::size_t>(radius-1)][static_cast<std::size_t>(index)],"missing native raster");const auto r=expected(radius,index);const auto& pixels=rasters[static_cast<std::size_t>(radius-1)][static_cast<std::size_t>(index)];
  for(I y=0;y<24;++y)for(I x=0;x<32;++x){const bool mark=r.w>0&&x>=r.x&&x<r.x+r.w&&y>=r.y&&y<r.y+r.h;need(pixels[static_cast<std::size_t>(y*32+x)]==(mark?fg:bg),"actual raster glyph/plot scissor mismatch");}
 }
 std::size_t at=0;int roots=0,global=0;
 while((at=raw.find("<svg ",at))!=std::string::npos){
  const auto end=raw.find("</svg>",at);need(end!=std::string::npos,"unclosed SVG");const auto body=raw.substr(at,end+6-at);
  const std::regex header(R"re(data-radius="([1-4])" data-case="([0-9]+)")re");std::smatch h;
  if(std::regex_search(body,h,header)){
   const int radius=std::stoi(h[1]),index=std::stoi(h[2]);need(index>=0&&index<12&&!svg_seen[static_cast<std::size_t>(radius-1)][static_cast<std::size_t>(index)],"duplicate/unknown SVG case");svg_seen[static_cast<std::size_t>(radius-1)][static_cast<std::size_t>(index)]=true;
   check_rect(body,expected(radius,index),"70,180,235");++roots;
  }else{
   ++global;need(body.find("width=\"640\" height=\"360\"")!=std::string::npos&&body.find("data-primary-radius=\"2\" data-secondary-radius=\"4\"")!=std::string::npos,"full composition marker metadata");
   const std::array<std::string,4> names{"scientific-layer-left-primary","scientific-layer-left-secondary","scientific-layer-right-primary","scientific-layer-right-secondary"};
   const std::array<Rect,4> marks{{{140,188,5,5},{214,126,9,9},{460,188,5,5},{534,126,9,9}}};
   for(std::size_t i=0;i<4;++i){const auto a=body.find("<g id=\""+names[i]+"\">"),b=body.find("</g>",a);need(a!=std::string::npos&&b!=std::string::npos,"composite layer missing");check_rect(body.substr(a,b-a),marks[i],i%2?"235,130,70":"70,180,235");}
   const std::regex legend(R"re(<rect data-overlay-legend="scatter" x="([-0-9.]+)" y="([-0-9.]+)" width="([0-9]+)" height="([0-9]+)" fill="rgb\(([0-9]+,[0-9]+,[0-9]+)\)"/>)re");std::size_t n=0;
   for(std::sregex_iterator i(body.begin(),body.end(),legend),e;i!=e;++i){need(n<4,"extra legend");const I radius=n%2?4:2;need(std::stoll((*i)[3])==radius*2+1&&std::stoll((*i)[4])==radius*2+1,"legend/data size disagreement");++n;}need(n==4,"sized legends absent");
  }
  at=end+6;
 }
 need(roots==48&&global==1,"incomplete parameterized geometry set");
 const std::regex pixel(R"(GLOBAL_SIZE_PIXELS ([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+))");std::smatch p;need(std::regex_search(raw,p,pixel),"global Canvas boundary probes absent");
 need(std::stoll(p[1])==fg&&std::stoll(p[2])==((22LL<<16)|(28LL<<8)|36)&&std::stoll(p[3])==((235LL<<16)|(130LL<<8)|70)&&std::stoll(p[4])==((22LL<<16)|(28LL<<8)|36),"global marker Canvas size");
 need(raw.find("STDLIB_DEEP_VIZ_SIZED_SCATTER_PASS")!=std::string::npos,"missing native completion");
}
int main(int argc,char** argv){try{
 if(argc==3&&std::string(argv[1])=="--before"){footprint(read(argv[2]),true);std::cout<<"SCATTER_FOOTPRINT_BEFORE_DISCRIMINATED\n";return 0;}
 need(argc==3,"usage: ORACLE sized.stdout corrected-footprint.stdout | --before before.stdout");sized(read(argv[1]));footprint(read(argv[2]),false);
 std::cout<<"SIZED_SCATTER_REFERENCE_PASS radii=4 cases=48 raster_pixels=36864 full_document=1 preflight_zero=3\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
