// Font convention/actual backend oracle. All numerical packing runs in C++23.
// Pinned public-domain primary: dhepper/font8x8 commit8e279d2d864e79128e96188a6b9526cfa3fbfef9.
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using Glyphs=std::array<std::uint64_t,128>;
static void need(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
static std::string load(const char* p){std::ifstream f(p);need(bool(f),std::string("open ")+p);return {std::istreambuf_iterator<char>(f),{}};}
static Glyphs canonical(const char* path){
 const auto raw=load(path);Glyphs out{};std::array<bool,128> seen{};
 const std::regex line(R"(\{([^{}]+)\},?\s*// U\+([0-9A-Fa-f]{4}))"),byte(R"(0x([0-9A-Fa-f]+))");
 for(std::sregex_iterator i(raw.begin(),raw.end(),line),end;i!=end;++i){
  const auto ch=std::stoul((*i)[2],nullptr,16);need(ch<128&&!seen[ch],"canonical duplicate/range");seen[ch]=true;
  const std::string values=(*i)[1];unsigned row=0;
  for(std::sregex_iterator j(values.begin(),values.end(),byte);j!=end;++j){
   need(row<8,"too many canonical rows");const auto b=std::stoul((*j)[1],nullptr,16);need(b<=255,"canonical byte");
   out[ch]|=std::uint64_t(b)<<(8*row);++row;
  }
  need(row==8,"missing canonical rows");
 }
 for(bool flag:seen)need(flag,"missing canonical glyph");
 return out;
}
static Glyphs legacy(const char* path){
 const auto raw=load(path);Glyphs out{};std::array<bool,128> seen{};
 const std::regex p(R"(if ch == ([0-9]+) \{ (0x[0-9A-Fa-f]+) \})");
 for(std::sregex_iterator i(raw.begin(),raw.end(),p),end;i!=end;++i){
  const auto ch=std::stoul((*i)[1]);need(ch<128&&!seen[ch],"source glyph duplicate/range");seen[ch]=true;
  out[ch]=std::stoull((*i)[2],nullptr,16);
 }
 for(unsigned ch=32;ch<127;++ch)need(seen[ch],"source glyph missing");
 return out;
}
struct Grid {int w=0,h=0;std::size_t next=0;std::vector<std::int64_t> pixels;};
struct Dump {Glyphs glyphs{};std::array<bool,128> seen{};std::array<Grid,3> grids{};bool done=false;};
static Dump parse(const char* path){
 std::istringstream in(load(path));Dump out;std::string line;
 while(std::getline(in,line)){
  std::istringstream row(line);std::string tag,extra;row>>tag;
  if(tag=="FONT_GLYPH"){
   int ch=-1;std::int64_t bits=0;need(bool(row>>ch>>bits)&&!(row>>extra),"glyph record");
   need(ch>=32&&ch<128&&!out.seen[ch],"glyph order/range");out.seen[ch]=true;out.glyphs[ch]=std::bit_cast<std::uint64_t>(bits);
  }else if(tag=="FONT_GRID"){
   int kind=-1,w=0,h=0;need(bool(row>>kind>>w>>h)&&!(row>>extra),"grid record");
   need(kind>=0&&kind<3&&w>0&&h>0&&w<=128&&h<=48,"grid bound");
   auto& g=out.grids[kind];need(g.pixels.empty(),"duplicate grid");g.w=w;g.h=h;g.pixels.resize(static_cast<std::size_t>(w*h));
  }else if(tag=="FONT_PIXEL"){
   int kind=-1;std::size_t index=0;std::int64_t value=0;need(bool(row>>kind>>index>>value)&&!(row>>extra),"pixel record");
   need(kind>=0&&kind<3,"pixel kind");auto& g=out.grids[kind];
   need(index==g.next&&index<g.pixels.size(),"pixel gap/overlap");g.pixels[index]=value;++g.next;
  }else if(tag=="FONT_NATIVE_DUMP_COMPLETE")out.done=true;
 }
 need(out.done,"incomplete native dump");
 for(unsigned ch=32;ch<128;++ch)need(out.seen[ch],"missing actual glyph");
 const std::array<int,3> ws{128,16,24},hs{48,16,18};
 for(unsigned k=0;k<3;++k)need(out.grids[k].w==ws[k]&&out.grids[k].h==hs[k]&&out.grids[k].next==static_cast<std::size_t>(ws[k]*hs[k]),"incomplete raster");
 return out;
}
static void draw(std::vector<std::int64_t>& p,int w,int h,std::uint64_t glyph,int x,int y,bool lsb){
 for(unsigned row=0;row<8;++row)for(unsigned col=0;col<8;++col){
  const auto bits=(glyph>>(row*8))&255U;
  const auto bit=(bits>>(lsb?col:7-col))&1U;
  const auto sx=x+static_cast<int>(col),sy=y+static_cast<int>(row);
  if(bit&&sx>=0&&sx<w&&sy>=0&&sy<h)p[static_cast<std::size_t>(sy*w+sx)]=0xffffff;
 }
}
static void verify(const Glyphs& glyphs,const Dump& actual,bool lsb){
 for(unsigned ch=32;ch<128;++ch)need(actual.glyphs[ch]==glyphs[ch],"packed glyph mismatch ch="+std::to_string(ch));
 for(unsigned kind=0;kind<3;++kind){
  const auto& g=actual.grids[kind];std::vector<std::int64_t> expected(g.pixels.size());
  if(kind==0){for(unsigned ch=32;ch<128;++ch)draw(expected,g.w,g.h,glyphs[ch],static_cast<int>((ch-32)%16)*8,static_cast<int>((ch-32)/16)*8,lsb);}
  if(kind==1){draw(expected,g.w,g.h,glyphs[70],-3,-2,lsb);draw(expected,g.w,g.h,glyphs[53],13,14,lsb);}
  if(kind==2){draw(expected,g.w,g.h,glyphs[70],0,0,lsb);draw(expected,g.w,g.h,glyphs[53],8,0,lsb);draw(expected,g.w,g.h,glyphs[60],0,9,lsb);draw(expected,g.w,g.h,glyphs[76],8,9,lsb);}
  for(std::size_t i=0;i<expected.size();++i)need(expected[i]==g.pixels[i],"actual pixel mismatch grid="+std::to_string(kind)+" x="+std::to_string(i%g.w)+" y="+std::to_string(i/g.w));
 }
}
int main(int argc,char** argv){
 try{
  need(argc>=3,"usage: ORACLE --emit-font HEADER | --source-parity SOURCE ACTUAL | --canonical HEADER ACTUAL");
  const std::string mode=argv[1];
  if(mode=="--emit-font"){
   need(argc==3,"emit arguments");const auto glyphs=canonical(argv[2]);
   std::cout<<"pub fn font_glyph(ch: i64) -> i64 {\n";
   for(unsigned ch=32;ch<128;++ch)std::cout<<(ch==32?"    if":"    else if")<<" ch == "<<std::dec<<ch<<" { 0x"<<std::hex<<std::setw(16)<<std::setfill('0')<<glyphs[ch]<<std::dec<<" }\n";
   std::cout<<"    else { 0 }\n}\n";return 0;
  }
  need(argc==4,"verify arguments");const auto actual=parse(argv[3]);
  const auto pixels=actual.grids[0].pixels.size()+actual.grids[1].pixels.size()+actual.grids[2].pixels.size();
  if(mode=="--source-parity"){verify(legacy(argv[2]),actual,false);std::cout<<"FONT_SOURCE_BACKEND_PARITY_PASS glyphs=96 pixels="<<pixels<<" source_row0_top_MSB_left\n";}
  else if(mode=="--canonical"){verify(canonical(argv[2]),actual,true);std::cout<<"FONT_CANONICAL_PASS glyphs=96 pixels="<<pixels<<" row0_top_LSB_left clipping_string_advance\n";}
  else throw std::runtime_error("unknown mode");
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
