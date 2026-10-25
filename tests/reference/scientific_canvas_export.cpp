// Lossless export of ACTUAL native Canvas packed RGB, never an SVG redraw.
// Build: g++ -std=c++23 -O2 -Wall -Wextra scientific_canvas_export.cpp -lz -o EXPORT
// Omit -lz if zlib.h is absent: the bounded stored-deflate fallback is standalone.
#include <cstdint>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if __has_include(<zlib.h>)
#include <zlib.h>
#define CANVAS_USE_ZLIB 1
#else
#define CANVAS_USE_ZLIB 0
#endif
using Bytes=std::vector<unsigned char>;
static void need(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
static void be32(Bytes& b,std::uint32_t x){b.push_back(static_cast<unsigned char>(x>>24));b.push_back(static_cast<unsigned char>(x>>16));b.push_back(static_cast<unsigned char>(x>>8));b.push_back(static_cast<unsigned char>(x));}
static std::uint32_t crc(const unsigned char* p,std::size_t n){
 std::uint32_t c=0xffffffffU;
 for(std::size_t i=0;i<n;++i){c^=p[i];for(int j=0;j<8;++j)c=(c>>1)^(0xedb88320U&std::uint32_t(0U-(c&1U)));}
 return c^0xffffffffU;
}
static void chunk(Bytes& png,const char* type,const Bytes& payload){
 need(payload.size()<=0x7fffffffU,"PNG chunk budget");
 be32(png,static_cast<std::uint32_t>(payload.size()));
 const auto start=png.size();
 for(int i=0;i<4;++i)png.push_back(static_cast<unsigned char>(type[i]));
 png.insert(png.end(),payload.begin(),payload.end());
 be32(png,crc(png.data()+start,png.size()-start));
}
static Bytes deflate_rows(const Bytes& rows){
#if CANVAS_USE_ZLIB
 uLongf size=compressBound(static_cast<uLong>(rows.size()));Bytes out(size);
 need(compress2(out.data(),&size,rows.data(),static_cast<uLong>(rows.size()),Z_BEST_SPEED)==Z_OK,"zlib compression");
 out.resize(size);return out;
#else
 Bytes out{0x78,0x01};std::size_t at=0;
 while(at<rows.size()){
  const auto amount=std::min<std::size_t>(65535,rows.size()-at);
  const auto n=static_cast<std::uint16_t>(amount),inverse=static_cast<std::uint16_t>(~n);
  out.push_back(at+amount==rows.size()?1:0);
  out.push_back(static_cast<unsigned char>(n));out.push_back(static_cast<unsigned char>(n>>8));
  out.push_back(static_cast<unsigned char>(inverse));out.push_back(static_cast<unsigned char>(inverse>>8));
  out.insert(out.end(),rows.begin()+static_cast<std::ptrdiff_t>(at),rows.begin()+static_cast<std::ptrdiff_t>(at+amount));at+=amount;
 }
 std::uint32_t a=1,b=0;
 for(auto byte:rows){a=(a+byte)%65521U;b=(b+a)%65521U;}
 be32(out,(b<<16)|a);return out;
#endif
}
int main(int argc,char** argv){
 try{
  need(argc==4,"usage: EXPORT actual-native.stdout actual.ppm actual.png");
  std::ifstream input(argv[1]);need(bool(input),"open actual stdout");
  bool seen=false,ended=false;long long w=0,h=0,total=0,next=0,runs=0;Bytes rgb;std::string line;
  while(std::getline(input,line)){
   if(line.rfind("CANVAS_",0)!=0)continue;
   std::istringstream in(line);std::string tag,extra;in>>tag;
   if(tag=="CANVAS_SIZE"){
    need(!seen&&bool(in>>w>>h)&&!(in>>extra),"duplicate/malformed raster header");
    need(w>0&&w<=4096&&h>0&&h<=4096,"raster dimensions");
    total=w*h;need(total<=16777216,"raster payload budget");
    rgb.resize(static_cast<std::size_t>(total)*3);seen=true;
   }else if(tag=="CANVAS_RUN"){
    long long start=0,length=0,color=0;
    need(seen&&!ended&&bool(in>>start>>length>>color)&&!(in>>extra),"malformed/out-of-order run");
    need(start==next&&length>0&&length<=total-next,"run gap/overlap/range");
    need(color>=0&&color<=0xffffff,"packed RGB out of range");
    for(long long i=0;i<length;++i){
     const auto at=static_cast<std::size_t>(next+i)*3;
     rgb[at]=static_cast<unsigned char>(color>>16);
     rgb[at+1]=static_cast<unsigned char>(color>>8);
     rgb[at+2]=static_cast<unsigned char>(color);
    }
    next+=length;++runs;
   }else if(tag=="CANVAS_END"){
    need(seen&&!ended&&next==total&&!(in>>extra),"incomplete/duplicate raster terminator");ended=true;
   }else throw std::runtime_error("unknown raster record");
  }
  need(seen&&ended&&next==total,"missing/incomplete native raster");
  std::ofstream ppm(argv[2],std::ios::binary);need(bool(ppm),"open PPM");
  ppm<<"P6\n"<<w<<" "<<h<<"\n255\n";ppm.write(reinterpret_cast<const char*>(rgb.data()),static_cast<std::streamsize>(rgb.size()));
  ppm.close();need(bool(ppm),"write PPM");
  Bytes rows;rows.reserve(rgb.size()+static_cast<std::size_t>(h));
  for(long long y=0;y<h;++y){
   rows.push_back(0);
   const auto at=static_cast<std::size_t>(y*w)*3;
   rows.insert(rows.end(),rgb.begin()+static_cast<std::ptrdiff_t>(at),rgb.begin()+static_cast<std::ptrdiff_t>(at+static_cast<std::size_t>(w)*3));
  }
  Bytes png{137,80,78,71,13,10,26,10},ihdr;
  be32(ihdr,static_cast<std::uint32_t>(w));be32(ihdr,static_cast<std::uint32_t>(h));
  ihdr.insert(ihdr.end(),{8,2,0,0,0});chunk(png,"IHDR",ihdr);
  const auto compressed=deflate_rows(rows);chunk(png,"IDAT",compressed);chunk(png,"IEND",{});
  std::ofstream image(argv[3],std::ios::binary);need(bool(image),"open PNG");
  image.write(reinterpret_cast<const char*>(png.data()),static_cast<std::streamsize>(png.size()));image.close();need(bool(image),"write PNG");
  std::cout<<"SCIENTIFIC_CANVAS_EXPORT_PASS width="<<w<<" height="<<h<<" pixels="<<total<<" runs="<<runs<<" png_zlib="<<CANVAS_USE_ZLIB<<"\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
