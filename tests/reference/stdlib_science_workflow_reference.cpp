#include <algorithm>
#include <bit>
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>
using I=std::int64_t;
using Bytes=std::vector<unsigned char>;
std::string read(const char* path){std::ifstream f(path,std::ios::binary);assert(f);return {std::istreambuf_iterator<char>(f),{}};}
I integer(const std::string& text,std::string_view key,bool quoted=false){
 std::string marker="\""+std::string(key)+"\":"+(quoted?"\"":"");
 auto begin=text.find(marker);assert(begin!=std::string::npos);begin+=marker.size();
 I value=0;auto [end,error]=std::from_chars(text.data()+begin,text.data()+text.size(),value);
 assert(error==std::errc{}&&end>text.data()+begin);if(quoted)assert(*end=='"');else assert(*end==','||*end=='}');
 return value;
}
double real(const std::string& text,std::string_view key){return std::bit_cast<double>(integer(text,key,true));}
std::uint32_t be(const Bytes& b,std::size_t p){assert(p+4<=b.size());return (std::uint32_t(b[p])<<24)|(std::uint32_t(b[p+1])<<16)|(std::uint32_t(b[p+2])<<8)|b[p+3];}
std::uint32_t crc(const Bytes& b,std::size_t begin,std::size_t end){
 std::uint32_t value=0xffffffffU;for(auto i=begin;i<end;++i){value^=b[i];for(int bit=0;bit<8;++bit)value=(value>>1)^((value&1)?0xedb88320U:0);}
 return value^0xffffffffU;
}
void png(const char* path){
 const auto text=read(path);const Bytes data(text.begin(),text.end());
 const Bytes signature{137,80,78,71,13,10,26,10};assert(data.size()>8&&std::equal(signature.begin(),signature.end(),data.begin()));
 Bytes compressed;std::size_t position=8;bool header=false,ended=false;
 while(position<data.size()){
  assert(position+12<=data.size());const auto length=be(data,position);assert(length<=data.size()-position-12);
  const std::string type(data.begin()+position+4,data.begin()+position+8);
  assert(crc(data,position+4,position+8+length)==be(data,position+8+length));
  if(type=="IHDR"){assert(!header&&length==13);header=true;assert(be(data,position+8)==160&&be(data,position+12)==120);assert(data[position+16]==8&&data[position+17]==2);assert(data[position+18]==0&&data[position+19]==0&&data[position+20]==0);}
  else if(type=="IDAT"){assert(header&&!ended);compressed.insert(compressed.end(),data.begin()+position+8,data.begin()+position+8+length);}
  else if(type=="IEND"){assert(length==0);ended=true;assert(position+12==data.size());}
  else assert(false);
  position+=length+12;
 }
 assert(header&&ended&&compressed.size()>6);
 assert((unsigned(compressed[0])*256+compressed[1])%31==0&&(compressed[0]&15)==8&&(compressed[1]&32)==0);
 Bytes raw;position=2;bool final=false;
 while(!final){
  assert(position+5<=compressed.size()-4);
  const auto flags=compressed[position++];final=flags&1;assert(((flags>>1)&3)==0);
  const auto length=unsigned(compressed[position])|(unsigned(compressed[position+1])<<8);
  const auto inverse=unsigned(compressed[position+2])|(unsigned(compressed[position+3])<<8);position+=4;
  assert((length^inverse)==65535&&length<=compressed.size()-position-4);
  raw.insert(raw.end(),compressed.begin()+position,compressed.begin()+position+length);position+=length;
 }
 assert(position+4==compressed.size());
 std::uint32_t a=1,b=0;for(auto c:raw){a=(a+c)%65521;b=(b+a)%65521;}
 assert((b<<16|a)==be(compressed,position));assert(raw.size()==120*(1+160*3));
 std::size_t changed=0;
 for(std::size_t y=0;y<120;++y){const auto offset=y*(1+160*3);assert(raw[offset]==0);for(std::size_t x=0;x<160;++x){const auto p=offset+1+x*3;if(raw[p]!=255||raw[p+1]!=255||raw[p+2]!=255)++changed;}}
 assert(changed>100&&changed<160*120);
}
int main(int argc,char** argv){
 std::vector<I> x,y;for(I i=0;i<300;++i){x.push_back(i%4);y.push_back(2*(i%4)+1);}
 if(argc==2&&std::string_view(argv[1])=="--fixture"){
  std::ofstream f("workflow-input.csv",std::ios::binary);assert(f);f<<"x,y,label\r\n";
  for(std::size_t i=0;i<x.size();++i)f<<x[i]<<','<<y[i]<<",\"measurement, "<<i<<"\"\r\n";
  assert(f);return 0;
 }
 assert(argc==4);const auto report=read(argv[1]),svg=read(argv[2]);
 assert(report.find("\"schema\":\"sounio.stdlib.science-workflow.v1\"")!=std::string::npos);
 const I n=static_cast<I>(x.size());assert(integer(report,"rows")==n&&integer(report,"df")==n-2);
 assert(integer(report,"qr_rank")==2&&integer(report,"svd_rank")==2&&integer(report,"raster_primitives")==n-1);
 I sx=0,sy=0,sxx=0,sxy=0;for(std::size_t i=0;i<x.size();++i){sx+=x[i];sy+=y[i];sxx+=x[i]*x[i];sxy+=x[i]*y[i];}
 const I denominator=n*sxx-sx*sx;assert(denominator>0);
 const long double slope=static_cast<long double>(n*sxy-sx*sy)/denominator;
 const long double intercept=(sy-slope*sx)/n;
 const auto close=[](long double a,long double b,long double tol){return std::isfinite(a)&&std::abs(a-b)<=tol;};
 assert(close(real(report,"slope_bits"),slope,1e-10L)&&close(real(report,"intercept_bits"),intercept,1e-10L));
 assert(close(real(report,"residual_norm_bits"),0,1e-10L));
 const long double s0=static_cast<long double>(real(report,"sigma0_scaled_bits"))*real(report,"sigma_scale_bits");
 const long double s1=static_cast<long double>(real(report,"sigma1_scaled_bits"))*real(report,"sigma_scale_bits");
 assert(s0>=s1&&s1>0);assert(close(s0*s0+s1*s1,n+sxx,1e-8L));assert(close(s0*s0*s1*s1,denominator,1e-5L));
 std::sort(y.begin(),y.end());
 auto quantile=[&](I numerator,I denominator){I rank=(n-1)*numerator;I lo=rank/denominator;I fraction=rank%denominator;return static_cast<long double>(y[lo])+(static_cast<long double>(y[lo+1])-y[lo])*fraction/denominator;};
 assert(close(real(report,"q1_bits"),quantile(1,4),0));assert(close(real(report,"median_bits"),quantile(1,2),0));assert(close(real(report,"q3_bits"),quantile(3,4),0));
 assert(real(report,"xmin_bits")==0&&real(report,"xmax_bits")==3&&real(report,"ymin_bits")==1&&real(report,"ymax_bits")==7);
 assert(svg.starts_with("<svg ")&&svg.find("</svg>")!=std::string::npos&&svg.find("data-dynamic-series=\"ordered-line\"")!=std::string::npos);
 std::size_t lines=0,pos=0;while((pos=svg.find("<line ",pos))!=std::string::npos){++lines;pos+=6;}assert(lines==x.size()-1);
 png(argv[3]);std::puts("SCIENCE_WORKFLOW_CPP_OK");
}
