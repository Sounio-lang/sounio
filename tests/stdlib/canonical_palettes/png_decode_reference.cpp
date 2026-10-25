// Independent PNG filter/chunk decoder, full-raster/native/primary comparison.
// zlib ABI per https://zlib.net/manual.html#Utility; existing runtime, no install.
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int uncompress2(unsigned char*,unsigned long*,const unsigned char*,unsigned long*);
using B=unsigned char;using V=std::vector<B>;using I=long long;
constexpr I W=720,H=480;
void need(bool v,const std::string&s){if(!v)throw std::runtime_error(s);}
V bytes(const std::string&p){std::ifstream f(p,std::ios::binary);need(bool(f),"open "+p);return {std::istreambuf_iterator<char>(f),{}};}
std::string text(const std::string&p){const auto b=bytes(p);return {b.begin(),b.end()};}
std::uint32_t u32(const B*p){return(std::uint32_t(p[0])<<24)|(std::uint32_t(p[1])<<16)|(std::uint32_t(p[2])<<8)|p[3];}
std::uint32_t checksum(const B*p,std::size_t n){std::uint32_t c=0xffffffff;for(std::size_t i=0;i<n;++i){c^=p[i];for(int b=0;b<8;++b){if(c&1)c=(c>>1)^0xedb88320;else c>>=1;}}return~c;}
int paeth(int a,int b,int c){const int p=a+b-c,da=std::abs(p-a),db=std::abs(p-b),dc=std::abs(p-c);return da<=db&&da<=dc?a:db<=dc?b:c;}
V png(const std::string& path){const auto file=bytes(path);const V signature{137,80,78,71,13,10,26,10};need(file.size()>=8&&std::equal(signature.begin(),signature.end(),file.begin()),"PNG signature");std::size_t at=8;V compressed;bool header=false,end=false;while(at<file.size()){
 need(file.size()-at>=12,"PNG truncated chunk");const auto n=u32(file.data()+at);need(n<=file.size()-at-12,"PNG chunk bounds");const std::string type(reinterpret_cast<const char*>(file.data()+at+4),4);need(checksum(file.data()+at+4,n+4)==u32(file.data()+at+n+8),"PNG CRC "+type);const B*p=file.data()+at+8;
 std::cout<<"PNG_CHUNK "<<type<<" bytes="<<n<<'\n';
 need(type=="IHDR"||type=="IDAT"||type=="IEND","PNG ancillary profiles/transparency outside raw RGB contract");
 if(type=="IHDR"){need(!header&&n==13,"PNG header identity");need(u32(p)==W&&u32(p+4)==H&&p[8]==8&&p[9]==2&&p[10]==0&&p[11]==0&&p[12]==0,"PNG expected RGB8 noninterlaced dimensions");header=true;}
 if(type=="IDAT"){need(header&&!end,"PNG data order");compressed.insert(compressed.end(),p,p+n);}
 if(type=="IEND"){need(n==0,"PNG end length");end=true;at+=12;break;}
 at+=static_cast<std::size_t>(n)+12;
 }
 need(header&&end&&at==file.size(),"PNG completeness/trailing data");V filtered(static_cast<std::size_t>((W*3+1)*H));unsigned long out=filtered.size(),in=compressed.size();need(uncompress2(filtered.data(),&out,compressed.data(),&in)==0&&out==filtered.size()&&in==compressed.size(),"zlib size/input completeness");V rgb(static_cast<std::size_t>(W*H*3));
 for(I y=0;y<H;++y){const I line=y*(W*3+1);const int filter=filtered[line];need(filter<=4,"PNG row filter");for(I x=0;x<W*3;++x){const I index=y*W*3+x;const int a=x>=3?rgb[index-3]:0,b=y?rgb[index-W*3]:0,c=x>=3&&y?rgb[index-W*3-3]:0;const int predictor=filter==0?0:filter==1?a:filter==2?b:filter==3?(a+b)/2:paeth(a,b,c);rgb[index]=static_cast<B>((int(filtered[line+1+x])+predictor)&255);}}
 return rgb;
}
V ppm(const std::string&path){std::ifstream f(path,std::ios::binary);std::string magic;I w=0,h=0,max=0;f>>magic>>w>>h>>max;need(magic=="P6"&&w==W&&h==H&&max==255,"PPM header");need(f.get()=='\n',"PPM separator");V rgb(static_cast<std::size_t>(W*H*3));f.read(reinterpret_cast<char*>(rgb.data()),static_cast<std::streamsize>(rgb.size()));need(f.gcount()==static_cast<std::streamsize>(rgb.size())&&f.peek()==EOF,"PPM exact raster");return rgb;}
void put(V&rgb,I x,I y,I color){need(x>=0&&x<W&&y>=0&&y<H,"oracle pixel bounds");const auto i=static_cast<std::size_t>((y*W+x)*3);rgb[i]=B(color>>16);rgb[i+1]=B(color>>8);rgb[i+2]=B(color);}
I get(const V&rgb,I x,I y){const auto i=static_cast<std::size_t>((y*W+x)*3);return(I(rgb[i])<<16)|(I(rgb[i+1])<<8)|rgb[i+2];}
V native(const std::string&path){V rgb(static_cast<std::size_t>(W*H*3));std::vector<bool> seen(static_cast<std::size_t>(W*H));std::istringstream f(text(path));std::string row;I count=0;bool done=false;while(std::getline(f,row)){std::istringstream in(row);std::string tag;in>>tag;if(tag=="PALETTE_PIXEL"){I x,y,color;in>>x>>y>>color;need(bool(in)&&x>=0&&x<W&&y>=0&&y<H&&color>0&&color<=0xffffff,"native RGB record");const auto i=static_cast<std::size_t>(y*W+x);need(!seen[i],"native duplicate");seen[i]=true;put(rgb,x,y,color);++count;}if(tag=="CANONICAL_PALETTES_PASS")done=true;}need(done&&count>50000,"native completeness");std::cout<<"NATIVE_NONZERO "<<count<<'\n';return rgb;}
using Maps=std::array<std::array<I,256>,4>;
Maps primary(const std::string&path){const auto raw=text(path);const std::array<std::string,4> names{"viridis","plasma","inferno","magma"};Maps maps{};const std::regex triplet(R"(\[\s*([01]\.[0-9]{6})\s*,\s*([01]\.[0-9]{6})\s*,\s*([01]\.[0-9]{6})\s*\])");
 for(std::size_t p=0;p<4;++p){auto begin=raw.find("_"+names[p]+"_data = [");need(begin!=std::string::npos,"primary palette");begin=raw.find('[',begin);auto end=begin;int depth=0;do{need(end<raw.size(),"primary list bounds");if(raw[end]=='[')++depth;if(raw[end]==']')--depth;++end;}while(depth);const auto block=raw.substr(begin,end-begin);std::size_t index=0;for(std::sregex_iterator it(block.begin(),block.end(),triplet),last;it!=last;++it){need(index<256,"primary extra entry");I packed=0;for(int ch=1;ch<=3;++ch){std::string decimal=(*it)[ch];decimal.erase(1,1);const I numerator=std::stoll(decimal);need(numerator<=1000000,"primary channel");packed=(packed<<8)|(numerator*255/1000000);}maps[p][index++]=packed;}need(index==256,"primary entry count");}
 return maps;
}
void glyph(V&rgb,const std::array<std::array<B,8>,128>&font,const std::string&s,I x,I y){for(std::size_t i=0;i<s.size();++i){const auto ch=static_cast<unsigned char>(s[i]);for(I row=0;row<8;++row)for(I col=0;col<8;++col)if((font[ch][row]>>col)&1)put(rgb,x+static_cast<I>(i)*8+col,y+row,0xffffff);}}
V expected(const Maps&m,const std::string&fontpath){V out(static_cast<std::size_t>(W*H*3));for(I p=0;p<4;++p)for(I y=20+80*p;y<36+80*p;++y)for(I x=80;x<592;++x)put(out,x,y,m[p][(x-80)/2]);for(I y=340;y<356;++y)for(I x=80;x<592;++x)put(out,x,y,m[0][128]);
 const std::array<I,10> colors{0xff00ff,m[0][0],m[0][64],m[0][128],m[0][192],m[0][255],0x00ffff,0x808080,m[0][128],0x808080};for(I row=0;row<2;++row)for(I col=0;col<5;++col)for(I y=420+row*16;y<436+row*16;++y)for(I x=80+col*512/5;x<80+(col+1)*512/5;++x)put(out,x,y,colors[row*5+col]);
 std::array<std::array<B,8>,128> font{};std::array<bool,128> seen{};std::istringstream source(text(fontpath));std::string line;const std::regex hex(R"(0x([0-9a-fA-F]{2}))");while(std::getline(source,line)){const auto u=line.find("// U+");if(u==std::string::npos)continue;const auto ch=std::stoul(line.substr(u+5,4),nullptr,16);need(ch<128&&!seen[ch],"font row identity");seen[ch]=true;std::size_t row=0;for(std::sregex_iterator it(line.begin(),line.end(),hex),last;it!=last;++it){need(row<8,"font bytes");font[ch][row++]=static_cast<B>(std::stoul((*it)[1],nullptr,16));}need(row==8,"font row count");}for(bool v:seen)need(v,"font complete");
 const std::array<std::string,4> names{"viridis","plasma","inferno","magma"};for(I p=0;p<4;++p){glyph(out,font,"-1.00000e+0",36,44+p*80);glyph(out,font,"1.00000e+0",551,44+p*80);glyph(out,font,names[p],80,8+p*80);}glyph(out,font,"2.50000e+0",295,364);glyph(out,font,"viridis",80,328);return out;
}
void compare(const V&a,const V&b,const std::string&label){need(a.size()==b.size(),"raster size");for(std::size_t i=0;i<a.size();++i)if(a[i]!=b[i])throw std::runtime_error(label+" x="+std::to_string((i/3)%W)+" y="+std::to_string(i/(3*W))+" channel="+std::to_string(i%3)+" actual="+std::to_string(a[i])+" expected="+std::to_string(b[i]));std::cout<<"FULL_RASTER_MATCH "<<label<<" pixels="<<W*H<<" bytes="<<a.size()<<'\n';}
int main(int argc,char**argv){try{need(argc==6,"usage: DECODE PRIMARY FONT NATIVE_STDOUT PNG PPM");const auto palette=primary(argv[1]);const auto wanted=expected(palette,argv[2]);const auto observed=native(argv[3]);const auto decoded=png(argv[4]);const auto raw=ppm(argv[5]);compare(observed,wanted,"native_vs_independent_primary_fullcanvas");compare(decoded,wanted,"PNG_vs_independent_primary_fullcanvas");compare(raw,wanted,"PPM_vs_independent_primary_fullcanvas");for(I p=0;p<4;++p){for(I x:std::array<I,4>{80,200,400,590}){const I y=28+80*p;std::cout<<"PIXEL_SAMPLE palette="<<p+1<<" x="<<x<<" y="<<y<<" png="<<get(decoded,x,y)<<" native="<<get(observed,x,y)<<" primary="<<palette[p][(x-80)/2]<<'\n';}bool found=false;for(I y=44+80*p;y<52+80*p&&!found;++y)for(I x=551;x<631;++x)if(get(wanted,x,y)==0xffffff){std::cout<<"WHITE_ENDPOINT_LABEL palette="<<p+1<<" x="<<x<<" y="<<y<<" png="<<get(decoded,x,y)<<" native="<<get(observed,x,y)<<" expected=16777215\n";found=true;break;}need(found,"white endpoint expected");}std::cout<<"PNG_NATIVE_PRIMARY_REFERENCE_PASS\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
