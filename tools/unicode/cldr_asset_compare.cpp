// C++23 byte-for-byte preservation of the qualified locale CE payload.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>
using Bytes=std::vector<unsigned char>;
Bytes read(const char*p){std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("asset missing");return Bytes(std::istreambuf_iterator<char>(in),{});}
std::uint32_t u32(const Bytes&b,std::size_t p){if(p>b.size()||b.size()-p<4)throw std::runtime_error("u32 bounds");return std::uint32_t(b[p])|(std::uint32_t(b[p+1])<<8)|(std::uint32_t(b[p+2])<<16)|(std::uint32_t(b[p+3])<<24);}
struct Section{std::uint32_t width,count,offset;std::size_t bytes()const{return std::size_t(width)*count;}};
Section section(const Bytes&b,unsigned id){if(id==0||id>u32(b,20))throw std::runtime_error("section id");auto p=32+(id-1)*16;if(u32(b,p)!=id)throw std::runtime_error("directory identity");Section s{u32(b,p+4),u32(b,p+8),u32(b,p+12)};if(s.offset>b.size()||s.bytes()>b.size()-s.offset)throw std::runtime_error("section bounds");return s;}
int main(int argc,char**argv){try{if(argc!=3)return 64;auto a=read(argv[1]),b=read(argv[2]);if(u32(a,8)!=1||u32(b,8)!=2||u32(a,12)!=0x120000||u32(b,12)!=0x120000||u32(a,16)!=49||u32(b,16)!=49||u32(a,24)!=u32(b,24))throw std::runtime_error("header versions");std::size_t bytes=0;for(unsigned id=2;id<=8;++id){auto x=section(a,id),y=section(b,id);if(x.width!=y.width||x.count!=y.count||!std::equal(a.begin()+x.offset,a.begin()+x.offset+x.bytes(),b.begin()+y.offset))throw std::runtime_error("qualified CE payload changed");bytes+=x.bytes();}
auto locales=section(a,5),strings_a=section(a,1),strings_b=section(b,1);std::size_t prefix=0;for(std::size_t i=0;i<locales.count;++i)for(unsigned field:{0U,2U,4U}){auto at=locales.offset+i*locales.width+field*4;auto first=u32(a,at),n=u32(a,at+4);if(first>strings_a.count||n>strings_a.count-first)throw std::runtime_error("locale string bounds");prefix=std::max(prefix,std::size_t(first)+n);}if(prefix>strings_b.count||!std::equal(a.begin()+strings_a.offset,a.begin()+strings_a.offset+prefix,b.begin()+strings_b.offset))throw std::runtime_error("qualified locale identities changed");
std::cout<<"CLDR_QUALIFIED_CE_PAYLOAD_PRESERVED_OK sections=7 bytes="<<bytes<<" public_locales="<<locales.count<<" identity_prefix="<<prefix<<" aliases="<<section(b,11).count<<" likely="<<section(b,12).count<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"CLDR_ASSET_COMPARE_ERROR "<<e.what()<<"\n";return 1;}}

