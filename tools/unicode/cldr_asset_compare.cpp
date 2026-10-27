// C++23 byte-for-byte preservation of the qualified locale CE payload.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>
#include <string>
#include <utility>
using Bytes=std::vector<unsigned char>;
Bytes read(const char*p){std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("asset missing");return Bytes(std::istreambuf_iterator<char>(in),{});}
std::uint32_t u32(const Bytes&b,std::size_t p){if(p>b.size()||b.size()-p<4)throw std::runtime_error("u32 bounds");return std::uint32_t(b[p])|(std::uint32_t(b[p+1])<<8)|(std::uint32_t(b[p+2])<<16)|(std::uint32_t(b[p+3])<<24);}
struct Section{std::uint32_t width,count,offset;std::size_t bytes()const{return std::size_t(width)*count;}};
Section section(const Bytes&b,unsigned id){if(id==0||id>u32(b,20))throw std::runtime_error("section id");auto p=32+(id-1)*16;if(u32(b,p)!=id)throw std::runtime_error("directory identity");Section s{u32(b,p+4),u32(b,p+8),u32(b,p+12)};if(s.offset>b.size()||s.bytes()>b.size()-s.offset)throw std::runtime_error("section bounds");return s;}

std::string key(const Bytes&b,Section record,std::size_t row,unsigned field){auto strings=section(b,1);auto at=record.offset+row*record.width+field*4;auto first=u32(b,at),n=u32(b,at+4);if(first>strings.count||n>strings.count-first)throw std::runtime_error("metadata string bounds");std::string out;for(std::size_t i=0;i<n;++i){auto c=b[strings.offset+first+i];if(c>127)throw std::runtime_error("non-ASCII metadata key");if(c>='A'&&c<='Z')c=static_cast<unsigned char>(c-'A'+'a');if(c=='-')c='_';out.push_back(static_cast<char>(c));}return out;}
void metadata_order(const Bytes&b){auto aliases=section(b,11);std::pair<std::uint32_t,std::string> previous{};std::string language;
for(std::size_t i=0;i<aliases.count;++i){auto kind=u32(b,aliases.offset+i*aliases.width);auto name=key(b,aliases,i,1);auto current=std::pair(kind,name);if(kind<1||kind>4||(i&&current<=previous))throw std::runtime_error("alias (kind,key) order");if(kind==1){auto first=name.substr(0,name.find('_'));if(!language.empty()&&first<language)throw std::runtime_error("alias language prefix order");language=first;}previous=current;}
auto locales=section(b,9);language.clear();for(std::size_t i=0;i<locales.count;++i){auto name=key(b,locales,i,0);auto first=name.substr(0,name.find('_'));if(!language.empty()&&first<language)throw std::runtime_error("locale language prefix order");language=first;}
auto likely=section(b,12);std::string last;for(std::size_t i=0;i<likely.count;++i){auto name=key(b,likely,i,0);if(i&&name<=last)throw std::runtime_error("likely key order");last=name;}
std::cout<<"CLDR_METADATA_BINARY_SEARCH_ORDER_OK aliases="<<aliases.count<<" locale_rows="<<locales.count<<" likely="<<likely.count<<"\n";
}

int main(int argc,char**argv){try{if(argc!=3)return 64;auto a=read(argv[1]),b=read(argv[2]);if(u32(a,8)!=1||u32(b,8)!=2||u32(a,12)!=0x120000||u32(b,12)!=0x120000||u32(a,16)!=49||u32(b,16)!=49||u32(a,24)!=u32(b,24))throw std::runtime_error("header versions");metadata_order(b);std::size_t bytes=0;for(unsigned id=2;id<=8;++id){auto x=section(a,id),y=section(b,id);if(x.width!=y.width||x.count!=y.count||!std::equal(a.begin()+x.offset,a.begin()+x.offset+x.bytes(),b.begin()+y.offset))throw std::runtime_error("qualified CE payload changed");bytes+=x.bytes();}
auto locales=section(a,5),strings_a=section(a,1),strings_b=section(b,1);std::size_t prefix=0;for(std::size_t i=0;i<locales.count;++i)for(unsigned field:{0U,2U,4U}){auto at=locales.offset+i*locales.width+field*4;auto first=u32(a,at),n=u32(a,at+4);if(first>strings_a.count||n>strings_a.count-first)throw std::runtime_error("locale string bounds");prefix=std::max(prefix,std::size_t(first)+n);}if(prefix>strings_b.count||!std::equal(a.begin()+strings_a.offset,a.begin()+strings_a.offset+prefix,b.begin()+strings_b.offset))throw std::runtime_error("qualified locale identities changed");
std::cout<<"CLDR_QUALIFIED_CE_PAYLOAD_PRESERVED_OK sections=7 bytes="<<bytes<<" public_locales="<<locales.count<<" identity_prefix="<<prefix<<" aliases="<<section(b,11).count<<" likely="<<section(b,12).count<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"CLDR_ASSET_COMPARE_ERROR "<<e.what()<<"\n";return 1;}}

