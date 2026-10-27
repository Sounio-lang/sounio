// Test-only parser for unmodified Unicode18 UAX9 conformance data.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace bidi_cases {
using U=std::uint32_t;
struct Case {std::vector<U> cps;std::vector<int> levels;std::vector<U> order;U mode=0,base=0,kind=0,line=0;};
inline void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
inline std::string trim(std::string s){const auto a=s.find_first_not_of(" \t\r\n");if(a==std::string::npos)return {};return s.substr(a,s.find_last_not_of(" \t\r\n")-a+1);}
inline std::vector<std::string> split(const std::string& s,char delimiter){
 std::vector<std::string> out;std::size_t p=0;
 for(;;){const auto end=s.find(delimiter,p);out.push_back(trim(s.substr(p,end==std::string::npos?end:end-p)));if(end==std::string::npos)return out;p=end+1;}
}
inline U number(const std::string& s,int base){
 need(!s.empty()&&s[0]!='-'&&s[0]!='+',"invalid unsigned field");
 std::size_t n=0;const auto x=std::stoull(s,&n,base);need(n==s.size()&&x<=0xffffffffULL,"unsigned field overflow");return static_cast<U>(x);
}
inline std::vector<std::string> words(const std::string& s){std::istringstream in(s);std::vector<std::string> out;std::string w;while(in>>w)out.push_back(w);return out;}
inline std::vector<int> levels(const std::string& s){std::vector<int> out;for(const auto& w:words(s)){if(w=="x"){out.push_back(-1);}else{const U v=number(w,10);need(v<=126,"invalid expected level");out.push_back(static_cast<int>(v));}}return out;}
inline std::vector<U> order(const std::string& s){std::vector<U> out;for(const auto& w:words(s))out.push_back(number(w,10));return out;}
inline const std::array<std::string,23> names{"L","R","AL","EN","ES","ET","AN","CS","NSM","BN","B","S","WS","ON","LRE","LRO","RLE","RLO","PDF","LRI","RLI","FSI","PDI"};
inline const std::array<U,23> representatives{0x61,0x5d0,0x627,0x30,0x2b,0x24,0x660,0x2c,0x300,0xad,0x2029,9,0x20,0x21,0x202a,0x202d,0x202b,0x202e,0x202c,0x2066,0x2067,0x2068,0x2069};
inline U class_id(const std::string& s){const auto at=std::find(names.begin(),names.end(),s);need(at!=names.end(),"unknown Bidi_Class token: "+s);return static_cast<U>(at-names.begin());}
inline void valid(const Case& c){
 need(c.cps.size()==c.levels.size(),"levels/input length mismatch");std::vector<bool> used(c.cps.size(),false);std::size_t wanted=0;
 for(std::size_t i=0;i<c.cps.size();++i){need(c.cps[i]<=0x10ffff&&!(c.cps[i]>=0xd800&&c.cps[i]<=0xdfff),"non-scalar official input");if(c.levels[i]>=0)++wanted;}
 need(c.order.size()==wanted,"expected reordering cardinality");
 for(U i:c.order){need(i<c.cps.size()&&!used[i]&&c.levels[i]>=0,"invalid/duplicate reordering index");used[i]=true;}
}
inline U auto_base(const std::vector<U>& types){
 U depth=0;
 for(U c:types){if(c==19||c==20||c==21){++depth;continue;}if(c==22){if(depth)--depth;continue;}if(depth)continue;if(c==0)return 0;if(c==1||c==2)return 1;}
 return 0;
}
template<class Callback> inline void read(const std::filesystem::path& root,Callback emit){
 for(U kind=0;kind<2;++kind){
  const auto file=root/(kind==0?"BidiTest.txt":"BidiCharacterTest.txt");std::ifstream in(file);need(bool(in),"open "+file.string());
  std::string line;U line_number=0;bool version=false,have_levels=false,have_order=false;std::vector<int> lv;std::vector<U> ro;
  while(std::getline(in,line)){
   ++line_number;if(line.find(kind==0?"BidiTest-18.0.0.txt":"BidiCharacterTest-18.0.0.txt")!=std::string::npos)version=true;
   line=trim(line.substr(0,line.find('#')));if(line.empty())continue;
   if(kind==0&&line.starts_with("@Levels:")){lv=levels(line.substr(8));have_levels=true;continue;}
   if(kind==0&&line.starts_with("@Reorder:")){ro=order(line.substr(9));have_order=true;continue;}
   if(line.starts_with("@"))continue;
   need(version,"unversioned/wrong UCD corpus");
   const auto fields=split(line,';');
   if(kind==0){
    need(fields.size()==2&&have_levels&&have_order,"BidiTest row schema");std::vector<U> types,cps;
    for(const auto& word:words(fields[0])){const U type=class_id(word);types.push_back(type);cps.push_back(representatives[type]);}
    const U modes=number(fields[1],16);need(modes>0&&(modes&~7U)==0,"invalid paragraph bitset");
    for(U bit=1;bit<=4;bit*=2)if(modes&bit){
     Case c;c.cps=cps;c.levels=lv;c.order=ro;c.mode=bit==1?2:bit==2?0:1;c.base=c.mode==2?auto_base(types):c.mode;c.kind=kind;c.line=line_number;valid(c);emit(c);
    }
   }else{
    need(fields.size()==5,"BidiCharacterTest row schema");Case c;
    for(const auto& word:words(fields[0]))c.cps.push_back(number(word,16));
    c.mode=number(fields[1],10);c.base=number(fields[2],10);need(c.mode<=2&&c.base<=1,"invalid paragraph fields");
    c.levels=levels(fields[3]);c.order=order(fields[4]);c.kind=kind;c.line=line_number;valid(c);emit(c);
   }
  }
  need(version,"missing Unicode18 corpus header");
 }
}
inline void put(std::ostream& out,U value){for(int i=0;i<4;++i)out.put(static_cast<char>((value>>(i*8))&255));need(bool(out),"write corpus");}
inline std::uint64_t encoded_size(const Case& c){return 24ULL+c.cps.size()*8ULL+c.order.size()*4ULL;}
inline void write(std::ostream& out,const Case& c){
 put(out,static_cast<U>(c.cps.size()));put(out,c.mode);put(out,c.base);put(out,static_cast<U>(c.order.size()));put(out,c.kind);put(out,c.line);
 for(U x:c.cps){put(out,x);}
 for(int x:c.levels){put(out,static_cast<U>(x));}
 for(U x:c.order){put(out,x);}
}
}
