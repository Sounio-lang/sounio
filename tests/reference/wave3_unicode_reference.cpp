// Independent ICU79.1 oracle for the serialized Unicode18 table, C++23 only.
// This executable is a test tool. Sounio production code never links ICU.
#include <unicode/uchar.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <unicode/uversion.h>
#include <unicode/ulocdata.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
using U=std::uint32_t;
using Seq=std::vector<U>;
[[noreturn]] void fail(const std::string& s){throw std::runtime_error(s);}
struct Section{U width,count,offset;};
struct Blob{
 std::vector<std::uint8_t> bytes;std::map<U,Section> sections;
 explicit Blob(const std::string& path){
  std::ifstream in(path,std::ios::binary);bytes.assign(std::istreambuf_iterator<char>(in),{});if(!in.eof()&&in.fail())fail("read blob");
  if(bytes.size()<24||std::string(bytes.begin(),bytes.begin()+8)!="SOUNUCD1"||u32(8)!=1||u32(12)!=0x00120000)fail("blob header");
  U count=u32(16);if(count!=8)fail("section count");
  for(U i=0;i<count;++i){U p=24+16*i,id=u32(p);Section s{u32(p+4),u32(p+8),u32(p+12)};
   if(std::uint64_t{s.offset}+std::uint64_t{s.width}*s.count>bytes.size()||!sections.emplace(id,s).second)fail("section bounds/duplicate");
  }
 }
 U u32(std::size_t p) const {
  if(p>bytes.size()||bytes.size()-p<4)fail("u32 bounds");U v=0;for(U i=0;i<4;++i)v|=U{bytes[p+i]}<<(8*i);return v;
 }
 U at(U id,U i,U col) const {const auto& s=sections.at(id);if(i>=s.count||col*4>=s.width)fail("record bounds");return u32(s.offset+i*s.width+col*4);}
 std::string str(U p) const {
  const auto& s=sections.at(1);if(p>=s.count)fail("string offset");auto start=s.offset+p,end=start;
  while(end<s.offset+s.count&&bytes[end])++end;if(end==s.offset+s.count)fail("unterminated string");return {bytes.begin()+start,bytes.begin()+end};
 }
 U property(const std::string& name) const {
  for(U i=0;i<sections.at(2).count;++i)if(str(at(2,i,0))==name)return i;fail("missing property: "+name);
 }
 std::string value(U prop,U cp) const {
  U first=at(2,prop,1),count=at(2,prop,2),lo=0,hi=count;
  while(lo<hi){U m=lo+(hi-lo)/2,atrow=first+m;if(cp<at(3,atrow,0))hi=m;else if(cp>at(3,atrow,1))lo=m+1;else return str(at(3,atrow,2));}
  return str(at(2,prop,3));
 }
 Seq mapping(U row) const {Seq s;U first=at(4,row,3),n=at(4,row,4);for(U i=0;i<n;++i)s.push_back(at(5,first+i,0));return s;}
};
std::string trim(std::string s){auto p=s.find_first_not_of(" \t\r\n");if(p==s.npos)return {};return s.substr(p,s.find_last_not_of(" \t\r\n")-p+1);}
std::vector<std::string> split(const std::string& s,char c){std::vector<std::string> r;std::size_t p=0;for(;;){auto e=s.find(c,p);r.push_back(trim(s.substr(p,e==s.npos?e:e-p)));if(e==s.npos)return r;p=e+1;}}
Seq parse(const std::string& s){Seq r;std::istringstream in(s);std::string x;while(in>>x)r.push_back(static_cast<U>(std::stoul(x,nullptr,16)));return r;}
std::vector<UChar> utf16(const Seq& s){std::vector<UChar> r;for(U c:s){if(c<0x10000)r.push_back(static_cast<UChar>(c));else{c-=0x10000;r.push_back(static_cast<UChar>(0xD800+(c>>10)));r.push_back(static_cast<UChar>(0xDC00+(c&1023)));}}return r;}
Seq scalars(std::span<const UChar> s){Seq r;for(std::size_t i=0;i<s.size();++i){U c=s[i];if(c>=0xD800&&c<=0xDBFF){if(++i>=s.size()||s[i]<0xDC00||s[i]>0xDFFF)fail("oracle UTF16");c=0x10000+((c-0xD800)<<10)+(s[i]-0xDC00);}r.push_back(c);}return r;}
Seq icu_normalize(const UNormalizer2* norm,const Seq& in){
 auto input=utf16(in);std::array<UChar,8192> out{};UErrorCode error=U_ZERO_ERROR;
 auto n=unorm2_normalize(norm,input.data(),static_cast<int32_t>(input.size()),out.data(),out.size(),&error);
 if(U_FAILURE(error)||n<0||static_cast<std::size_t>(n)>out.size())fail("ICU normalization error");
 return scalars(std::span<const UChar>(out.data(),static_cast<std::size_t>(n)));
}
struct Normalizer{
 const Blob& b;U ccc_property;std::map<U,std::pair<bool,Seq>> decomposition;std::map<std::pair<U,U>,U> composition;
 explicit Normalizer(const Blob& blob):b(blob),ccc_property(b.property("Canonical_Combining_Class")){
  for(U i=0;i<b.sections.at(4).count;++i)if(b.at(4,i,0)==1)decomposition.emplace(b.at(4,i,1),std::pair{b.at(4,i,2)!=0,b.mapping(i)});
  for(U i=0;i<b.sections.at(6).count;++i)if(!composition.emplace(std::pair{b.at(6,i,0),b.at(6,i,1)},b.at(6,i,2)).second)fail("duplicate composition");
 }
 U ccc(U c)const{return static_cast<U>(std::stoul(b.value(ccc_property,c)));}
 void decompose(U c,bool compat,Seq& out,U depth=0) const {
  if(depth>64)fail("cyclic/deep decomposition");
  if(c>=0xAC00&&c<0xD7A4){U i=c-0xAC00;out.push_back(0x1100+i/588);out.push_back(0x1161+(i%588)/28);if(i%28)out.push_back(0x11A7+i%28);return;}
  auto it=decomposition.find(c);if(it!=decomposition.end()&&(compat||!it->second.first)){for(U d:it->second.second)decompose(d,compat,out,depth+1);}else out.push_back(c);
 }
 U compose(U a,U c)const{
  if(a>=0x1100&&a<0x1113&&c>=0x1161&&c<0x1176)return 0xAC00+(a-0x1100)*588+(c-0x1161)*28;
  if(a>=0xAC00&&a<0xD7A4&&(a-0xAC00)%28==0&&c>0x11A7&&c<0x11C3)return a+c-0x11A7;
  auto i=composition.find({a,c});return i==composition.end()?0:i->second;
 }
 Seq normalize(const Seq& input,bool compat,bool composed)const {
  Seq out;for(U c:input)decompose(c,compat,out);
  for(std::size_t i=1;i<out.size();++i){const U cls=ccc(out[i]);if(!cls)continue;std::size_t j=i;while(j>0&&ccc(out[j-1])>cls){std::swap(out[j],out[j-1]);--j;}}
  if(!composed||out.empty())return out;
  Seq result={out[0]};std::size_t starter=0;U last=ccc(out[0]);
  for(std::size_t i=1;i<out.size();++i){U c=out[i],cls=ccc(c),joint=compose(result[starter],c);
   if(joint&&(last<cls||last==0))result[starter]=joint;
   else {if(cls==0)starter=result.size();result.push_back(c);last=cls;}
  }
  return result;
 }
};
void version_guard(){
 UVersionInfo uv{},cldr{};u_getUnicodeVersion(uv);UErrorCode e=U_ZERO_ERROR;ulocdata_getCLDRVersion(cldr,&e);
 if(U_ICU_VERSION_MAJOR_NUM!=79||U_ICU_VERSION_MINOR_NUM!=1||uv[0]!=18||uv[1]!=0||cldr[0]!=49||U_FAILURE(e))fail("expected ICU79.1/Unicode18.0/CLDR49");
 std::cout<<"ORACLE_VERSION ICU79.1 Unicode18.0 CLDR49\n";
}
void properties(const Blob& b){
 std::uint64_t comparisons=0;U checked=0,skipped=0;
 for(U p=0;p<b.sections.at(2).count;++p){
  const auto name=b.str(b.at(2,p,0));UProperty prop=u_getPropertyEnum(name.c_str());
  bool binary=prop>=UCHAR_BINARY_START&&prop<UCHAR_BINARY_LIMIT;
  bool integer=prop>=UCHAR_INT_START&&prop<UCHAR_INT_LIMIT;
  if(!binary&&!integer){++skipped;std::cout<<"PROPERTY_OUTSIDE_ICU_INT "<<name<<"\n";continue;}
  std::map<std::string,int32_t> enums;
  U first=b.at(2,p,1),count=b.at(2,p,2);std::vector<U> ids={b.at(2,p,3)};for(U i=0;i<count;++i)ids.push_back(b.at(3,first+i,2));
  for(U id:ids){auto value=b.str(id);if(value.empty())continue;int32_t v=prop==UCHAR_CANONICAL_COMBINING_CLASS?std::stoi(value):u_getPropertyValueEnum(prop,value.c_str());if(v==UCHAR_INVALID_CODE)fail("ICU value alias "+name+"="+value);enums.emplace(value,v);}
  U row=first,end=first+count;std::string current;
  for(U cp=0;cp<0x110000;++cp){
   while(row<end&&b.at(3,row,1)<cp)++row;
   U id=row<end&&b.at(3,row,0)<=cp?b.at(3,row,2):b.at(2,p,3);
   auto value=b.str(id);if(value.empty())fail("undefined property default "+name);
   int32_t actual=binary?int32_t{u_hasBinaryProperty(static_cast<UChar32>(cp),prop)}:u_getIntPropertyValue(static_cast<UChar32>(cp),prop);
   if(actual!=enums.at(value))fail("property mismatch "+name+" cp="+std::to_string(cp)+" table="+value+" ICU="+std::to_string(actual));
   ++comparisons;
  }
  ++checked;
 }
 if(checked<50)fail("property oracle coverage unexpectedly small");
 std::cout<<"PROPERTY_ORACLE_OK properties="<<checked<<" outside_ICU_int="<<skipped<<" comparisons="<<comparisons<<"\n";
}
void mappings(const Blob& b){
 std::map<U,Seq> folded,turkic;
 std::uint64_t simple=0;
 for(U i=0;i<b.sections.at(4).count;++i){
  U kind=b.at(4,i,0),cp=b.at(4,i,1);auto result=b.mapping(i);
  if(kind>=3&&kind<=5){U actual=static_cast<U>(kind==3?u_toupper(cp):kind==4?u_tolower(cp):u_totitle(cp));if(result!=Seq{actual})fail("simple case mapping mismatch");++simple;}
  if(kind==6&&!folded.emplace(cp,result).second)fail("duplicate fold");
  if(kind==7&&!turkic.emplace(cp,result).second)fail("duplicate Turkic fold");
 }
 for(U cp=0;cp<0x110000;++cp){
  if(cp>=0xD800&&cp<=0xDFFF)continue;
  auto input=utf16(Seq{cp});
  for(int mode=0;mode<2;++mode){Seq expected=folded.contains(cp)?folded.at(cp):Seq{cp};if(mode&&turkic.contains(cp))expected=turkic.at(cp);
   std::array<UChar,32> out{};UErrorCode e=U_ZERO_ERROR;
   int32_t n=u_strFoldCase(out.data(),out.size(),input.data(),static_cast<int32_t>(input.size()),mode?U_FOLD_CASE_EXCLUDE_SPECIAL_I:U_FOLD_CASE_DEFAULT,&e);
   if(U_FAILURE(e)||scalars(std::span<const UChar>(out.data(),static_cast<std::size_t>(n)))!=expected)fail("fold mismatch cp="+std::to_string(cp)+" mode="+std::to_string(mode));
  }
 }
 std::cout<<"CASE_ORACLE_OK simple="<<simple<<" fold_scalar_modes=2224128\n";
}
void normalization(const Blob& b,const std::string& path){
 Normalizer n(b);UErrorCode e=U_ZERO_ERROR;
 const std::array<const UNormalizer2*,4> norms={unorm2_getNFCInstance(&e),unorm2_getNFDInstance(&e),unorm2_getNFKCInstance(&e),unorm2_getNFKDInstance(&e)};
 if(U_FAILURE(e))fail("normalizer instances");
 std::ifstream in(path);if(!in)fail("NormalizationTest open");std::string line;U rows=0;
 while(std::getline(in,line)){
  line=trim(line.substr(0,line.find('#')));if(line.empty()||line.front()=='@')continue;
  auto v=split(line,';');if(v.size()<5)fail("NormalizationTest fields");std::array<Seq,5> seq;
  for(U i=0;i<5;++i)seq[i]=parse(v[i]);
  for(U i=0;i<5;++i)for(U form=0;form<4;++form){
   const Seq& expected=form==0?(i<3?seq[1]:seq[3]):form==1?(i<3?seq[2]:seq[4]):form==2?seq[3]:seq[4];
   auto actual=n.normalize(seq[i],form>=2,form%2==0);
   if(actual!=expected)fail("table normalization mismatch row="+std::to_string(rows)+" form="+std::to_string(form)+" input="+std::to_string(i));
   if(icu_normalize(norms[form],seq[i])!=expected)fail("ICU conformance mismatch row="+std::to_string(rows));
  }
  ++rows;
 }
 if(rows<19000)fail("normalization corpus unexpectedly small");
 std::cout<<"NORMALIZATION_TABLE_ORACLE_OK rows="<<rows<<" table_and_ICU_checks="<<std::uint64_t{rows}*40<<"\n";
}
int main(int argc,char** argv){
 try{
  if(argc==4&&std::string(argv[1])=="--mutate"){
   Blob b(argv[2]);U p=b.property("General_Category"),first=b.at(2,p,1),count=b.at(2,p,2),target=0,value=0;
   bool found=false;
   for(U i=0;i<count;++i){U row=first+i;if(b.at(3,row,0)<=65&&b.at(3,row,1)>=65){target=row;found=true;}if(b.at(3,row,0)<=97&&b.at(3,row,1)>=97)value=b.at(3,row,2);}
   if(!found||!value)fail("mutation target");
   U offset=b.sections.at(3).offset+target*12+8;
   for(U i=0;i<4;++i)b.bytes[offset+i]=static_cast<std::uint8_t>((value>>(8*i))&255);
   std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<const char*>(b.bytes.data()),static_cast<std::streamsize>(b.bytes.size()));if(!out)fail("mutation output");
   std::cout<<"MUTATED_ASCII_UPPERCASE_CATEGORY\n";return 0;
  }
  if(argc!=3){std::cerr<<"usage: unicode_reference UCD.bin NormalizationTest.txt\n";return 64;}
  version_guard();Blob b(argv[1]);properties(b);mappings(b);normalization(b,argv[2]);std::cout<<"UNICODE_TABLE_REFERENCE_OK\n";return 0;
 }
 catch(const std::exception& e){std::cerr<<"UNICODE_TABLE_REFERENCE_ERROR "<<e.what()<<"\n";return 1;}
}
