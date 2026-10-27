// Independent ICU79.1 oracle for the serialized Unicode18 table, C++23 only.
// This executable is a test tool. Sounio production code never links ICU.
#include <unicode/uchar.h>
#include <unicode/ubrk.h>
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

struct CaseMapper {
 const Blob& b;U cased,ignorable,soft,ccc;
 std::map<std::pair<U,U>,std::vector<U>> rows;
 explicit CaseMapper(const Blob& blob):b(blob),cased(b.property("Cased")),ignorable(b.property("Case_Ignorable")),soft(b.property("Soft_Dotted")),ccc(b.property("Canonical_Combining_Class")){
  for(U i=0;i<b.sections.at(4).count;++i)rows[{b.at(4,i,0),b.at(4,i,1)}].push_back(i);
 }
 bool has(U prop,U cp)const{return b.value(prop,cp)=="Y";}
 U combining(U cp)const{return static_cast<U>(std::stoul(b.value(ccc,cp)));}
 bool cased_context(const Seq& in,std::size_t index,int direction)const{
  auto at=static_cast<std::int64_t>(index)+direction;
  while(at>=0&&static_cast<std::size_t>(at)<in.size()){
   U cp=in[static_cast<std::size_t>(at)];if(!has(ignorable,cp))return has(cased,cp);at+=direction;
  }return false;
 }
 bool after(const Seq& in,std::size_t index,bool dotted)const{
  for(std::size_t at=index;at>0;){U cp=in[--at];if(dotted?has(soft,cp):cp==0x49)return true;U cc=combining(cp);if(cc==0||cc==230)return false;}return false;
 }
 bool before(const Seq& in,std::size_t index,bool dot)const{
  for(std::size_t at=index+1;at<in.size();++at){U cp=in[at];if(dot&&cp==0x307)return true;U cc=combining(cp);if(!dot&&cc==230)return true;if(cc==0||cc==230)return false;}return false;
 }
 bool condition(const Seq& in,std::size_t at,const std::string& conditions,U locale)const{
  bool matches=true;std::istringstream input(conditions);std::string token;
  while(input>>token){
   bool result=false;
   if(token=="tr")result=locale==1;
   else if(token=="az")result=locale==2;
   else if(token=="lt")result=locale==3;
   else if(token=="Final_Sigma")result=cased_context(in,at,-1)&&!cased_context(in,at,1);
   else if(token=="After_Soft_Dotted")result=after(in,at,true);
   else if(token=="After_I")result=after(in,at,false);
   else if(token=="More_Above")result=before(in,at,false);
   else if(token=="Not_Before_Dot")result=!before(in,at,true);
   else fail("unknown SpecialCasing condition: "+token);
   matches=matches&&result;
  }return matches;
 }
 Seq map(const Seq& in,bool upper,U locale)const{
  Seq result;U kind=upper?10:8,simple=upper?3:4;
  for(std::size_t at=0;at<in.size();++at){
   auto group=rows.find({kind,in[at]});std::int64_t selected=-1,fallback=-1;
   if(group!=rows.end())for(U row:group->second){
    auto cond=b.str(b.at(4,row,5));if(cond.empty())fallback=row;
    else if(condition(in,at,cond,locale)){selected=row;break;}
   }
   if(selected<0)selected=fallback;
   if(selected<0){auto s=rows.find({simple,in[at]});if(s!=rows.end())selected=s->second.front();}
   if(selected<0)result.push_back(in[at]);
   else {auto mapped=b.mapping(static_cast<U>(selected));result.insert(result.end(),mapped.begin(),mapped.end());}
  }return result;
 }
};
Seq icu_case(const Seq& in,bool upper,U locale){
 static constexpr const char* locales[]={"","tr","az","lt"};
 auto input=utf16(in);std::array<UChar,512> out{};UErrorCode e=U_ZERO_ERROR;
 int32_t n=upper?u_strToUpper(out.data(),out.size(),input.data(),static_cast<int32_t>(input.size()),locales[locale],&e):u_strToLower(out.data(),out.size(),input.data(),static_cast<int32_t>(input.size()),locales[locale],&e);
 if(U_FAILURE(e)||n<0||static_cast<std::size_t>(n)>out.size())fail("ICU full casing error");
 return scalars(std::span<const UChar>(out.data(),static_cast<std::size_t>(n)));
}
void full_casing(const Blob& b){
 CaseMapper mapper(b);std::uint64_t checks=0;
 auto check=[&](const Seq& in){
  for(U locale=0;locale<4;++locale)for(bool upper:{false,true}){
   if(mapper.map(in,upper,locale)!=icu_case(in,upper,locale)){
    std::ostringstream s;s<<"full case mismatch locale="<<locale<<" upper="<<upper<<" input=";for(U cp:in)s<<std::hex<<cp<<" ";fail(s.str());
   }++checks;
  }
 };
 for(U cp=0;cp<0x110000;++cp)if(cp<0xD800||cp>0xDFFF)check(Seq{cp});
 const Seq alphabet={0,0x20,0x21,0x41,0x61,0x49,0x69,0x4A,0xDF,0x130,0x12E,0x307,0x300,0x323,0x345,0x2B0,0x3A3};
 const Seq targets={0x49,0x4A,0x12E,0xCC,0xCD,0x128,0x130,0x69,0x307,0x3A3,0xDF,0xFB03};
 for(U target:targets)for(U before:alphabet)for(U after:alphabet){
  check(Seq{before,target,after});
  check(Seq{before,0x323,target,0x323,after});
  check(Seq{before,0x300,target,0x300,after});
 }
 std::cout<<"FULL_CASING_TABLE_ORACLE_OK locales=default,tr,az,lt checks="<<checks<<"\n";
}


struct Segmenter {
 const Blob& b;
 U gcb,wb,sb,ep,incb;
 explicit Segmenter(const Blob& x):b(x),gcb(b.property("Grapheme_Cluster_Break")),wb(b.property("Word_Break")),sb(b.property("Sentence_Break")),ep(b.property("Extended_Pictographic")),incb(b.property("Indic_Conjunct_Break")){}
 static bool word_ignore(int c){return c==4||c==5||c==6;}
 static bool sentence_ignore(int c){return c==4||c==5;}
 static bool word_newline(int c){return c>=1&&c<=3;}
 static bool ah(int c){return c==8||c==9;}
 static bool midletter(int c){return c==11||c==13||c==14;}
 static bool midnum(int c){return c==12||c==13||c==14;}
 static bool para(int c){return c>=1&&c<=3;}
 static bool sat(int c){return c==6||c==7;}
 static int cls(const std::vector<int>& codes,std::int64_t i){return i<0||static_cast<std::size_t>(i)>=codes.size()?-1:codes[static_cast<std::size_t>(i)]&255;}
 static int previous(const std::vector<int>& codes,std::int64_t i,int mode){
  while(i>=0&&(mode==1?word_ignore(cls(codes,i)):sentence_ignore(cls(codes,i))))--i;return static_cast<int>(i);
 }
 static int next(const std::vector<int>& codes,std::int64_t i,int mode){
  while(static_cast<std::size_t>(i)<codes.size()&&(mode==1?word_ignore(cls(codes,i)):sentence_ignore(cls(codes,i))))++i;return static_cast<int>(i);
 }
 std::vector<int> encode(const Seq& input,int mode)const{
  static const std::array<std::map<std::string,int>,3> names={{
   {{"XX",0},{"CR",1},{"LF",2},{"CN",3},{"L",4},{"V",5},{"T",6},{"LV",7},{"LVT",8},{"EX",9},{"ZWJ",10},{"SM",11},{"PP",12},{"RI",13}},
   {{"XX",0},{"CR",1},{"LF",2},{"NL",3},{"Extend",4},{"FO",5},{"ZWJ",6},{"WSegSpace",7},{"LE",8},{"HL",9},{"NU",10},{"ML",11},{"MN",12},{"MB",13},{"SQ",14},{"DQ",15},{"KA",16},{"EX",17},{"RI",18}},
   {{"XX",0},{"CR",1},{"LF",2},{"SE",3},{"EX",4},{"FO",5},{"AT",6},{"ST",7},{"CL",8},{"SP",9},{"LO",10},{"UP",11},{"LE",12},{"NU",13},{"SC",14}}
  }};
  U prop=mode==0?gcb:mode==1?wb:sb;std::vector<int> result;int ri=0,close=0,space=0;
  for(U cp:input){
   const auto name=b.value(prop,cp);auto it=names[static_cast<std::size_t>(mode)].find(name);if(it==names[mode].end())fail("unknown segmentation class "+name);int c=it->second,v=c;
   if(mode<2&&b.value(ep,cp)=="Y")v|=256;
   if(mode==0){auto in=b.value(incb,cp);int k=in=="None"?0:in=="Consonant"?1:in=="Extend"?2:in=="Linker"?3:-1;if(k<0)fail("unknown InCB");v|=k<<9;ri=c==13?1-ri:0;}
   else if(mode==1){if(c==18)ri=1-ri;else if(!word_ignore(c))ri=0;}
   else{
    if(c==6||c==7){close=c-5;space=close;}
    else if(c==8){space=close;}
    else if(c==9){close=0;}
    else if(!sentence_ignore(c)){close=0;space=0;}
    v|=close<<12;v|=space<<14;
   }
   v|=ri<<11;result.push_back(v);
  }return result;
 }
 static bool grapheme(const std::vector<int>& c,std::size_t i){
  int a=cls(c,i-1),z=cls(c,i);
  if(a==1&&z==2)return false;
  if(a>=1&&a<=3||z>=1&&z<=3)return true;
  if(a==4&&(z==4||z==5||z==7||z==8))return false;
  if((a==5||a==7)&&(z==5||z==6))return false;
  if((a==6||a==8)&&z==6)return false;
  if(z==9||z==10||z==11||a==12)return false;
  if(((c[i]>>9)&3)==1){
   auto j=static_cast<std::int64_t>(i)-1;
   while(j>=0&&((c[static_cast<std::size_t>(j)]>>9)&3)==2)--j;
   if(j>=0&&((c[static_cast<std::size_t>(j)]>>9)&3)==3)return false;
  }
  if(a==10&&(c[i]&256)){auto j=static_cast<std::int64_t>(i)-2;while(j>=0&&cls(c,j)==9)--j;if(j>=0&&(c[static_cast<std::size_t>(j)]&256))return false;}
  if(a==13&&z==13&&(c[i-1]&2048))return false;
  return true;
 }
 static bool word(const std::vector<int>& c,std::size_t i){
  int a=cls(c,i-1),z=cls(c,i);
  if(a==1&&z==2)return false;
  if(word_newline(a)||word_newline(z))return true;
  if(a==6&&(c[i]&256))return false;
  if(a==7&&z==7)return false;
  if(word_ignore(z))return false;
  int left=previous(c,static_cast<std::int64_t>(i)-1,1);if(left<0)return true;
  a=cls(c,left);int before=cls(c,previous(c,left-1,1)),after=cls(c,next(c,static_cast<std::int64_t>(i)+1,1));
  if(ah(a)&&ah(z))return false;
  if(ah(a)&&midletter(z)&&ah(after))return false;
  if(midletter(a)&&ah(z)&&ah(before))return false;
  if(a==9&&z==14)return false;
  if(a==9&&z==15&&after==9)return false;
  if(a==15&&z==9&&before==9)return false;
  if(a==10&&z==10||ah(a)&&z==10||a==10&&ah(z))return false;
  if(a==10&&midnum(z)&&after==10||midnum(a)&&z==10&&before==10)return false;
  if(a==16&&z==16)return false;
  if((ah(a)||a==10||a==16||a==17)&&z==17||a==17&&(ah(z)||z==10||z==16))return false;
  if(a==18&&z==18&&(c[static_cast<std::size_t>(left)]&2048))return false;
  return true;
 }
 static bool sentence(const std::vector<int>& c,std::size_t i){
  int actual=cls(c,i-1),z=cls(c,i);
  if(actual==1&&z==2)return false;
  if(para(actual))return true;
  if(sentence_ignore(z))return false;
  int left=previous(c,static_cast<std::int64_t>(i)-1,2),a=cls(c,left);
  if(a==6&&z==13)return false;
  int before=cls(c,previous(c,left-1,2));
  if((before==10||before==11)&&a==6&&z==11)return false;
  int close=(c[i-1]>>12)&3,space=(c[i-1]>>14)&3;
  // These two unconditional no-break rules also avoid repeated lookahead over spaces.
  if(close&&(z==8||z==9||para(z)))return false;
  if(space&&(z==9||para(z)))return false;
  if(space==1){
   int at=static_cast<int>(i);
   while(static_cast<std::size_t>(at)<c.size()){
    int k=cls(c,at);if(k==10)return false;
    if(k==11||k==12||para(k)||sat(k))break;
    ++at;
   }
  }
  if(space&&(z==14||sat(z)))return false;
  if(space)return true;
  return false;
 }
 std::vector<std::size_t> boundaries(const Seq& input,int mode)const{
  auto codes=encode(input,mode);std::vector<std::size_t> out={0};
  for(std::size_t i=1;i<input.size();++i)if(mode==0?grapheme(codes,i):mode==1?word(codes,i):sentence(codes,i))out.push_back(i);
  if(!input.empty())out.push_back(input.size());return out;
 }
};
std::vector<std::size_t> icu_boundaries(const Seq& input,int mode){
 auto text=utf16(input);UErrorCode error=U_ZERO_ERROR;
 UBreakIterator* it=ubrk_open(mode==0?UBRK_CHARACTER:mode==1?UBRK_WORD:UBRK_SENTENCE,"root",text.data(),static_cast<int32_t>(text.size()),&error);
 if(U_FAILURE(error)||!it)fail("ICU break iterator");
 std::vector<int32_t> prefix={0};for(U cp:input)prefix.push_back(prefix.back()+(cp<0x10000?1:2));
 std::vector<std::size_t> result;
 for(int32_t at=ubrk_first(it);at!=UBRK_DONE;at=ubrk_next(it)){
  auto found=std::lower_bound(prefix.begin(),prefix.end(),at);if(found==prefix.end()||*found!=at)fail("ICU break inside scalar");result.push_back(static_cast<std::size_t>(found-prefix.begin()));
 }ubrk_close(it);return result;
}
void segmentation(const Blob& b,const std::string& ucdroot){
 Segmenter segmenter(b);U total=0;std::array<const char*,3> names={"Grapheme","Word","Sentence"};
 for(int mode=0;mode<3;++mode){
  std::string file=ucdroot+"/auxiliary/"+names[mode]+"BreakTest.txt";std::ifstream in(file);if(!in)fail("segmentation test open "+file);std::string line;U row=0;
  while(std::getline(in,line)){
   line=trim(line.substr(0,line.find('#')));if(line.empty())continue;
   Seq seq;std::vector<std::size_t> expected;std::istringstream tokens(line);std::string token;
   while(tokens>>token){if(token=="\xC3\xB7")expected.push_back(seq.size());else if(token!="\xC3\x97")seq.push_back(static_cast<U>(std::stoul(token,nullptr,16)));}
   if(segmenter.boundaries(seq,mode)!=expected){std::ostringstream e;e<<"segmentation table mismatch "<<names[mode]<<" row="<<row<<" input=";for(U cp:seq)e<<std::hex<<cp<<" ";e<<" expected=";for(auto i:expected)e<<std::dec<<i<<",";e<<" actual=";for(auto i:segmenter.boundaries(seq,mode))e<<i<<",";fail(e.str());}
   if(icu_boundaries(seq,mode)!=expected)fail("ICU segmentation mismatch "+std::string(names[mode])+" row="+std::to_string(row));
   ++row;++total;
  }
  std::cout<<"SEGMENTATION_TABLE_ORACLE_OK mode="<<names[mode]<<" rows="<<row<<"\n";
 }
 if(total<2000)fail("segmentation corpus too small");
}

int main(int argc,char** argv){
 try{
  if(argc==4&&std::string(argv[1])=="--seg-only"){version_guard();Blob b(argv[2]);segmentation(b,argv[3]);std::cout<<"SEGMENTATION_REFERENCE_OK\n";return 0;}
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
  version_guard();Blob b(argv[1]);properties(b);mappings(b);full_casing(b);normalization(b,argv[2]);segmentation(b,std::filesystem::path(argv[2]).parent_path().string());std::cout<<"UNICODE_TABLE_REFERENCE_OK\n";return 0;
 }
 catch(const std::exception& e){std::cerr<<"UNICODE_TABLE_REFERENCE_ERROR "<<e.what()<<"\n";return 1;}
}
