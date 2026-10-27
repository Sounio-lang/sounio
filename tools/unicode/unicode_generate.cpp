// Deterministic Unicode 18 table compiler. No ICU/runtime host Unicode dependency.
// Compile with C++23. Input is the pinned, unmodified UCD source directory.
// Output uses explicit little-endian fields; no C++ object layout is serialized.
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>
namespace fs=std::filesystem;
using U=std::uint32_t;
using Bytes=std::vector<std::uint8_t>;
constexpr U limit=0x110000;
[[noreturn]] void fail(const std::string& s){throw std::runtime_error(s);}
std::string trim(std::string s){
 auto p=s.find_first_not_of(" \t\r\n");if(p==s.npos)return {};
 return s.substr(p,s.find_last_not_of(" \t\r\n")-p+1);
}
std::vector<std::string> split(const std::string& s,char sep){
 std::vector<std::string> r;std::size_t p=0;
 for(;;){auto e=s.find(sep,p);r.push_back(trim(s.substr(p,e==s.npos?e:e-p)));if(e==s.npos)return r;p=e+1;}
}
std::string loose(std::string_view s){
 std::string r;for(unsigned char c:s)if(c!='_'&&c!='-'&&c!=' ')r.push_back(static_cast<char>(std::tolower(c)));return r;
}
U hex(const std::string& s){
 std::size_t n=0;auto v=std::stoul(s,&n,16);if(n!=s.size()||v>=limit)fail("invalid scalar field: "+s);return static_cast<U>(v);
}
std::pair<U,U> interval(const std::string& s){
 auto p=s.find("..");U a=hex(s.substr(0,p));U b=p==s.npos?a:hex(s.substr(p+2));if(a>b)fail("reversed range");return {a,b};
}
std::vector<U> sequence(const std::string& s){
 std::vector<U> r;std::istringstream in(s);std::string x;while(in>>x)r.push_back(hex(x));return r;
}
template<class F> void lines(const fs::path& p,F f){
 std::ifstream in(p);if(!in)fail("cannot read "+p.string());std::string s;U n=0;
 while(std::getline(in,s)){++n;try{f(s);}catch(const std::exception& e){fail(p.string()+":"+std::to_string(n)+": "+e.what());}}
 if(!in.eof())fail("read failed: "+p.string());
}
struct Range{U a,b;std::string value;};
struct Property{std::string fallback;std::vector<Range> defaults,ranges;};
struct Mapping{U kind,cp,flags;std::vector<U> value;std::string condition;};
struct Data{
 std::map<std::string,Property> properties;
 std::map<std::string,std::string> property_alias;
 std::map<std::pair<std::string,std::string>,std::string> value_alias;
 std::vector<Mapping> mappings;
 std::set<U> exclusions;
 std::map<U,U> ccc;
 std::map<U,std::vector<U>> canonical_decomposition;
 std::map<std::string,U> strings;
 fs::path root;
 std::string pname(const std::string& s) const {
  auto i=property_alias.find(loose(s));return i==property_alias.end()?s:i->second;
 }
 std::string vname(const std::string& prop,const std::string& v) const {
  auto i=value_alias.find({pname(prop),loose(v)});return i==value_alias.end()?v:i->second;
 }
 void add(const std::string& prop,U a,U b,std::string value,bool missing=false){
  const auto name=pname(prop);value=vname(name,value);
  auto& p=properties[name];auto& out=missing?p.defaults:p.ranges;out.push_back({a,b,std::move(value)});
 }
 void read_aliases(){
  lines(root/"PropertyAliases.txt",[&](std::string s){
   s=trim(s.substr(0,s.find('#')));if(s.empty())return;auto v=split(s,';');if(v.size()<2)fail("property alias fields");
   for(const auto& a:v)property_alias[loose(a)]=v[1];
  });
  lines(root/"PropertyValueAliases.txt",[&](std::string s){
   s=trim(s.substr(0,s.find('#')));if(s.empty())return;auto v=split(s,';');if(v.size()<3)fail("value alias fields");
   const auto prop=pname(v[0]);for(std::size_t i=1;i<v.size();++i)if(!v[i].empty())value_alias[{prop,loose(v[i])}]=v[1];
  });
 }
 void property_file(const std::string& file,const std::string& forced=""){
  lines(root/file,[&](std::string s){
   bool missing=false;const auto m=s.find("@missing:");
   if(m!=s.npos){missing=true;s=s.substr(m+9);}else{s=trim(s.substr(0,s.find('#')));if(s.empty())return;}
   s=trim(s.substr(0,s.find('#')));auto v=split(s,';');if(v.size()<2)fail("property fields");
   const auto [a,b]=interval(v[0]);
   if(forced.empty()){add(v[1],a,b,v.size()>2?v[2]:"Y",missing);}
   else add(forced,a,b,forced=="Numeric_Value"&&v.size()>3&&!v[3].empty()?v[3]:v[1],missing);
  });
 }
 void unicode_data(){
  std::vector<std::string> pending;U first=0;
  lines(root/"UnicodeData.txt",[&](const std::string& s){
   auto v=split(s,';');if(v.size()!=15)fail("UnicodeData must have 15fields");
   U cp=hex(v[0]),last=cp;
   if(v[1].ends_with(", First>")){if(!pending.empty())fail("nested First");pending=v;first=cp;return;}
   if(v[1].ends_with(", Last>")){if(pending.empty())fail("Last without First");last=cp;cp=first;v=pending;pending.clear();}
   else if(!pending.empty())fail("unclosed First");
   add("General_Category",cp,last,v[2]);add("Canonical_Combining_Class",cp,last,v[3]);add("Bidi_Class",cp,last,v[4]);
   add("Bidi_Mirrored",cp,last,v[9]);ccc[cp]=static_cast<U>(std::stoul(v[3]));
   if(!v[6].empty())add("Decimal_Digit_Value",cp,last,v[6]);
   if(!v[7].empty())add("Digit_Value",cp,last,v[7]);
   if(!v[8].empty())add("Numeric_Value",cp,last,v[8]);
   // Unicode algorithmic names and formal aliases are intentionally separate from character properties.
   for(U col=12;col<15;++col)if(!v[col].empty())mappings.push_back({col-9,cp,0,sequence(v[col]),{}});
   if(!v[5].empty()){
    std::string decomp=v[5];U compat=0;
    if(decomp.front()=='<'){auto e=decomp.find('>');if(e==decomp.npos)fail("decomposition tag");decomp=trim(decomp.substr(e+1));compat=1;}
    auto seq=sequence(decomp);mappings.push_back({1,cp,compat,seq,{}});
    if(!compat)canonical_decomposition.emplace(cp,std::move(seq));
   }
  });
  if(!pending.empty())fail("unclosed UnicodeData First range");
 }
 void folding(){
  lines(root/"CaseFolding.txt",[&](std::string s){
   s=trim(s.substr(0,s.find('#')));if(s.empty())return;auto v=split(s,';');if(v.size()<3)fail("casefold fields");
   const U cp=hex(v[0]);if(v[1]=="C"||v[1]=="F")mappings.push_back({6,cp,0,sequence(v[2]),{}});
   else if(v[1]=="T")mappings.push_back({7,cp,0,sequence(v[2]),{}});
   else if(v[1]!="S")fail("unknown casefold status");
  });
 }
 void special_casing(){
  lines(root/"SpecialCasing.txt",[&](std::string s){
   s=trim(s.substr(0,s.find('#')));if(s.empty())return;auto v=split(s,';');if(v.size()<5)fail("special casing fields");
   const U cp=hex(v[0]);for(U col=1;col<=3;++col)mappings.push_back({7+col,cp,0,sequence(v[col]),v[4]});
  });
 }
 void exclusions_file(){
  lines(root/"CompositionExclusions.txt",[&](std::string s){s=trim(s.substr(0,s.find('#')));if(s.empty())return;auto [a,b]=interval(s);for(U cp=a;cp<=b;++cp)exclusions.insert(cp);});
 }
 void load(){
  std::ifstream r(root/"ReadMe.txt");std::ostringstream text;text<<r.rdbuf();
  if(text.str().find("final data files for version 18.0.0")==std::string::npos)fail("expected final UCD18.0.0 ReadMe");
  read_aliases();unicode_data();
  const std::array<std::pair<const char*,const char*>,28> files={{
   {"PropList.txt",""},{"DerivedCoreProperties.txt",""},{"DerivedNormalizationProps.txt",""},
   {"Scripts.txt","Script"},{"ScriptExtensions.txt","Script_Extensions"},{"Blocks.txt","Block"},
   {"EastAsianWidth.txt","East_Asian_Width"},{"DerivedAge.txt","Age"},
   {"auxiliary/GraphemeBreakProperty.txt","Grapheme_Cluster_Break"},
   {"auxiliary/WordBreakProperty.txt","Word_Break"},{"auxiliary/SentenceBreakProperty.txt","Sentence_Break"},
   {"emoji/emoji-data.txt",""},{"extracted/DerivedBidiClass.txt","Bidi_Class"},
   {"extracted/DerivedBinaryProperties.txt",""},{"extracted/DerivedCombiningClass.txt","Canonical_Combining_Class"},
   {"extracted/DerivedDecompositionType.txt","Decomposition_Type"},{"extracted/DerivedGeneralCategory.txt","General_Category"},
   {"extracted/DerivedJoiningGroup.txt","Joining_Group"},{"extracted/DerivedJoiningType.txt","Joining_Type"},
   {"extracted/DerivedNumericType.txt","Numeric_Type"},{"extracted/DerivedNumericValues.txt","Numeric_Value"},
   {"IndicPositionalCategory.txt","Indic_Positional_Category"},{"IndicSyllabicCategory.txt","Indic_Syllabic_Category"},
   {"VerticalOrientation.txt","Vertical_Orientation"},{"HangulSyllableType.txt","Hangul_Syllable_Type"},
   {"LineBreak.txt","Line_Break"},{"EquivalentUnifiedIdeograph.txt","Equivalent_Unified_Ideograph"},
   {"extracted/DerivedEastAsianWidth.txt","East_Asian_Width"}
  }};
  for(const auto& [file,prop]:files)property_file(file,prop);
  // Paired bracket type and paired glyph are independent properties.
  lines(root/"BidiBrackets.txt",[&](std::string s){s=trim(s.substr(0,s.find('#')));if(s.empty())return;auto v=split(s,';');if(v.size()!=3)fail("BidiBrackets fields");U cp=hex(v[0]);add("Bidi_Paired_Bracket",cp,cp,v[1]);add("Bidi_Paired_Bracket_Type",cp,cp,v[2]);});
  lines(root/"BidiMirroring.txt",[&](std::string s){s=trim(s.substr(0,s.find('#')));if(s.empty())return;auto v=split(s,';');if(v.size()!=2)fail("BidiMirroring fields");U cp=hex(v[0]);add("Bidi_Mirroring_Glyph",cp,cp,v[1]);});
  properties[pname("Bidi_Paired_Bracket_Type")].fallback=vname("Bidi_Paired_Bracket_Type","n");
  folding();special_casing();exclusions_file();
  auto ex=properties.find(pname("Full_Composition_Exclusion"));if(ex==properties.end())fail("missing Full_Composition_Exclusion");
  for(const auto& x:ex->second.ranges)if(x.value==vname(ex->first,"Y"))for(U c=x.a;c<=x.b;++c)exclusions.insert(c);
  std::sort(mappings.begin(),mappings.end(),[](const Mapping& a,const Mapping& b){return std::tie(a.kind,a.cp,a.condition,a.flags,a.value)<std::tie(b.kind,b.cp,b.condition,b.flags,b.value);});
 }
};
void put(Bytes& b,U n){for(U i=0;i<4;++i)b.push_back(static_cast<std::uint8_t>((n>>(8*i))&255));}
void patch(Bytes& b,U p,U n){if(p+4>b.size())fail("patch bounds");for(U i=0;i<4;++i)b[p+i]=static_cast<std::uint8_t>((n>>(8*i))&255);}
U narrow(std::size_t n){if(n>std::numeric_limits<U>::max())fail("blob exceeds32-bit format");return static_cast<U>(n);}
struct Section{U id,width,count;Bytes bytes;};
Bytes generate(Data& d,std::ostream& report){
 std::set<std::string> words={""};
 for(auto& [name,p]:d.properties){words.insert(name);words.insert(p.fallback);for(auto& x:p.defaults)words.insert(x.value);for(auto& x:p.ranges)words.insert(x.value);}
 for(const auto& [a,b]:d.property_alias){words.insert(a);words.insert(b);}
 for(const auto& [a,b]:d.value_alias){words.insert(a.first);words.insert(a.second);words.insert(b);}
 for(const auto& m:d.mappings)words.insert(m.condition);
 Section str{1,1,0,{}};for(const auto& s:words){d.strings.emplace(s,narrow(str.bytes.size()));str.bytes.insert(str.bytes.end(),s.begin(),s.end());str.bytes.push_back(0);}str.count=narrow(str.bytes.size());
 Section prop{2,16,0,{}},ranges{3,12,0,{}},mapping{4,28,0,{}},pool{5,4,0,{}},composition{6,12,0,{}},pa{7,8,0,{}},va{8,12,0,{}};
 std::vector<U> values(limit);
 for(auto& [name,p]:d.properties){
  std::string fallback=p.fallback;
  if(fallback.empty()){
   // Binary properties have No defaults; numeric/optional mapping have empty defaults.
   bool binary=!p.ranges.empty()&&std::all_of(p.ranges.begin(),p.ranges.end(),[&](const Range& r){return r.value==d.vname(name,"Y")||r.value==d.vname(name,"N");});
   if(binary)fallback=d.vname(name,"N");
  }
  if(!d.strings.contains(fallback))fail("uninterned fallback: "+fallback);
  const U def=d.strings.at(fallback);std::fill(values.begin(),values.end(),def);
  for(const auto& r:p.defaults)std::fill(values.begin()+r.a,values.begin()+r.b+1,d.strings.at(r.value));
  for(const auto& r:p.ranges)std::fill(values.begin()+r.a,values.begin()+r.b+1,d.strings.at(r.value));
  U first=ranges.count;
  for(U a=0;a<limit;){U b=a;while(b+1<limit&&values[b+1]==values[a])++b;
   if(values[a]!=def){put(ranges.bytes,a);put(ranges.bytes,b);put(ranges.bytes,values[a]);++ranges.count;}a=b+1;
  }
  put(prop.bytes,d.strings.at(name));put(prop.bytes,first);put(prop.bytes,ranges.count-first);put(prop.bytes,def);++prop.count;
 }
 for(const auto& m:d.mappings){
  put(mapping.bytes,m.kind);put(mapping.bytes,m.cp);put(mapping.bytes,m.flags);put(mapping.bytes,pool.count);put(mapping.bytes,narrow(m.value.size()));put(mapping.bytes,d.strings.at(m.condition));put(mapping.bytes,0);
  for(U v:m.value){put(pool.bytes,v);++pool.count;}++mapping.count;
 }
 std::set<std::tuple<U,U,U>> comps;
 for(const auto& [cp,seq]:d.canonical_decomposition)if(seq.size()==2&&!d.exclusions.contains(cp))comps.emplace(seq[0],seq[1],cp);
 for(const auto& [a,b,c]:comps){put(composition.bytes,a);put(composition.bytes,b);put(composition.bytes,c);++composition.count;}
 for(const auto& [a,b]:d.property_alias){put(pa.bytes,d.strings.at(a));put(pa.bytes,d.strings.at(b));++pa.count;}
 for(const auto& [a,b]:d.value_alias){put(va.bytes,d.strings.at(a.first));put(va.bytes,d.strings.at(a.second));put(va.bytes,d.strings.at(b));++va.count;}
 std::vector<Section> sections={std::move(str),std::move(prop),std::move(ranges),std::move(mapping),std::move(pool),std::move(composition),std::move(pa),std::move(va)};
 Bytes out={'S','O','U','N','U','C','D','1'};put(out,1);put(out,0x00120000);put(out,narrow(sections.size()));put(out,0);
 const U directory=narrow(out.size());out.resize(out.size()+16*sections.size(),0);
 for(U i=0;i<sections.size();++i){
  auto& s=sections[i];while(out.size()%4)out.push_back(0);
  patch(out,directory+16*i,s.id);patch(out,directory+16*i+4,s.width);patch(out,directory+16*i+8,s.count);patch(out,directory+16*i+12,narrow(out.size()));
  report<<"section\t"<<s.id<<"\t"<<s.width<<"\t"<<s.count<<"\t"<<out.size()<<"\n";out.insert(out.end(),s.bytes.begin(),s.bytes.end());
 }
 report<<"unicode\t18.0.0\nformat\t1\nproperties\t"<<prop.count<<"\nbytes\t"<<out.size()<<"\n";
 return out;
}
int main(int argc,char** argv){
 try{
  if(argc!=4){std::cerr<<"usage: unicode_generate UCD_DIRECTORY OUTPUT.bin REPORT.tsv\n";return 64;}
  Data d;d.root=argv[1];d.load();std::ofstream report(argv[3]);if(!report)fail("report output");
  auto bytes=generate(d,report);std::ofstream out(argv[2],std::ios::binary);if(!out)fail("binary output");
  out.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));if(!out||!report)fail("output write");
  std::cout<<"UNICODE_GENERATOR_OK "<<bytes.size()<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<"UNICODE_GENERATOR_ERROR "<<e.what()<<"\n";return 1;}
}
