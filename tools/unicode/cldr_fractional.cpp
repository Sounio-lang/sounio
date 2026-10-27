// CLDR49 FractionalUCA production input reader. No ICU headers or derived data.
// Weights keep all source bytes (including case bits); references remain explicit.
#define SOUNIO_CLDR_REGISTRY_NO_MAIN
#include "cldr_registry.cpp"
#include <array>
#include <limits>
using Fcp=std::uint32_t;
using Fseq=std::vector<Fcp>;
Fcp fhex(const std::string&s){if(s.empty()||s.size()>8)bad("fractional hex width");Fcp v=0;for(unsigned char c:s){int n=c>='0'&&c<='9'?c-'0':c>='A'&&c<='F'?c-'A'+10:c>='a'&&c<='f'?c-'a'+10:-1;if(n<0)bad("fractional hex digit");v=(v<<4)|n;}return v;}
Fseq fhexseq(const std::string&s){Fseq out;for(const auto&w:words(s)){auto cp=fhex(w);if(cp>0x10ffff)bad("fractional codepoint range");out.push_back(cp);}return out;}
struct Frune{Fcp cp;bool escaped;};
std::vector<Frune> frunes(const std::string&s){std::vector<Frune> out;std::size_t at=0;while(at<s.size()){auto c=static_cast<unsigned char>(s[at++]);Fcp cp=0;bool escaped=false;if(c==92){escaped=true;if(at==s.size())bad("rule escape EOF");auto kind=s[at++];if(kind=='u'||kind=='U'){auto n=kind=='u'?4U:8U;if(n>s.size()-at)bad("rule hex EOF");cp=fhex(s.substr(at,n));at+=n;}else{switch(kind){case 't':cp=9;break;case 'n':cp=10;break;case 'r':cp=13;break;case 'f':cp=12;break;case 'b':cp=8;break;default:cp=static_cast<unsigned char>(kind);}}}else if(c<128)cp=c;else{int n=c>=194&&c<=223?1:c>=224&&c<=239?2:c>=240&&c<=244?3:-1;if(n<0)bad("fractional UTF8 lead");cp=c&((1U<<(6-n))-1);for(int j=0;j<n;++j){if(at==s.size())bad("fractional UTF8 short");auto b=static_cast<unsigned char>(s[at++]);if((b&192)!=128)bad("fractional UTF8 continuation");cp=(cp<<6)|(b&63);}if(cp<(n==1?128U:n==2?2048U:65536U))bad("fractional UTF8 overlong");}if(cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff))bad("non-scalar fractional literal");out.push_back({cp,escaped});}return out;}
Fseq fliteral(const std::string&s){Fseq out;for(auto c:frunes(s))out.push_back(c.cp);return out;}
Fseq fstarred(const std::string&s){auto r=frunes(s);Fseq out;for(std::size_t i=0;i<r.size();++i){if(r[i].cp==45&&!r[i].escaped)bad("starred missing range start");auto first=r[i].cp;if(i+1<r.size()&&r[i+1].cp==45&&!r[i+1].escaped){if(i+2>=r.size()||r[i+2].cp<first)bad("starred descending/incomplete range");auto last=r[i+2].cp;if(first<=0xdfff&&last>=0xd800)bad("starred surrogate range");for(Fcp cp=first;cp<=last;++cp)out.push_back(cp);i+=2;}else out.push_back(first);}return out;}
struct Fweight { Fcp value=0;unsigned bytes=0;bool wildcard=false;auto operator<=>(const Fweight&)const=default; };
Fweight fweight(std::string s,bool wildcard=false){s=trim(s);if(s=="X"){if(!wildcard)bad("unexpected wildcard CE");return {0,0,true};}Fweight out;for(const auto&w:words(s)){if(w.size()!=2||out.bytes==4)bad("fractional weight byte width");out.value=(out.value<<8)|fhex(w);++out.bytes;}if(out.bytes)out.value<<=(4-out.bytes)*8;return out;}
std::vector<std::string> fsplit(const std::string&s,char delimiter){std::vector<std::string> out;std::size_t at=0;while(true){auto end=s.find(delimiter,at);out.push_back(trim(s.substr(at,end==s.npos?s.npos:end-at)));if(end==s.npos)break;at=end+1;}return out;}
struct Fce{std::array<Fweight,3> w;Fcp reference=0;unsigned reference_fields=0;};
std::vector<Fce> fces(const std::string&s,bool wildcard=false){std::vector<Fce> out;std::size_t at=0;while(at<s.size()){while(at<s.size()&&std::isspace(static_cast<unsigned char>(s[at])))++at;if(at==s.size())break;if(s[at++]!='[')bad("fractional CE opener");auto end=s.find(']',at);if(end==s.npos)bad("fractional CE EOF");auto parts=fsplit(s.substr(at,end-at),',');Fce ce{};if(parts[0].starts_with("U+")){if(parts.size()>3)bad("fractional reference fields");ce.reference=fhex(parts[0].substr(2));ce.reference_fields=static_cast<unsigned>(parts.size());if(ce.reference>0x10ffff)bad("fractional reference range");if(parts.size()==2)ce.w[2]=fweight(parts[1]);if(parts.size()==3){ce.w[1]=fweight(parts[1]);ce.w[2]=fweight(parts[2]);}}else{if(parts.size()!=3)bad("fractional CE fields");for(unsigned j=0;j<3;++j)ce.w[j]=fweight(parts[j],wildcard);}out.push_back(ce);at=end+1;}if(out.empty())bad("empty fractional CE mapping");return out;}
struct Fkey{Fseq text,prefix;auto operator<=>(const Fkey&)const=default;};
struct Fractional {
 std::map<Fkey,std::vector<Fce>> entries;
 Fseq unified,radical;
 std::map<std::string,std::vector<Fce>> logical;
 std::map<unsigned,std::vector<std::string>> top_bytes;
 std::map<std::string,std::string> reorder_tokens,fixed,categories;
 Fcp variable_top=0;
 explicit Fractional(const fs::path&path){
  std::istringstream in(load(path));std::string line;bool version=false,radical_end=false;unsigned radical_number=0,line_number=0;std::set<Fcp> han_seen;
  while(std::getline(in,line)){++line_number;line=trim(line.substr(0,line.find('#')));if(line.empty())continue;try{
   if(line.starts_with("[UCA version = ")){if(version||line!="[UCA version = 18.0.0]")bad("expected one UCA18 version");version=true;continue;}
   if(line.starts_with("[Unified_Ideograph ")){if(!unified.empty()||line.back()!=']')bad("Unified_Ideograph directive");for(auto w:words(line.substr(19,line.size()-20))){auto sep=w.find("..");auto a=fhex(w.substr(0,sep)),b=sep==w.npos?a:fhex(w.substr(sep+2));if(b<a||b>0x10ffff)bad("Unified_Ideograph range");for(auto cp=a;cp<=b;++cp)unified.push_back(cp);}std::set<Fcp> unique(unified.begin(),unified.end());if(unique.size()!=unified.size())bad("duplicate Unified_Ideograph");continue;}
   if(line=="[radical end]"){if(radical_end||radical_number!=214)bad("radical terminator");radical_end=true;continue;}
   if(line.starts_with("[radical ")){auto equal=line.find('='),colon=line.find(':',equal);if(radical_end||equal==line.npos||colon==line.npos||line.back()!=']')bad("radical syntax");if(std::stoul(line.substr(9,equal-9))!=++radical_number)bad("radical numeric order");auto expanded=fstarred(line.substr(colon+1,line.size()-colon-2));for(auto cp:expanded){if(!han_seen.insert(cp).second)bad("duplicate radical Han");radical.push_back(cp);}continue;}
   if(line.starts_with("[top_byte")){if(line.back()!=']')bad("top byte syntax");auto w=words(line.substr(1,line.size()-2));if(w.size()<3)bad("top byte fields");auto byte=fhex(w[1]);if(byte>255||!top_bytes.emplace(byte,std::vector<std::string>(w.begin()+2,w.end())).second)bad("duplicate/invalid top byte");continue;}
   if(line.starts_with("[reorderingTokens")){if(line.back()!=']')bad("reorder syntax");auto w=words(line.substr(1,line.size()-2));if(w.size()<3||!reorder_tokens.emplace(w[1],trim(line.substr(line.find(w[1])+w[1].size(),line.size()-line.find(w[1])-w[1].size()-1))).second)bad("reorder token fields");continue;}
   if(line.starts_with("[categories")){if(line.back()!=']')bad("categories syntax");auto w=words(line.substr(1,line.size()-2));if(w.size()<3||!categories.emplace(w[1],line.substr(line.find(w[1])+w[1].size(),line.size()-line.find(w[1])-w[1].size()-1)).second)bad("category metadata");continue;}
   if(line.starts_with("[fixed ")){auto end=line.find_last_of(" \t");if(end==line.npos||line.back()!=']'||!fixed.emplace(line.substr(7,end-7),line.substr(end+1,line.size()-end-2)).second)bad("fixed directive");fweight(line.substr(end+1,line.size()-end-2));continue;}
   if(line.starts_with("[variable top = ")){if(variable_top||line.back()!=']')bad("variable top directive");variable_top=fweight(line.substr(16,line.size()-17)).value;continue;}
   if(line.front()=='['){auto nested=line.find('[',1);if(nested==line.npos||line.size()<3||line.substr(line.size()-2)!="]]")bad("unknown fractional directive");auto name=trim(line.substr(1,nested-1));if(!logical.emplace(name,fces(line.substr(nested,line.size()-nested-1),true)).second)bad("duplicate logical CE");continue;}
   auto semi=line.find(';');if(semi==line.npos)bad("mapping separator");auto key=fsplit(line.substr(0,semi),'|');if(key.size()>2)bad("mapping prefix");Fkey k{fhexseq(key.back()),key.size()==2?fhexseq(key.front()):Fseq{}};if(k.text.empty()||!entries.emplace(k,fces(line.substr(semi+1))).second)bad("empty/duplicate fractional mapping");
  }catch(const std::exception&e){bad(path.filename().string()+":"+std::to_string(line_number)+": "+e.what());}}
  if(!version||!radical_end||unified.empty()||entries.empty()||!variable_top||fixed.empty()||reorder_tokens.empty())bad("incomplete fractional table");
  auto a=unified,b=radical;std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());if(a!=b)bad("radical/Unified_Ideograph domain differs");
  for(const auto&[key,ces]:entries)for(const auto&ce:ces)if(ce.reference_fields){auto target=entries.find(Fkey{{ce.reference},{}});if(target==entries.end()){if(!han_seen.contains(ce.reference))bad("unresolved fractional reference");}else if(target->second.size()!=1)bad("reference target not single CE");}
 }
};
void fractional_negative_controls(const fs::path&source,const fs::path&directory){
 fs::create_directories(directory);const auto original=load(source);
 const std::vector<std::array<std::string,3>> mutants={
  {"version","[UCA version = 18.0.0]","[UCA version = 17.0.0]"},
  {"duplicate-radical","[radical 1=⼀一:一","[radical 1=⼀一:一一"},
  {"weight-overflow","0000; [,,]","0000; [01 02 03 04 05,,]"},
  {"reference-range","2F00; [U+4E00, 10]","2F00; [U+110000, 10]"},
  {"unknown-directive","[fixed tertiary common byte 05]","[unsupported production directive]"}
 };
 for(const auto&test:mutants){auto text=original;auto at=text.find(test[1]);if(at==text.npos||text.find(test[1],at+1)!=text.npos)bad("non-unique negative control anchor "+test[0]);text.replace(at,test[1].size(),test[2]);auto path=directory/(test[0]+".txt");{std::ofstream out(path);out<<text;if(!out)bad("negative source write");}bool rejected=false;try{Fractional wrong(path);}catch(const std::exception&){rejected=true;}if(!rejected)bad("wrong fractional input accepted "+test[0]);}
 std::cout<<"CLDR_FRACTIONAL_NEGATIVES_OK count="<<mutants.size()<<"\n";
}
#ifndef SOUNIO_CLDR_FRACTIONAL_NO_MAIN
int main(int argc,char**argv){try{if(argc!=3)return 64;Fractional f(argv[1]);fractional_negative_controls(argv[1],fs::path(argv[2]).parent_path()/"fractional-negatives");std::ofstream out(argv[2]);out<<"kind\tcount\nmappings\t"<<f.entries.size()<<"\nunified\t"<<f.unified.size()<<"\nradical\t"<<f.radical.size()<<"\nlogical\t"<<f.logical.size()<<"\nreorder_tokens\t"<<f.reorder_tokens.size()<<"\nfixed\t"<<f.fixed.size()<<"\n";std::size_t prefixes=0,refs=0,ces=0;for(const auto&[key,value]:f.entries){prefixes+=!key.prefix.empty();ces+=value.size();for(const auto&ce:value)refs+=ce.reference_fields!=0;}out<<"prefixes\t"<<prefixes<<"\nreferences\t"<<refs<<"\nCEs\t"<<ces<<"\n";if(!out)bad("fractional report");std::cout<<"CLDR_FRACTIONAL_INPUT_OK mappings="<<f.entries.size()<<" unified="<<f.unified.size()<<" radical="<<f.radical.size()<<" ces="<<ces<<" prefixes="<<prefixes<<" references="<<refs<<"\n";return 0;}catch(const std::exception&e){std::cerr<<"CLDR_FRACTIONAL_INPUT_ERROR "<<e.what()<<"\n";return 1;}}
#endif
