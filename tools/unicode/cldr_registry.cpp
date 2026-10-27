// CLDR49 registry/import compiler preparation. No ICU-derived production data.
#define SOUNIO_CLDR_NO_MAIN
#include "cldr_generate.cpp"
#include <functional>
#include <array>
#include <tuple>
#include <limits>
std::string without_comments(std::string s){std::size_t at=0;while((at=s.find("<!--",at))!=s.npos){auto end=s.find("-->",at+4);if(end==s.npos)bad("unclosed metadata comment");s.erase(at,end-at+3);}return s;}
std::vector<std::string> words(const std::string&s){std::istringstream in(s);std::vector<std::string> out;std::string w;while(in>>w)out.push_back(w);return out;}
std::size_t tag_end(const std::string&s,std::size_t at){char quote=0;for(std::size_t i=at;i<s.size();++i){char c=s[i];if(quote){if(c==quote)quote=0;}else if(c==39||c==34)quote=c;else if(c=='>')return i;}bad("metadata tag EOF");}
std::string locale_id(std::string s){std::replace(s.begin(),s.end(),'-','_');if(s=="und")s="root";return s;}
struct Registry {
 std::vector<Tailoring> entries;std::map<std::string,std::size_t> ids;
 std::map<std::string,std::string> defaults,parents,type_aliases;
 explicit Registry(const fs::path&root){
  auto data=without_comments(load(root/"common/supplemental/supplementalData.xml"));std::size_t at=0;
  while((at=data.find("<parentLocales",at))!=data.npos){auto end=tag_end(data,at);auto a=attrs(data.substr(at+1,end-at-1));auto close=data.find("</parentLocales>",end);if(close==data.npos)bad("parentLocales close");if(a.contains("component")&&a.at("component")=="collations"){auto block=data.substr(end+1,close-end-1);std::size_t p=0;while((p=block.find("<parentLocale ",p))!=block.npos){auto e=tag_end(block,p);auto v=attrs(block.substr(p+1,e-p-1));if(!v.contains("parent")||!v.contains("locales")||v.contains("localeRules"))bad("collation parent metadata");for(const auto&locale:words(v.at("locales")))if(!parents.emplace(locale_id(locale),locale_id(v.at("parent"))).second)bad("duplicate collation parent");p=e+1;}}at=close+16;}
  auto bcp=without_comments(load(root/"common/bcp47/collation.xml"));at=0;bool have_co=false;
  while((at=bcp.find("<key ",at))!=bcp.npos){auto e=tag_end(bcp,at);auto a=attrs(bcp.substr(at+1,e-at-1));auto close=bcp.find("</key>",e);if(close==bcp.npos)bad("BCP47 key close");if(a.contains("name")&&a.at("name")=="co"){if(have_co)bad("duplicate co registry");have_co=true;auto block=bcp.substr(e+1,close-e-1);std::size_t p=0;while((p=block.find("<type ",p))!=block.npos){auto finish=tag_end(block,p);auto v=attrs(block.substr(p+1,finish-p-1));if(!v.contains("name"))bad("collation type name");std::string canonical=v.at("name");if(v.contains("alias")){auto aliases=words(v.at("alias"));if(aliases.size()!=1)bad("unsupported multiple co aliases");canonical=aliases.front();}type_aliases[v.at("name")]=canonical;type_aliases[canonical]=canonical;p=finish+1;}}at=close+6;}if(!have_co)bad("missing BCP47 co registry");
  std::vector<fs::path> files;for(const auto&e:fs::directory_iterator(root/"common/collation"))if(e.path().extension()==".xml")files.push_back(e.path());std::sort(files.begin(),files.end());
  for(const auto&p:files){std::string def;auto parsed=xml(p,&def);defaults.emplace(p.stem().string(),def);entries.insert(entries.end(),parsed.begin(),parsed.end());}
  std::sort(entries.begin(),entries.end(),[](const auto&a,const auto&b){return std::tie(a.locale,a.type,a.variant)<std::tie(b.locale,b.type,b.variant);});
  for(std::size_t i=0;i<entries.size();++i)if(!ids.emplace(identity(entries[i].locale,entries[i].type,entries[i].variant),i).second)bad("duplicate tailoring identity");
  if(entries.empty()||!defaults.contains("zh_Hant")||defaults.at("zh_Hant")!="stroke")bad("default-only locale lost");
 }
 static std::string identity(const std::string&locale,const std::string&type,const std::string&variant=""){return locale+"/"+type+"/"+variant;}
 std::string canonical_type(const std::string&s)const{auto found=type_aliases.find(s);return found==type_aliases.end()?s:found->second;}
 std::string parent(const std::string&locale)const{auto explicit_parent=parents.find(locale);if(explicit_parent!=parents.end())return explicit_parent->second;if(locale=="root")return "";auto p=locale.find_last_of('_');return p==locale.npos?"root":locale.substr(0,p);}
 std::vector<std::string> chain(std::string locale)const{std::vector<std::string> out;std::set<std::string> seen;while(!locale.empty()){if(!seen.insert(locale).second)bad("locale parent cycle");out.push_back(locale);locale=parent(locale);}return out;}
 std::string default_type(const std::string&locale)const{for(const auto&part:chain(locale)){auto d=defaults.find(part);if(d!=defaults.end()&&!d->second.empty())return canonical_type(d->second);}return "standard";}
 std::size_t find_type(const std::string&locale,const std::string&type)const{for(const auto&part:chain(locale)){auto i=ids.find(identity(part,type));if(i!=ids.end())return i->second;}return entries.size();}
 std::size_t resolve(const std::string&requested_locale,const std::string&requested_type,bool internal=false)const{
  auto locale=locale_id(requested_locale);auto def=default_type(locale);auto type=canonical_type(requested_type.empty()?def:requested_type);
  if(!internal&&type.starts_with("private-"))bad("private collation is incomplete");
  auto found=find_type(locale,type);if(found<entries.size())return found;
  if(type.starts_with("search")&&type.size()>6){found=find_type(locale,"search");if(found<entries.size())return found;}
  if(type!=def){found=find_type(locale,def);if(found<entries.size())return found;}
  if(type!="standard"){found=find_type(locale,"standard");if(found<entries.size())return found;}
  found=find_type("root","standard");if(found==entries.size())bad("missing root collation");return found;
 }
 std::size_t imported(const std::string&setting)const{
  if(!setting.starts_with("import "))bad("expected import setting");auto value=trim(setting.substr(7));auto split=value.find("-u-co-");auto locale=locale_id(split==value.npos?value:value.substr(0,split));auto type=canonical_type(split==value.npos?"standard":value.substr(split+6));auto found=find_type(locale,type);if(found==entries.size())bad("unresolved import "+value);return found;
 }
 std::vector<std::uint64_t> expanded_counts()const{
  std::vector<int> states(entries.size());std::vector<std::uint64_t> result(entries.size());std::function<std::uint64_t(std::size_t)> walk=[&](std::size_t i){if(states[i]==1)bad("collation import cycle");if(states[i]==2)return result[i];states[i]=1;std::uint64_t n=0;for(const auto&r:entries[i].rules){auto add=r.op==4?walk(imported(r.text)):1;if(add>2000000-n)bad("expanded instruction budget");n+=add;}states[i]=2;return result[i]=n;};for(std::size_t i=0;i<entries.size();++i)walk(i);return result;
 }
};
void registry_controls(const Registry&r){
 for(const auto&test:std::vector<std::array<std::string,4>>{{"da","phonebook","da","standard"},{"zh","","zh","pinyin"},{"zh","standard","root","standard"},{"zh","phonebk","zh","pinyin"},{"zh_Hant","phonebk","zh","stroke"},{"da","searchjl","da","search"},{"el","search","root","search"},{"el","searchjl","root","search"},{"ko","searchjl","ko","searchjl"},{"yue","","zh","stroke"},{"yue_CN","","zh","pinyin"}}){const auto&t=r.entries.at(r.resolve(test[0],test[1]));if(t.locale!=test[2]||t.type!=test[3])bad("locale/type fallback control "+test[0]+"/"+test[1]+" -> "+t.locale+"/"+t.type);}
 bool rejected=false;try{r.resolve("ja","private-kana");}catch(const std::exception&){rejected=true;}if(!rejected)bad("private public request accepted");
 Registry wrong=r;auto id=wrong.find_type("root","standard");Rule missing;missing.op=4;missing.text="import zz-u-co-definitelymissing";wrong.entries[id].rules.push_back(missing);rejected=false;try{wrong.expanded_counts();}catch(const std::exception&){rejected=true;}if(!rejected)bad("missing import accepted");
 wrong=r;Rule cycle;cycle.op=4;cycle.text="import und-u-co-standard";wrong.entries[id].rules.push_back(cycle);rejected=false;try{wrong.expanded_counts();}catch(const std::exception&){rejected=true;}if(!rejected)bad("import cycle accepted");
 std::cout<<"CLDR_REGISTRY_CONTROLS_OK fallback_alias_defaults_private_missing_cycle\n";
}
#ifndef SOUNIO_CLDR_REGISTRY_NO_MAIN
int main(int argc,char**argv){try{if(argc!=3)return 64;Registry r(argv[1]);auto counts=r.expanded_counts();registry_controls(r);std::ofstream out(argv[2]);out<<"locale\ttype\talt\texplicit_default\tpublic\texpanded_instructions\n";for(std::size_t i=0;i<r.entries.size();++i){const auto&t=r.entries[i];out<<t.locale<<'\t'<<t.type<<'\t'<<t.variant<<'\t'<<t.default_type<<'\t'<<(!t.type.starts_with("private-")&&t.variant.empty())<<'\t'<<counts[i]<<'\n';}if(!out)bad("registry output");std::cout<<"CLDR_REGISTRY_OK locales="<<r.defaults.size()<<" entries="<<r.entries.size()<<"\n";return 0;}catch(const std::exception&e){std::cerr<<"CLDR_REGISTRY_ERROR "<<e.what()<<"\n";return 1;}}
#endif
