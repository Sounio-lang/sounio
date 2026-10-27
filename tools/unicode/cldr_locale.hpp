#pragma once
// Requires Registry from cldr_registry.cpp. Canonicalization uses official
// CLDR49 alias/likely-subtag data; no ICU-derived production locale metadata.
struct Cparts{
 std::string language,script,region;std::vector<std::string> variants;
 static bool alpha(const std::string&s){return !s.empty()&&std::all_of(s.begin(),s.end(),[](unsigned char c){return (c>='a'&&c<='z')||(c>='A'&&c<='Z');});}
 static bool digits(const std::string&s){return !s.empty()&&std::all_of(s.begin(),s.end(),[](unsigned char c){return c>='0'&&c<='9';});}
 static bool alnum(const std::string&s){return !s.empty()&&std::all_of(s.begin(),s.end(),[](unsigned char c){return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9');});}
 static std::string lower(std::string s){for(char&c:s){auto u=static_cast<unsigned char>(c);if(u>127)bad("locale identifier must be ASCII");if(c>='A'&&c<='Z')c=static_cast<char>(c-'A'+'a');if(c=='-')c='_';}return s;}
 static std::string type_lower(std::string s){for(char&c:s){if(static_cast<unsigned char>(c)>127)bad("collation identifier ASCII");if(c>='A'&&c<='Z')c=static_cast<char>(c-'A'+'a');if(c=='_')c='-';}return s;}
 static std::vector<std::string> split(const std::string&s){std::vector<std::string> out;std::size_t p=0;while(true){auto end=s.find('_',p);auto token=s.substr(p,end==s.npos?s.npos:end-p);if(token.empty())bad("empty locale subtag");out.push_back(token);if(end==s.npos)break;p=end+1;}return out;}
 static Cparts parse(const std::string&raw){auto s=lower(raw);if(s.empty())s="root";auto tokens=split(s);Cparts out;out.language=tokens[0];if(out.language.size()<2||out.language.size()>8||!alpha(out.language))bad("locale language syntax");if(out.language=="root"&&tokens.size()!=1)bad("root locale subtags");bool variants=false;for(std::size_t i=1;i<tokens.size();++i){auto token=tokens[i];if(token=="x"){if(i+1==tokens.size())bad("empty private-use extension");for(++i;i<tokens.size();++i)if(tokens[i].size()>8||!alnum(tokens[i]))bad("private-use syntax");break;}if(!variants&&out.script.empty()&&out.region.empty()&&token.size()==4&&alpha(token)){out.script=token;continue;}if(!variants&&out.region.empty()&&((token.size()==2&&alpha(token))||(token.size()==3&&(digits(token)||alpha(token))))){out.region=token;continue;}if(!alnum(token)||token.size()>8||!(token.size()>=5||(token.size()==4&&token[0]>='0'&&token[0]<='9')))bad("locale variant syntax");variants=true;out.variants.push_back(token);}std::sort(out.variants.begin(),out.variants.end());if(std::adjacent_find(out.variants.begin(),out.variants.end())!=out.variants.end())bad("duplicate locale variant");return out;}
 std::string id()const{if(language=="root")return "root";auto out=language;if(!script.empty()){auto s=script;s[0]=static_cast<char>(s[0]-'a'+'A');out+="_"+s;}if(!region.empty()){auto r=region;for(auto&c:r)if(c>='a'&&c<='z')c=static_cast<char>(c-'a'+'A');out+="_"+r;}for(const auto&v:variants)out+="_"+v;return out;}
};
struct ClocaleRegistry:Registry{
 std::map<std::string,std::string> language_aliases,script_aliases,territory_aliases,variant_aliases,likely,source_locales;
 explicit ClocaleRegistry(const fs::path&root):Registry(root){
  auto metadata=without_comments(load(root/"common/supplemental/supplementalMetadata.xml"));
  for(auto spec:std::vector<std::pair<std::string,std::map<std::string,std::string>*>>{{"languageAlias",&language_aliases},{"scriptAlias",&script_aliases},{"territoryAlias",&territory_aliases},{"variantAlias",&variant_aliases}}){std::size_t p=0;auto tag="<"+spec.first+" ";while((p=metadata.find(tag,p))!=metadata.npos){auto end=tag_end(metadata,p);auto a=attrs(metadata.substr(p+1,end-p-1));if(!a.contains("type")||!a.contains("replacement"))bad("locale alias metadata fields");if(!spec.second->emplace(Cparts::lower(a.at("type")),a.at("replacement")).second)bad("duplicate locale alias");p=end+1;}}
  auto data=without_comments(load(root/"common/supplemental/likelySubtags.xml"));std::size_t p=0;while((p=data.find("<likelySubtag ",p))!=data.npos){auto end=tag_end(data,p);auto a=attrs(data.substr(p+1,end-p-1));if(!a.contains("from")||!a.contains("to")||!likely.emplace(Cparts::lower(a.at("from")),a.at("to")).second)bad("likely-subtag metadata");p=end+1;}
  // Collations inherit ordinary main parents but not the main nonlikelyScript
  // rule. Explicit collation-component parents already loaded by Registry win.
  data=without_comments(load(root/"common/supplemental/supplementalData.xml"));p=0;while((p=data.find("<parentLocales",p))!=data.npos){auto end=tag_end(data,p),close=data.find("</parentLocales>",end);if(close==data.npos)bad("parentLocales close");auto a=attrs(data.substr(p+1,end-p-1));if(!a.contains("component")){auto block=data.substr(end+1,close-end-1);std::size_t at=0;while((at=block.find("<parentLocale ",at))!=block.npos){auto e=tag_end(block,at);auto v=attrs(block.substr(at+1,e-at-1));if(!v.contains("parent")||!v.contains("locales"))bad("main locale parent fields");if(!v.contains("localeRules"))for(const auto&locale:words(v.at("locales")))parents.try_emplace(locale,v.at("parent"));else if(v.at("localeRules")!="nonlikelyScript")bad("unsupported main parent rule");at=e+1;}}p=close+16;}
  for(const auto&[name,type]:defaults){(void)type;source_locales.emplace(Cparts::lower(name),name);}for(const auto&[name,parent]:parents){source_locales.try_emplace(Cparts::lower(name),name);source_locales.try_emplace(Cparts::lower(parent),parent);}source_locales.emplace("root","root");if(likely.empty()||language_aliases.empty())bad("empty locale metadata");
 }
 Cparts maximize(Cparts p)const{
  // Explicitly exclude und from language inference for collation bundle lookup.
  if(p.language=="root"||p.language=="und")return p;
  std::vector<std::string> keys;
  if(!p.script.empty()&&!p.region.empty())keys.push_back(p.language+"_"+p.script+"_"+p.region);
  if(!p.script.empty())keys.push_back(p.language+"_"+p.script);
  if(!p.region.empty())keys.push_back(p.language+"_"+p.region);
  keys.push_back(p.language);
  if(!p.script.empty())keys.push_back("und_"+p.script);
  if(!p.region.empty())keys.push_back("und_"+p.region);
  keys.push_back("und");
  for(const auto&key:keys){auto found=likely.find(Cparts::lower(key));if(found==likely.end())continue;auto target=Cparts::parse(found->second);if(p.script.empty())p.script=target.script;if(p.region.empty())p.region=target.region;return p;}return p;
 }
 static bool pattern(const Cparts&p,const Cparts&alias){if(p.language!=alias.language||(!alias.script.empty()&&p.script!=alias.script)||(!alias.region.empty()&&p.region!=alias.region))return false;for(const auto&v:alias.variants)if(std::find(p.variants.begin(),p.variants.end(),v)==p.variants.end())return false;return true;}
 Cparts canonical(std::string raw)const{
  raw=Cparts::lower(raw);if(raw.empty())raw="root";Cparts p;bool parsed=false;std::set<std::string> legacy_seen;
  for(unsigned attempt=0;attempt<16&&!parsed;++attempt){try{p=Cparts::parse(raw);parsed=true;}catch(const std::exception&){if(!legacy_seen.insert(raw).second)bad("legacy locale alias cycle");std::string best,replacement;for(const auto&[key,value]:language_aliases)if((raw==key||raw.starts_with(key+"_"))&&key.size()>best.size()){best=key;replacement=value;}if(best.empty())bad("invalid locale identifier");raw=Cparts::lower(replacement)+raw.substr(best.size());}}
  if(!parsed)bad("legacy locale alias depth");std::set<std::string> seen;
  for(unsigned attempt=0;attempt<16;++attempt){auto identity=p.id();if(!seen.insert(identity).second)bad("locale language alias cycle");std::size_t score=0;bool have=false;Cparts source,target;
   for(const auto&[key,value]:language_aliases){Cparts alias;try{alias=Cparts::parse(key);}catch(const std::exception&){continue;}if(!pattern(p,alias))continue;auto weight=1+(!alias.script.empty()?1U:0U)+(!alias.region.empty()?1U:0U)+alias.variants.size();if(have&&weight<=score)continue;auto candidates=words(value);if(candidates.size()!=1)bad("multiple language alias targets");source=alias;target=Cparts::parse(candidates.front());score=weight;have=true;}
   if(!have)break;auto next=p;next.language=target.language;if(!source.script.empty())next.script=target.script;else if(next.script.empty())next.script=target.script;if(!source.region.empty())next.region=target.region;else if(next.region.empty())next.region=target.region;for(const auto&v:source.variants)next.variants.erase(std::remove(next.variants.begin(),next.variants.end(),v),next.variants.end());for(const auto&v:target.variants)if(std::find(next.variants.begin(),next.variants.end(),v)==next.variants.end())next.variants.push_back(v);std::sort(next.variants.begin(),next.variants.end());if(next.id()==identity)break;p=std::move(next);if(attempt==15)bad("language alias depth");
  }
  if(auto i=script_aliases.find(p.script);i!=script_aliases.end())p.script=Cparts::lower(i->second);
  if(auto i=territory_aliases.find(p.region);i!=territory_aliases.end()){auto choices=words(Cparts::lower(i->second));if(choices.empty())bad("empty territory replacement");auto preferred=p;preferred.region.clear();preferred.variants.clear();preferred=maximize(preferred);p.region=std::find(choices.begin(),choices.end(),preferred.region)!=choices.end()?preferred.region:choices.front();}
  std::vector<std::string> variants;for(const auto&v:p.variants){auto i=variant_aliases.find(v);if(i==variant_aliases.end())variants.push_back(v);else for(const auto&replacement:words(Cparts::lower(i->second)))variants.push_back(replacement);}std::sort(variants.begin(),variants.end());variants.erase(std::unique(variants.begin(),variants.end()),variants.end());p.variants=std::move(variants);if(!p.region.empty()&&!((p.region.size()==2&&Cparts::alpha(p.region))||(p.region.size()==3&&Cparts::digits(p.region))))bad("uncanonicalizable region");return p;
 }
 std::string bundle(Cparts p)const{
  if(p.language=="root"||(p.language=="und"&&p.script.empty()&&p.region.empty()&&p.variants.empty()))return "root";
  if(!p.script.empty()){auto missing=p;missing.script.clear();if(maximize(missing).script==p.script)p.script.clear();}
  auto expanded=maximize(p);auto regional=p;if(regional.region.empty())regional.region=expanded.region;
  std::string best;std::size_t best_variants=0;
  for(const auto&[folded,original]:source_locales){(void)folded;Cparts candidate;try{candidate=Cparts::parse(original);}catch(const std::exception&){continue;}bool same_region=candidate.language==regional.language&&candidate.script==regional.script&&candidate.region==regional.region;bool same_expanded=candidate.language==expanded.language&&candidate.script==expanded.script&&candidate.region==expanded.region;if((!same_region&&!same_expanded)||candidate.variants.size()>p.variants.size())continue;bool subset=true;for(const auto&v:candidate.variants)if(std::find(p.variants.begin(),p.variants.end(),v)==p.variants.end())subset=false;if(subset&&(best.empty()||candidate.variants.size()>best_variants)){best=original;best_variants=candidate.variants.size();}}
  if(!best.empty())return best;
  std::string current=expanded.id();std::set<std::string> seen;while(!current.empty()){if(!seen.insert(Cparts::lower(current)).second)bad("bundle parent cycle");auto found=source_locales.find(Cparts::lower(current));if(found!=source_locales.end())return found->second;current=parent(current);}return "root";
 }
 std::pair<std::string,std::string> request(const std::string&input,const std::string&explicit_type)const{
  auto raw=Cparts::lower(input);if(raw.size()>255)bad("locale identifier length");if(raw.empty())raw="root";auto tokens=Cparts::split(raw);std::string base,inline_type;bool posix=false;
  for(std::size_t i=0;i<tokens.size();++i){if(i&&tokens[i].size()==1){
    if(tokens[i]=="x"){if(i+1==tokens.size())bad("empty private use");for(++i;i<tokens.size();++i)if(tokens[i].size()>8||!Cparts::alnum(tokens[i]))bad("private-use syntax");break;}
    if(tokens[i]!="u")bad("unsupported locale extension");std::map<std::string,std::string> keywords;++i;
    while(i<tokens.size()){if(tokens[i]=="x"){if(i+1==tokens.size())bad("empty private use");for(++i;i<tokens.size();++i)if(tokens[i].size()>8||!Cparts::alnum(tokens[i]))bad("private-use syntax");break;}
     auto key=tokens[i++];if(key!="co"&&key!="va")bad("unsupported Unicode locale keyword");std::string value;
     while(i<tokens.size()&&tokens[i].size()!=1&&tokens[i].size()!=2){if(tokens[i].size()<3||tokens[i].size()>8||!Cparts::alnum(tokens[i]))bad("Unicode locale keyword syntax");if(!value.empty())value+='-';value+=tokens[i++];}
     if(value.empty()||!keywords.emplace(key,value).second)bad("empty/duplicate Unicode locale keyword");
    }
    if(auto k=keywords.find("co");k!=keywords.end())inline_type=k->second;if(auto k=keywords.find("va");k!=keywords.end()){if(k->second!="posix")bad("unsupported variant keyword");posix=true;}break;
   }
   if(!base.empty())base+='_';base+=tokens[i];
  }
  if(posix){auto b=Cparts::split(base);if(std::find(b.begin()+1,b.end(),"posix")==b.end())base+="_posix";}
  auto folded_type=Cparts::type_lower(explicit_type);if(folded_type.size()>64)bad("collation type length");if(!folded_type.empty()){if(folded_type.front()=='-'||folded_type.back()=='-'||folded_type.find("--")!=folded_type.npos)bad("collation type syntax");for(unsigned char ch:folded_type)if(!((ch>='a'&&ch<='z')||(ch>='0'&&ch<='9')||ch=='-'))bad("collation type syntax");}auto type=canonical_type(folded_type);if(!inline_type.empty()){auto inline_canonical=canonical_type(inline_type);if(!type.empty()&&type!=inline_canonical)bad("conflicting collation selectors");type=inline_canonical;}return {bundle(canonical(base)),type};
 }
 std::size_t resolve_request(const std::string&locale,const std::string&type)const{auto request_value=request(locale,type);return resolve(request_value.first,request_value.second);}
};

