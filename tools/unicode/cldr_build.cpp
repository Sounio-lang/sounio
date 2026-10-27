// Production CLDR49 CE graph construction from pinned root and rule sources.
// No ICU dependency: canonical decomposition, contextual lookup and relation insertion
// are computed here from UnicodeData.txt, FractionalUCA.txt and CLDR XML.
#define SOUNIO_CLDR_FRACTIONAL_NO_MAIN
#include "cldr_fractional.cpp"
#include <list>
#include <deque>
#include <numeric>
void cldr_input_versions(const fs::path&cldr,const fs::path&unicode_data){
 auto dtd=load(cldr/"common/dtd/ldml.dtd");std::string compact;for(unsigned char c:dtd)if(!std::isspace(c))compact.push_back(static_cast<char>(c));if(compact.find(R"(cldrVersionCDATA#FIXED"49")")==compact.npos)bad("expected pinned CLDR49");
 if(load(unicode_data.parent_path()/"ReadMe.txt").find("18.0.0")==std::string::npos)bad("expected pinned UCD18");
}
struct Fnormalize{
 std::map<Fcp,Fseq> decomposition;std::map<Fcp,unsigned> classes;std::vector<std::tuple<Fcp,Fcp,std::string>> scripts;std::map<std::string,std::string> script_aliases;
 static std::string alias_key(std::string s){s.erase(std::remove(s.begin(),s.end(),'_'),s.end());std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});return s;}
 explicit Fnormalize(const fs::path&unicode_data){std::istringstream in(load(unicode_data));std::string row;while(std::getline(in,row)){auto f=fsplit(row,';');if(f.size()!=15)bad("UnicodeData field count");auto cp=fhex(f[0]);auto c=std::stoul(f[3]);if(c>255)bad("UnicodeData CCC");if(c)classes.emplace(cp,static_cast<unsigned>(c));if(!f[5].empty()&&f[5].front()!='<')decomposition.emplace(cp,fhexseq(f[5]));}
  std::istringstream aliases(load(unicode_data.parent_path()/"PropertyValueAliases.txt"));while(std::getline(aliases,row)){auto f=fsplit(row.substr(0,row.find('#')),';');if(f.size()<3||f[0]!="sc")continue;for(std::size_t i=1;i<f.size();++i)script_aliases[alias_key(f[i])]=f[1];}
  std::istringstream script_file(load(unicode_data.parent_path()/"Scripts.txt"));while(std::getline(script_file,row)){auto f=fsplit(row.substr(0,row.find('#')),';');if(f.size()!=2)continue;auto split=f[0].find("..");auto a=fhex(f[0].substr(0,split)),b=split==std::string::npos?a:fhex(f[0].substr(split+2));auto name=script_aliases.find(alias_key(f[1]));if(name==script_aliases.end())bad("unresolved script alias");scripts.emplace_back(a,b,name->second);}std::sort(scripts.begin(),scripts.end());}
 std::string script(Fcp cp)const{auto i=std::upper_bound(scripts.begin(),scripts.end(),cp,[](Fcp value,const auto&r){return value<std::get<0>(r);});if(i==scripts.begin())return "Zzzz";--i;return cp<=std::get<1>(*i)?std::get<2>(*i):"Zzzz";}
 unsigned ccc(Fcp cp)const{auto i=classes.find(cp);return i==classes.end()?0:i->second;}
 void decompose(Fcp cp,Fseq&out,unsigned depth=0)const{if(depth>32)bad("canonical decomposition cycle/depth");if(cp>=0xac00&&cp<0xd7a4){auto s=cp-0xac00;out.push_back(0x1100+s/588);out.push_back(0x1161+(s%588)/28);if(s%28)out.push_back(0x11a7+s%28);return;}auto i=decomposition.find(cp);if(i==decomposition.end()){out.push_back(cp);return;}for(auto child:i->second)decompose(child,out,depth+1);}
 Fseq nfd(const Fseq&input)const{Fseq out;for(auto cp:input)decompose(cp,out);for(std::size_t i=1;i<out.size();++i){auto c=ccc(out[i]);if(!c)continue;std::size_t j=i;while(j&&ccc(out[j-1])>c){std::swap(out[j],out[j-1]);--j;}}return out;}
};
struct Fnode{std::size_t previous=0,next=0;Fcp original=0;bool root=false;};
struct Forder{
 std::deque<Fnode> nodes{{0,0,0,true}};std::map<Fcp,std::size_t> roots{{0,0}};
 std::size_t root(Fcp value){auto found=roots.find(value);if(found!=roots.end())return found->second;auto next=roots.upper_bound(value);std::size_t after=0;if(next!=roots.end())after=nodes[next->second].previous;else after=nodes[0].previous;auto id=insert(after);nodes[id].original=value;nodes[id].root=true;roots.emplace(value,id);return id;}
 std::size_t insert(std::size_t after){if(after>=nodes.size())bad("CE graph anchor");auto next=nodes[after].next;auto id=nodes.size();nodes.push_back({after,next,nodes[after].original,false});nodes[after].next=id;nodes[next].previous=id;return id;}
 std::size_t before(std::size_t id)const{if(id==0||id>=nodes.size())bad("before ignorable CE");return nodes[id].previous;}
 std::vector<std::uint64_t> weights()const{std::vector<std::uint64_t> out(nodes.size());std::uint64_t offset=0;Fcp original=0;auto id=nodes[0].next;std::size_t seen=0;while(id){if(nodes[id].root){original=nodes[id].original;offset=0;}else if(++offset>=4194304)bad("CE interval capacity");out[id]=static_cast<std::uint64_t>(original)*4194304+offset;id=nodes[id].next;if(++seen>=nodes.size())bad("CE graph cycle");}return out;}
 std::vector<Fcp> ranks()const{std::vector<Fcp> out(nodes.size());Fcp rank=0;auto id=nodes[0].next;while(id){if(++rank==std::numeric_limits<Fcp>::max())bad("CE graph rank overflow");out[id]=rank;id=nodes[id].next;if(rank>=nodes.size())bad("CE graph cycle");}if(static_cast<std::size_t>(rank)+1!=nodes.size())bad("CE graph disconnected");return out;}
};
struct Gce{std::array<std::size_t,4> w{};unsigned casing=0;};
using Gces=std::vector<Gce>;
struct Gsettings{unsigned strength=3;bool shifted=false,backwards=false,case_level=false;unsigned case_first=0;std::vector<std::string> reorder;std::string max_variable="punct";};
struct Fbuilder{
 const Fractional&fractional;const Fnormalize&normalizer;std::array<Forder,4> order;
 std::map<Fkey,Gces> root_mappings,mappings;
 std::map<Fcp,std::vector<Fkey>> index,root_index;
 std::map<Fcp,Fcp> han_ranks;
 Gsettings settings;
 Gces current;unsigned pending_before=0;std::size_t instructions=0;
 Fbuilder(const Fractional&f,const Fnormalize&n):fractional(f),normalizer(n){
  for(std::size_t i=0;i<f.radical.size();++i)han_ranks.emplace(f.radical[i],static_cast<Fcp>(i));
  for(const auto&[key,ces]:f.entries){if(key.text.front()==0xfdd0)continue;Fkey normalized{n.nfd(key.text),n.nfd(key.prefix)};if(key.text.size()==1&&key.prefix.empty()&&normalized.text!=key.text)continue;auto converted=convert(ces);auto found=root_mappings.find(normalized);if(found==root_mappings.end())root_mappings.emplace(normalized,converted);else if(!same(found->second,converted))bad("canonically conflicting fractional root mapping");}
  mappings=root_mappings;reindex();root_index=index;
 }
 bool same(const Gces&a,const Gces&b)const{if(a.size()!=b.size())return false;for(std::size_t i=0;i<a.size();++i)if(a[i].w!=b[i].w||a[i].casing!=b[i].casing)return false;return true;}
 Gce implicit(Fcp cp){if(cp>0x10ffff)bad("implicit codepoint range");auto han=han_ranks.find(cp);Fcp p=han==han_ranks.end()?0xe4000000U+cp*8U:0x82040000U+han->second*4096U;return {{{order[0].root(p),order[1].root(0x05000000),order[2].root(0x05000000),0}},0};}
 Gces convert(const std::vector<Fce>&ces,unsigned depth=0){if(depth>32)bad("fractional reference cycle");Gces out;for(const auto&ce:ces){Gce g{};if(ce.reference_fields){auto found=fractional.entries.find(Fkey{{ce.reference},{}});if(han_ranks.contains(ce.reference)||found==fractional.entries.end())g=implicit(ce.reference);else{auto base=convert(found->second,depth+1);if(base.size()!=1)bad("fractional reference CE cardinality");g=base[0];}if(ce.reference_fields>=2){auto t=ce.w[2].value;g.w[2]=order[2].root(t&0x3fffffffU);g.casing=t>>30;}if(ce.reference_fields==3)g.w[1]=order[1].root(ce.w[1].value);}else{for(unsigned j=0;j<3;++j){if(ce.w[j].wildcard)bad("wildcard concrete CE");auto value=ce.w[j].value;if(j==2){g.casing=value>>30;value&=0x3fffffffU;}g.w[j]=order[j].root(value);}}if(g.w[0]||g.w[1]||g.w[2])out.push_back(g);}return out;}
 void reindex(){index.clear();for(const auto&[key,ces]:mappings){(void)ces;if(key.text.empty())bad("empty CE mapping key");index[key.text.front()].push_back(key);}for(auto&[cp,keys]:index){(void)cp;std::sort(keys.begin(),keys.end(),[](const auto&a,const auto&b){return std::pair(a.text.size(),a.prefix.size())>std::pair(b.text.size(),b.prefix.size());});}}
 bool prefix_matches(const Fseq&text,std::size_t at,const Fseq&prefix)const{return prefix.size()<=at&&std::equal(prefix.begin(),prefix.end(),text.begin()+static_cast<std::ptrdiff_t>(at-prefix.size()));}
 Gces lookup(const Fseq&text,bool root_only=false){
  Gces out;const auto&table=root_only?root_mappings:mappings;const auto&lookup_index=root_only?root_index:index;std::vector<bool> consumed(text.size());std::size_t at=0;
  while(at<text.size()){if(consumed[at]){++at;continue;}const Gces*best=nullptr;std::vector<std::size_t> best_positions;std::pair<std::size_t,std::size_t> best_score{};auto candidates=lookup_index.find(text[at]);
   if(candidates!=lookup_index.end())for(const auto&key:candidates->second){auto found=table.find(key);if(found==table.end()||!prefix_matches(text,at,key.prefix))continue;std::vector<std::size_t> positions{at};std::size_t scan=at+1;bool matches=true;
    for(std::size_t j=1;j<key.text.size();++j){auto target=key.text[j];auto cls=normalizer.ccc(target);bool matched=false;while(scan<text.size()){if(consumed[scan]){++scan;continue;}auto other=text[scan];if(other==target){positions.push_back(scan++);matched=true;break;}auto other_class=normalizer.ccc(other);if(!cls||!other_class||other_class>=cls)break;++scan;}if(!matched){matches=false;break;}}
    auto score=std::pair(key.prefix.size(),key.text.size());if(matches&&(!best||score>best_score)){best=&found->second;best_score=score;best_positions=std::move(positions);}}
   if(best){out.insert(out.end(),best->begin(),best->end());for(auto i:best_positions)consumed[i]=true;}else out.push_back(implicit(text[at]));++at;}
  return out;
 }
 void set_mapping(Fkey key,Gces ces){bool fresh=!mappings.contains(key);mappings[key]=std::move(ces);if(fresh){auto&v=index[key.text.front()];v.push_back(key);std::sort(v.begin(),v.end(),[](const auto&a,const auto&b){return std::pair(a.text.size(),a.prefix.size())>std::pair(b.text.size(),b.prefix.size());});}}
 static unsigned strength(const Gce&ce){for(unsigned i=0;i<4;++i)if(ce.w[i])return i+1;return 5;}
 void casing(Gces&ces,const Fseq&text){auto base=lookup(text,true);std::vector<unsigned> source;for(const auto&ce:base)if(ce.w[0])source.push_back(ce.casing);std::vector<std::size_t> target;for(std::size_t i=0;i<ces.size();++i)if(ces[i].w[0])target.push_back(i);
  for(std::size_t i=0;i<target.size();++i){unsigned c=i<source.size()?source[i]:0;if(i+1==target.size()&&source.size()>target.size()){for(std::size_t j=i+1;j<source.size();++j)if(source[j]!=c)c=1;}ces[target[i]].casing=c;}
  for(auto&ce:ces)if(!ce.w[0])ce.casing=ce.w[1]?0:ce.w[2]?2:0;
 }
 void relation(const Rule&r,const Fseq&text){
  if(pending_before&&static_cast<unsigned>(r.strength)!=pending_before)bad("reset-before relation strength differs");
  if(r.strength){std::size_t last=current.size();while(last&&strength(current[last-1])>static_cast<unsigned>(r.strength))--last;if(!last){current.assign(1,Gce{});last=1;}else current.resize(last);
   auto&ce=current.back();unsigned level=static_cast<unsigned>(r.strength-1);auto anchor=ce.w[level];if(!anchor){if(level==0)anchor=order[0].root(0x02000000);else if(level==1&&!ce.w[0])anchor=order[1].root(0x80000000);else if(level==2&&!ce.w[0]&&!ce.w[1])anchor=order[2].root(0x3c000000);}if(pending_before)anchor=order[level].before(anchor);ce.w[level]=order[level].insert(anchor);for(unsigned j=level+1;j<4;++j)ce.w[j]=j<3?order[j].root(0x05000000):0;
  }
  pending_before=0;casing(current,text);auto all=current;if(!r.extension.empty()){auto ext=lookup(normalizer.nfd(fliteral(r.extension)));all.insert(all.end(),ext.begin(),ext.end());}set_mapping({text,normalizer.nfd(fliteral(r.prefix))},std::move(all));
 }
 void reset(const Rule&r){pending_before=r.before;if(!r.text.starts_with('[')){current=lookup(normalizer.nfd(fliteral(r.text)));return;}auto name=r.text.substr(1,r.text.size()-2);if(name=="last regular"){auto boundary=fractional.entries.find(Fkey{{0xfdd1,0x5b57},{}});if(boundary==fractional.entries.end())bad("missing Hani boundary");current=convert(boundary->second);return;}auto found=fractional.logical.find(name);if(found==fractional.logical.end())bad("unknown logical reset "+name);current=convert(found->second);}
 void setting(const std::string&s){auto split=s.find(' ');auto name=s.substr(0,split),value=trim(s.substr(split+1));if(name=="alternate")settings.shifted=value=="shifted";else if(name=="backwards")settings.backwards=true;else if(name=="caseLevel")settings.case_level=value=="on";else if(name=="caseFirst")settings.case_first=value=="upper"?2:value=="lower"?1:0;else if(name=="strength")settings.strength=value=="I"?5:static_cast<unsigned>(std::stoul(value));else if(name=="normalization"){if(value!="on"&&value!="off")bad("normalization setting");}else if(name=="reorder")settings.reorder=words(value);else if(name=="maxVariable")settings.max_variable=value;else if(name=="optimize"){if(value.front()!='['||value.back()!=']')bad("optimize syntax");}else if(name=="suppressContractions"){if(value.front()!='['||value.back()!=']')bad("suppress syntax");auto body=value.substr(1,value.size()-2);body.erase(std::remove_if(body.begin(),body.end(),[](unsigned char c){return c==32||c==9||c==10||c==13;}),body.end());auto cps=fstarred(body);std::set<Fcp> suppress(cps.begin(),cps.end());for(auto i=mappings.begin();i!=mappings.end();){if(suppress.contains(i->first.text.front())&&(i->first.text.size()>1||!i->first.prefix.empty()))i=mappings.erase(i);else ++i;}reindex();}else bad("unsupported CE builder setting "+name);}
 void apply(const Registry&r,std::size_t id,std::set<std::size_t>&stack){if(!stack.insert(id).second)bad("CE builder import cycle");for(const auto&rule:r.entries.at(id).rules){if(++instructions>2000000)bad("CE builder instruction budget");if(rule.op==4)apply(r,r.imported(rule.text),stack);else if(rule.op==3)setting(rule.text);else if(rule.op==1)reset(rule);else if(rule.op==2){if(rule.starred){if(!rule.prefix.empty()||!rule.extension.empty())bad("starred prefix/extension");for(auto cp:fstarred(rule.text)){if(normalizer.ccc(cp)||normalizer.nfd({cp})!=Fseq{cp})bad("non-inert starred rule");relation(rule,{cp});}}else relation(rule,normalizer.nfd(fliteral(rule.text)));}else bad("CE builder opcode");}stack.erase(id);}
 void apply(const Registry&r,std::size_t id){std::set<std::size_t> stack;apply(r,id,stack);if(pending_before)bad("unconsumed reset-before");}
};
std::string canonical_group(std::string name){if(name=="Hira"||name=="Hrkt")return "Kana";if(name=="Hans"||name=="Hant")return "Hani";if(name=="Zzzz")return "others";return name;}
struct Ggroups{
 std::map<Fcp,std::string> starts;std::map<std::string,std::uint64_t> rank;
 Ggroups(const Fbuilder&b){
  for(const auto&[key,ces]:b.fractional.entries)if(key.text.size()==2&&key.text[0]==0xfdd1){if(ces.size()!=1||ces[0].reference_fields)bad("reorder boundary CE");auto cp=key.text[1];std::string name=cp==0xa0?"space":cp==0x201c?"punct":cp==0x263a?"symbol":cp==0x20ac?"currency":cp==0x34?"digit":cp==0xfdd0?"others":canonical_group(b.normalizer.script(cp));auto p=ces[0].w[0].value;auto prior=starts.find(p);if(prior!=starts.end()&&prior->second!=name){if((prior->second=="Merc"&&name=="Mero")||(prior->second=="Mero"&&name=="Merc"))name="Mero";else bad("shared reorder boundary "+prior->second+"/"+name);}starts[p]=name;}
  std::vector<std::string> original;for(const auto&[p,name]:starts){(void)p;if(std::find(original.begin(),original.end(),name)==original.end())original.push_back(name);}
  std::vector<std::string> request;std::set<std::string> seen;for(auto name:b.settings.reorder){name=canonical_group(name);if(!seen.insert(name).second)bad("duplicate reorder group");if(name!="others"&&std::find(original.begin(),original.end(),name)==original.end())bad("unknown reorder group "+name);request.push_back(name);}
  if(request.empty())request={"others"};for(const auto&core:std::vector<std::string>{"digit","currency","symbol","punct","space"})if(!seen.contains(core)){request.insert(request.begin(),core);seen.insert(core);}if(!seen.contains("others"))request.push_back("others");
  std::vector<std::string> resolved;for(const auto&name:request){if(name!="others")resolved.push_back(name);else for(const auto&part:original)if(part=="others"||!seen.contains(part))resolved.push_back(part);}
  for(std::size_t i=0;i<resolved.size();++i)rank.emplace(resolved[i],i+1);if(rank.empty())bad("empty reorder result");
 }
 std::uint64_t group(Fcp primary)const{if(primary<0x03000000)return 0;if(primary>=0xe5000000)return rank.size()+2;auto i=starts.upper_bound(primary);if(i==starts.begin())bad("missing primary reorder boundary");--i;auto found=rank.find(i->second);if(found==rank.end())bad("missing resolved group "+i->second);return found->second;}
};
#ifndef SOUNIO_CLDR_BUILD_NO_MAIN
int main(int argc,char**argv){try{if(argc!=4)return 64;cldr_input_versions(argv[1],argv[2]);Fractional f(fs::path(argv[1])/"common/uca/FractionalUCA.txt");Fnormalize n(argv[2]);Registry registry(argv[1]);std::ofstream report(argv[3]);report<<"locale\ttype\talt\tmappings\tprimary_nodes\tsecondary_nodes\ttertiary_nodes\tquaternary_nodes\n";for(std::size_t i=0;i<registry.entries.size();++i){Fbuilder b(f,n);try{b.apply(registry,i);}catch(const std::exception&e){bad(Registry::identity(registry.entries[i].locale,registry.entries[i].type,registry.entries[i].variant)+": "+e.what());}for(const auto&o:b.order)o.ranks();const auto&t=registry.entries[i];report<<t.locale<<'\t'<<t.type<<'\t'<<t.variant<<'\t'<<b.mappings.size();for(const auto&o:b.order)report<<'\t'<<o.nodes.size();report<<'\n';std::cout<<"CLDR_CE_GRAPH_ENTRY_OK "<<Registry::identity(t.locale,t.type,t.variant)<<" instructions="<<b.instructions<<"\n";}if(!report)bad("CE graph report");std::cout<<"CLDR_CE_GRAPH_BUILD_OK entries="<<registry.entries.size()<<"\n";return 0;}catch(const std::exception&e){std::cerr<<"CLDR_CE_GRAPH_BUILD_ERROR "<<e.what()<<"\n";return 1;}}
#endif
