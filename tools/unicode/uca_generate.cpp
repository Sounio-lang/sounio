// Pinned UCA18 DUCET + CLDR49 root CE compiler. No ICU-derived production data.
#define main sounio_ucd_generator_main
#include "unicode_generate.cpp"
#undef main
using Cell=std::array<U,4>;
using Cells=std::vector<Cell>;
struct ImplicitRange{U a,b,lead,offset;};
struct Allkeys{std::map<std::vector<U>,Cells> entries;std::vector<ImplicitRange> implicit;};
U uca_weight(const std::string& text){U n=hex(text);if(n>0xFFFF)fail("collation weight exceeds16bits");return n;}
Allkeys read_allkeys(const fs::path& path,bool cldr,bool compact_zero=true){
 std::ifstream raw(path);std::ostringstream full;full<<raw.rdbuf();if(!raw)fail("allkeys open");
 if(full.str().find(cldr?"# allkeys_CLDR.txt":"# allkeys-18.0.0.txt")==std::string::npos)fail("unexpected allkeys source header");
 Allkeys out;bool version=false;
 lines(path,[&](std::string s){
  s=trim(s.substr(0,s.find_first_of("#%")));if(s.empty())return;
  if(s.starts_with("@version")){if(trim(s.substr(8))!="18.0.0"||version)fail("expected one UCA18.0.0 version directive");version=true;return;}
  if(s.starts_with("@implicitweights")){
   auto fields=split(trim(s.substr(16)),';');if(fields.size()!=2)fail("implicit fields");auto[a,b]=interval(fields[0]);U lead=uca_weight(fields[1]);out.implicit.push_back({a,b,lead,a});return;
  }
  if(s.front()=='@')fail("unsupported allkeys directive");
  auto fields=split(s,';');if(fields.size()!=2)fail("allkeys fields");auto key=sequence(fields[0]);if(key.empty())fail("empty collation key");
  Cells cells;std::size_t at=0;const auto& weights=fields[1];
  while(at<weights.size()){
   while(at<weights.size()&&std::isspace(static_cast<unsigned char>(weights[at])))++at;if(at==weights.size())break;
   if(weights[at++]!='['||at==weights.size())fail("CE opener");char marker=weights[at++];if(marker!='.'&&marker!='*')fail("CE variable marker");
   auto end=weights.find(']',at);if(end==weights.npos)fail("CE closer");auto values=split(weights.substr(at,end-at),'.');if(values.size()!=3)fail("CE must contain exactly3weights");
   cells.push_back({uca_weight(values[0]),uca_weight(values[1]),uca_weight(values[2]),marker=='*'?1U:0U});at=end+1;
  }
  // Entirely zero CEs are ignorable padding. Preserve explicit empty mappings.
  if(compact_zero)cells.erase(std::remove_if(cells.begin(),cells.end(),[](const Cell& c){if(c[0]||c[1]||c[2])return false;if(c[3])fail("zero CE marked variable");return true;}),cells.end());
  if(!out.entries.emplace(std::move(key),std::move(cells)).second)fail("empty CE or duplicate key");
 });
 if(!version||out.entries.empty())fail("empty/unversioned allkeys");
 std::map<U,U> minima;for(const auto&r:out.implicit){auto i=minima.find(r.lead);if(i==minima.end())minima.emplace(r.lead,r.a);else i->second=std::min(i->second,r.a);}
 std::sort(out.implicit.begin(),out.implicit.end(),[](const auto&a,const auto&b){return a.a<b.a;});
 U previous=0;bool first=true;for(auto&r:out.implicit){r.offset=minima.at(r.lead);if(!first&&r.a<=previous)fail("overlapping implicit ranges");previous=r.b;first=false;}
 return out;
}
void cldr_version(const fs::path& dtd){
 std::ifstream in(dtd);std::ostringstream all;all<<in.rdbuf();if(!in)fail("CLDR DTD open");std::string compact;
 for(unsigned char c:all.str())if(!std::isspace(c))compact.push_back(static_cast<char>(c));
 if(compact.find("cldrVersionCDATA#FIXED\"49\"")==std::string::npos)fail("expected pinned CLDR49 DTD");
}
Bytes uca_generate(const Allkeys& ducet,const Allkeys& cldr,std::ostream& report){
 if(ducet.entries.size()!=cldr.entries.size())fail("CLDR49/DUCET18 root keysets differ");
 Section entries{1,16,0,{}},keys{2,4,0,{}},ces{3,16,0,{}},implicit{4,16,0,{}};
 std::map<Cells,U> intern;U variable_differences=0,max_key=0,max_expansion=0;
 for(const auto&[key,weights]:ducet.entries){
  auto found=cldr.entries.find(key);if(found==cldr.entries.end()||found->second.size()!=weights.size())fail("CLDR49/DUCET18 root mapping shapes differ");
  Cells both=weights;
  for(std::size_t i=0;i<weights.size();++i){
   const auto& b=found->second[i];for(U col=0;col<3;++col)if(weights[i][col]!=b[col])fail("CLDR49/DUCET18 root weights differ; no substitution permitted");
   both[i][3]=weights[i][3]|(b[3]<<1);if(weights[i][3]!=b[3])++variable_differences;
  }
  U first_ce=0;auto prior=intern.find(both);
  if(prior==intern.end()){first_ce=ces.count;intern.emplace(both,first_ce);for(const auto&ce:both){for(U value:ce)put(ces.bytes,value);++ces.count;}}
  else first_ce=prior->second;
  put(entries.bytes,keys.count);put(entries.bytes,narrow(key.size()));put(entries.bytes,first_ce);put(entries.bytes,narrow(weights.size()));++entries.count;
  for(U cp:key){put(keys.bytes,cp);++keys.count;}
  max_key=std::max(max_key,narrow(key.size()));max_expansion=std::max(max_expansion,narrow(weights.size()));
 }
 for(const auto&r:ducet.implicit){put(implicit.bytes,r.a);put(implicit.bytes,r.b);put(implicit.bytes,r.lead);put(implicit.bytes,r.offset);++implicit.count;}
 std::vector<Section> sections={std::move(entries),std::move(keys),std::move(ces),std::move(implicit)};
 Bytes out={'S','O','U','N','U','C','A','1'};put(out,1);put(out,0x00120000);put(out,4);put(out,49);U directory=narrow(out.size());out.resize(out.size()+64,0);
 for(U i=0;i<sections.size();++i){auto&s=sections[i];while(out.size()%4)out.push_back(0);patch(out,directory+i*16,s.id);patch(out,directory+i*16+4,s.width);patch(out,directory+i*16+8,s.count);patch(out,directory+i*16+12,narrow(out.size()));report<<"section\t"<<s.id<<'\t'<<s.width<<'\t'<<s.count<<'\t'<<out.size()<<'\n';out.insert(out.end(),s.bytes.begin(),s.bytes.end());}
 report<<"uca\t18.0.0\ncldr\t49\nformat\t1\nbytes\t"<<out.size()<<"\nmax_key_scalars\t"<<max_key<<"\nmax_expansion_CEs\t"<<max_expansion<<"\nvariable_flag_differences\t"<<variable_differences<<"\n";
 return out;
}
#ifndef SOUNIO_UCA_NO_MAIN
int main(int argc,char**argv){
 try{
  if(argc!=6){std::cerr<<"usage: uca_generate DUCET18_ALLKEYS CLDR49_ALLKEYS CLDR49_LDML_DTD OUTPUT.bin REPORT.tsv\n";return 64;}
  cldr_version(argv[3]);auto ducet=read_allkeys(argv[1],false),cldr=read_allkeys(argv[2],true);std::ofstream report(argv[5]);if(!report)fail("report output");auto data=uca_generate(ducet,cldr,report);std::ofstream out(argv[4],std::ios::binary);out.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(data.size()));if(!out||!report)fail("UCA output");std::cout<<"UCA_DATA_GENERATOR_OK bytes="<<data.size()<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<"UCA_DATA_GENERATOR_ERROR "<<e.what()<<"\n";return 1;}
}

#endif

