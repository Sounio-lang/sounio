// Independent C++23 UCA18 corpus/oracle runner. Test-only ICU normalization/properties.
#define SOUNIO_UCA_NO_MAIN
#include "../../tools/unicode/uca_generate.cpp"
#include <unicode/unorm2.h>
#include <unicode/uchar.h>
#include <unicode/ustring.h>
#include <unicode/ucol.h>
#include <iterator>
using Seq=std::vector<U>;
struct UcaBlob {
 Bytes bytes;std::array<std::array<U,3>,5> dirs{};
 explicit UcaBlob(const std::string&p){std::ifstream f(p,std::ios::binary);bytes.assign(std::istreambuf_iterator<char>(f),{});if(bytes.size()<88||std::string(bytes.begin(),bytes.begin()+8)!="SOUNUCA1"||read(8)!=1||read(12)!=0x120000||read(16)!=4||read(20)!=49)fail("UCA binary header");for(U i=1;i<=4;++i){U p=24+(i-1)*16;U width=read(p+4),n=read(p+8),off=read(p+12);if(read(p)!=i||width!=(i==2?4:16)||off<88||off>bytes.size()||std::uint64_t(n)*width>bytes.size()-off)fail("UCA directory");dirs[i]={width,n,off};}}
 U read(std::size_t p)const{if(p>bytes.size()||bytes.size()-p<4)fail("UCA u32 bound");return U(bytes[p])|(U(bytes[p+1])<<8)|(U(bytes[p+2])<<16)|(U(bytes[p+3])<<24);}
 U field(U id,U row,U col)const{const auto&d=dirs.at(id);if(row>=d[1]||col*4>=d[0])fail("UCA field bound");return read(d[2]+std::uint64_t(row)*d[0]+col*4);}
 std::map<Seq,Cells> mappings(bool cldr)const{std::map<Seq,Cells> result;for(U row=0;row<dirs[1][1];++row){Seq key;Cells ces;for(U i=0;i<field(1,row,1);++i)key.push_back(field(2,field(1,row,0)+i,0));for(U i=0;i<field(1,row,3);++i){U r=field(1,row,2)+i;ces.push_back({field(3,r,0),field(3,r,1),field(3,r,2),(field(3,r,3)>>(cldr?1:0))&1});}if(!result.emplace(key,ces).second)fail("duplicate binary key");}return result;}
};
Seq nfd_valid(const Seq&s){if(s.empty())return {};std::vector<UChar> u;for(U cp:s){if(cp<=0xFFFF)u.push_back(cp);else{u.push_back(0xD800+((cp-0x10000)>>10));u.push_back(0xDC00+((cp-0x10000)&1023));}}UErrorCode e=U_ZERO_ERROR;auto norm=unorm2_getNFDInstance(&e);auto len=unorm2_normalize(norm,u.data(),static_cast<int32_t>(u.size()),nullptr,0,&e);if(e!=U_BUFFER_OVERFLOW_ERROR&&U_FAILURE(e))fail("NFD sizing");e=U_ZERO_ERROR;std::vector<UChar> out(len);unorm2_normalize(norm,u.data(),u.size(),out.data(),len,&e);if(U_FAILURE(e))fail("NFD");Seq result;for(int32_t i=0;i<len;){UChar32 cp;U16_NEXT(out.data(),i,len,cp);result.push_back(cp);}return result;}
// Raw UCA codepoint sequences preserve surrogate values individually. ICU is
// used only on valid scalar spans, so adjacent surrogate values cannot fuse.
Seq nfd(const Seq&s){Seq result,span;for(U cp:s){if(cp>=0xD800&&cp<=0xDFFF){auto part=nfd_valid(span);result.insert(result.end(),part.begin(),part.end());span.clear();result.push_back(cp);}else span.push_back(cp);}auto part=nfd_valid(span);result.insert(result.end(),part.begin(),part.end());return result;}
struct RootCollator {
 std::map<Seq,Cells> mappings;std::map<U,std::vector<std::pair<Seq,Cells>>> candidates;std::vector<ImplicitRange> implicit;
 RootCollator(std::map<Seq,Cells> m,std::vector<ImplicitRange> i):mappings(std::move(m)),implicit(std::move(i)){for(const auto&[k,v]:mappings)candidates[k.front()].push_back({k,v});}
 Cells missing(U cp)const{
  U lead=0,second=0;
  if(u_charType(cp)!=U_UNASSIGNED)for(const auto&r:implicit)if(cp>=r.a&&cp<=r.b){lead=r.lead;second=(cp-r.offset)|0x8000;break;}
  if(!lead){lead=(u_hasBinaryProperty(cp,UCHAR_UNIFIED_IDEOGRAPH)?((cp>=0x4E00&&cp<=0x9FFF)||(cp>=0xF900&&cp<=0xFAFF)?0xFB40:0xFB80):0xFBC0)+(cp>>15);second=(cp&0x7FFF)|0x8000;}
  return {{lead,0x20,2,0},{second,0,0,0}};
 }
 Cells extract(const Seq&s)const{
  std::vector<bool> consumed(s.size());Cells result;
  for(std::size_t pos=0;pos<s.size();++pos){if(consumed[pos])continue;auto range=candidates.find(s[pos]);const Cells*best=nullptr;std::vector<std::size_t> best_positions;
   if(range!=candidates.end())for(const auto&[key,ces]:range->second){std::vector<std::size_t> matched={pos};std::size_t scan=pos+1;bool ok=true;
    for(std::size_t j=1;j<key.size();++j){U target=key[j];int ccc=u_getCombiningClass(target);bool found=false;while(scan<s.size()){if(consumed[scan]){++scan;continue;}int intervening=u_getCombiningClass(s[scan]);if(s[scan]==target){matched.push_back(scan++);found=true;break;}if(!ccc||!intervening||intervening>=ccc)break;++scan;}if(!found){ok=false;break;}}
    if(ok&&matched.size()>best_positions.size()){best=&ces;best_positions=std::move(matched);}
   }
   if(best){auto start=result.size();result.insert(result.end(),best->begin(),best->end());if(s[pos]==0xFFFE)for(auto i=start;i<result.size();++i)result[i][3]|=4;for(auto at:best_positions)consumed[at]=true;}else{auto ces=missing(s[pos]);result.insert(result.end(),ces.begin(),ces.end());}
  }return result;
 }
 Seq key(const Seq&input,bool shifted,int strength=3,bool backwards=false)const{
  auto normalized=nfd(input);auto ces=extract(normalized);std::array<Seq,4> levels;bool variable=false;std::size_t secondary_start=0;
  for(const auto&ce:ces){auto[p,s,t,flags]=ce;U q=0;if(shifted){if((flags&1)&&p){q=p;p=s=t=0;variable=true;}else if(!p&&variable){p=s=t=0;}else{if(p)variable=false;q=p?0xFFFF:0;}}if(p)levels[0].push_back(p);if(backwards&&(flags&4))std::reverse(levels[1].begin()+secondary_start,levels[1].end());if(s)levels[1].push_back(s);if(backwards&&(flags&4))secondary_start=levels[1].size();if(t)levels[2].push_back(t);if(q)levels[3].push_back(q);}
  if(backwards)std::reverse(levels[1].begin()+secondary_start,levels[1].end());Seq out;for(int i=0;i<std::min(strength,4);++i){out.insert(out.end(),levels[i].begin(),levels[i].end());out.push_back(0);}if(strength==5)for(U cp:normalized)out.push_back(cp==0xFFFE?1:cp+2);return out;
 }
};

int order(const Seq&a,const Seq&b){return a<b?-1:b<a?1:0;}
void all_options_equivalence(const std::string&ducet_path,const std::string&cldr_path,const UcaBlob&blob){
 std::uint64_t comparisons=0;
 for(bool cldr:{false,true}){
  auto raw=read_allkeys(cldr?cldr_path:ducet_path,cldr,false);
  RootCollator before(raw.entries,read_allkeys(ducet_path,false).implicit),after(blob.mappings(cldr),before.implicit);
  for(const auto&[key,unused]:raw.entries)for(bool shifted:{false,true})for(int strength=1;strength<=5;++strength)for(bool backwards:{false,true}){
   if(before.key(key,shifted,strength,backwards)!=after.key(key,shifted,strength,backwards))fail("zero CE semantic normalization changed key");
   ++comparisons;
  }
 }
 std::cout<<"UCA_ZERO_CE_NORMALIZATION_OK all_strengths_shifted_backwards_comparisons="<<comparisons<<"\n";
}
std::vector<Seq> root_control_inputs(){
 std::vector<Seq> inputs={{},{0},{0,0},{0,0xFFFE},{0xFFFE,0},{0x34F,0xFFFE},{0xFFFE,0x34F},{0xFFFE},{0xFFFF},{0xFFFD},{0x10FFFF},{0x1FFFF},{0xFDD0},{0x4E00},{0x3400},{0x17000},{0x18D00},{0x18D80},{0x18E00},{0x3D000},{0x1B170},{0x1B300},{0x10FFFE}};
 for(U cp:{0x20U,0x2DU,0x24U,0x41U,0x61U,0x62U,0x65U,0xE9U,0x301U,0x300U,0x308U,0x315U,0x0F71U,0x0F72U,0x0F73U,0x1D15EU,0x1D165U,0x4E00U,0x1F600U}){
  inputs.push_back({cp});inputs.push_back({cp,0x61});inputs.push_back({0x61,cp});inputs.push_back({0x61,cp,0x301});inputs.push_back({0x61,0xFFFE,cp});inputs.push_back({cp,0xFFFE,0x61});inputs.push_back({0x61,cp,0x20});inputs.push_back({0x20,cp,0x301});}
 for(U left:{0U,0x300U,0x301U,0x308U})for(U right:{0U,0x300U,0x301U,0x308U}){Seq s={0x61};if(left)s.push_back(left);s.push_back(0xFFFE);s.push_back(0x62);if(right)s.push_back(right);inputs.push_back(s);}
 inputs.push_back({0x0F71,0x0F72});inputs.push_back({0x0F71,0x334,0x0F72});inputs.push_back({0x0F71,0x301,0x0F72});
 return inputs;
}
void icu_root_controls(const RootCollator&root){
 auto inputs=root_control_inputs();
 UErrorCode e=U_ZERO_ERROR;UCollator*c=ucol_open("root",&e);if(U_FAILURE(e))fail("ICU root open");std::uint64_t count=0;
 std::vector<std::vector<UChar>> strings;for(const auto&s:inputs){std::vector<UChar> u;for(U cp:s){if(cp<=0xFFFF)u.push_back(cp);else{u.push_back(0xD800+((cp-0x10000)>>10));u.push_back(0xDC00+((cp-0x10000)&1023));}}strings.push_back(u);}
 for(int strength=1;strength<=5;++strength)for(bool shifted:{false,true})for(bool backwards:{false,true}){
  ucol_setStrength(c,strength==5?UCOL_IDENTICAL:static_cast<UCollationStrength>(strength-1));
  ucol_setAttribute(c,UCOL_NORMALIZATION_MODE,UCOL_ON,&e);
  ucol_setAttribute(c,UCOL_ALTERNATE_HANDLING,shifted?UCOL_SHIFTED:UCOL_NON_IGNORABLE,&e);
  ucol_setAttribute(c,UCOL_FRENCH_COLLATION,backwards?UCOL_ON:UCOL_OFF,&e);if(U_FAILURE(e))fail("ICU settings");
  std::vector<Seq> keys;for(const auto&s:inputs)keys.push_back(root.key(s,shifted,strength,backwards));
  for(std::size_t a=0;a<inputs.size();++a)for(std::size_t b=0;b<inputs.size();++b){
   auto expected=ucol_strcoll(c,strings[a].empty()?u"":strings[a].data(),strings[a].size(),strings[b].empty()?u"":strings[b].data(),strings[b].size());int actual=order(keys[a],keys[b]);
   if(actual!=static_cast<int>(expected)){std::cerr<<"ICU_ROOT_ORDER_ERROR strength="<<strength<<" shifted="<<shifted<<" backwards="<<backwards<<" a="<<a<<" b="<<b<<" expected="<<expected<<" actual="<<actual<<" a_cps=";for(U cp:inputs[a])std::cerr<<std::hex<<cp<<",";std::cerr<<" b_cps=";for(U cp:inputs[b])std::cerr<<std::hex<<cp<<",";std::cerr<<"\n";fail("ICU root control");}++count;
  }
 }
 ucol_close(c);std::cout<<"UCA_ICU_ROOT_CONTROLS_OK comparisons="<<count<<"\n";
}

bool ordering_defect(const RootCollator&root,const std::string&path,bool shifted){
 std::ifstream in(path);std::string line;Seq previous;bool have=false;std::uint64_t count=0;
 while(std::getline(in,line)){line=trim(line.substr(0,line.find('#')));if(line.empty())continue;auto input=sequence(line.substr(0,line.find(';')));auto key=root.key(input,shifted,shifted?4:3);if(have&&key<previous)return true;previous=std::move(key);have=true;++count;}
 if(!count)fail("negative oracle empty corpus");return false;
}
void negative_controls(const UcaBlob&blob,const std::vector<ImplicitRange>&implicit,const std::string&normal,const std::string&shifted){
 auto weights=blob.mappings(false);auto changed=weights;auto a=changed.find(Seq{0x61});if(a==changed.end()||a->second.empty())fail("negative ASCII anchor");a->second[0][0]+=0x10000;
 if(!ordering_defect(RootCollator(changed,implicit),normal,false))fail("wrong primary weight not rejected");
 changed=weights;for(auto&[key,ces]:changed)for(auto&ce:ces)ce[3]=0;
 if(!ordering_defect(RootCollator(changed,implicit),shifted,true))fail("wrong shifted handling not rejected");
 auto corrupt=blob.bytes;corrupt[12]^=1;std::ofstream file("negative-version.bin",std::ios::binary);file.write(reinterpret_cast<const char*>(corrupt.data()),corrupt.size());file.close();bool rejected=false;try{UcaBlob bad("negative-version.bin");}catch(const std::exception&){rejected=true;}if(!rejected)fail("wrong UCA version accepted");fs::remove("negative-version.bin");
 std::cout<<"UCA_NEGATIVE_CONTROLS_OK wrong_primary wrong_shifted wrong_version\n";
}
#ifndef SOUNIO_UCA_REFERENCE_NO_MAIN
int main(int argc,char**argv){try{
 if(argc!=6){std::cerr<<"usage: reference DUCET CLDR_ALLKEYS BLOB UCA_NON_IGNORABLE UCA_SHIFTED\n";return 64;}
 auto ducet=read_allkeys(argv[1],false),cldr=read_allkeys(argv[2],true);UcaBlob blob(argv[3]);
 if(blob.mappings(false)!=ducet.entries||blob.mappings(true)!=cldr.entries)fail("binary/source meaningful CE roundtrip");
 std::cout<<"UCA_BINARY_ROUNDTRIP_OK mappings="<<ducet.entries.size()<<"\n";
 RootCollator root(blob.mappings(false),ducet.implicit);
 for(int file=4;file<6;++file){std::ifstream in(argv[file]);std::string line;Seq previous,previous_input;std::uint64_t count=0;bool version=false;while(std::getline(in,line)){if(line.find("# UCA Version: 18.0.0")!=line.npos)version=true;line=trim(line.substr(0,line.find('#')));if(line.empty())continue;auto input=sequence(line.substr(0,line.find(';')));auto key=root.key(input,file==5,file==5?4:3);if(count&&key<previous){std::cerr<<"ORDER_ERROR row="<<count+1<<" file="<<argv[file]<<" previous=";for(U cp:previous_input)std::cerr<<std::hex<<cp<<' ';std::cerr<<" input=";for(U cp:input)std::cerr<<std::hex<<cp<<' ';std::cerr<<"\n";fail("official UCA conformance ordering");}previous=std::move(key);previous_input=std::move(input);++count;}if(!version||!count)fail("empty/versionless UCA corpus");std::cout<<"UCA_CORPUS_OK shifted="<<(file==5)<<" rows="<<count<<"\n";}
 if(nfd(Seq{0xD800,0xDC00})!=Seq{0xD800,0xDC00}||root.key(Seq{0xD800,0xDC00},false,5)==root.key(Seq{0x10000},false,5))fail("surrogate codepoint preservation");
 std::cout<<"UCA_RAW_CODEPOINT_DOMAIN_OK adjacent_surrogates_preserved\n";
 all_options_equivalence(argv[1],argv[2],blob);
 RootCollator cldr_root(blob.mappings(true),ducet.implicit);icu_root_controls(cldr_root);
 negative_controls(blob,ducet.implicit,argv[4],argv[5]);
 std::cout<<"UCA_REFERENCE_OK\n";return 0;
}catch(const std::exception&e){std::cerr<<"UCA_REFERENCE_ERROR "<<e.what()<<"\n";return 1;}}

#endif

