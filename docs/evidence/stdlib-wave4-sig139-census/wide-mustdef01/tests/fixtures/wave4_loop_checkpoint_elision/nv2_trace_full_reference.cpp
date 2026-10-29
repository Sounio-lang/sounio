// C++23 schema-2 stage-bound full-arena diagnostic parser and adversarial controls.
#include <algorithm>
#include <bit>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
using I=std::int64_t;
struct Row{std::string tag;std::vector<I> v;};
[[noreturn]] void bad(const std::string&s){throw std::runtime_error(s);}
std::vector<std::string> load(const char*p){std::ifstream f(p);if(!f)bad("missing input");std::vector<std::string> r;for(std::string s;std::getline(f,s);)r.push_back(s);return r;}
Row row(const std::string&s){std::istringstream in(s);Row r;if(!(in>>r.tag))bad("empty row");for(std::string t;in>>t;){I x=0;auto [e,ec]=std::from_chars(t.data(),t.data()+t.size(),x);if(ec!=std::errc{}||e!=t.data()+t.size())bad("noninteger field");r.v.push_back(x);}return r;}
void size(const Row&r,std::size_t n){if(r.v.size()!=n)bad("field count "+r.tag);}
std::uint64_t bits(I hi,I lo){if(hi<0||hi>4294967295LL||lo<0||lo>4294967295LL)bad("word limbs");return(std::uint64_t(hi)<<32)|std::uint64_t(lo);}
struct Parser{
 std::vector<Row> rows;std::size_t at=0;
 explicit Parser(const std::vector<std::string>&lines){for(const auto&s:lines)if(s.starts_with("NV2_IR_FULL_"))rows.push_back(row(s));}
 Row take(std::string_view tag){if(at>=rows.size()||rows[at].tag!=tag)bad("missing or misordered "+std::string(tag));return rows[at++];}
 I name(std::string_view tag,I fi,I ii){const auto r=take(tag);if(r.v.size()<3||r.v[0]!=fi||r.v[1]!=ii||r.v[2]<0||r.v[2]>384||r.v.size()!=std::size_t(3+r.v[2]))bad("name shape");for(std::size_t i=3;i<r.v.size();++i)if(r.v[i]<0||r.v[i]>255)bad("name byte");return r.v[2];}
 void verify(bool probe,I expected_stage){
  const auto begin=take("NV2_IR_FULL_BEGIN");size(begin,10);
  if(begin.v[0]!=2||begin.v[1]!=expected_stage)bad("capture stage/version mismatch");
  auto h=begin.v;h.erase(h.begin()+1);h[0]=1;
  if(h[0]!=1||h[1]<0||h[1]>65536||h[2]<0||h[2]>131072||h[3]<0||h[3]>524288||h[4]<0||h[5]<=0||h[5]>2048||h[6]<0||h[7]<=0||h[8]<0)bad("header");
  I ni=0,na=0,nb=0;bool minimum=false,negative_zero=false,nan_payload=false,args_from_slot=false,float_bitmap=false;
  for(I fi=0;fi<h[1];++fi){
   const auto f=take("NV2_IR_FULL_FN");size(f,23);
   if(f.v[0]!=fi||f.v[7]<0||f.v[8]<0||f.v[10]<0||f.v[10]>64||f.v[7]>h[2]-ni)bad("function bounds");
   nb+=name("NV2_IR_FULL_FN_NAME",fi,-1);
   nb+=name("NV2_IR_FULL_RETURN_NAME",fi,-1);
   const auto params=take("NV2_IR_FULL_PARAMS");size(params,65);if(params.v[0]!=fi)bad("parameter function");
   const auto fp=take("NV2_IR_FULL_FLOAT_BITS");size(fp,std::size_t(2+h[5]));
   if(fp.v[0]!=fi||fp.v[1]!=h[5])bad("float-word count");
   for(std::size_t w=2;w<fp.v.size();++w)if(fp.v[w]<0||fp.v[w]>4294967295LL)bad("float bitmap word");
   if(fi==0&&fp.v.size()>3&&(std::uint64_t(fp.v[2])&(UINT64_C(1)<<31))&&(fp.v[3]&1))float_bitmap=true;
   for(I ii=0;ii<f.v[7];++ii){
    const auto ins=take("NV2_IR_FULL_INSTR");
    if(ins.v.size()<20||ins.v[0]!=fi||ins.v[1]!=ii||ins.v[2]<0||ins.v[2]>=h[7]||ins.v[3]<0)bad("instruction identity");
    const I argc=ins.v[18],base=ins.v[19];
    if(argc<0||argc>64||base<0||(argc>0&&base>h[6]-argc)||ins.v.size()!=std::size_t(20+argc))bad("argument shape");
    const auto integer=bits(ins.v[7],ins.v[8]),floating=bits(ins.v[9],ins.v[10]);
    if(ins.v[17]<0||ins.v[17]>h[8])bad("name identity");
    if(fi==0&&ins.v[3]==0&&ins.v[4]==30&&integer==UINT64_C(0x8000000000000000))minimum=true;
    if(fi==0&&ins.v[3]==1&&ins.v[4]==31&&floating==UINT64_C(0x8000000000000000))negative_zero=true;
    if(fi==0&&ins.v[3]==1&&ins.v[4]==32&&floating==UINT64_C(0x7ff8000000000042))nan_payload=true;
    if(fi==0&&ins.v[3]==10&&ins.v[14]==64077&&argc==2&&ins.v[20]==2&&ins.v[21]==5)args_from_slot=true;
    nb+=name("NV2_IR_FULL_INSTR_NAME",fi,ii);
    ++ni;na+=argc;
   }
  }
  const auto end=take("NV2_IR_FULL_END");size(end,6);
  if(end.v!=std::vector<I>{2,expected_stage,h[1],ni,na,nb}||ni!=h[2]||na!=h[3]||nb!=h[4]||at!=rows.size())bad("truncated/duplicate capture or census mismatch");
  if(probe&&!(h[1]==(expected_stage==0?6:7)&&minimum&&negative_zero&&nan_payload&&args_from_slot&&float_bitmap))bad("exact probe payloads or slot args lost");
  std::cout<<"NV2_FULL_TRACE_SCHEMA2_CPP23_OK stage="<<expected_stage<<" functions="<<h[1]<<" instructions="<<ni<<" arguments="<<na<<'\n';
 }
};
void write(const char*p,const std::vector<std::string>&lines){std::ofstream f(p);for(const auto&s:lines)f<<s<<'\n';if(!f)bad("output");}
std::string render(const Row&r){std::string s=r.tag;for(I x:r.v)s+=" "+std::to_string(x);return s;}
void mutate(std::string_view kind,const char*input,const char*output){
 auto lines=load(input);
 auto seek=[&](std::string_view prefix){return std::find_if(lines.begin(),lines.end(),[&](const auto&s){return s.starts_with(prefix);});};
 if(kind=="truncate"){auto p=seek("NV2_IR_FULL_END ");if(p==lines.end())bad("no end");lines.erase(p,lines.end());}
 else if(kind=="duplicate"){auto p=seek("NV2_IR_FULL_INSTR ");if(p==lines.end())bad("no instruction");const auto s=*p;lines.insert(p,s);}
 else if(kind=="total"){auto p=seek("NV2_IR_FULL_BEGIN ");if(p==lines.end())bad("no begin");auto r=row(*p);++r.v[3];*p=render(r);}
 else if(kind=="args"){auto p=seek("NV2_IR_FULL_INSTR 0 4 ");if(p==lines.end())bad("no slot");auto r=row(*p);r.v.pop_back();*p=render(r);}
 else if(kind=="bits"){bool found=false;for(auto&s:lines)if(s.starts_with("NV2_IR_FULL_INSTR ")){auto r=row(s);if(r.v.size()>=20&&r.v[0]==0&&r.v[4]==31){r.v[9]=0;s=render(r);found=true;break;}}if(!found)bad("no negative-zero payload");}
 else if(kind=="name"){auto p=seek("NV2_IR_FULL_FN_NAME ");if(p==lines.end())bad("no name");auto r=row(*p);++r.v[2];*p=render(r);}
 else if(kind=="float-word"){auto p=seek("NV2_IR_FULL_FLOAT_BITS ");if(p==lines.end())bad("no float bitmap");auto r=row(*p);r.v.pop_back();*p=render(r);}
 else if(kind=="unknown"){auto p=seek("NV2_IR_FULL_PARAMS ");if(p==lines.end())bad("no params");lines.insert(p,"NV2_IR_FULL_UNKNOWN 1");}
 else if(kind=="end"){auto p=seek("NV2_IR_FULL_END ");if(p==lines.end())bad("no end");lines.push_back(*p);}
 else if(kind=="stage"){auto p=seek("NV2_IR_FULL_BEGIN ");auto q=seek("NV2_IR_FULL_END ");if(p==lines.end()||q==lines.end())bad("no group");auto a=row(*p),b=row(*q);a.v[1]=1-a.v[1];b.v[1]=a.v[1];*p=render(a);*q=render(b);}
 else bad("unknown control");
 write(output,lines);
}
I stage(std::string_view s){if(s=="source")return 0;if(s=="prepared")return 1;return -2;}
int main(int argc,char**argv){try{
 if(argc==4&&(std::string_view(argv[1])=="--check"||std::string_view(argv[1])=="--probe")){
  const auto wanted=stage(argv[2]);if(wanted<0)return 64;Parser p(load(argv[3]));p.verify(std::string_view(argv[1])=="--probe",wanted);return 0;
 }
 if(argc==5&&std::string_view(argv[1])=="--mutate"){mutate(argv[2],argv[3],argv[4]);return 0;}
 if(argc==5&&std::string_view(argv[1])=="--refusal"){
  const I wanted=std::string_view(argv[2])=="unknown"?-1:stage(argv[2]);if(wanted< -1)return 64;
  const auto lines=load(argv[4]);if(lines.size()!=1)bad("refusal must have no partial capture");const auto r=row(lines[0]);size(r,5);I reason=0;const std::string text=argv[3];auto [p,e]=std::from_chars(text.data(),text.data()+text.size(),reason);if(e!=std::errc{}||p!=text.data()+text.size()||r.tag!="NV2_IR_FULL_REFUSED"||r.v[0]!=2||r.v[1]!=wanted||r.v[2]!=reason)bad("wrong stage/refusal");std::cout<<"NV2_FULL_TRACE_REFUSAL_CPP23_OK\n";return 0;
 }
 return 64;
}catch(const std::exception&e){std::cerr<<"NV2_FULL_TRACE_REFUSED "<<e.what()<<'\n';return 1;}}
