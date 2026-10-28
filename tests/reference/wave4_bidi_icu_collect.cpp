// Full-corpus diagnostic collector. The locked wave3 reference stays unchanged.
// Same expected base/map and mixed-level rules; counts failures instead of
// stopping at the first mismatch. Mutants alter observed RAM values only.
#include "wave3_bidi_cases.hpp"
#include <unicode/ubidi.h>
#include <unicode/uchar.h>
#include <unicode/uversion.h>
#include <iostream>
#include <memory>
#include <string_view>

namespace {
using bidi_cases::Case;
using bidi_cases::U;
using bidi_cases::need;
int class_id(UCharDirection d) {
 switch(d) {
  case U_LEFT_TO_RIGHT:return 0; case U_RIGHT_TO_LEFT:return 1;
  case U_RIGHT_TO_LEFT_ARABIC:return 2; case U_EUROPEAN_NUMBER:return 3;
  case U_EUROPEAN_NUMBER_SEPARATOR:return 4; case U_EUROPEAN_NUMBER_TERMINATOR:return 5;
  case U_ARABIC_NUMBER:return 6; case U_COMMON_NUMBER_SEPARATOR:return 7;
  case U_DIR_NON_SPACING_MARK:return 8; case U_BOUNDARY_NEUTRAL:return 9;
  case U_BLOCK_SEPARATOR:return 10; case U_SEGMENT_SEPARATOR:return 11;
  case U_WHITE_SPACE_NEUTRAL:return 12; case U_OTHER_NEUTRAL:return 13;
  case U_LEFT_TO_RIGHT_EMBEDDING:return 14; case U_LEFT_TO_RIGHT_OVERRIDE:return 15;
  case U_RIGHT_TO_LEFT_EMBEDDING:return 16; case U_RIGHT_TO_LEFT_OVERRIDE:return 17;
  case U_POP_DIRECTIONAL_FORMAT:return 18; case U_LEFT_TO_RIGHT_ISOLATE:return 19;
  case U_RIGHT_TO_LEFT_ISOLATE:return 20; case U_FIRST_STRONG_ISOLATE:return 21;
  case U_POP_DIRECTIONAL_ISOLATE:return 22;
  default:throw std::runtime_error("unknown ICU Bidi_Class");
 }
}
struct Observed {
 U base=0;
 bool mixed=false;
 std::vector<int> levels;
 std::vector<U> order;
};
struct Counts {
 std::uint64_t read=0,executed=0,cases=0,bases_bad=0,levels_bad=0,orders_bad=0;
 std::uint64_t levels_checked=0,order_checked=0,optimized=0,optimized_bad=0;
 std::uint64_t mutant_hits=0,samples=0;
 std::array<std::uint64_t,2> kinds{};
 std::array<std::uint64_t,2> read_kinds{};
};
std::string origin(const Case& c) {
 return "kind="+std::to_string(c.kind)+" line="+std::to_string(c.line)+
        " mode="+std::to_string(c.mode);
}
void sample(Counts& n,std::ostream& report,const Case& c,const std::string& why) {
 if(n.samples<128) {report<<origin(c)<<'\t'<<why<<'\n';++n.samples;}
}
Observed observe(UBiDi* b,const Case& c) {
 std::vector<UChar> text;
 std::vector<int> logical;
 std::vector<std::size_t> starts;
 for(std::size_t i=0;i<c.cps.size();++i) {
  const U cp=c.cps[i];
  starts.push_back(text.size());
  logical.push_back(static_cast<int>(i));
  if(cp<=0xffff) {text.push_back(static_cast<UChar>(cp));}
  else {
   text.push_back(static_cast<UChar>(0xd800+((cp-0x10000)>>10)));
   text.push_back(static_cast<UChar>(0xdc00+((cp-0x10000)&1023)));
   logical.push_back(-1);
  }
 }
 UErrorCode error=U_ZERO_ERROR;
 const UChar empty=0;
 ubidi_setPara(b,text.empty()?&empty:text.data(),static_cast<int32_t>(text.size()),
               c.mode==2?UBIDI_DEFAULT_LTR:static_cast<UBiDiLevel>(c.mode),nullptr,&error);
 need(U_SUCCESS(error),"ICU setPara "+origin(c));
 Observed out;
 out.base=ubidi_getParaLevel(b);
 const auto* levels=ubidi_getLevels(b,&error);
 need(U_SUCCESS(error),"ICU getLevels "+origin(c));
 out.mixed=ubidi_getDirection(b)==UBIDI_MIXED;
 for(const auto start:starts) {out.levels.push_back(levels[start]);}
 std::vector<int32_t> map(text.size());
 if(!map.empty()) {ubidi_getVisualMap(b,map.data(),&error);}
 need(U_SUCCESS(error),"ICU getVisualMap "+origin(c));
 for(const auto unit:map) {
  need(unit>=0&&static_cast<std::size_t>(unit)<logical.size(),"ICU map bounds");
  const int scalar=logical[static_cast<std::size_t>(unit)];
  if(scalar>=0&&c.levels[static_cast<std::size_t>(scalar)]>=0) {
   out.order.push_back(static_cast<U>(scalar));
  }
 }
 return out;
}
void compare(const Case& c,const Observed& actual,Counts& n,std::ostream& report) {
 ++n.cases;
 ++n.kinds[c.kind];
 if(actual.base!=c.base) {
  ++n.bases_bad;
  sample(n,report,c,"base expected="+std::to_string(c.base)+" actual="+std::to_string(actual.base));
 }
 need(actual.levels.size()==c.levels.size(),"observed level cardinality");
 for(std::size_t i=0;i<c.levels.size();++i) {
  if(c.levels[i]<0) {continue;}
  if(actual.mixed) {
   ++n.levels_checked;
   if(actual.levels[i]!=c.levels[i]) {
    ++n.levels_bad;
    sample(n,report,c,"level scalar="+std::to_string(i)+" expected="+
                       std::to_string(c.levels[i])+" actual="+std::to_string(actual.levels[i]));
   }
  } else {
   ++n.optimized;
   if(actual.levels[i]!=c.levels[i]) {++n.optimized_bad;}
  }
 }
 n.order_checked+=actual.order.size();
 if(actual.order!=c.order) {
  ++n.orders_bad;
  sample(n,report,c,"visual order disagreement");
 }
}
bool focus(const Case& c) {return c.kind==1&&c.line==67&&c.mode==2;}
} // namespace

int main(int argc,char** argv) {
 try {
  need(argc==3||argc==4,"usage: bidi-collect UCD_DIRECTORY FRESH_REPORT [none|level|base|order|missing|duplicate]");
  const std::string mode=argc==4?argv[3]:"none";
  need(mode=="none"||mode=="level"||mode=="base"||mode=="order"||
       mode=="missing"||mode=="duplicate","unknown observation mutant");
  need(!std::filesystem::exists(argv[2]),"refuse existing report");
  std::ofstream report(argv[2]);
  need(bool(report),"open mismatch report");
  report<<"origin\tmismatch\n";
  UVersionInfo version{};
  u_getVersion(version);
  need(version[0]==79&&version[1]==1,"ICU79.1 required");
  u_getUnicodeVersion(version);
  need(version[0]==18&&version[1]==0&&version[2]==0,"Unicode18 final required");
  for(std::size_t i=0;i<bidi_cases::representatives.size();++i) {
   need(class_id(u_charDirection(static_cast<UChar32>(bidi_cases::representatives[i])))==
        static_cast<int>(i),"class representative mismatch");
  }
  UErrorCode error=U_ZERO_ERROR;
  std::unique_ptr<UBiDi,decltype(&ubidi_close)> b(ubidi_openSized(4096,0,&error),ubidi_close);
  need(U_SUCCESS(error)&&bool(b),"open ICU bidi");
  Counts n;
  bidi_cases::read(argv[1],[&](const Case& c) {
   ++n.read;
   ++n.read_kinds[c.kind];
   auto actual=observe(b.get(),c);
   ++n.executed;
   const bool inject=mode!="none"&&focus(c);
   if(inject) {
    ++n.mutant_hits;
    need(actual.mixed,"mutation witness must be MIXED");
    need(c.levels.size()>6&&c.levels[6]==2&&c.base==1&&c.order.size()>=2,
         "mutation witness source identity");
    if(mode=="level") {actual.levels[6]=3;}
    if(mode=="base") {actual.base=0;}
    if(mode=="order") {std::swap(actual.order[0],actual.order[1]);}
    if(mode=="missing") {return;}
   }
   compare(c,actual,n,report);
   if(inject&&mode=="duplicate") {compare(c,actual,n,report);}
  });
  need(n.read==861948&&n.read_kinds[0]==770241&&n.read_kinds[1]==91707,
       "locked official input inventory mismatch");
  need(n.executed==n.read,"every source case must execute in ICU");
  if(mode!="none") {need(n.mutant_hits==1,"observation mutant must fire exactly once");}
  const bool counts_bad=n.cases!=861948||n.kinds[0]!=770241||n.kinds[1]!=91707;
  report.close();
  need(bool(report),"close mismatch report");
  std::cout<<"BIDI_COLLECT_COMPLETE cases="<<n.cases<<" class_cases="<<n.kinds[0]
           <<" character_cases="<<n.kinds[1]<<" read_cases="<<n.read
           <<" executed_cases="<<n.executed<<'\n';
  std::cout<<"BIDI_COLLECT_BAD bases="<<n.bases_bad<<" levels="<<n.levels_bad
           <<" orders="<<n.orders_bad<<" counts="<<(counts_bad?1:0)<<'\n';
  std::cout<<"BIDI_COLLECT_LEVELS mixed_exact="<<n.levels_checked
           <<" optimized_observations="<<n.optimized
           <<" optimized_mismatches="<<n.optimized_bad
           <<" order_checked="<<n.order_checked<<'\n';
  std::cout<<"BIDI_COLLECT_MUTATION mode="<<mode<<" hits="<<n.mutant_hits<<'\n';
  const bool failed=n.bases_bad||n.levels_bad||n.orders_bad||counts_bad;
  std::cout<<(failed?"BIDI_COLLECT_VERDICT FAIL\n":"BIDI_COLLECT_VERDICT PASS\n");
  return failed?1:0;
 } catch(const std::exception& e) {
  std::cerr<<e.what()<<'\n';
  return 2;
 }
}
