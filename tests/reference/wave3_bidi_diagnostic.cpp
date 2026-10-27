// Diagnostic only: no conformance-acceptance marker, no input/expectation changes.
#include "wave3_bidi_cases.hpp"
#include <unicode/ubidi.h>
#include <unicode/uchar.h>
#include <unicode/uversion.h>
#include <iomanip>
#include <iostream>
#include <memory>
// Independent bounded UAX9 analysis for this exact all-BMP witness.
// After X1-X8 every retained character is L or R, so W/N rules are identities.
static std::pair<std::vector<int>,std::vector<bidi_cases::U>> normative_witness(const std::vector<int>& types,int base){
 using bidi_cases::need;using U=bidi_cases::U;
 enum {L=0,R=1,LRE=14,LRO=15,RLE=16,RLO=17,PDF=18,LRI=19,RLI=20,PDI=22};
 struct State{int level,override;bool isolate;};
 std::vector<State> stack{{base,-1,false}};
 std::vector<int> resolved(types.size(),-1),levels(types.size(),-1);
 for(std::size_t i=0;i<types.size();++i){
  const int original=types[i];const auto current=stack.back();
  if(original==RLO||original==LRO){const int next=original==RLO?current.level+1+(current.level&1):current.level+2-(current.level&1);stack.push_back({next,original==RLO?R:L,false});continue;}
  if(original==LRE||original==RLE){const int next=original==RLE?current.level+1+(current.level&1):current.level+2-(current.level&1);stack.push_back({next,-1,false});continue;}
  if(original==PDF){need(stack.size()>1&&!stack.back().isolate,"witness PDF");stack.pop_back();continue;}
  levels[i]=current.level;
  if(original==LRI||original==RLI){
   // X5a applies the current override to the initiator, before entering isolation.
   resolved[i]=current.override;
   stack.push_back({original==RLI?current.level+1+(current.level&1):current.level+2-(current.level&1),-1,true});
  }else if(original==PDI){
   // X6a leaves the valid isolate and reapplies the surrounding override.
   while(stack.size()>1&&!stack.back().isolate)stack.pop_back();
   need(stack.size()>1&&stack.back().isolate,"witness PDI");stack.pop_back();
   levels[i]=stack.back().level;resolved[i]=stack.back().override;
  }else{need(original==L||original==R,"witness retained type");resolved[i]=current.override<0?original:current.override;}
  need(resolved[i]==L||resolved[i]==R,"bounded witness does not need W/N resolution");
 }
 std::vector<U> order;int maximum=0,lowest_odd=127;
 for(std::size_t i=0;i<levels.size();++i)if(levels[i]>=0){
  if((levels[i]&1)==0&&resolved[i]==R)++levels[i];
  else if((levels[i]&1)==1&&resolved[i]==L)++levels[i];
  maximum=std::max(maximum,levels[i]);if(levels[i]&1)lowest_odd=std::min(lowest_odd,levels[i]);order.push_back(static_cast<U>(i));
 }
 for(int level=maximum;level>=lowest_odd;--level){
  std::size_t at=0;
  while(at<order.size()){
   if(levels[order[at]]<level){++at;continue;}
   const auto first=at;while(at<order.size()&&levels[order[at]]>=level)++at;
   std::reverse(order.begin()+static_cast<std::ptrdiff_t>(first),order.begin()+static_cast<std::ptrdiff_t>(at));
  }
 }
 return {levels,order};
}
int main(int argc,char** argv){
 try{
  using namespace bidi_cases;need(argc==2,"usage: line61-diagnostic UCD_DIRECTORY");Case source;bool found=false;
  read(argv[1],[&](const Case& c){if(c.kind==1&&c.line==61){need(!found,"duplicate source selection");source=c;found=true;}});
  need(found,"missing exact official line61");
  const std::vector<U> direct{0x202e,0x0061,0x202a,0x0062,0x202c,0x2066,0x0063,0x2069,0x202a,0x0064,0x202c,0x0065,0x202c};
  const std::vector<int> expected{-1,1,-1,2,-1,1,2,1,-1,2,-1,1,-1};
  const std::vector<U> order{11,9,7,6,5,3,1};
  need(source.cps==direct&&source.levels==expected&&source.order==order&&source.mode==2&&source.base==0,"official line61 differs from independently transcribed witness");
  const auto normative=normative_witness({17,0,14,0,18,19,0,22,14,0,18,0,18},0);need(normative.first==expected&&normative.second==order,"independent X5a/X6a/I/L2 witness disagreement");
  std::cout<<"BIDI_UAX9_LINE61_CPP_NORMATIVE_OK\n";
  const auto dual=normative_witness({15,1,16,1,18,20,1,22,16,1,18,1,18},1);
  need(dual.first==std::vector<int>{-1,2,-1,3,-1,2,3,2,-1,3,-1,2,-1}&&dual.second==std::vector<U>{1,3,5,6,7,9,11},"independent RLI/LRO dual control");
  const auto nested=normative_witness({17,0,19,17,0,20,0,22,0,18,22,0,18},0);
  need(nested.first==std::vector<int>{-1,1,1,-1,3,3,6,3,3,-1,1,1,-1}&&nested.second==std::vector<U>{11,10,8,7,6,5,4,2,1},"nested isolate/PDI override control");
  std::cout<<"BIDI_UAX9_DUAL_NESTED_CPP_OK cases=3\n";
  UVersionInfo version{};u_getVersion(version);std::cout<<"ICU "<<unsigned(version[0])<<'.'<<unsigned(version[1])<<'.'<<unsigned(version[2])<<'.'<<unsigned(version[3])<<'\n';
  u_getUnicodeVersion(version);std::cout<<"UNICODE "<<unsigned(version[0])<<'.'<<unsigned(version[1])<<'.'<<unsigned(version[2])<<'.'<<unsigned(version[3])<<'\n';
  bool discrepancy=false;
  // Auto-LTR is official line61; explicit LTR is an additional identical-base control.
  for(U mode:{0U,2U}){
   UErrorCode error=U_ZERO_ERROR;std::unique_ptr<UBiDi,decltype(&ubidi_close)> bidi(ubidi_openSized(64,0,&error),ubidi_close);
   std::vector<UChar> text;for(U cp:direct){need(cp<=0xffff,"witness is intentionally all BMP");text.push_back(static_cast<UChar>(cp));}
   ubidi_setPara(bidi.get(),text.data(),static_cast<int32_t>(text.size()),mode==2?UBIDI_DEFAULT_LTR:0,nullptr,&error);need(U_SUCCESS(error),"ICU setPara");
   std::cout<<"MODE "<<mode<<" BASE "<<unsigned(ubidi_getParaLevel(bidi.get()))<<" DIRECTION "<<int(ubidi_getDirection(bidi.get()))<<" REORDER_MODE "<<int(ubidi_getReorderingMode(bidi.get()))<<" OPTIONS "<<ubidi_getReorderingOptions(bidi.get())<<'\n';
   std::vector<unsigned> before;for(std::size_t i=0;i<text.size();++i)before.push_back(ubidi_getLevelAt(bidi.get(),static_cast<int32_t>(i)));
   const auto* levels=ubidi_getLevels(bidi.get(),&error);need(U_SUCCESS(error),"getLevels");
   std::cout<<"scalar utf16_index codepoint original_class expected getLevelAt_before getLevels getLevelAt_after\n";
   for(std::size_t i=0;i<text.size();++i){
    const unsigned after=ubidi_getLevelAt(bidi.get(),static_cast<int32_t>(i));
    std::cout<<i<<' '<<i<<" U+"<<std::hex<<std::uppercase<<direct[i]<<std::dec<<' '<<int(u_charDirection(static_cast<UChar32>(direct[i])))<<' '<<expected[i]<<' '<<before[i]<<' '<<unsigned(levels[i])<<' '<<after<<'\n';
    if(expected[i]>=0&&unsigned(expected[i])!=levels[i])discrepancy=true;
   }
   std::vector<int32_t> map(text.size());ubidi_getVisualMap(bidi.get(),map.data(),&error);need(U_SUCCESS(error),"getVisualMap");
   std::cout<<"RAW_UTF16_VISUAL_MAP";for(int x:map)std::cout<<' '<<x;std::cout<<'\n';
   std::vector<U> filtered;for(int x:map){need(x>=0&&static_cast<std::size_t>(x)<expected.size(),"map bounds");if(expected[static_cast<std::size_t>(x)]>=0)filtered.push_back(static_cast<U>(x));}
   std::cout<<"FILTERED_SCALAR_MAP";for(U x:filtered)std::cout<<' '<<x;std::cout<<'\n';
   std::cout<<"OFFICIAL_SCALAR_MAP";for(U x:order)std::cout<<' '<<x;std::cout<<'\n';
   if(filtered!=order)discrepancy=true;
  }
  std::cout<<(discrepancy?"BIDI_ICU_LINE61_DISAGREEMENT\n":"BIDI_ICU_LINE61_AGREEMENT\n");
  std::cout<<"BIDI_ICU_LINE61_DIAGNOSTIC_COMPLETE\n";
 }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
