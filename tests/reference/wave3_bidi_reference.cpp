#include "wave3_bidi_cases.hpp"
#include <unicode/ubidi.h>
#include <unicode/uchar.h>
#include <unicode/uversion.h>
#include <iostream>
#include <memory>
static int type(UCharDirection d){
 switch(d){
  case U_LEFT_TO_RIGHT:return 0;case U_RIGHT_TO_LEFT:return 1;case U_RIGHT_TO_LEFT_ARABIC:return 2;
  case U_EUROPEAN_NUMBER:return 3;case U_EUROPEAN_NUMBER_SEPARATOR:return 4;case U_EUROPEAN_NUMBER_TERMINATOR:return 5;
  case U_ARABIC_NUMBER:return 6;case U_COMMON_NUMBER_SEPARATOR:return 7;case U_DIR_NON_SPACING_MARK:return 8;
  case U_BOUNDARY_NEUTRAL:return 9;case U_BLOCK_SEPARATOR:return 10;case U_SEGMENT_SEPARATOR:return 11;
  case U_WHITE_SPACE_NEUTRAL:return 12;case U_OTHER_NEUTRAL:return 13;case U_LEFT_TO_RIGHT_EMBEDDING:return 14;
  case U_LEFT_TO_RIGHT_OVERRIDE:return 15;case U_RIGHT_TO_LEFT_EMBEDDING:return 16;case U_RIGHT_TO_LEFT_OVERRIDE:return 17;
  case U_POP_DIRECTIONAL_FORMAT:return 18;case U_LEFT_TO_RIGHT_ISOLATE:return 19;case U_RIGHT_TO_LEFT_ISOLATE:return 20;
  case U_FIRST_STRONG_ISOLATE:return 21;case U_POP_DIRECTIONAL_ISOLATE:return 22;default:throw std::runtime_error("unknown ICU Bidi_Class");
 }
}
int main(int argc,char** argv){
 try{
  using namespace bidi_cases;need(argc==3,"usage: bidi-ICU-reference UCD_DIRECTORY FRESH_OPTIMIZED_LEVEL_REPORT");
  need(!std::filesystem::exists(argv[2]),"refuse existing optimized-level report");
  std::ofstream optimized(argv[2]);need(bool(optimized),"open optimized-level report");
  optimized<<"kind\tline\tmode\tscalar\tofficial_level\ticu_optimized_level\n";
  UVersionInfo version{};u_getVersion(version);need(version[0]==79&&version[1]==1,"ICU79.1 required");
  u_getUnicodeVersion(version);need(version[0]==18&&version[1]==0&&version[2]==0,"Unicode18 final required");
  for(std::size_t i=0;i<representatives.size();++i)need(type(u_charDirection(static_cast<UChar32>(representatives[i])))==static_cast<int>(i),"class representative mismatch");
  UErrorCode error=U_ZERO_ERROR;std::unique_ptr<UBiDi,decltype(&ubidi_close)> bidi(ubidi_openSized(4096,0,&error),ubidi_close);
  need(U_SUCCESS(error)&&bool(bidi),"open ICU BIDI");std::uint64_t cases=0,levels_checked=0,order_checked=0;std::array<std::uint64_t,2> kinds{};std::uint64_t optimized_observations=0,optimized_mismatches=0;
  read(argv[1],[&](const Case& c){
   std::vector<UChar> text;std::vector<int> logical;std::vector<std::size_t> starts;
   for(std::size_t i=0;i<c.cps.size();++i){
    starts.push_back(text.size());const U cp=c.cps[i];logical.push_back(static_cast<int>(i));
    if(cp<=0xffff)text.push_back(static_cast<UChar>(cp));
    else{text.push_back(static_cast<UChar>(0xd800+((cp-0x10000)>>10)));text.push_back(static_cast<UChar>(0xdc00+((cp-0x10000)&1023)));logical.push_back(-1);}
   }
   error=U_ZERO_ERROR;const UChar empty=0;
   ubidi_setPara(bidi.get(),text.empty()?&empty:text.data(),static_cast<int32_t>(text.size()),c.mode==2?UBIDI_DEFAULT_LTR:static_cast<UBiDiLevel>(c.mode),nullptr,&error);
   need(U_SUCCESS(error),"ICU setPara");
   const std::string origin="kind="+std::to_string(c.kind)+" line="+std::to_string(c.line)+" mode="+std::to_string(c.mode);
   need(ubidi_getParaLevel(bidi.get())==c.base,"paragraph base disagreement "+origin);
   const auto* levels=ubidi_getLevels(bidi.get(),&error);need(U_SUCCESS(error),"ICU levels");
   for(std::size_t i=0;i<c.cps.size();++i)if(c.levels[i]>=0){
    if(ubidi_getDirection(bidi.get())==UBIDI_MIXED){need(levels[starts[i]]==c.levels[i],"mixed level disagreement "+origin+" scalar="+std::to_string(i));++levels_checked;}
    else{
     ++optimized_observations;
     if(levels[starts[i]]!=c.levels[i]){++optimized_mismatches;optimized<<c.kind<<'\t'<<c.line<<'\t'<<c.mode<<'\t'<<i<<'\t'<<c.levels[i]<<'\t'<<static_cast<unsigned>(levels[starts[i]])<<'\n';}
    }
   }
   std::vector<int32_t> map(text.size());if(!map.empty())ubidi_getVisualMap(bidi.get(),map.data(),&error);need(U_SUCCESS(error),"ICU visual map");
   std::vector<U> actual;
   for(int32_t unit:map){need(unit>=0&&static_cast<std::size_t>(unit)<logical.size(),"ICU map bounds");const int scalar=logical[static_cast<std::size_t>(unit)];if(scalar>=0&&c.levels[static_cast<std::size_t>(scalar)]>=0)actual.push_back(static_cast<U>(scalar));}
   need(actual==c.order,"visual order disagreement "+origin);order_checked+=actual.size();++cases;++kinds[c.kind];
  });
  need(cases&&kinds[0]&&kinds[1],"empty official corpus");
  optimized.close();need(bool(optimized),"close optimized-level report");
  std::cout<<"BIDI_ICU_OFFICIAL_OK cases="<<cases<<" class_cases="<<kinds[0]<<" character_cases="<<kinds[1]<<" mixed_exact_levels="<<levels_checked<<" optimized_level_observations="<<optimized_observations<<" optimized_level_mismatches="<<optimized_mismatches<<" order="<<order_checked<<"\n";
 }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
