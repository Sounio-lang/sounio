// Isolated SDK diagnostics: BD16 must discard all earlier pairs on overflow.
// Expected relations follow UAX9 rev52 BD16/N0; canonical corpus is unchanged.
#include <unicode/ubidi.h>
#include <unicode/uchar.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct Observation {std::vector<UBiDiLevel> levels;std::vector<int32_t> order;};
void need(bool v,const char* what) {if(!v) throw std::runtime_error(what);}
Observation observe(const std::vector<UChar>& text) {
 UErrorCode e=U_ZERO_ERROR;
 std::unique_ptr<UBiDi,decltype(&ubidi_close)> b(ubidi_open(),ubidi_close);
 need(bool(b),"open");
 ubidi_setPara(b.get(),text.data(),static_cast<int32_t>(text.size()),1,nullptr,&e);
 need(U_SUCCESS(e),"setPara");
 const auto* levels=ubidi_getLevels(b.get(),&e);
 need(U_SUCCESS(e),"levels");
 Observation out;
 out.levels.assign(levels,levels+text.size());
 out.order.resize(text.size());
 ubidi_getVisualMap(b.get(),out.order.data(),&e);
 need(U_SUCCESS(e),"visualMap");
 return out;
}
std::vector<UChar> text(int depth,bool canonical) {
 std::vector<UChar> out={'a','(','b',')',0x05d0};
 for(int i=0;i<depth;++i)out.push_back(canonical?(i%2?0x2329:0x3008):'(');
 out.push_back('c');
 for(int i=0;i<depth;++i)out.push_back(canonical?(i%2?0x3009:0x232a):')');
 return out;
}
std::vector<UChar> neutral(std::vector<UChar> input) {
 for(auto& c:input)if(u_getIntPropertyValue(c,UCHAR_BIDI_PAIRED_BRACKET_TYPE)!=U_BPT_NONE)c='!';
 return input;
}
}
int main() {
 try {
  int cases=0,failures=0;
  auto verdict=[&](const char* family,int depth,bool canonical,bool ok) {
   ++cases;if(!ok)++failures;
   std::cout<<"SDK_CONTROL family="<<family<<" depth="<<depth
            <<" canonical="<<canonical<<" status="<<(ok?"PASS":"FAIL")<<'\n';
  };
  for(bool canonical:{false,true})for(int depth:{62,63,64,65,129}) {
   const auto input=text(depth,canonical);
   const auto actual=observe(input),plain=observe(neutral(input));
   const bool overflow=depth>63;
   const bool ok=overflow
    ?actual.levels==plain.levels&&actual.order==plain.order&&actual.levels[3]==1
    :actual.levels[3]==2&&plain.levels[3]==1;
   verdict("completed-before-overflow",depth,canonical,ok);
  }
  for(bool canonical:{false,true}) {
   std::vector<UChar> input={'a','(','b',')',0x05d0,0x2067};
   const auto inner=text(64,canonical);
   input.insert(input.end(),inner.begin(),inner.end());
   input.push_back(0x2069);input.push_back(0x05d1);
   const auto actual=observe(input);
   verdict("isolate-overflow-keeps-outer-pair",64,canonical,
           actual.levels[3]==2&&actual.levels[6+3]==3);
  }
  std::cout<<"BIDI_SDK_CONTROLS cases="<<cases<<" failures="<<failures<<'\n';
  return failures?1:0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
}
