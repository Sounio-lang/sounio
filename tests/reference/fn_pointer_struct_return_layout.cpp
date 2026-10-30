#include <cassert>
#include <cstdint>
#include <cstdio>
using I=std::int64_t;
struct Earlier{I value;I outcome;};
struct Later{I state;I outcome;I value;I wake;};
Later make(I state){return {state,1,42,99};}
I via_parameter(Later(*callback)(I),I state){
 const auto step=callback(state);
 assert(step.state==state&&step.outcome==1&&step.wake==99);
 return step.value;
}
int main(){
 assert(via_parameter(make,13)==42);
 auto typed=&make;const auto first=typed(17);assert(first.value==42&&first.state==17&&first.wake==99);
 auto alias=typed;const auto second=alias(23);assert(second.value==42&&second.state==23);
 const auto raw=reinterpret_cast<std::uintptr_t>(&make);
 auto casted=reinterpret_cast<Later(*)(I)>(raw);const auto third=casted(29);
 assert(third.value==42&&third.state==29&&third.wake==99);
 Earlier earlier{5,7};assert(earlier.value==5);
 std::puts("FN_POINTER_LAYOUT_CPP_OK");
}
