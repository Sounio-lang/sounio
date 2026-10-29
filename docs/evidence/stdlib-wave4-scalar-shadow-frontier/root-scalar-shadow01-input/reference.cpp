#include <cstdint>
#include <iostream>
constexpr std::int64_t core(std::int64_t seed){auto s=seed;if(s==0)s=1;s=s*3+1;return s;}
static_assert(core(0)==4);static_assert(core(5)==16);
int main(){std::cout<<"CORE 0 "<<core(0)<<"\nCORE 5 "<<core(5)<<"\n";return (core(0)==4&&core(5)==16)?0:1;}
