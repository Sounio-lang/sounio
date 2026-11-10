// Exact construction + independent long-double reference for 20k x 512 counts.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
int main() {
    constexpr std::int64_t genes=20000,samples=512;
    long double logsum=0;
    for(std::int64_t s=0;s<samples;++s) {
        const std::int64_t factor=std::int64_t{1}<<(s%4);
        const long double centered=std::log(static_cast<long double>(factor))-1.5L*std::log(2.L);
        const long double normalized=std::exp(centered);
        assert(std::abs(normalized*std::sqrt(8.L)-factor)<1e-15L);
        logsum+=centered;
        for(std::int64_t g=0;g<genes;++g){
            const auto count=(g+1)*factor;
            assert(count>0 && count<9007199254740992LL);
            assert(count/(g+1)==factor);
        }
    }
    assert(std::abs(logsum)<1e-14L);
    std::cout<<"DEEP_NORMALIZATION_EXACT_REFERENCE_PASS\n";
}
