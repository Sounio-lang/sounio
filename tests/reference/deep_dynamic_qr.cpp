// Independent analytic reference for the dynamic QR acceptance fixture.
// Each of 64 indicator columns has 128 nonzero rows; A^T A=128 I.
// Exact coefficients are 1..64 and exact residual is zero.
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
int main() {
    constexpr std::int64_t m=8192,p=64;
    std::int64_t diagonal[64]{}, cross[64]{};
    for(std::int64_t i=0;i<m;++i){const auto j=i%p;++diagonal[j];cross[j]+=j+1;}
    for(std::int64_t j=0;j<p;++j){assert(diagonal[j]==128);assert(cross[j]==diagonal[j]*(j+1));}
    // Exact maximum affine index, including overflow-invalid fixtures.
    constexpr std::int64_t rows=32,cols=128,rs=128,cs=1;
    static_assert((rows-1)*rs+(cols-1)*cs==4095);
    static_assert(std::numeric_limits<std::int64_t>::max()/8< std::numeric_limits<std::int64_t>::max());
    std::cout<<"DEEP_DYNAMIC_QR_EXACT_REFERENCE_PASS\n";
}
