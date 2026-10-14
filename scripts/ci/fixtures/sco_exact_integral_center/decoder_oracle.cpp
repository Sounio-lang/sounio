#include <bit>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <array>
#include <limits>
#include <cstring>
static std::pair<bool,double> decode(std::uint64_t bits) {
 const std::uint64_t sign=(bits>>63)&1, exponent=(bits>>52)&2047, fraction=bits&4503599627370495ULL;
 if(exponent==0 && fraction==0) return {true, sign ? -0.0 : 0.0};
 if(exponent<993 || exponent>1077) return {false,0.0};
 double value=1.0+static_cast<double>(fraction)/4503599627370496.0;
 const auto power=static_cast<std::int64_t>(exponent)-1023;
 if(power>=0) for(std::int64_t i=0;i<power;++i) value*=2.0;
 else for(std::int64_t i=0;i<-power;++i) value*=0.5;
 return {true,sign ? -value : value};
}
int main(int argc, char** argv) {
 const bool emit = argc==2 && std::strcmp(argv[1], "--sounio-fixture")==0;
 // Independent reference is C++23 bit_cast, not the arithmetic decoder.
 const std::array<std::uint64_t,22> cases={
  0,0x8000000000000000ULL,0x3ff0000000000000ULL,0xbff0000000000000ULL,
  0x4341c37937e08000ULL,0xc341c37937e08000ULL,0x4350000000000000ULL,0xc350000000000000ULL,
  0x3e10000000000000ULL,0xbe10000000000000ULL,0x3fe0000000000000ULL,0x3fd0000000000000ULL,
  0x3e112e0be826d695ULL,0x3ff8000000000000ULL,0x3e0fffffffffffffULL,
  0x4360000000000000ULL,0x7ff0000000000000ULL,0xfff0000000000000ULL,
  0x7ff8000000000001ULL,1,0x000fffffffffffffULL,0x7fefffffffffffffULL};
 if(emit) std::printf("use ir::opt_cleanup::{ocp_private_bounded_decoder}\nfn test_decode(bits:i64, expected:bool) -> bool with Mut, Div {\n let actual = ocp_private_bounded_decoder(bits)\n if actual.0 != expected { return false }\n if expected { return f64_to_bits(actual.1) == bits }\n true\n}\nfn main() -> i32 with IO, Mut, Div {\n");
 int accepted=0,refused=0;
 for(const auto bits:cases) {
  const std::uint64_t exponent=(bits>>52)&2047, fraction=bits&4503599627370495ULL;
  const bool domain=(exponent==0 && fraction==0)||(exponent>=993&&exponent<=1077);
  const auto actual=decode(bits);
  if(actual.first!=domain) return 1;
  if(domain) {
   if(std::bit_cast<std::uint64_t>(actual.second)!=bits) return 2;
   ++accepted;
  } else ++refused;
  if(emit) {
   if(bits==0x8000000000000000ULL) std::printf(" if !test_decode((-9223372036854775807 - 1), %s) { return 1 }\n",domain?"true":"false");
   else std::printf(" if !test_decode(%lld, %s) { return 1 }\n",static_cast<long long>(std::bit_cast<std::int64_t>(bits)),domain?"true":"false");
  } else std::printf("%016llx %s\n",static_cast<unsigned long long>(bits),domain?"EXACT":"REFUSED");
 }
 if(emit) std::printf(" print(\"BOUNDED_DECODER_SOUNIO_22_VECTORS_PASS\\n\")\n 0\n}\n");
 else std::printf("BOUNDED_DECODER_CPP23_PASS accepted=%d refused=%d\n",accepted,refused);
}
