// Oracle for tests/run-pass/wide_int_fn_abi_high_limbs.sio: the two limbs the
// test prints. Build: c++ -std=c++23 wide_int_fn_abi_high_limbs.cpp && ./a.out
#include <cstdio>
#include <cstdint>
typedef __int128 I; typedef unsigned __int128 U;
long long hi64(I v){ return (long long)(v >> 64); } long long lo64(I v){ return (long long)(uint64_t)v; }
int main(){
  I two100 = (I)1 << 100;
  I a = two100 + 1;                       // add_one(2^100)
  printf("add_one_hi=%lld add_one_lo=%lld\n", hi64(a), lo64(a));
  I f=1; for(int i=2;i<=30;i++) f*=i;     // 30!
  printf("fact30_div_2pow64=%lld fact30_mod_2pow64=%llu\n", (long long)(f >> 64), (unsigned long long)(uint64_t)f);
  U u = (U)(uint64_t)0xFFFFFFFFFFFFFFFFULL;  // u64 max widened to u128
  printf("u128_from_u64max_hi=%llu\n", (unsigned long long)(u >> 64));
  I n = (I)(-1);                           // i64 -1 widened
  printf("i128_from_neg1_hi=%lld\n", hi64(n));
  I sq = (I)3037000500LL * 3037000500LL;   // > 2^63
  printf("square_hi=%lld square_lo=%llu\n", hi64(sq), (unsigned long long)(uint64_t)sq);
}
