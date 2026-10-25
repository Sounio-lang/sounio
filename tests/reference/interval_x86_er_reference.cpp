// Test-only C++23 EVEX arithmetic witness. NOT a Sounio interval backend.
// Finite scalar inputs; directed overflow/subnormal/signed-zero outputs allowed.
// MPFR is the independent test oracle, never a compiler/runtime dependency.
#include <immintrin.h>
#include <mpfr.h>
#include <array>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

using U64=std::uint64_t;
constexpr int round_nearest=_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC;
constexpr int round_down=_MM_FROUND_TO_NEG_INF|_MM_FROUND_NO_EXC;
constexpr int round_up=_MM_FROUND_TO_POS_INF|_MM_FROUND_NO_EXC;
enum class Mutation {none,rn,swapped,flush,bypass};
template<class T> using UInt=std::conditional_t<std::is_same_v<T,float>,std::uint32_t,U64>;
template<class T> constexpr UInt<T> sign_bit=UInt<T>(1)<<(sizeof(T)*8-1);
template<class T> constexpr UInt<T> exponent=std::is_same_v<T,float>?UInt<T>(0x7f800000U):UInt<T>(0x7ff0000000000000ULL);
template<class T> constexpr UInt<T> one=std::is_same_v<T,float>?UInt<T>(0x3f800000U):UInt<T>(0x3ff0000000000000ULL);
template<class T> constexpr UInt<T> half_ulp=std::is_same_v<T,float>?UInt<T>(0x33800000U):UInt<T>(0x3ca0000000000000ULL);
template<class T> bool finite(UInt<T> u){return (u&exponent<T>)!=exponent<T>;}
template<class T> bool zero(UInt<T> u){return (u&~sign_bit<T>)==0;}
template<class T> UInt<T> bits(T x){return std::bit_cast<UInt<T>>(x);}
template<class T> T value(UInt<T> u){return std::bit_cast<T>(u);}
static void require(bool condition,const char* why){if(!condition)throw std::runtime_error(why);}

// Each leaf's rounding immediate is part of its instruction, with SAE.
// Separate noinline targets let the gate inspect the actual emitted instructions.
template<class T,int Op,int R>
[[gnu::target("avx512f"),gnu::noinline]] T scalar_er(T x,T y){
 if constexpr(std::is_same_v<T,float>){
  const auto a=_mm_set_ss(x),b=_mm_set_ss(y);
  if constexpr(Op==0)return _mm_cvtss_f32(_mm_add_round_ss(a,b,R));
  if constexpr(Op==1)return _mm_cvtss_f32(_mm_sub_round_ss(a,b,R));
  if constexpr(Op==2)return _mm_cvtss_f32(_mm_mul_round_ss(a,b,R));
  if constexpr(Op==3)return _mm_cvtss_f32(_mm_div_round_ss(a,b,R));
 }else{
  const auto a=_mm_set_sd(x),b=_mm_set_sd(y);
  if constexpr(Op==0)return _mm_cvtsd_f64(_mm_add_round_sd(a,b,R));
  if constexpr(Op==1)return _mm_cvtsd_f64(_mm_sub_round_sd(a,b,R));
  if constexpr(Op==2)return _mm_cvtsd_f64(_mm_mul_round_sd(a,b,R));
  if constexpr(Op==3)return _mm_cvtsd_f64(_mm_div_round_sd(a,b,R));
 }
}
template<class T,int R> T scalar_dispatch(int op,T x,T y){
 switch(op){case 0:return scalar_er<T,0,R>(x,y);case 1:return scalar_er<T,1,R>(x,y);case 2:return scalar_er<T,2,R>(x,y);case 3:return scalar_er<T,3,R>(x,y);default:throw std::runtime_error("operation");}
}
template<class T,int Op,int R>
[[gnu::target("avx512f"),gnu::noinline]] void zmm_er(const T* x,const T* y,T* out){
 if constexpr(std::is_same_v<T,float>){
  const auto a=_mm512_loadu_ps(x),b=_mm512_loadu_ps(y);
  if constexpr(Op==0)_mm512_storeu_ps(out,_mm512_add_round_ps(a,b,R));
  if constexpr(Op==1)_mm512_storeu_ps(out,_mm512_sub_round_ps(a,b,R));
  if constexpr(Op==2)_mm512_storeu_ps(out,_mm512_mul_round_ps(a,b,R));
  if constexpr(Op==3)_mm512_storeu_ps(out,_mm512_div_round_ps(a,b,R));
 }else{
  const auto a=_mm512_loadu_pd(x),b=_mm512_loadu_pd(y);
  if constexpr(Op==0)_mm512_storeu_pd(out,_mm512_add_round_pd(a,b,R));
  if constexpr(Op==1)_mm512_storeu_pd(out,_mm512_sub_round_pd(a,b,R));
  if constexpr(Op==2)_mm512_storeu_pd(out,_mm512_mul_round_pd(a,b,R));
  if constexpr(Op==3)_mm512_storeu_pd(out,_mm512_div_round_pd(a,b,R));
 }
}
template<class T,int R> void zmm_dispatch(int op,const T* a,const T* b,T* out){
 switch(op){case 0:zmm_er<T,0,R>(a,b,out);break;case 1:zmm_er<T,1,R>(a,b,out);break;case 2:zmm_er<T,2,R>(a,b,out);break;case 3:zmm_er<T,3,R>(a,b,out);break;default:throw std::runtime_error("operation");}
}
template<class T> void mutate(UInt<T>& lo,UInt<T>& hi,Mutation mode){
 if(mode==Mutation::swapped)std::swap(lo,hi);
 if(mode==Mutation::flush){
  if((lo&exponent<T>)==0&&!zero<T>(lo))lo&=sign_bit<T>;
  if((hi&exponent<T>)==0&&!zero<T>(hi))hi&=sign_bit<T>;
 }
}
// Status1 bad environment,2 nonfinite input,3 indeterminate0/0; no writes.
// Driver normalization around the MPFR oracle is separate from this guard.
template<class T> [[gnu::noinline]] int guarded_scalar(int op,T x,T y,UInt<T>& lo,UInt<T>& hi,Mutation mode){
 if(mode!=Mutation::bypass&&_mm_getcsr()!=0x1f80U)return 1;
 if(!finite<T>(bits(x))||!finite<T>(bits(y)))return 2;
 if(op==3&&zero<T>(bits(x))&&zero<T>(bits(y)))return 3;
 asm volatile("":::"memory");
 if(mode==Mutation::rn){lo=bits(scalar_dispatch<T,round_nearest>(op,x,y));hi=bits(scalar_dispatch<T,round_nearest>(op,x,y));}
 else {lo=bits(scalar_dispatch<T,round_down>(op,x,y));hi=bits(scalar_dispatch<T,round_up>(op,x,y));}
 mutate<T>(lo,hi,mode);return 0;
}
template<class T,std::size_t N> [[gnu::noinline]] int guarded_zmm(int op,const std::array<T,N>& a,const std::array<T,N>& b,std::array<T,N>& lo,std::array<T,N>& hi,Mutation mode){
 if(mode!=Mutation::bypass&&_mm_getcsr()!=0x1f80U)return 1;
 for(std::size_t i=0;i<N;++i){if(!finite<T>(bits(a[i]))||!finite<T>(bits(b[i])))return 2;if(op==3&&zero<T>(bits(a[i]))&&zero<T>(bits(b[i])))return 3;}
 asm volatile("":::"memory");
 if(mode==Mutation::rn){zmm_dispatch<T,round_nearest>(op,a.data(),b.data(),lo.data());zmm_dispatch<T,round_nearest>(op,a.data(),b.data(),hi.data());}
 else {zmm_dispatch<T,round_down>(op,a.data(),b.data(),lo.data());zmm_dispatch<T,round_up>(op,a.data(),b.data(),hi.data());}
 for(std::size_t i=0;i<N;++i){auto l=bits(lo[i]),h=bits(hi[i]);mutate<T>(l,h,mode);lo[i]=value<T>(l);hi[i]=value<T>(h);}
 return 0;
}
struct Oracle {
 mpfr_t a,b,r;
 mpfr_exp_t old_emin,old_emax;
 mpfr_flags_t old_flags;
 explicit Oracle(bool binary32):old_emin(mpfr_get_emin()),old_emax(mpfr_get_emax()),old_flags(mpfr_flags_save()){
  // MPFR4.2.1 manual, mpfr_subnormalize: exact IEEE precision/exponent model.
  // This test oracle emulates gradual underflow using the operation's ternary
  // return, rather than relying on a host-format cast of a rounded result.
  require(mpfr_set_emin(binary32?-148:-1073)==0,"MPFR emin");
  require(mpfr_set_emax(binary32?128:1024)==0,"MPFR emax");
  mpfr_inits2(binary32?24:53,a,b,r,(mpfr_ptr)0);
 }
 ~Oracle(){mpfr_clears(a,b,r,(mpfr_ptr)0);mpfr_set_emin(old_emin);mpfr_set_emax(old_emax);mpfr_flags_restore(old_flags,MPFR_FLAGS_ALL);}
 Oracle(const Oracle&)=delete;Oracle& operator=(const Oracle&)=delete;
 template<class T> UInt<T> endpoint(int op,T x,T y,mpfr_rnd_t rnd){
  if constexpr(std::is_same_v<T,float>){mpfr_set_flt(a,x,MPFR_RNDN);mpfr_set_flt(b,y,MPFR_RNDN);}
  else {mpfr_set_d(a,x,MPFR_RNDN);mpfr_set_d(b,y,MPFR_RNDN);}
  int ternary=0;
  switch(op){case 0:ternary=mpfr_add(r,a,b,rnd);break;case 1:ternary=mpfr_sub(r,a,b,rnd);break;case 2:ternary=mpfr_mul(r,a,b,rnd);break;case 3:ternary=mpfr_div(r,a,b,rnd);break;default:throw std::runtime_error("oracle operation");}
  mpfr_subnormalize(r,ternary,rnd);
  // r is now itself an exactly representable target-format value.
  if constexpr(std::is_same_v<T,float>)return bits(mpfr_get_flt(r,MPFR_RNDN));
  else return bits(mpfr_get_d(r,MPFR_RNDN));
 }
};
struct Rng {
 U64 state;
 U64 next(){state+=0x9e3779b97f4a7c15ULL;auto z=state;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
};
template<class T> auto edges(){
 using U=UInt<T>;
 const U s=sign_bit<T>,e=exponent<T>,o=one<T>,mn=e&~(e-1);
 return std::array<U,18>{o,half_ulp<T>,o|s,U(0),s,U(1),U(1)|s,mn,mn-1,e-1,(e-1)|s,o-1,o+1,o*2,U(2),U(2)|s,mn|s,(mn-1)|s};
}
static void mix(U64& hash,U64 v){hash=(hash^v)*0x100000001b3ULL;}
template<class T> void controls(Mutation mode){
 using U=UInt<T>;constexpr std::size_t N=64/sizeof(T);
 const U sentinel=U(0x1234567);
 std::array<T,N>a,b,lo,hi;a.fill(value<T>(one<T>));b.fill(value<T>(half_ulp<T>));lo.fill(value<T>(sentinel));hi.fill(value<T>(sentinel));
 for(const unsigned bad:{0x9f80U,0x1fc0U,0x3f80U,0x1f00U,0x1f81U}){
  _mm_setcsr(bad);U l=sentinel,h=sentinel;
  const auto rc=guarded_scalar(0,a[0],b[0],l,h,mode);
  require(rc==1&&l==sentinel&&h==sentinel&&_mm_getcsr()==bad,"environment refusal missing (scalar)");
  require(guarded_zmm(0,a,b,lo,hi,mode)==1&&_mm_getcsr()==bad,"environment refusal missing (ZMM)");
  for(std::size_t i=0;i<N;++i)require(bits(lo[i])==sentinel&&bits(hi[i])==sentinel,"environment refusal wrote outputs");
 }
 _mm_setcsr(0x1f80U);
 for(const U nonfinite:{exponent<T>,exponent<T>|sign_bit<T>,exponent<T>|U(1)}){
  U l=sentinel,h=sentinel;
  require(guarded_scalar(0,value<T>(nonfinite),a[0],l,h,mode)==2&&l==sentinel&&h==sentinel,"nonfinite refusal missing");
  a[N-1]=value<T>(nonfinite);
  require(guarded_zmm(0,a,b,lo,hi,mode)==2,"nonfinite ZMM refusal missing");
  for(std::size_t i=0;i<N;++i)require(bits(lo[i])==sentinel&&bits(hi[i])==sentinel,"nonfinite ZMM refusal wrote outputs");
  a[N-1]=value<T>(one<T>);
 }
 U l=sentinel,h=sentinel;
 require(guarded_scalar(3,value<T>(U(0)),value<T>(sign_bit<T>),l,h,mode)==3&&l==sentinel&&h==sentinel,"indeterminate refusal missing");
 a[N-1]=value<T>(UInt<T>(0));b[N-1]=value<T>(sign_bit<T>);
 require(guarded_zmm(3,a,b,lo,hi,mode)==3,"indeterminate ZMM refusal missing");
 for(std::size_t i=0;i<N;++i)require(bits(lo[i])==sentinel&&bits(hi[i])==sentinel,"indeterminate ZMM refusal wrote outputs");
 _mm_setcsr(0x1f80U);
}
template<class T> U64 qualify(U64 count,int op,Mutation mode){
 using U=UInt<T>;constexpr std::size_t N=64/sizeof(T);
 std::array<T,N>a,b,zlo,zhi;std::array<U,N>elo,ehi;
 Oracle oracle(std::is_same_v<T,float>);Rng rng{0xd340000000000000ULL+U64(sizeof(T))*256+U64(op)};
 const auto edge=edges<T>();U64 hash=0xcbf29ce484222325ULL;U64 checked=0;
 for(U64 base=0;base<count;base+=N){
  const auto active=std::min<U64>(N,count-base);
  for(std::size_t lane=0;lane<N;++lane){
   const auto index=base+lane;U x,y;
   if(index<edge.size()*edge.size()){x=edge[std::size_t(index)/edge.size()];y=edge[std::size_t(index)%edge.size()];}
   else {x=U(rng.next());y=U(rng.next());if(!finite<T>(x))x^=exponent<T>&~(exponent<T>-1);if(!finite<T>(y))y^=exponent<T>&~(exponent<T>-1);}
   if(op==3&&zero<T>(x)&&zero<T>(y))y=one<T>;
   a[lane]=value<T>(x);b[lane]=value<T>(y);
   elo[lane]=oracle.endpoint<T>(op,a[lane],b[lane],MPFR_RNDD);ehi[lane]=oracle.endpoint<T>(op,a[lane],b[lane],MPFR_RNDU);
  }
  _mm_setcsr(0x1f80U);
  require(guarded_zmm(op,a,b,zlo,zhi,mode)==0,"valid ZMM refused");
  require(_mm_getcsr()==0x1f80U,"ZMM altered MXCSR");
  for(std::size_t lane=0;lane<active;++lane){
   U lo=0,hi=0;_mm_setcsr(0x1f80U);
   require(guarded_scalar(op,a[lane],b[lane],lo,hi,mode)==0,"valid scalar refused");
   require(_mm_getcsr()==0x1f80U,"scalar altered MXCSR");
   if(lo!=elo[lane]||hi!=ehi[lane]||bits(zlo[lane])!=elo[lane]||bits(zhi[lane])!=ehi[lane]){
    std::cerr<<"rounding mismatch format="<<sizeof(T)*8<<" op="<<op<<" case="<<base+lane<<std::hex<<" a="<<bits(a[lane])<<" b="<<bits(b[lane])<<" expected="<<elo[lane]<<','<<ehi[lane]<<" scalar="<<lo<<','<<hi<<" ZMM="<<bits(zlo[lane])<<','<<bits(zhi[lane])<<std::dec<<'\n';throw std::runtime_error("directed endpoint disagreement");
   }
   mix(hash,bits(a[lane]));mix(hash,bits(b[lane]));mix(hash,lo);mix(hash,hi);++checked;
  }
 }
 require(checked==count,"actual case coverage mismatch");
 std::cout<<"ER_CASES format="<<sizeof(T)*8<<" op="<<op<<" cases="<<count<<" scalar=PASS ZMM=PASS bit_identity=PASS digest_fnv64="<<std::hex<<hash<<std::dec<<'\n';
 return checked;
}
int main(int argc,char** argv){
 const unsigned original=_mm_getcsr();
 struct Restore{unsigned original;~Restore(){_mm_setcsr(original);}}restore{original};
 try{
  if(!__builtin_cpu_supports("avx512f")){std::cerr<<"AVX512F unavailable; no qualification\n";return 77;}
  U64 count=1000000;Mutation mode=Mutation::none;
  for(int i=1;i<argc;++i){const std::string arg=argv[i];if(arg=="--cases"&&i+1<argc)count=std::stoull(argv[++i]);else if(arg=="--mutant-rn")mode=Mutation::rn;else if(arg=="--mutant-swapped")mode=Mutation::swapped;else if(arg=="--mutant-flush")mode=Mutation::flush;else if(arg=="--mutant-env-bypass")mode=Mutation::bypass;else return 64;}
  if(count<1||count>1000000)return 64;
  std::cout<<"ER_SCOPE test_only_finite_primitives_NOT_SOUNIO_INTERVAL_BACKEND MPFR="<<mpfr_get_version()<<" precision=IEEE24_or_53 subnormalize=explicit requested_cases_per_format_operation="<<count<<'\n';
  controls<float>(mode);controls<double>(mode);
  U64 total=0;
  for(int op=0;op<4;++op){total+=qualify<float>(count,op,mode);total+=qualify<double>(count,op,mode);}
  std::cout<<"ER_TOTAL indexed_cases="<<total<<" scalar_cases="<<total<<" zmm_lanes="<<total<<'\n';
  std::cout<<"EVEX_ER_MPFR_PASS formats=2 operations=4 variants=scalar,ZMM environment_controls=5_per_format\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 8;}
}
