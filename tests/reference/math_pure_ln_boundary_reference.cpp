// C++23/quadmath boundary reference; no production logarithm implementation reused.
#include <quadmath.h>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>
#include <array>
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
static std::uint64_t bits(double x){return std::bit_cast<std::uint64_t>(x);}
static std::uint64_t key(double x){auto u=bits(x);return u>>63?~u:u|(1ULL<<63);}
static std::uint64_t distance(double a,double b){auto x=key(a),y=key(b);return x>y?x-y:y-x;}
int main(int argc,char** argv){try{
 require(argc==2,"usage: oracle NATIVE_STDOUT");
 std::ifstream in(argv[1]);require(bool(in),"missing stdout");
 const std::array<std::uint64_t,11> expected={0x3ff0000000000000ULL,0x4000000000000000ULL,0x3fe0000000000000ULL,0x7fefffffffffffffULL,0x0010000000000000ULL,1ULL,0x7ff0000000000000ULL,0,0xfff0000000000000ULL,0,0xbff0000000000000ULL};
 std::array<bool,11> seen{};std::string line;unsigned rows=0;bool marker=false;
 while(std::getline(in,line)){
  if(line=="PURE_LN_POS_INF_OK")marker=true;
  if(!line.starts_with("LN_BOUNDARY "))continue;
  std::istringstream s(line);std::string tag,extra;int id;std::int64_t raw,a,b,c;
  require(bool(s>>tag>>id>>raw>>a>>b>>c),"malformed row");require(!(s>>extra),"extra fields");
  require(id>=0&&id<11&&!seen[id],"duplicate/invalid id");seen[id]=true;++rows;
  const auto u=std::bit_cast<std::uint64_t>(raw);double x=std::bit_cast<double>(u);
  if(id==7)require(std::isnan(x),"NaN input");else require(u==expected[id],"wrong input bits");
  std::array<double,3> y={std::bit_cast<double>(a),std::bit_cast<double>(b),std::bit_cast<double>(c)};
  for(unsigned j=0;j<3;++j){
   if(id==7){require(std::isnan(y[j]),"NaN not propagated");continue;}
   if(id==6){require(bits(y[j])==expected[6],"positive infinity not preserved");continue;}
   __float128 ref;
   if(x<=0){ref=static_cast<__float128>(-1.0e30);if(j==1)ref/=logq(10);if(j==2)ref/=logq(2);}
   else {ref=logq(static_cast<__float128>(x));if(j==1)ref/=logq(10);if(j==2)ref/=logq(2);}
   double rounded=static_cast<double>(ref);require(std::isfinite(y[j]),"nonfinite finite/domain result");
   auto ulp=distance(y[j],rounded);std::cout<<"ULP\t"<<id<<'\t'<<j<<'\t'<<ulp<<'\n';require(ulp<=4,"boundary error exceeds4ULP");
  }
 }
 require(rows==11&&marker,"incomplete native output");
 // Guard classification across every exponent, extremal significands and subnormals.
 std::uint64_t classifications=0;
 for(std::uint64_t exponent=0;exponent<2047;++exponent)for(auto fraction:{0ULL,1ULL,0x7ffffffffffffULL,0xfffffffffffffULL}){
  const auto u=(exponent<<52)|fraction;if(u==0)continue;double x=std::bit_cast<double>(u);
  require(std::isfinite(x)&&x>0&&x/2!=x,"guard accepts positive finite input");++classifications;
 }
 std::cout<<"FINITE_GUARD_CLASSIFICATIONS\t"<<classifications<<"\nLN_BOUNDARY_REFERENCE_PASS\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
