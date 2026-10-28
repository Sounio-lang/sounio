// Three-dimensional oracle composes the binary C++ hypot with its Annex F
// special values. The three-argument overload has underspecified special-value
// behavior (LWG3172); the qualified host returns NaN for hypot(inf,2,3).
// Sounio explicitly chooses infinity dominance, otherwise NaN propagation.
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
int main(int argc,char**argv){if(argc<2||argc>3)return 64;std::ifstream in(argv[1]);if(!in)return 65;const double inf=std::numeric_limits<double>::infinity(),nan=std::bit_cast<double>(std::uint64_t{9221120237041090561ULL});const std::array<std::array<double,3>,15>cases={{{0,0,0},{3,4,0},{1,2,2},{-3,-4,-12},{1e308,1e308,0},{1e308,1e308,1e308},{1e-308,1e-308,1e-308},{1e-320,1e-320,0},{1.7976931348623157e308,1e292,0},{-0.0,0,-0.0},{1e-200,1e200,-1e-100},{inf,2,3},{nan,2,3},{nan,inf,3},{2,nan,inf}}};std::uint64_t worst=0;std::size_t count=0;for(const auto&v:cases)for(int dims=2;dims<=3;++dims){double expected=dims==2?std::hypot(v[0],v[1]):std::hypot(std::hypot(v[0],v[1]),v[2]);if(argc==3&&count==0)expected=1;std::int64_t raw=0;if(!(in>>raw))return 2;const double actual=std::bit_cast<double>(static_cast<std::uint64_t>(raw));bool ok=false;if(std::isnan(expected))ok=std::isnan(actual);else if(std::isinf(expected))ok=actual==expected;else if(std::isfinite(actual)&&actual>=0){const auto a=std::bit_cast<std::uint64_t>(actual),b=std::bit_cast<std::uint64_t>(expected),distance=a>b?a-b:b-a;worst=distance>worst?distance:worst;ok=distance<=4;}if(!ok){std::cerr<<"distance mismatch case="<<count/2<<" dimensions="<<dims<<" actual="<<actual<<" expected="<<expected<<'\n';return 1;}++count;}std::string extra;if(in>>extra)return 3;std::cout<<"GEO_CPP23_OK distances="<<count<<" max_ulp="<<worst<<'\n';}
