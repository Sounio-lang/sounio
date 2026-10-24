#include <bit>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <boost/multiprecision/cpp_int.hpp>
using I=long long;using boost::multiprecision::cpp_int;
double from_parts(I sig,I exponent){double v=static_cast<double>(sig);while(exponent>0){v*=2;--exponent;}while(exponent<0){v/=2;++exponent;}return v;}
double value(I i,I j){if(i==0)return 0;if(i==1){if(j==4)return 0;return j%2==0?1:-1;}if(i>=2&&i<=7)return from_parts(4503599627370496LL+(j==4?0:j*2-3+(j==3?i-4:0)),i%2==0?971:-1074);if(i==8)return from_parts(j%2==0?4503599627370496LL:-4503599627370496LL,-1126);if(i==9)return j==4?-0.0:from_parts(4503599627370496LL+j,971);I sig=4503599627370496LL+(i*1000000007LL+j*100000000003LL)%4503599627370496LL;return from_parts((i+j)%3==0?-sig:sig,(i*37+j*251)%2046-1074);}
struct D{cpp_int m;I e;};
D decode(double x){if(!std::isfinite(x))throw std::runtime_error("nonfinite fixture");const auto bits=std::bit_cast<std::uint64_t>(x);const auto exponent=(bits>>52)&2047;std::uint64_t m=bits&((std::uint64_t{1}<<52)-1);I e=-1074;if(exponent){m|=std::uint64_t{1}<<52;e=static_cast<I>(exponent)-1023-52;}cpp_int signed_m=m;if(bits>>63)signed_m=-signed_m;return {signed_m,e};}
I expected(I i){const D a=decode(value(i,0)),b=decode(value(i,1)),c=decode(value(i,2)),d=decode(value(i,3)),l=decode(value(i,4));
 auto product=[](const D&x,const D&y)->cpp_int{return (x.m*y.m)<<(x.e+y.e+2252);};
 const cpp_int q=product(a,c)-product(b,d)-product(l,a)-product(l,c)+product(l,b)+product(l,d);return q>0?1:q<0?-1:0;
}
int main(int argc,char**argv){try{if(argc!=2)throw std::runtime_error("usage: checker native-stdout");std::ifstream f(argv[1]);if(!f)throw std::runtime_error("input");std::string row;I count=0;bool complete=false;while(std::getline(f,row)){std::istringstream in(row);std::string tag;in>>tag;if(tag=="SADDLE_CASE"){I index,code,sign;in>>index>>code>>sign;if(!in||index!=count||code!=0||sign!=expected(index))throw std::runtime_error("exact predicate mismatch case "+std::to_string(count));++count;}if(tag=="STDLIB_DEEP_VIZ_CONTOUR_PREDICATE_PASS")complete=true;}if(count!=96||!complete)throw std::runtime_error("count/completion");std::cout<<"DEEP_VIZ_CONTOUR_PREDICATE_REFERENCE_PASS cases=96 exact_cpp_int\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
