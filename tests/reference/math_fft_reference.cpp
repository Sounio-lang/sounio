// ADR009 C++23: independent direct DFT (small), periodic16+impulse analytic DFT
// (4096), Parseval, circular convolution, real DC/Nyquist and output coverage.
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <map>
#include <numbers>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
using C=std::complex<long double>;
C signal(int i,bool real=false){long double r=static_cast<long double>(i%16-8)/8;if(i==37)r+=.5L;return {r,real?0:static_cast<long double>(i%8-4)/16};}
C phase(long double x){return {std::cos(x),std::sin(x)};}
C direct(int n,int k,bool real){
 C answer{};
 if(n<4096){for(int j=0;j<n;++j)answer+=signal(j,real)*phase(-2*std::numbers::pi_v<long double>*k*j/n);return answer;}
 if(k%256==0){for(int j=0;j<16;++j)answer+=256.L*signal(j,real)*phase(-2*std::numbers::pi_v<long double>*k*j/n);}
 return answer+.5L*phase(-2*std::numbers::pi_v<long double>*k*37/n);
}
bool close(long double a,long double b){return std::isfinite(a)&&std::abs(a-b)<=2e-10L*(1+std::abs(b));}
int main(int argc,char**argv){
 if(argc==3 && std::string(argv[1])=="--bench") {
  std::ifstream b(argv[2]); long double actual; std::string marker;
  if(!(b>>actual))return 20;
  std::getline(b,marker);std::getline(b,marker);
  long double expected=0;for(int i=0;i<4096;++i)expected+=std::norm(signal(i));
  if(!close(actual,expected)||marker!="FFT_BENCH_PASS n=4096 roundtrip_pairs=32")return 21;
  std::cout<<"FFT_BENCH_NUMERIC_PASS\n";return 0;
 }
 if(argc!=2)return 64;
 std::ifstream f(argv[1]);
 if(!f)return 66;
 std::set<std::tuple<int,int,int>>seen;std::map<std::pair<int,int>,int>counts;
 std::string line;bool complete=false;int rows=0;
 while(std::getline(f,line)){
  if(line=="FFT_CHECKED_PASS"){complete=true;continue;}
  if(!line.starts_with("FFT ")){if(!line.empty()){std::cerr<<"unexpected output: "<<line<<'\n';return 1;}continue;}
  std::istringstream head(line);std::string tag;int n,kind,k;if(!(head>>tag>>n>>kind>>k))return 2;
  if(n!=1&&n!=8&&n!=4096)return 3;
  if(!seen.emplace(n,kind,k).second)return 4;
  long double re,im;try{if(!std::getline(f,line))return 5;re=std::stold(line);if(!std::getline(f,line))return 5;im=std::stold(line);}catch(...){return 6;}
  C expected{};
  int bound=n;
  if(kind==0)expected=direct(n,k,false);
  else if(kind==1){bound=n/2+1;expected=direct(n,k,true);}
  else if(kind==2){if(n>8)return 7;expected=direct(n,k,false)/std::sqrt(static_cast<long double>(n));}
  else if(kind==3){if(n>8)return 7;for(int j=0;j<n;++j)expected+=signal(j)*signal((k-j+n)%n);}
  else if(kind==4){bound=1;long double e=0;for(int j=0;j<n;++j)e+=std::norm(signal(j));expected={n*e,0};}
  else if(kind==5){bound=1;if(!std::isfinite(re)||re<0||re>1e-10L||im!=0)return 8;expected={re,0};}
  else return 9;
  if(k<0||k>=bound)return 10;
  if(!close(re,expected.real())||!close(im,expected.imag())){std::cerr<<"mismatch n="<<n<<" kind="<<kind<<" k="<<k<<" actual="<<re<<','<<im<<" expected="<<expected<<'\n';return 11;}
  ++counts[{n,kind}];++rows;
 }
 for(int n:{1,8,4096}){
  for(int kind=0;kind<=5;++kind){if(n>8&&(kind==2||kind==3))continue;int required=kind==1?n/2+1:(kind>=4?1:n);if(counts[{n,kind}]!=required)return 12;}
 }
 if(!complete)return 13;
 std::cout<<"FFT_CPP_CONCORDANCE_PASS rows="<<rows<<" real4096_bins=2049 direct_small=1,8 parseval=1,8,4096 roundtrip=1,8,4096 circular_convolution=1,8 tolerance=2e-10*(1+abs(expected))\n";
}
