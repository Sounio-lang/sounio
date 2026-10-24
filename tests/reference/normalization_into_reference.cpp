// C++23 independent normalization oracle: direct high-precision log counts,
// sorted sample ratios, represented log-median, then geometric centering.
// No production anchors, selection algorithm, logarithm approximation, or fit reuse.
// Slurm: g++ -std=c++23 -O2 normalization_into_reference.cpp -o norm_reference
// norm_reference [actual-native-stdout]; requires final marker and all five cases.
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using B=boost::multiprecision::number<boost::multiprecision::cpp_dec_float<100>>;
static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
struct Reference{int samples,used,excluded;std::vector<B> logs,factors;};
static long long count(int kind,int g,int s){static const std::array<long long,15> small{1,4,9,2,8,0,0,3,12,6,12,6,0,0,0};if(kind<2)return small[g*3+s];if(kind==2)return (g+1)*(s%7+1);if(kind==3){constexpr long long q=9007199254740992LL;return g==0?(s==0?q:q-1):(s==0?q-2:q);}return 8;}
static Reference reference(int kind){int genes=kind==2?513:kind==3?2:kind==4?3:5;int samples=kind==2?17:kind==3?2:kind==4?4:3;Reference r{samples,0,0,{},{}};std::vector<std::vector<B>> ratios(samples);
 for(int g=0;g<genes;g++){bool valid=true;int positive=0;B logs=0;for(int s=0;s<samples;s++){long long c=count(kind,g,s);if(c>0){positive++;logs+=log(B(c));}else if(kind!=1)valid=false;}if(positive==0)valid=false;if(!valid){r.excluded++;continue;}r.used++;B center=logs/positive;for(int s=0;s<samples;s++){auto c=count(kind,g,s);if(c>0)ratios[s].push_back(log(B(c))-center);}}
 B center=0;for(auto& values:ratios){need(!values.empty(),"eligible sample");std::sort(values.begin(),values.end());size_t n=values.size();B median=values[n/2];if(n%2==0)median=(values[n/2-1]+values[n/2])/2;r.logs.push_back(median);center+=median;}center/=samples;for(B& l:r.logs){l-=center;r.factors.push_back(exp(l));}
 // Repeated-view case has exact analytic log ratio0 and factors1. Record
 // that identity separately rather than treating decimal log roundoff as truth.
 if(kind==4){for(int s=0;s<samples;s++){need(abs(r.logs[s])<B("1e-95"),"direct constant log roundoff");r.logs[s]=0;r.factors[s]=1;}}
 return r;
}
static std::string field(std::ifstream& in,const std::string& prefix){std::string line;need(bool(std::getline(in,line)),"complete native stream");need(line.starts_with(prefix),"ordered native field");return line.substr(prefix.size());}
int main(int argc,char** argv){try{need(argc<=2,"usage");std::ifstream in;if(argc==2){in.open(argv[1]);need(bool(in),"native open");}
 for(int k=0;k<5;k++){Reference r=reference(k);if(in.is_open()){need(std::stoi(field(in,"normalization into case: "))==k,"case order");need(std::stoi(field(in,"normalization into used: "))==r.used,"genes used");need(std::stoi(field(in,"normalization into excluded: "))==r.excluded,"genes excluded");}
 for(int s=0;s<r.samples;s++){std::cout<<k<<'\t'<<s<<'\t'<<std::setprecision(45)<<r.factors[s]<<'\t'<<r.logs[s]<<'\n';if(in.is_open()){B f(field(in,"normalization into factor: ")),l(field(in,"normalization into log: "));need(abs(f-r.factors[s])<=abs(r.factors[s])*B("3e-12"),"native factor");if(k==3)need(abs(l-r.logs[s])<=abs(r.logs[s])*B("2e-11"),"tiny near-2^53 native log factor");else if(k==4)need(l==0&&f==1,"constant analytic identity");else need(abs(l-r.logs[s])<=B("3e-12"),"native centered log");}}
 }
 if(in.is_open()){std::string line;need(bool(std::getline(in,line))&&line=="SCALE NORMALIZATION INTO PASS","complete native marker");while(std::getline(in,line))need(line.empty(),"no extra native output");}
 std::cout<<"NORMALIZATION_INTO_REFERENCE_COMPLETE\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
