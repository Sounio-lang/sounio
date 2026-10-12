#include <boost/multiprecision/cpp_int.hpp>
#include <cstdint>
#include <iostream>
#include <string>
using boost::multiprecision::cpp_int;
struct Decimal { cpp_int significand; int exp10; };
Decimal parse(std::string s) {
  bool neg=false; if(s[0]=='-'){neg=true;s.erase(0,1);}
  auto ep=s.find_first_of("eE"); int exp=0;
  if(ep!=std::string::npos){exp=std::stoi(s.substr(ep+1));s.resize(ep);}
  auto dot=s.find('.'); int frac=0;
  if(dot!=std::string::npos){frac=(int)s.size()-(int)dot-1;s.erase(dot,1);}
  cpp_int n=0; for(char c:s) if(c>='0'&&c<='9') n=n*10+(c-'0');
  if(neg)n=-n; return {n,exp-frac};
}
cpp_int powi(cpp_int b,int e){cpp_int r=1;for(int i=0;i<e;i++)r*=b;return r;}
uint64_t round_subnormal(const Decimal& d) {
  cpp_int num=d.significand, den=1;
  if(d.exp10>=0) num*=powi(10,d.exp10); else den=powi(10,-d.exp10);
  num <<= 1074;
  cpp_int q=num/den, rem=num%den;
  if(2*rem>den || (2*rem==den && (q&1)!=0)) ++q;
  return q.convert_to<uint64_t>();
}
int main(){
 const char* xs[]={"4.9406564584124654e-324","5e-324","3e-324","2.4703282292062328e-324","2.470328229206232720882538e-324","9.8813129168249309e-324","7.4109846876186982e-324","2.2250738585072014e-308","1e-400"};
 for(auto s:xs) std::cout<<s<<" bits="<<round_subnormal(parse(s))<<"\n";
}
