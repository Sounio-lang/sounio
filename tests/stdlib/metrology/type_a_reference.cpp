// ADR009 C++23 independent Type-A reference and actual Sounio-output comparison.
// c++ -std=c++23 -O2 type_a_reference.cpp -o oracle; oracle native.stdout
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <array>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <stdexcept>
using Q=boost::multiprecision::cpp_dec_float_100;
struct Result { Q mean,sd,u; };
Result eval(std::span<const Q> x) {
 if(x.size()<2||x.size()>16) throw std::invalid_argument("count");
 Q total=0; for(const auto& v:x) total+=v;
 Q mean=total/Q(x.size()), ss=0;
 for(const auto& v:x) {Q d=v-mean;ss+=d*d;}
 Q sd=sqrt(ss/Q(x.size()-1)); return {mean,sd,sd/sqrt(Q(x.size()))};
}
bool close(const Q&a,const Q&b){return b==0?a==0:abs(a-b)<=abs(b)*Q("1e-12");}
int main(int argc,char**argv) {
 if(argc!=2)return 64;
 std::array<Q,5>a={-2,-1,0,1,2};
 std::array<Q,2>b={Q(1e-12),Q(2e-12)},c={Q(-1e150),Q(1e150)},e={1,3};
 std::array<Q,16>d;d.fill(4);
 std::array<Result,5> expected={eval(a),eval(b),eval(c),eval(d),eval(e)};
 if(!close(expected[0].sd*expected[0].sd,Q("2.5"))||!close(expected[1].u,Q("5e-13"))||!close(expected[2].u,Q("1e150"))||expected[3].u!=0)return 2;
 for(int n:{0,1,17}) {std::array<Q,17>v{};try{(void)eval(std::span<const Q>(v.data(),n));return 3;}catch(const std::invalid_argument&) {}}
 std::ifstream f(argv[1]); if(!f)return 66;
 std::string line;int seen=0;bool done=false;
 while(std::getline(f,line)) {
  if(line=="TYPE_A_BOUNDARY_OK"){done=true;continue;}
  if(!line.starts_with("TYPE_A "))continue;
  int id=std::stoi(line.substr(7));if(id!=seen||id>=5)return 4;
  Q actual[3];for(auto &v:actual){if(!std::getline(f,line))return 5;try{v=Q(line);}catch(...){return 6;}}
  const auto&r=expected[id];if(!close(actual[0],r.mean)||!close(actual[1],r.sd)||!close(actual[2],r.u)){std::cerr<<"mismatch case="<<id<<'\n';return 7;}
  ++seen;
 }
 if(seen!=5||!done)return 8;
 std::cout<<"TYPE_A_CPP_CONCORDANCE_PASS cases=5 values=15 relative_tolerance=1e-12 controls=3_invalid_counts\n";
}
