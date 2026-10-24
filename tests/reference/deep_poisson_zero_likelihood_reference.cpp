// C++23 exact IEEE-754 concordance for the analytic identity log Poisson(0;mu)=-mu.
// Build/run only centrally: c++ -std=c++23 -O2 -Wall -Wextra -pedantic FILE -o EXE
// EXE NATIVE_STDOUT verifies raw integer bit reports; EXE without args emits reference.
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
int main(int argc,char** argv) try {
 static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559);
 const std::array<double,10> means={std::numeric_limits<double>::denorm_min(),
  2*std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::min(),
  .125,.5,1,2,10,std::ldexp(1.,512),std::numeric_limits<double>::max()};
 std::array<bool,10> seen{};
 if(argc==1){
  for(std::size_t i=0;i<means.size();++i){
   const auto input=std::bit_cast<std::uint64_t>(means[i]);
   const auto expected=input ^ (std::uint64_t(1)<<63);
   if(std::bit_cast<std::uint64_t>(-means[i])!=expected)throw std::runtime_error("host negation");
   std::cout<<"P0 "<<i<<" "<<std::bit_cast<std::int64_t>(means[i])<<" "
    <<std::bit_cast<std::int64_t>(expected)<<" 0 0 0 "<<(i==9?1:0)<<" true\n";
  }
  return 0;
 }
 if(argc!=2)throw std::runtime_error("usage: oracle [NATIVE_STDOUT]");
 std::ifstream in(argv[1]);if(!in)throw std::runtime_error("input open");
 bool complete=false;std::string line;
 while(std::getline(in,line)){
  if(line=="STDLIB_DEEP_POISSON_ZERO_LIKELIHOOD_PASS"){complete=true;continue;}
  if(!line.starts_with("P0 ")){if(!line.empty())throw std::runtime_error("unexpected row");continue;}
  std::istringstream row(line);std::string tag,flag,extra;std::size_t i;
  std::int64_t input,actual,status,normal,residual,deviance;
  if(!(row>>tag>>i>>input>>actual>>status>>normal>>residual>>deviance>>flag))throw std::runtime_error("row shape");
  if(row>>extra)throw std::runtime_error("extra field");
  if(i>=means.size())throw std::runtime_error("index");
  if(seen[i])throw std::runtime_error("duplicate");
  seen[i]=true;
  const auto bits=std::bit_cast<std::uint64_t>(means[i]);
  if(input!=std::bit_cast<std::int64_t>(means[i]) || std::bit_cast<std::uint64_t>(actual)!=(bits^(std::uint64_t(1)<<63)))throw std::runtime_error("bit mismatch case "+std::to_string(i));
  if(status!=0||normal!=0||residual!=0||deviance!=(i==9?1:0)||flag!="true")throw std::runtime_error("metadata case "+std::to_string(i));
 }
 if(!complete)throw std::runtime_error("missing completion");
 for(bool value:seen)if(!value)throw std::runtime_error("missing case");
 std::cout<<"POISSON_ZERO_IEEE_CONCORDANCE_PASS cases=10\n";
 return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
