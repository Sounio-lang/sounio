#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
static void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
static double from(std::int64_t bits){return std::bit_cast<double>(static_cast<std::uint64_t>(bits));}
static void concordance(double got,double expected,const char* metric,double input){
 if(std::isinf(expected)){require(got==expected,"infinite result disagreement");return;}
 if(!std::isfinite(got)||std::abs(got-expected)>1e-11){
  std::cerr<<std::setprecision(17)<<metric<<" input="<<input<<" got="<<got<<" expected="<<expected<<"\n";
  throw std::runtime_error("C++23/Sounio logarithm disagreement (absolute tolerance 1e-11)");
 }
}
int main(int argc,char** argv){
 try{
  require(argc==2,"usage: active-log-reference stdout");
  const auto bits=[](std::uint64_t x){return std::bit_cast<double>(x);};
  const std::vector<double> inputs{bits(1),bits(2),bits(4503599627370495),bits(4503599627370496),
   1e-300,1e-200,1e-100,1e-20,1e-7,.01,.1,.25,.49999999999999994,.5,.5000000000000001,
   .9999999999999999,1.,1.0000000000000002,1.5,1.9999999999999998,2.,2.0000000000000004,
   4.,16.,1e6,1e20,1e100,1e200,1e300,bits(9218868437227405311)};
  const std::array<double,4> domain{0.,-1.,bits(9218868437227405312),bits(9221120237041090560)};
  std::ifstream in(argv[1]);require(bool(in),"open Sounio stdout");std::string line;std::size_t row=0,edge=0;int pass=0;
  while(std::getline(in,line)){
   if(line=="ACTIVE_LOG_CONCORDANCE_OK"){++pass;continue;}
   if(line.starts_with("ACTIVE_LOG ")){
    std::istringstream words(line);std::string tag,extra;std::int64_t x=0,l=0,e=0;
    require(bool(words>>tag>>x>>l>>e)&&!(words>>extra),"malformed logarithm row");
    require(row<inputs.size(),"extra logarithm row");
    const double value=from(x);require(std::bit_cast<std::uint64_t>(value)==std::bit_cast<std::uint64_t>(inputs[row]),"input bits/order changed");
    const double expected=std::log(inputs[row]);concordance(from(l),expected,"canonical log",value);
    concordance(from(e),std::max(0.,.5*expected),"active EFE",value);++row;continue;
   }
   if(line.starts_with("ACTIVE_DOMAIN ")){
    std::istringstream words(line);std::string tag,extra;std::int64_t x=0,e=0;
    require(bool(words>>tag>>x>>e)&&!(words>>extra),"malformed domain row");require(edge<domain.size(),"extra domain row");
    require(static_cast<std::uint64_t>(x)==std::bit_cast<std::uint64_t>(domain[edge]),"domain input bits/order changed");
    const double expected=edge==2?std::numeric_limits<double>::infinity():0.;
    concordance(from(e),expected,"active domain",from(x));++edge;continue;
   }
   require(line.empty(),"unexpected output row");
  }
  require(row==inputs.size()&&edge==domain.size()&&pass==1,"missing required numerical rows/marker");
  std::cout<<"ACTIVE_LOG_CPP_CONCORDANCE_OK finite_rows="<<row<<" domain_rows="<<edge<<" absolute_tolerance=1e-11\n";
 }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
