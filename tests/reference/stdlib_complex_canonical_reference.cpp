#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <map>
#include <numbers>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
using Z=std::complex<double>;
int main(int argc,char** argv){
 try{
  if(argc!=2)throw std::runtime_error("usage: reference native-stdout");
  const std::map<std::string,Z> expected{
   {"MUL",Z(1,2)*Z(3,4)},{"DIV",Z(1,2)/Z(3,4)},
   {"EULER",std::exp(Z(0,std::numbers::pi))+Z(1,0)},
   {"EXP10",std::exp(Z(10,.5))},{"LN_E",std::log(Z(std::numbers::e,0))},
   {"SQ_NEG",std::sqrt(Z(-1,0))},{"SQ_GENERAL",std::sqrt(Z(3,4))},
   {"SIN_I",std::sin(Z(0,1))},{"COS_GENERAL",std::cos(Z(.5,.75))},
   {"POW_REAL",std::pow(Z(1.5,-.25),2.5)},{"HYPER_SIN",std::sinh(Z(.25,.75))},
   {"QUAD1",Z(0,1)},{"QUAD2",Z(0,-1)}
  };
  std::ifstream input(argv[1]);if(!input)throw std::runtime_error("missing output");
  std::set<std::string> seen;std::string line;bool completed=false;
  while(std::getline(input,line)){
   if(completed)throw std::runtime_error("trailing output");
   if(line=="COMPLEX_CANONICAL_OK"){completed=true;continue;}
   std::istringstream row(line);std::string tag,extra;double re=0,im=0;
   if(!(row>>tag>>re>>im)||(row>>extra)||!expected.contains(tag)||!seen.insert(tag).second)throw std::runtime_error("malformed/duplicate row");
   const auto value=expected.at(tag);
   for(auto [actual,want]:{std::pair{re,value.real()},std::pair{im,value.imag()}}){
    if(!std::isfinite(actual)||std::abs(actual-want)>1e-10*std::max(1.0,std::abs(want)))throw std::runtime_error("complex mismatch "+tag);
   }
  }
  if(!completed||seen.size()!=expected.size())throw std::runtime_error("incomplete rows");
  std::cout<<"COMPLEX_CPP23_CONCORDANCE_OK rows="<<seen.size()<<"\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

