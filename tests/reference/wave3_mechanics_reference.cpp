#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
struct Result{int status;long double value,unc;};
static long double norm(long double a,long double b,long double c=0,long double d=0){return std::hypot(std::hypot(a,b),std::hypot(c,d));}
static Result gravity(long double g,long double ug,long double m1,long double u1,long double m2,long double u2,long double r,long double ur){
 const auto value=g*m1*m2/(r*r);
 return {0,value,norm(ug*m1*m2/(r*r),g*m2*u1/(r*r),g*m1*u2/(r*r),2*g*m1*m2*ur/(r*r*r))};
}
int main(int argc,char**argv){
 try{
  if(argc==4&&std::string(argv[1])=="--wrong-output"){
   std::ifstream input(argv[2]);std::ofstream output(argv[3]);if(!input||!output)throw std::runtime_error("mutation open");
   std::string line;bool changed=false;
   while(std::getline(input,line)){
    if(line.rfind("MECH_ROW ke ",0)==0){
     std::istringstream row(line);std::string tag,name,extra;int status=0,model=0,first=0;std::int64_t value=0,uncertainty=0;
     if(!(row>>tag>>name>>status>>value>>uncertainty>>model>>first)||(row>>extra))throw std::runtime_error("mutation row");
     line=tag+" "+name+" "+std::to_string(status)+" "+std::to_string(value)+" 0 "+std::to_string(model)+" "+std::to_string(first);changed=true;
    }
    output<<line<<'\n';
   }
   if(!changed||!output)throw std::runtime_error("mutation missing");
   std::cout<<"MECHANICS_WRONG_UNCERTAINTY_WRITTEN\n";return 0;
  }
  if(argc!=2)throw std::runtime_error("native output required");
  std::map<std::string,Result> expected{
   {"ke",{0,0.5L*2*3*3,norm(0.5L*3*3*0.1L,2*3*0.1L)}},
   {"spring",{0,100*0.5L,norm(0.5L*2,100*0.01L)}},
   {"work",{0,10*5,norm(5*0.1L,10*0.2L)}},
   {"power",{0,100.0L/2,norm(3.0L/2,100*0.1L/(2*2))}},
   {"gravity",gravity(6.67430e-11L,1.5e-15L,1,0.01L,5.972e24L,1e21L,6.371e6L,100)},
   {"gravity-explicit",gravity(2,0.01L,3,0.1L,5,0.2L,4,0.01L)},
   {"ke-zero",{0,0,0}},{"work-zero",{0,0,5}},{"power-zero",{0,0,1.5L}},
   {"gravity-zero",gravity(2,0.01L,0,0.2L,5,0.2L,4,0.01L)},
   {"bad-dim",{5,0,0}},{"bad-unc",{3,0,0}},{"bad-nan",{1,0,0}},{"bad-mass",{2,0,0}},
   {"bad-time",{2,0,0}},{"bad-radius",{2,0,0}},{"bad-spring",{2,0,0}},{"overflow",{4,0,0}},{"underflow",{4,0,0}}
  };
  std::ifstream in(argv[1]);if(!in)throw std::runtime_error("output open");
  std::string line;std::map<std::string,bool> seen;bool marker=false;
  while(std::getline(in,line)){
   if(line=="MECHANICS_CHECKED_ROWS_OK"){if(marker)throw std::runtime_error("duplicate marker");marker=true;continue;}
   std::istringstream row(line);std::string tag,name,extra;int status=0,model=0,first=0;std::int64_t v=0,u=0;
   if(!(row>>tag>>name>>status>>v>>u>>model>>first)||(row>>extra)||tag!="MECH_ROW"||!expected.contains(name)||seen.contains(name))throw std::runtime_error("row structure");
   seen[name]=true;const auto want=expected.at(name);const double value=std::bit_cast<double>(v),unc=std::bit_cast<double>(u);
   if(status!=want.status||model!=1||first!=1||!std::isfinite(value)||!std::isfinite(unc))throw std::runtime_error("status or metadata "+name);
   for(const auto& pair:std::array<std::pair<long double,long double>,2>{{{value,want.value},{unc,want.unc}}})
    if(std::fabs(pair.first-pair.second)>1e-12L*std::max(1.0L,std::fabs(pair.second))){std::cerr<<std::setprecision(24)<<name<<" got="<<pair.first<<" expected="<<pair.second<<"\n";throw std::runtime_error("mechanics numerical mismatch");}
  }
  if(!marker||seen.size()!=expected.size())throw std::runtime_error("missing rows");
  std::cout<<"MECHANICS_CPP_CONCORDANCE_OK rows="<<seen.size()<<" tolerance=1e-12\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
