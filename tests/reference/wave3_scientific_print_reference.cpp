// ADR009 C++23 draft. No source/test/formatter changes.
// Actual native rc0 + original assertions + original PASS are mandatory inputs.
// Small fixed6 zeros qualify formatting only, not underlying numerical accuracy.
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace fs=std::filesystem;
constexpr double fixed6_allowance=1.000001e-6;
void require(bool yes,const std::string& why){if(!yes)throw std::runtime_error(why);}
struct Dual {
 double value{};
 std::array<double,7> derivative{};
};
Dual operator+(const Dual& a,const Dual& b){
 Dual out;out.value=a.value+b.value;
 for(std::size_t i=0;i<7;++i)out.derivative[i]=a.derivative[i]+b.derivative[i];
 return out;
}
Dual operator-(const Dual& a,const Dual& b){
 Dual out;out.value=a.value-b.value;
 for(std::size_t i=0;i<7;++i)out.derivative[i]=a.derivative[i]-b.derivative[i];
 return out;
}
Dual operator-(const Dual& a){return Dual{}-a;}
Dual operator*(const Dual& a,const Dual& b){
 Dual out;out.value=a.value*b.value;
 for(std::size_t i=0;i<7;++i)out.derivative[i]=a.derivative[i]*b.value+a.value*b.derivative[i];
 return out;
}
Dual operator*(const Dual& a,double b){
 Dual out=a;out.value*=b;
 for(auto& d:out.derivative)d*=b;
 return out;
}
Dual operator/(const Dual& a,double b){
 Dual out=a;out.value/=b;
 for(auto& d:out.derivative)d/=b;
 return out;
}
using State=std::array<Dual,4>;
using Rates=std::array<Dual,3>;
State rhs(const State& y,const Rates& k){
 const auto first=k[0]*y[0],second=k[1]*y[1],third=k[2]*y[0]*y[1];
 return {-first-third,first-second-third,second,third};
}
State shifted(const State& y,const State& derivative,double step){
 State out;for(std::size_t i=0;i<4;++i)out[i]=y[i]+derivative[i]*step;return out;
}
struct Solution {std::array<double,4> values{},sigmas{};};
Solution integrate(const std::array<double,4>& initial,const std::array<double,4>& initial_variance,
                   const std::array<double,3>& means,const std::array<double,3>& standard_uncertainty,
                   double dt,int steps){
 State y;Rates rates;
 for(std::size_t i=0;i<4;++i){y[i].value=initial[i];y[i].derivative[i]=1.0;}
 for(std::size_t i=0;i<3;++i){rates[i].value=means[i];rates[i].derivative[4+i]=1.0;}
 for(int step=0;step<steps;++step){
  const auto k1=rhs(y,rates);
  const auto k2=rhs(shifted(y,k1,.5*dt),rates);
  const auto k3=rhs(shifted(y,k2,.5*dt),rates);
  const auto k4=rhs(shifted(y,k3,dt),rates);
  for(std::size_t i=0;i<4;++i)y[i]=y[i]+((k1[i]+k2[i]*2.0+k3[i]*2.0+k4[i])/6.0)*dt;
 }
 Solution result;
 for(std::size_t i=0;i<4;++i){
  result.values[i]=y[i].value;double variance=0.0;
  for(std::size_t j=0;j<4;++j)variance+=y[i].derivative[j]*y[i].derivative[j]*initial_variance[j];
  for(std::size_t j=0;j<3;++j)variance+=y[i].derivative[4+j]*y[i].derivative[4+j]*standard_uncertainty[j]*standard_uncertainty[j];
  require(std::isfinite(variance)&&variance>=0.0,"C++ reference covariance invalid");
  result.sigmas[i]=std::sqrt(variance);
 }
 const double mass=result.values[0]+result.values[1]+result.values[2]+2.0*result.values[3];
 require(std::fabs(mass-1.0)<=1e-12,"C++ RK4 mass control failed");
 return result;
}
std::array<double,12> chemistry_reference(){
 // Exact original inputs: initial uncertainty is VARIANCE; rate uncertainty is STD.
 const auto first=integrate({1,0,0,0},{.01,0,0,0},{.1,.05,.2},{.001,.0005,.002},.1,20);
 const auto second=integrate({1,0,0,0},{.05,0,0,0},{.05,.02,.01},{.004,.002,.0015},.33,72);
 std::array<double,12> out{};
 for(std::size_t i=0;i<4;++i){out[i]=first.values[i];out[4+i]=first.sigmas[i];out[8+i]=second.values[i];}
 return out;
}
std::vector<std::string> lines(const fs::path& path){
 require(fs::file_size(path)<=65536,"stdout exceeds bounded 64KiB format contract");
 std::ifstream input(path);require(bool(input),"cannot open stdout");
 std::vector<std::string> result;std::string line;
 while(std::getline(input,line)){
  if(!line.empty()&&line.back()=='\r')line.pop_back();
  if(line.empty())continue;
  require(result.size()<512,"stdout exceeds bounded line count");
  require(line.find("FAIL")==std::string::npos&&line.find("panic")==std::string::npos,"native failure text");
  result.push_back(line);
 }
 return result;
}
void require_native_exit_zero(const fs::path& path){
 std::ifstream input(path);int value=-1;std::string extra;
 require(bool(input>>value)&&value==0&&!(input>>extra),"native rc0 is mandatory; original assertions cannot be bypassed");
}
double fixed6(const std::string& token){
 static const std::regex format(R"(^[-+]?[0-9]+\.[0-9]{6}$)");
 require(std::regex_match(token,format),"expected complete fixed6 token: "+token);
 double value{};const auto result=std::from_chars(token.data(),token.data()+token.size(),value);
 require(result.ec==std::errc{}&&result.ptr==token.data()+token.size()&&std::isfinite(value),"invalid finite fixed6 token");
 return value;
}
void require_once(const std::vector<std::string>& output,std::string_view marker){
 require(std::count(output.begin(),output.end(),marker)==1,"exactly one original PASS marker required");
}
void check_printed(double actual,double expected,const std::string& field){
 require(std::isfinite(actual)&&std::isfinite(expected),"nonfinite "+field);
 const double allowance=fixed6_allowance+8.0*std::numeric_limits<double>::epsilon()*std::fabs(expected);
 require(std::fabs(actual-expected)<=allowance,"fixed6/reference disagreement "+field);
}
void check_chemistry(const fs::path& stdout_path,const fs::path& exit_path){
 require_native_exit_zero(exit_path);const auto output=lines(stdout_path);
 require(output.size()>=13,"missing original chemistry outputs");
 require_once(output,"CHEM ODE SYS5 PASS");
 const auto expected=chemistry_reference();
 for(std::size_t i=0;i<12;++i)check_printed(fixed6(output[i]),expected[i],"chemistry row "+std::to_string(i));
 std::cout<<"CHEM_CPP_DUAL_RK4_FIXED6_OK first12=checked native_assertions=mandatory\n";
}
void check_sigma(const fs::path& stdout_path,const fs::path& exit_path){
 require_native_exit_zero(exit_path);const auto output=lines(stdout_path);
 require(output.size()==21,"expected12 variance rows, worst, original regulatory audit, infinity and original PASS");
 require_once(output,"KIN SIGMA SMALL VARIANCE PASS");
 const std::array<double,12> variance{1.0,4.0,16.0,3.999,4.001,1e-20,1e-40,1e-100,1e-200,1e-300,1e6,1e-6};
 const std::regex row(R"(^var=([-+]?[0-9]+\.[0-9]{6}) sigma=([-+]?[0-9]+\.[0-9]{6}) rel_err=([-+]?[0-9]+\.[0-9]{6})$)");
 double maximum_printed_relative=0.0;std::size_t unobservable_rows=0;
 for(std::size_t i=0;i<12;++i){
  std::smatch match;require(std::regex_match(output[i],match,row),"malformed sigma row "+std::to_string(i));
  const double printed_variance=fixed6(match[1]),printed_sigma=fixed6(match[2]),relative=fixed6(match[3]);
  check_printed(printed_variance,variance[i],"variance "+std::to_string(i));
  check_printed(printed_sigma,std::sqrt(variance[i]),"sigma "+std::to_string(i));
  require(relative>=0.0&&relative<=1e-6,"printed relative field outside original bound");
  maximum_printed_relative=std::max(maximum_printed_relative,relative);
  if((printed_variance==0.0&&variance[i]>0.0)||(printed_sigma==0.0&&variance[i]>0.0))++unobservable_rows;
 }
 constexpr std::string_view worst_prefix="worst relative error=";
 require(output[12].starts_with(worst_prefix),"missing worst field");
 const double worst=fixed6(output[12].substr(worst_prefix.size()));
 require(worst>=0.0&&worst<=1e-6&&worst+fixed6_allowance>=maximum_printed_relative,"worst field violates original bound/format consistency");
 constexpr std::string_view inf_prefix="crn_result_to_audit with an overflowing part: gum_uc=";
 const std::array<std::string_view,6> audit{
  "=== REGULATORY AUDIT ===",
  "model_id=42 gum_u_c=inf expanded_U=inf(k=2.000000)",
  "contrib %: epistemic 0.000000 intrinsic 0.000000 structural 0.000000 other 0.000000",
  "lit_refs=1 effect_mode=0",
  "REG PROVENANCE EXPORT COMPLETE",
  "GUM report: y = 1.000000 , u(y) = inf (std), U = inf (expanded k=2.000000)"
 };
 for(std::size_t i=0;i<audit.size();++i)require(output[13+i]==audit[i],"original regulatory audit block changed");
 require(output[19].starts_with(inf_prefix),"missing overflow audit field");
 auto inf=output[19].substr(inf_prefix.size());
 for(auto& c:inf)if(c>='A'&&c<='Z')c=static_cast<char>(c-'A'+'a');
 require(inf=="inf"||inf=="+inf"||inf=="infinity"||inf=="+infinity","overflow audit must print positive infinity");
 require(output[20]=="KIN SIGMA SMALL VARIANCE PASS","original final PASS ordering changed");
 std::cout<<"KIN_CPP_FIXED6_FORMAT_OK rows=12 tiny_zero_rows="<<unobservable_rows
          <<" underlying_tiny_accuracy=not_observable original_square_back_assertions=mandatory\n";
}
int main(int argc,char** argv){
 try{
  if(argc==2&&std::string_view(argv[1])=="--references"){
   const auto expected=chemistry_reference();std::cout<<std::setprecision(17);
   for(std::size_t i=0;i<expected.size();++i)std::cout<<i<<" "<<expected[i]<<"\n";
   return 0;
  }
  require(argc==4,"usage: format_oracle --chem|--kin <native.stdout> <native.exit-status>");
  const std::string_view mode=argv[1];
  if(mode=="--chem")check_chemistry(argv[2],argv[3]);
  else if(mode=="--kin")check_sigma(argv[2],argv[3]);
  else throw std::runtime_error("unknown mode");
  return 0;
 }catch(const std::exception& error){std::cerr<<"FORMAT_ORACLE_REFUSAL "<<error.what()<<"\n";return 1;}
}
