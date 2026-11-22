// C++23 independent analytic correlated-volume entropy and SVG geometry.
// Opposite volume gradients: u^2=(nR uVf/Vf)^2+(nR uVi/Vi)^2
// -2*rho*(nR uVf/Vf)*(nR uVi/Vi), with un=0 and equal 1% errors.
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <regex>
#include <string>
#include <utility>
#include <vector>
struct Circle {long long x,y,radius,r,g,b;};
int main(int argc,char**argv){
 const std::array<double,5> rho{-1,0,.5,.9,1};
 const long double r=static_cast<long double>(8.31446261815324);
 const long double rel=static_cast<long double>(.01);
 std::array<long double,5> uncertainty{};
 std::array<long long,5> expected_x{};
 std::array<std::pair<long long,long long>,5> circle_y{},path_y{};
 std::cout<<std::setprecision(21);
 for(unsigned i=0;i<5;++i){
  uncertainty[i]=r*rel*std::sqrt(2*(1-static_cast<long double>(rho[i])));
  expected_x[i]=160+static_cast<long long>(((rho[i]+1)/2)*600);
  if(uncertainty[i]==0){circle_y[i]={370,370};path_y[i]={370,370};}
  else{
   const long double projected=(uncertainty[i]/.2L)*240;
   const long double physical_budget=3e-11L*uncertainty[i]+2*static_cast<long double>(std::numeric_limits<double>::denorm_min());
   const long double display_roundoff=8*std::numeric_limits<double>::epsilon()*(std::fabs(projected)+241);
   const long double half=(physical_budget/.2L)*240+display_roundoff;
   circle_y[i]={370-static_cast<long long>(std::floor(projected+half)),
                370-static_cast<long long>(std::floor(projected-half))};
   path_y[i]={130+static_cast<long long>(std::floor(240-projected+.5L-half)),
              130+static_cast<long long>(std::floor(240-projected+.5L+half))};
   if(circle_y[i].second-circle_y[i].first>1||path_y[i].second-path_y[i].first>1)return 2;
  }
  std::cout<<"CASE rho="<<rho[i]<<" u_J_per_K="<<uncertainty[i]<<" x="<<expected_x[i]
           <<" circle_y=["<<circle_y[i].first<<","<<circle_y[i].second<<"]"
           <<" path_y=["<<path_y[i].first<<","<<path_y[i].second<<"]\n";
 }
 if(argc==1){std::cout<<"ENTROPY_COVARIANCE_SVG_REFERENCE_READY\n";return 0;}
 if(argc!=2){std::cerr<<"Usage: reference [SVG-or-native-log]\n";return 2;}
 std::ifstream file(argv[1]);if(!file){std::cerr<<"Cannot read SVG artifact\n";return 2;}
 const std::string text{std::istreambuf_iterator<char>(file),{}};
 const std::regex any_circle(R"rx(<circle\b)rx");
 const auto total=std::distance(std::sregex_iterator(text.begin(),text.end(),any_circle),std::sregex_iterator());
 if(total!=5){std::cerr<<"Wrong total circle count "<<total<<"\n";return 1;}
 const std::regex circle(R"rx(<circle cx="(-?[0-9]+)" cy="(-?[0-9]+)" r="([0-9]+)" fill="rgb\(([0-9]+),([0-9]+),([0-9]+)\)"/>)rx");
 std::vector<Circle> actual;
 for(std::sregex_iterator i(text.begin(),text.end(),circle),end;i!=end;++i)
  actual.push_back({std::stoll((*i)[1]),std::stoll((*i)[2]),std::stoll((*i)[3]),std::stoll((*i)[4]),std::stoll((*i)[5]),std::stoll((*i)[6])});
 if(actual.size()!=5){std::cerr<<"Malformed circle attributes\n";return 1;}
 for(unsigned i=0;i<5;++i){
  const auto c=actual[i];
  if(c.x!=expected_x[i]||c.y<circle_y[i].first||c.y>circle_y[i].second||
     c.radius!=3||c.r!=70||c.g!=180||c.b!=235){std::cerr<<"Circle mismatch "<<i<<"\n";return 1;}
 }
 const std::regex path(R"rx(<path fill="none" stroke="rgb\(220,135,225\)"[^>]* d="([^"]+)"/>)rx");
 std::smatch path_match;
 if(!std::regex_search(text,path_match,path)){std::cerr<<"Missing connected-case path\n";return 1;}
 const std::string commands=path_match[1];
 const std::regex command(R"rx(([ML])(-?[0-9]+) (-?[0-9]+))rx");
 unsigned step=0;
 for(std::sregex_iterator i(commands.begin(),commands.end(),command),end;i!=end;++i,++step){
  if(step>=8){std::cerr<<"Too many path commands\n";return 1;}
  const unsigned point=(step+1)/2;
  const auto x=std::stoll((*i)[2]),y=std::stoll((*i)[3]);
  if((*i)[1].str()!=(step%2==0?"M":"L")||x!=expected_x[point]||
     y<path_y[point].first||y>path_y[point].second){std::cerr<<"Path geometry mismatch "<<step<<"\n";return 1;}
 }
 if(step!=8){std::cerr<<"Expected four connected-case segments\n";return 1;}
 const std::array<std::string,6> labels{
  "rho is declared, not estimated; first-order propagation",
  "standard uncertainty (J/K)","declared volume correlation rho",
  "both u(V)/V=1%","rho=1 common calibration cancels",
  "Standard uncertainty; first-order; no confidence interval"};
 for(const auto& label:labels)if(text.find(label)==std::string::npos){std::cerr<<"Missing label "<<label<<"\n";return 1;}
 if(text.find("<!-- scatter n=5 -->")==std::string::npos||
    text.find("clipped-line visible=4 outside=0 invalid=0")==std::string::npos){
  std::cerr<<"Wrong observation/segment metadata\n";return 1;}
 std::cout<<"ENTROPY_COVARIANCE_SVG_REFERENCE_PASS circles=5 segments=4\n";
}
