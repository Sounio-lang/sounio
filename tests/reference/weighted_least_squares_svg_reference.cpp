// C++23 independent weighted calibration and SVG geometry reference.
// Well-conditioned two-column fixture: solve weighted scalar projections
// in long double from the raw binary64 inputs. Independent of Householder QR.
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <utility>
#include <regex>
#include <string>
#include <vector>
using Values=std::array<long double,8>;
struct Fit {long double intercept,slope,norm,weighted_norm;Values fitted,residual;};
Fit solve(const Values& y,const Values& w){
 long double sw=0,sx=0,sy=0,sxx=0,sxy=0,maxw=0;
 for(unsigned i=0;i<8;++i){const long double x=i;sw+=w[i];sx+=w[i]*x;sy+=w[i]*y[i];
  sxx+=w[i]*x*x;sxy+=w[i]*x*y[i];if(w[i]>maxw)maxw=w[i];}
 const long double det=sw*sxx-sx*sx;
 if(!(det>0))throw std::runtime_error("singular reference design");
 Fit out{};out.slope=(sw*sxy-sx*sy)/det;out.intercept=(sy-out.slope*sx)/sw;
 long double ss=0,wss=0;
 for(unsigned i=0;i<8;++i){out.fitted[i]=out.intercept+out.slope*i;out.residual[i]=y[i]-out.fitted[i];
  ss+=out.residual[i]*out.residual[i];wss+=(w[i]/maxw)*out.residual[i]*out.residual[i];}
 out.norm=std::sqrt(ss);out.weighted_norm=std::sqrt(wss);return out;
}
struct Circle {long long x,y,r,g,b;};
int main(int argc,char**argv){
 const std::array<double,8> raw{1.2,2.7,5.4,6.6,9.6,10.4,13.5,14.5};
 Values response{},unit{},weights{};
 for(unsigned i=0;i<8;++i){response[i]=raw[i];unit[i]=1;weights[i]=i%2==0?1:4;}
 const auto ordinary=solve(response,unit),weighted=solve(response,weights);
 std::cout<<std::setprecision(21)<<"OLS_BETA "<<ordinary.intercept<<" "<<ordinary.slope
  <<"\nWLS_BETA "<<weighted.intercept<<" "<<weighted.slope
  <<"\nWLS_RAW_RESIDUAL_NORM "<<weighted.norm
  <<"\nWLS_NORMALIZED_WEIGHTED_RESIDUAL_NORM "<<weighted.weighted_norm<<"\n";
 const auto px=[](long long origin,long long extent,double value,double lo,double hi){
  return origin+static_cast<long long>(((value-lo)/(hi-lo))*static_cast<double>(extent));};
 const auto py=[](long long origin,long long extent,double value,double lo,double hi){
  return origin+extent-static_cast<long long>(((value-lo)/(hi-lo))*static_cast<double>(extent));};
 // Match the accepted numerical fixture's physical forward budget:
 // residual error <= 3e-11 * ||raw response||_2, measured in mV.
 // Propagate that interval through the affine display map before truncation.
 // Only an interval crossing an integer boundary admits adjacent pixels.
 long double response_ss=0;
 for(const auto value:response)response_ss+=value*value;
 const long double physical_budget=3e-11L*std::sqrt(response_ss);
 const auto residual_pixels=[&](long double value,unsigned point){
  const long double projected=(value+2.0L)*60.0L;
  const long double mapping_roundoff=8.0L*std::numeric_limits<double>::epsilon()
                                   *(std::fabs(projected)+240.0L+1.0L);
  const long double half=60.0L*physical_budget+mapping_roundoff;
  const long double projected_lo=projected-half,projected_hi=projected+half;
  const auto low=390-static_cast<long long>(std::floor(projected_hi));
  const auto high=390-static_cast<long long>(std::floor(projected_lo));
  if(high-low>1)throw std::runtime_error("physical budget spans more than adjacent pixels");
  std::cout<<"PIXEL_INTERVAL "<<point<<" residual_mV="<<value
           <<" projection="<<projected<<" physical_budget_mV="<<physical_budget
           <<" projection_bounds=["<<projected_lo<<","<<projected_hi<<"]"
           <<" allowed_y=["<<low<<","<<high<<"]\n";
  return std::pair{low,high};
 };
 std::vector<Circle> expected;
 std::vector<std::pair<long long,long long>> allowed_y;
 for(unsigned i=0;i<8;++i){
  expected.push_back({px(140,480,i,0,8),py(150,240,raw[i],0,18),70,180,235});
  allowed_y.emplace_back(expected.back().y,expected.back().y);
 }
 for(unsigned i=0;i<8;++i){
  expected.push_back({px(800,480,i+1,0,8),py(150,240,static_cast<double>(ordinary.residual[i]),-2,2),220,135,225});
  allowed_y.push_back(residual_pixels(ordinary.residual[i],8+i));
 }
 for(unsigned i=0;i<8;++i){
  expected.push_back({px(800,480,i+1,0,8),py(150,240,static_cast<double>(weighted.residual[i]),-2,2),245,170,65});
  allowed_y.push_back(residual_pixels(weighted.residual[i],16+i));
 }
 unsigned ambiguous=0;
 for(const auto& bounds:allowed_y)if(bounds.first!=bounds.second)++ambiguous;
 std::cout<<"BOUNDARY_STRADDLING_POINTS "<<ambiguous<<"\n";
 for(const auto& c:expected)std::cout<<"EXPECTED_CIRCLE "<<c.x<<" "<<c.y<<" "<<c.r<<" "<<c.g<<" "<<c.b<<"\n";
 if(argc==1){std::cout<<"WEIGHTED_LEAST_SQUARES_SVG_REFERENCE_READY\n";return 0;}
 if(argc!=2){std::cerr<<"Usage: reference [SVG-or-native-log]\n";return 2;}
 std::ifstream file(argv[1]);if(!file){std::cerr<<"Cannot read SVG artifact\n";return 2;}
 const std::string text{std::istreambuf_iterator<char>(file),{}};
 const std::regex any_circle(R"rx(<circle\b)rx");
 const auto total=std::distance(std::sregex_iterator(text.begin(),text.end(),any_circle),std::sregex_iterator());
 if(total!=24){std::cerr<<"Wrong total circle count: "<<total<<"\n";return 1;}
 const std::regex circle(R"rx(<circle cx="(-?[0-9]+)" cy="(-?[0-9]+)" r="3" fill="rgb\(([0-9]+),([0-9]+),([0-9]+)\)"/>)rx");
 std::vector<Circle> actual;
 for(std::sregex_iterator i(text.begin(),text.end(),circle),end;i!=end;++i)
  actual.push_back({std::stoll((*i)[1]),std::stoll((*i)[2]),std::stoll((*i)[3]),std::stoll((*i)[4]),std::stoll((*i)[5])});
 if(actual.size()!=expected.size()){std::cerr<<"Circle shape/attribute mismatch\n";return 1;}
 for(unsigned i=0;i<expected.size();++i){
  const auto a=actual[i],e=expected[i];
  if(a.x!=e.x||a.y<allowed_y[i].first||a.y>allowed_y[i].second||a.r!=e.r||a.g!=e.g||a.b!=e.b){
   std::cerr<<"Circle mismatch "<<i<<" actual("<<a.x<<","<<a.y<<") expected_x="<<e.x
            <<" allowed_y=["<<allowed_y[i].first<<","<<allowed_y[i].second<<"] actual_rgb="
            <<a.r<<","<<a.g<<","<<a.b<<" expected_rgb="<<e.r<<","<<e.g<<","<<e.b<<"\n";return 1;}}
 std::size_t at=0,count=0;
 while((at=text.find("<!-- scatter n=8 -->",at))!=std::string::npos){++count;++at;}
 if(count!=3){std::cerr<<"Expected three eight-observation scatter records\n";return 1;}
 const std::array<std::string,5> labels{"response (mV)","residual (mV)",
  "Relative objective weights; variance model unassessed",
  "No standard errors, p-values or confidence intervals",
  "alternating objective weights 1 and 4"};
 for(const auto& label:labels)if(text.find(label)==std::string::npos){std::cerr<<"Missing label "<<label<<"\n";return 1;}
 const std::array<long double,2> beta{weighted.intercept,weighted.slope};
 for(unsigned j=0;j<2;++j){
  const std::string prefix="WLS beta"+std::to_string(j)+" (mV)=";
  const auto position=text.find(prefix);
  if(position==std::string::npos){std::cerr<<"Missing coefficient label\n";return 1;}
  const long double shown=std::stold(text.substr(position+prefix.size()));
  if(std::fabs(shown-beta[j])>5e-6L*std::fabs(beta[j])){
   std::cerr<<"Incorrect formatted coefficient "<<j<<"\n";return 1;}
 }
 std::cout<<"WEIGHTED_LEAST_SQUARES_SVG_REFERENCE_PASS circles=24 panels=2\n";
}
