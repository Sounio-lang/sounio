// Independent C++23 closed-form Poisson log-offset model and SVG reference.
// Two groups: fitted rate_g = sum(counts_g)/sum(exposures_g).
// No iterative Sounio output supplies numerical or geometric expectations.
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
struct Circle {long long x,y,radius,r,g,b;};
struct Expected {long long x;std::pair<long long,long long> y;long long r,g,b;};
int main(int argc,char**argv){
 const std::array<long long,8> counts{1,3,5,9,3,5,9,17};
 const std::array<long double,8> exposure{1,2,4,8,1,2,4,8};
 std::array<long double,2> total_counts{},total_exposure{},rate{};
 for(unsigned i=0;i<8;++i){total_counts[i/4]+=counts[i];total_exposure[i/4]+=exposure[i];}
 for(unsigned g=0;g<2;++g)rate[g]=total_counts[g]/total_exposure[g];
 std::array<long double,8> means{},residual{},mean_budget{};
 long double deviance=0;
 for(unsigned i=0;i<8;++i){
  means[i]=rate[i/4]*exposure[i];
  residual[i]=(counts[i]-means[i])/std::sqrt(means[i]);
  mean_budget[i]=1e-8L*(1+means[i]);
  deviance+=2*(counts[i]*std::log(counts[i]/means[i])-(counts[i]-means[i]));
 }
 std::cout<<std::setprecision(21)<<"GROUP_RATES "<<rate[0]<<" "<<rate[1]
          <<"\nCOEFFICIENTS "<<std::log(rate[0])<<" "<<std::log(rate[1])-std::log(rate[0])
          <<"\nDEVIANCE "<<deviance<<"\n";
 // Circle display map: origin+height-trunc(normalized_y*height).
 // Propagate the agreed count-mean budget through the physical function
 // before pixel truncation. Observed rates are raw exact dyadic ratios.
 const auto y_range=[](long double low,long double high,long double ymin,long double ymax){
  const long double a=(low-ymin)/(ymax-ymin)*240;
  const long double b=(high-ymin)/(ymax-ymin)*240;
  const long double rounding=8*std::numeric_limits<double>::epsilon()
                            *(std::fabs(a)+std::fabs(b)+241);
  const auto bottom=390-static_cast<long long>(std::floor(b+rounding));
  const auto top=390-static_cast<long long>(std::floor(a-rounding));
  if(top-bottom>1)throw std::runtime_error("count-mean budget spans more than adjacent pixels");
  return std::pair{bottom,top};
 };
 std::vector<Expected> expected;
 for(unsigned i=0;i<8;++i){
  const double observed=static_cast<double>(counts[i])/static_cast<double>(exposure[i]);
  const auto y=390-static_cast<long long>((observed/4)*240);
  expected.push_back({140+60*static_cast<long long>(i+1),{y,y},70,180,235});
 }
 for(unsigned i=0;i<8;++i){
  const auto range=y_range((means[i]-mean_budget[i])/exposure[i],
                           (means[i]+mean_budget[i])/exposure[i],0,4);
  expected.push_back({140+60*static_cast<long long>(i+1),range,245,170,65});
 }
 const auto pearson=[](long double y,long double mu){return (y-mu)/std::sqrt(mu);};
 for(unsigned i=0;i<8;++i){
  // Pearson residual is strictly decreasing in mu for nonnegative counts.
  const auto range=y_range(pearson(counts[i],means[i]+mean_budget[i]),
                           pearson(counts[i],means[i]-mean_budget[i]),-1,1);
  expected.push_back({800+60*static_cast<long long>(i+1),range,220,135,225});
 }
 for(unsigned i=0;i<8;++i)std::cout<<"ROW "<<i<<" mean="<<means[i]<<" mean_budget="<<mean_budget[i]
   <<" observed_rate="<<counts[i]/exposure[i]<<" fitted_rate="<<rate[i/4]
   <<" Pearson_residual="<<residual[i]<<"\n";
 for(unsigned i=0;i<expected.size();++i){
  const auto e=expected[i];
  std::cout<<"EXPECTED_CIRCLE "<<i<<" x="<<e.x<<" y=["<<e.y.first<<","<<e.y.second
           <<"] RGB="<<e.r<<","<<e.g<<","<<e.b<<" radius=3\n";
 }
 if(argc==1){std::cout<<"POISSON_OFFSETS_SVG_REFERENCE_READY\n";return 0;}
 if(argc!=2){std::cerr<<"Usage: reference [SVG-or-native-log]\n";return 2;}
 std::ifstream file(argv[1]);if(!file){std::cerr<<"Cannot read SVG artifact\n";return 2;}
 const std::string text{std::istreambuf_iterator<char>(file),{}};
 const std::regex any_circle(R"rx(<circle\b)rx");
 const auto total=std::distance(std::sregex_iterator(text.begin(),text.end(),any_circle),std::sregex_iterator());
 if(total!=24){std::cerr<<"Wrong total circle count "<<total<<"\n";return 1;}
 const std::regex circle(R"rx(<circle cx="(-?[0-9]+)" cy="(-?[0-9]+)" r="([0-9]+)" fill="rgb\(([0-9]+),([0-9]+),([0-9]+)\)"/>)rx");
 std::vector<Circle> actual;
 for(std::sregex_iterator i(text.begin(),text.end(),circle),end;i!=end;++i)
  actual.push_back({std::stoll((*i)[1]),std::stoll((*i)[2]),std::stoll((*i)[3]),std::stoll((*i)[4]),std::stoll((*i)[5]),std::stoll((*i)[6])});
 if(actual.size()!=24){std::cerr<<"Malformed circle attributes\n";return 1;}
 std::set<std::tuple<long long,long long,long long,long long,long long>> unique;
 for(unsigned i=0;i<24;++i){
  const auto a=actual[i];const auto e=expected[i];
  if(a.x!=e.x||a.y<e.y.first||a.y>e.y.second||a.radius!=3||a.r!=e.r||a.g!=e.g||a.b!=e.b){
   std::cerr<<"Circle mismatch "<<i<<" actual("<<a.x<<","<<a.y<<") expected_x="<<e.x
            <<" y=["<<e.y.first<<","<<e.y.second<<"] expected_radius=3 RGB="
            <<e.r<<","<<e.g<<","<<e.b<<"\n";return 1;
  }
  unique.emplace(a.x,a.y,a.r,a.g,a.b);
 }
 if(unique.size()!=24){std::cerr<<"Duplicate colored marker center\n";return 1;}
 std::size_t at=0,records=0;
 while((at=text.find("<!-- scatter n=8 -->",at))!=std::string::npos){++records;++at;}
 if(records!=3){std::cerr<<"Expected three eight-observation scatter records\n";return 1;}
 const std::array<std::string,6> labels{"rate (events/s)","Pearson residual (unitless)",
  "Count model with known exposure offsets; log link","Observed count / exposure",
  "Fitted mean / exposure","No intervals; Poisson assumptions and overdispersion unassessed"};
 for(const auto& label:labels)if(text.find(label)==std::string::npos){std::cerr<<"Missing label "<<label<<"\n";return 1;}
 for(unsigned g=0;g<2;++g){
  const std::string prefix=std::string(g==0?"A":"B")+" rate (events/s)=";
  const auto position=text.find(prefix);
  if(position==std::string::npos){std::cerr<<"Missing group rate label\n";return 1;}
  const auto shown=std::stold(text.substr(position+prefix.size()));
  if(std::fabs(shown-rate[g])>5e-6L*rate[g]){std::cerr<<"Wrong formatted fitted rate\n";return 1;}
 }
 if(text.find("data-interval-kind=")!=std::string::npos){std::cerr<<"Unexpected interval metadata\n";return 1;}
 std::cout<<"POISSON_OFFSETS_SVG_REFERENCE_PASS circles=24 series=3\n";
}
