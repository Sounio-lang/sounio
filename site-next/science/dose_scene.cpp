// ADR009 scientific oracle. Source extraction and all dose arithmetic live in C++23.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using Real = long double;
struct Reading { Real mean, sd, lo, hi, variance; bool outside; };
Reading propagate(Real weight, Real per_kg, Real per_kg_var, Real weight_sd,
                  Real z, Real window_lo, Real window_hi) {
  const Real variance = per_kg*per_kg*weight_sd*weight_sd + weight*weight*per_kg_var;
  const Real sd = std::sqrt(variance), mean = weight*per_kg;
  const Real lo = mean-z*sd, hi = mean+z*sd;
  return {mean,sd,lo,hi,variance,lo<window_lo || hi>window_hi};
}
std::vector<Real> grab(const std::string& s, const std::string& pattern) {
  std::smatch m;
  if (!std::regex_search(s,m,std::regex(pattern))) throw std::runtime_error("source contract missing: "+pattern);
  std::vector<Real> r;
  for (std::size_t i=1;i<m.size();++i) {
    const auto raw=m[i].str(); std::size_t consumed=0;
    const auto value=std::stold(raw,&consumed);
    if(consumed!=raw.size() || !std::isfinite(value)) throw std::runtime_error("invalid source number");
    r.push_back(value);
  }
  return r;
}
void number(Real x) {
  if (!std::isfinite(x)) throw std::runtime_error("non-finite scientific result");
  std::cout << std::setprecision(18) << x;
}
void reading(const Reading& r) {
  std::cout << "{\"mean\":"; number(r.mean); std::cout << ",\"sd\":";number(r.sd);
  std::cout << ",\"lo\":";number(r.lo);std::cout << ",\"hi\":";number(r.hi);
  std::cout << ",\"variance\":";number(r.variance);
  std::cout << ",\"outside\":" << (r.outside?"1":"0") << '}';
}
int main(int argc,char**argv) try {
  if (argc==2 && std::string(argv[1])=="--self-test") {
    const auto r=propagate(70,10,0,2,1.96L,600,800);
    if(r.mean!=700 || r.variance!=400 || r.sd!=20 || std::abs(r.lo-660.8L)>1e-12L || std::abs(r.hi-739.2L)>1e-12L || r.outside) throw std::runtime_error("analytic dose reference failed");
    const auto both=propagate(60,10,1,2.5L,1.96L,400,800);
    if(both.mean!=600 || both.variance!=4225 || both.sd!=65) throw std::runtime_error("two nonzero uncertainty terms failed");
    if(!propagate(70,10,0,6,1.96L,600,800).outside) throw std::runtime_error("window refusal failed");
    if(propagate(70,10,0,0,1.96L,600,800).sd!=0) throw std::runtime_error("zero variance failed");
    std::cout << "PASS exact analytic mean/variance, interval, zero-variance and refusal controls\n"; return 0;
  }
  if (argc!=2) throw std::runtime_error("usage: dose_scene SOURCE.sio");
  std::ifstream input(argv[1]); if(!input) throw std::runtime_error("source unreadable");
  const std::string s((std::istreambuf_iterator<char>(input)),{});
  const auto w=grab(s,R"(let weight\s*=\s*epistemic_new\(([\d.]+),\s*([\d.]+)\))");
  const auto p=grab(s,R"(let dose_per_kg\s*=\s*epistemic_new\(([\d.]+),\s*([\d.]+)\))");
  const Real window_lo=grab(s,R"(let therapeutic_min\s*=\s*([\d.]+))")[0];
  const Real window_hi=grab(s,R"(let therapeutic_max\s*=\s*([\d.]+))")[0];
  const Real documented_sd=grab(s,R"(Total Dose:[^\n]*?σ=([\d.]+))")[0];
  const auto doc_ci=grab(s,R"(95% CI:\s*\[([\d.]+),\s*([\d.]+)\])");
  if(w[0]<=0 || p[0]<=0 || w[1]<0 || p[1]<0 || window_hi<=window_lo)
    throw std::runtime_error("invalid source domain");
  const Real z=1.96L, weight_sd=std::sqrt(w[1]), per_kg_sd=std::sqrt(p[1]);
  if (std::abs(weight_sd*10-std::round(weight_sd*10))>1e-12L)
    throw std::runtime_error("default sigma is not on the one-decimal slider grid");
  const auto base=propagate(w[0],p[0],p[1],weight_sd,z,window_lo,window_hi);
  const Real half=std::min(base.mean-window_lo,window_hi-base.mean)/z;
  const Real crossing_rad=(half*half-w[0]*w[0]*p[1])/(p[0]*p[0]);
  if(half<=0 || crossing_rad<=0) throw std::runtime_error("source has no slider crossing in supported domain");
  const Real crossing=std::sqrt(crossing_rad), sigma_max=std::ceil(crossing*4)/2;
  const auto widest=propagate(w[0],p[0],p[1],sigma_max,z,window_lo,window_hi);
  const Real tick=100, axis_lo=std::floor((widest.lo-tick/2)/tick)*tick;
  const Real axis_hi=std::ceil((widest.hi+tick/2)/tick)*tick;
  if(sigma_max>1000) throw std::runtime_error("slider exceeds bounded oracle contract");
  const std::map<std::string,Real> metrics={
    {"weight",w[0]},{"weight_sd",weight_sd},{"weight_var",w[1]},
    {"per_kg",p[0]},{"per_kg_sd",per_kg_sd},{"per_kg_var",p[1]},
    {"window_lo",window_lo},{"window_hi",window_hi},{"dose",base.mean},
    {"variance",base.variance},{"sd",base.sd},{"ci_lo",base.lo},{"ci_hi",base.hi},
    {"sigma_crossing",crossing},{"sigma_min",0.3L},{"sigma_max",sigma_max},{"sigma_step",0.1L},
    {"axis_lo",axis_lo},{"axis_hi",axis_hi},{"tick_step",tick},{"z",z},
    {"documented_sd",documented_sd},{"documented_ci_lo",doc_ci[0]},{"documented_ci_hi",doc_ci[1]},
    {"documented_width",doc_ci[1]-doc_ci[0]},{"computed_width",base.hi-base.lo}};
  std::cout << "{\"oracle\":\"C++23/dose_scene\",\"metrics\":{";bool first=true;
  for(const auto&[key,value]:metrics){if (!first) { std::cout << ','; }
    first=false;std::cout<<'"'<<key<<"\":";number(value);}
  std::cout << "},\"default_reading\":";reading(base);
  std::cout << ",\"readings\":{";first=true;
  // Integer stepping produces canonical slider keys; browser never interpolates scientific readings.
  for(int tenth=3; Real(tenth)/10<=sigma_max; ++tenth){
    if (!first) { std::cout << ','; }
    first=false;
    std::cout << '"' << tenth/10 << '.' << tenth%10 << "\":";
    reading(propagate(w[0],p[0],p[1],Real(tenth)/10,z,window_lo,window_hi));
  }
  std::cout << "}}\n";
} catch(const std::exception&e){std::cerr<<"dose oracle: "<<e.what()<<'\n';return 1;}
