// Numerical hypercomplex smoke oracle. JSON framing and all comparisons run
// in C++23; the orchestration layer never recalculates metrics.
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace {
std::string read(const std::string& p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot read "+p);return {std::istreambuf_iterator<char>(f),{}};}
double number(const std::string& s){
  static const std::regex token(R"(-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?)");
  if(!std::regex_match(s,token))throw std::runtime_error("invalid JSON number");
  char* end=nullptr;errno=0;const double x=std::strtod(s.c_str(),&end);
  if(end!=s.c_str()+s.size()||!std::isfinite(x)||errno==ERANGE)throw std::runtime_error("unrepresentable metric");
  return x;
}
struct JSON {
  std::string s;std::size_t i=0;
  std::map<std::string,double> numbers;
  std::map<std::string,std::string> strings;
  void space(){while(i<s.size()&&(s[i]==' '||s[i]=='\n'||s[i]=='\r'||s[i]=='\t'))++i;}
  void expect(char c){space();if(i>=s.size()||s[i++]!=c)throw std::runtime_error("invalid golden JSON framing");}
  std::string text(){
    expect('"');std::string out;
    while(i<s.size()){
      char c=s[i++];if(c=='"')return out;
      if(static_cast<unsigned char>(c)<32)throw std::runtime_error("control in JSON string");
      if(c=='\\'){
        if(i>=s.size())throw std::runtime_error("truncated JSON escape");
        c=s[i++];
        if(c=='"'||c=='\\'||c=='/')out+=c;
        else throw std::runtime_error("unsupported escape in golden key/schema");
      }else out+=c;
    }
    throw std::runtime_error("unclosed JSON string");
  }
  void value(const std::string& prefix){
    space();if(i>=s.size())throw std::runtime_error("missing JSON value");
    if(s[i]=='{'){
      ++i;space();std::set<std::string> keys;
      if(i<s.size()&&s[i]=='}'){++i;return;}
      for(;;){
        const auto key=text();
        if(key.empty()||key.find('.')!=std::string::npos||!keys.insert(key).second)throw std::runtime_error("ambiguous golden key");
        expect(':');value(prefix.empty()?key:prefix+"."+key);space();
        if(i<s.size()&&s[i]=='}'){++i;return;}expect(',');
      }
    }
    if(s[i]=='"'){strings.emplace(prefix,text());return;}
    const auto first=i;while(i<s.size()&&std::string_view("-+0123456789.eE").find(s[i])!=std::string_view::npos)++i;
    if(first==i)throw std::runtime_error("golden value must be object, string, or number");
    numbers.emplace(prefix,number(s.substr(first,i-first)));
  }
  explicit JSON(std::string input):s(std::move(input)){value("");space();if(i!=s.size())throw std::runtime_error("trailing JSON");}
};
}
int main(int argc,char**argv){
 try{
  if(argc<4||(argc-2)%2)throw std::runtime_error("usage: oracle golden.json lane stdout [lane stdout ...]");
  JSON golden(read(argv[1]));
  if(golden.strings["schema"]!="sounio.stdlib.hyper_execution_golden.v1")throw std::runtime_error("golden schema");
  const auto at=golden.numbers.find("tolerance.abs"),rt=golden.numbers.find("tolerance.rel");
  if(at==golden.numbers.end()||rt==golden.numbers.end()||at->second<0||rt->second<0)throw std::runtime_error("invalid tolerances");
  const std::set<std::string> lanes{"nn","onn","qnn","snn","spnn","quantnn","math"};
  std::map<std::string,std::map<std::string,double>> expected,observed;
  for(const auto&[key,x]:golden.numbers){
    const auto dot=key.find(".metrics.");if(dot==std::string::npos)continue;
    const auto lane=key.substr(0,dot);if(!lanes.contains(lane))throw std::runtime_error("unexpected golden lane");
    expected[lane].emplace(key.substr(dot+9),x);
  }
  if(expected.size()!=lanes.size())throw std::runtime_error("missing expected lane");
  // Independent exact reference for q=(1,2,3,4), q-0 and its squared norm.
  constexpr double q[4]={1.0,2.0,3.0,4.0};
  constexpr double difference=q[0]-0.0;
  constexpr double norm=q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
  static_assert(difference==1.0&&norm==30.0);
  if(expected["nn"].at("diff_w")!=difference||expected["nn"].at("norm_sq")!=norm)
    throw std::runtime_error("quaternion golden disagrees with exact C++ reference");
  std::set<std::string> supplied;std::set<std::string> bad;
  const std::regex metric(R"(^\s*HYPER_METRIC\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\s+([-+0-9.eE]+)\s*$)");
  for(int arg=2;arg<argc;arg+=2){
    const std::string lane=argv[arg];if(!lanes.contains(lane)||!supplied.insert(lane).second)throw std::runtime_error("unexpected/duplicate supplied lane");
    std::istringstream input(read(argv[arg+1]));std::string line;
    while(std::getline(input,line)){
      std::smatch match;
      if(!std::regex_match(line,match,metric)){if(line.find("HYPER_METRIC")!=std::string::npos)bad.insert(lane);continue;}
      if(match[1]!=lane||!expected[lane].contains(match[2])){bad.insert(lane);std::cout<<"INVALID_METRIC\t"<<lane<<"\n";continue;}
      try{
        std::string lexeme=match[3];if(!lexeme.empty()&&lexeme[0]=='+')lexeme.erase(0,1);
        const double x=number(lexeme);
        if(!observed[lane].emplace(match[2],x).second){bad.insert(lane);std::cout<<"DUPLICATE_METRIC\t"<<lane<<"\n";}
      }catch(const std::exception&){bad.insert(lane);std::cout<<"INVALID_NUMBER\t"<<lane<<"\n";}
    }
  }
  for(const auto& lane:lanes){
    if(!supplied.contains(lane))bad.insert(lane);
    for(const auto&[key,want]:expected[lane]){
      const auto found=observed[lane].find(key);
      if(found==observed[lane].end()){bad.insert(lane);std::cout<<"MISSING_METRIC\t"<<lane<<"\t"<<key<<"\n";continue;}
      const double diff=std::abs(found->second-want),tol=at->second+rt->second*std::abs(want);
      if(!std::isfinite(diff)||!std::isfinite(tol)||diff>tol){
        bad.insert(lane);std::cout<<std::setprecision(17)<<"MISMATCH\t"<<lane<<"\t"<<key<<"\t"<<found->second<<"\t"<<want<<"\n";
      }
    }
    std::cout<<"LANE\t"<<lane<<"\t"<<(bad.contains(lane)?"fail":"pass")<<"\n";
  }
  if(bad.empty())std::cout<<"HYPER_METRICS_CPP_OK\n";
  return bad.empty()?0:1;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}
}
