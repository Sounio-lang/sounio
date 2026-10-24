// C++23 canonical transformation of recorded C++23/Sounio chemistry tables.
// This does not rerun the chemistry solver or claim current-compiler parity.
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
using Real=long double;
using Row=std::vector<Real>;
std::string section(const std::string& text,const std::string& start,const std::string& end){
  const auto a=text.find(start), b=text.find(end,a==std::string::npos?0:a+start.size());
  if(a==std::string::npos || b==std::string::npos) throw std::runtime_error("missing source section "+start);
  return text.substr(a,b-a);
}
Real parse(std::string s){
  s=std::regex_replace(s,std::regex("[* ]"),"");
  std::size_t n=0; const Real x=std::stold(s,&n);
  if(n!=s.size() || !std::isfinite(x) || x<=0) throw std::runtime_error("invalid positive table value");
  return x;
}
std::map<std::string,Row> table(const std::string& text){
  std::map<std::string,Row> out;
  const std::regex line(R"(\|\s*(H2O2|H2O|HO2|OH|H2|O2|H|O)\s*\|([^|\n]+)\|([^|\n]+)\|([^|\n]+)\|)");
  for(auto i=std::sregex_iterator(text.begin(),text.end(),line);i!=std::sregex_iterator();++i){
    const auto&m=*i; if(out.contains(m[1]))throw std::runtime_error("duplicate species");
    out[m[1]]={parse(m[2]),parse(m[3]),parse(m[4])};
  }
  if(out.size()!=8)throw std::runtime_error("incomplete eight-species table");
  return out;
}
void number(Real x){if(!std::isfinite(x))throw std::runtime_error("nonfinite output");std::cout<<std::setprecision(18)<<x;}
Row normalized(const Row& r,bool absolute){
  const Real half1=absolute?r[0]/r[1]:r[0],half2=absolute?r[1]/r[2]:r[1],quarter=absolute?r[0]/r[2]:r[2];
  return {half1,half2,quarter,1,1/half1,1/quarter};
}
void emit(const std::map<std::string,Row>& values,bool absolute){
  std::cout<<'{'; bool first=true;
  for(const auto&[name,r]:values){
    if(!first){std::cout<<',';}first=false;
    const auto widths=normalized(r,absolute);
    const Real half1=widths[0],half2=widths[1],quarter=widths[2];
    std::cout<<'"'<<name<<"\":{\"half1\":";number(half1);
    std::cout<<",\"half2\":";number(half2);std::cout<<",\"quarter\":";number(quarter);
    std::cout<<",\"dt4\":1,\"dt2\":";number(widths[4]);std::cout<<",\"dt1\":";number(widths[5]);std::cout<<'}';
  }
  std::cout<<'}';
}
Row contrast(const std::string& text,const std::string& label){
  std::smatch m;const std::regex line("\\|[^|\\n]*"+label+"[^|\\n]*\\|\\s*\\*\\*([\\d.]+)\\s*(?:–|-)\\s*([\\d.]+)\\*\\*");
  if(!std::regex_search(text,m,line))throw std::runtime_error("missing canonical contrast "+label);
  return {parse(m[1]),parse(m[2])};
}
int main(int argc,char**argv)try{
  if(argc==2 && std::string(argv[1])=="--self-test"){
    const auto absolute=normalized({2,1,0.5L},true),relative=normalized({2,2,4},false);
    if(absolute!=Row{2,2,4,1,0.5L,0.25L} || relative!=absolute) throw std::runtime_error("exact rational normalization failed");
    if(parse(" **2.0** ")!=2)throw std::runtime_error("decimal parse");
    bool rejected=false;try{parse("0");}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("zero band accepted");
    rejected=false;try{table("| H | 1 | 2 | 3 |");}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("missing species accepted");
    std::cout<<"PASS finite positive parser and incomplete-source refusal controls\n";return 0;
  }
  if(argc!=2)throw std::runtime_error("usage: h2_scene RESULTS.md");
  std::ifstream in(argv[1]);if(!in)throw std::runtime_error("source unreadable");
  const std::string text((std::istreambuf_iterator<char>(in)),{});
  const auto q=table(section(text,"### 5.1 Scaling in dt","### 5.2"));
  const auto n=table(section(text,"### 5.4 The Sounio native band","### 5.5"));
  const auto c=section(text,"This is the whole contrast","### 5.5");
  const auto cpp=contrast(c,"C\\+\\+23 cross-check"),native=contrast(c,"Sounio native");
  int non_acc=0;for(const auto&[name,r]:q){(void)name;if(std::abs(r[2]-1)<0.01L)++non_acc;}
  std::cout<<"{\"oracle\":\"C++23/h2_scene\",\"metrics\":{\"species\":10,\"reactions\":29,\"cpp_lo\":";number(cpp[0]);
  std::cout<<",\"cpp_hi\":";number(cpp[1]);std::cout<<",\"native_lo\":";number(native[0]);std::cout<<",\"native_hi\":";number(native[1]);
  std::cout<<",\"predicted_quarter\":2,\"predicted_half\":";number(std::sqrt(2.0L));
  std::cout<<",\"species_reported\":8,\"non_accumulating_count\":"<<non_acc<<",\"quadrature\":";emit(q,false);
  std::cout<<",\"native\":";emit(n,true);std::cout<<"}}\n";
}catch(const std::exception&e){std::cerr<<"h2 oracle: "<<e.what()<<'\n';return 1;}
