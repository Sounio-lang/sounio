// C++23 structural/numerical oracle for the correlation scatter SVG.
// Reads an extracted SVG. Independent six raw pairs and viewport projection.
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>
std::string attribute(const std::string& tag,const std::string& key) {
 std::smatch m;std::regex r(key+"=\"([^\"]+)\"");
 return std::regex_search(tag,m,r)?m[1].str():"";
}
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 std::ifstream input(argv[1]);if(!input)return 2;
 std::string svg((std::istreambuf_iterator<char>(input)),{});
 if(svg.find("<!-- scatter n=6 -->")==std::string::npos){std::cerr<<"bad scatter count metadata\n";return 1;}
 std::regex circles("<circle\\b[^>]*>");
 std::array<double,6> x{0,1,2,3,4,4};
 std::size_t count=0;
 for(auto i=std::sregex_iterator(svg.begin(),svg.end(),circles);i!=std::sregex_iterator();++i) {
  if(count>=x.size()){std::cerr<<"extra circle\n";return 1;}
  const std::string tag=i->str();
  const auto cx=std::llround(160+600*x[count]/4);
  const auto cy=std::llround(360-240*x[count]*x[count]/16);
  if(attribute(tag,"cx")!=std::to_string(cx)||attribute(tag,"cy")!=std::to_string(cy)
   ||attribute(tag,"r")!="3"||attribute(tag,"fill")!="rgb(70,180,235)"){
   std::cerr<<"circle geometry/style mismatch at "<<count<<"\n";return 1;
  }
  ++count;
 }
 if(count!=x.size()){std::cerr<<"missing circles\n";return 1;}
 std::cout<<"CORRELATION_SVG_REFERENCE_PASS count="<<count<<"\n";
}
