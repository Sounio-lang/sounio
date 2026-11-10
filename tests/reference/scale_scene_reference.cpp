#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using std::string;
void check(bool b,const char*m){if(!b)throw std::runtime_error(m);}
string read(const string&p){std::ifstream f(p);check(bool(f),"read");return {std::istreambuf_iterator<char>(f),{}};}
std::vector<string> docs(const string&s){
 std::vector<string> v;std::size_t p=0;
 while(true){auto a=s.find("<svg ",p);if(a==string::npos)break;auto b=s.find("</svg>",a);
 check(b!=string::npos,"svg close");v.push_back(s.substr(a,b+6-a));p=b+6;}
 check(!v.empty(),"no SVG");return v;
}
int main(int argc,char**argv)try{
 if(argc==4&&string(argv[1])=="--same"){
  auto a=docs(read(argv[2])),b=docs(read(argv[3]));
  check(a==b,"legacy SVG bytes changed");std::cout<<"LEGACY_SVG_IDENTICAL documents="<<a.size()<<"\n";return 0;
 }
 if(argc!=3||string(argv[1])!="--compton")return 2;
 auto all=docs(read(argv[2]));check(all.size()==2,"two consumers");
 const auto&s=all[1];auto a=s.find(" d=\"");check(a!=string::npos,"path data");a+=4;
 auto b=s.find('"',a);std::istringstream path(s.substr(a,b-a));
 struct P{long long x,y;};std::vector<P> expected;
 long double theta=(double)0.001;
 const long double lambda=6.62607015e-34L/(9.1093837139e-31L*299792458.0L);
 const long double xl=std::log10((long double)(double).001),xh=std::log10(10.0L);
 const long double yl=std::log10((long double)(double)1e-6),yh=std::log10(10.0L);
 for(int i=0;i<12;i++){
  long double sine=std::sin(theta/2),pm=2*lambda*sine*sine*1e12L;
  long double u=(std::log10(theta)-xl)/(xh-xl),v=(std::log10(pm)-yl)/(yh-yl);
  expected.push_back({130+std::llround(560*u),35+std::llround(180*(1-v))});theta*=2;
 }
 for(int i=0;i<11;i++){
  char m,l;long long x0,y0,x1,y1;check(bool(path>>m>>x0>>y0>>l>>x1>>y1),"segment parse");
  check(m=='M'&&l=='L',"independent segment");
  check(x0==expected[i].x&&y0==expected[i].y&&x1==expected[i+1].x&&y1==expected[i+1].y,"log/log geometry mismatch");
 }
 path>>std::ws;check(path.eof(),"extra segments");
 check(s.find("data-x-scale=\"log10\" data-y-scale=\"log10\"")!=string::npos,"scale metadata");
 std::cout<<"COMPTON_LOG_GEOMETRY_PASS points=12 segments=11 independent long-double sine/log10/projection\n";return 0;
}catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<"\n";return 1;}
