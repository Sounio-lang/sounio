#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <boost/multiprecision/cpp_dec_float.hpp>
using Q=boost::multiprecision::cpp_dec_float_100;
int main(int argc,char**argv){
 try{
 if(argc!=2)throw std::runtime_error("fixture argument");
 std::ifstream f(argv[1]);std::string line;bool h=false,b=false,w=false;
 std::vector<long long> counts(2);std::vector<int> xs{-2,-1,0,1,2,3,4,8};
 for(auto x:xs){if(x>=0&&x<=4)++counts[x<2?0:1];}
 std::vector<Q> box{1,2,3,4,5,6,7,100};std::sort(box.begin(),box.end());
 auto quant=[&](Q p)->Q{Q r=Q(box.size()-1)*p;auto i=floor(r).convert_to<unsigned>();Q t=r-i;return box[i]*(1-t)+box[std::min(i+1,unsigned(box.size()-1))]*t;};
 Q q1=quant(Q(1)/4),med=quant(Q(1)/2),q3=quant(Q(3)/4);
 Q lo=q1-Q("1.5")*(q3-q1),hi=q3+Q("1.5")*(q3-q1);std::vector<Q> inner;
 for(auto x:box)if(x>=lo&&x<=hi)inner.push_back(x);
 auto near=[](Q actual,Q expected){if(abs(actual-expected)>abs(expected)*Q("2e-13")+Q("1e-322")){std::cerr<<"actual="<<actual<<" expected="<<expected<<"\n";throw std::runtime_error("numeric concordance");}};
 while(std::getline(f,line)){
 if(line.rfind("HIST ",0)==0){Q c0(line.substr(5));std::getline(f,line);Q c1(line);std::getline(f,line);Q d0(line);std::getline(f,line);Q d1(line);if(c0!=counts[0]||c1!=counts[1])throw std::runtime_error("counts");near(d0,Q(counts[0])/16);near(d1,Q(counts[1])/16);h=true;}
 if(line.rfind("BOX ",0)==0){near(Q(line.substr(4)),q1);std::getline(f,line);near(Q(line),med);std::getline(f,line);near(Q(line),q3);std::getline(f,line);near(Q(line),inner.front());std::getline(f,line);near(Q(line),inner.back());std::getline(f,line);if(Q(line)!=box.size()-inner.size())throw std::runtime_error("outliers");b=true;}
 if(line.rfind("WIDE ",0)==0){near(Q(line.substr(5)),Q(1)/(Q("3.4e308")));w=true;}
 }
 if(!h||!b||!w)throw std::runtime_error("missing records");
 std::cout<<"SUMMARY_HISTOGRAM_REFERENCE_PASS\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}
}
