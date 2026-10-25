#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
int main(int argc,char**argv){if(argc!=2)return 64;std::ifstream in(argv[1]);std::string line,name;std::map<std::string,std::map<std::string,int>>counts;std::map<std::string,int>headers;
while(std::getline(in,line)){
 if(line.rfind("NV2_IR function ",0)==0){auto begin=line.find(" name=");auto end=line.find(" instr_count=",begin);name=begin==std::string::npos?"":line.substr(begin+6,end-begin-6);if(name.find("ln_checked")!=std::string::npos||name.find("log1p_checked")!=std::string::npos||name=="sv_step"||name=="sv_emit")headers[name]++;else name.clear();}
 else if(!name.empty()&&line.rfind("NV2_IR fn=",0)==0){auto begin=line.find(" op=");auto end=line.find(' ',begin+4);if(begin!=std::string::npos)counts[name][line.substr(begin+4,end-begin-4)]++;}
}
if(headers.empty()){std::cerr<<"ALLOCATION_CENSUS_MISSING_TRACE\n";return 10;}
for(auto&[fn,n]:headers){std::cout<<"ALLOCATION_CENSUS function="<<fn<<" headers="<<n;for(auto&[op,c]:counts[fn])std::cout<<' '<<op<<'='<<c;std::cout<<'\n';}return 0;}
