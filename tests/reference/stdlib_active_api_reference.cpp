#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
void need(bool okay,const char* message){if(!okay)throw std::runtime_error(message);}
int main(int argc,char** argv){
 try{
  need(argc==2,"usage: active-reference <Sounio stdout>");
  std::ifstream input(argv[1]);need(bool(input),"open stdout");
  std::string line;std::array<long long,9> actual{};bool found=false;
  while(std::getline(input,line)){
   if(line.rfind("ACTIVE_CONTROL ",0)!=0)continue;
   need(!found,"duplicate control row");found=true;
   std::size_t at=15;
   for(auto& value:actual){std::size_t used=0;value=std::stoll(line.substr(at),&used);at+=used;while(at<line.size()&&line[at]==' ')++at;}
   need(at==line.size(),"extra control data");
  }
  need(found,"missing nonempty control");
  const auto sample=[](long double d,long double e,long double delta){return static_cast<long long>(std::ceil((d+std::log(1.0L/delta))/(e*e)));};
  const long double prior_precision=1.0L/4.0L,observation_precision=1.0L/4.0L;
  const auto mean=static_cast<long long>((prior_precision*10.0L+observation_precision*14.0L)/(prior_precision+observation_precision));
  const auto variance=static_cast<long long>(1.0L/(prior_precision+observation_precision));
  const std::array<long long,9> expected{8,7,4,mean,variance,800,sample(1,.5L,.5L),sample(10,.1L,.1L),sample(2,.25L,.25L)};
  need(actual==expected,"C++23/Sounio active/PAC control disagreement");
  std::cout<<"ACTIVE_CPP_CONCORDANCE_OK\n";
 }catch(const std::exception& error){std::cerr<<error.what()<<"\n";return 1;}
}
