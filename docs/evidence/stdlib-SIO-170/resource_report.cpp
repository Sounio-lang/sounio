// C++23 calculation of summaries from the frozen paired process observations.
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
struct Observation {double elapsed=std::numeric_limits<double>::quiet_NaN();long long rss=-1;int exit=-1;};
Observation read(const std::filesystem::path& path){
 std::ifstream in(path);if(!in)throw std::runtime_error("missing "+path.string());Observation out;std::string line;
 while(std::getline(in,line)){auto end=line.find('=');if(end==std::string::npos)continue;auto key=line.substr(0,end),value=line.substr(end+1);if(key=="elapsed_seconds")out.elapsed=std::stod(value);else if(key=="max_rss_kib")out.rss=std::stoll(value);else if(key=="exit_code")out.exit=std::stoi(value);}
 if(!(out.elapsed>0)||out.rss<=0||out.exit!=0){throw std::runtime_error("incomplete/failed resource "+path.string());}
 return out;
}
int main(int argc,char** argv){try{
 if(argc!=2){return 64;}
 const std::filesystem::path base=argv[1];std::ifstream status(base/"STATUS");int code=-1;status>>code;if(code!=0)throw std::runtime_error("campaign has not completed successfully");
 std::cout<<std::fixed<<std::setprecision(3);
 std::cout<<"density_percent\tpair\tdense_seconds\tcsr_seconds\tdense_rss_kib\tcsr_rss_kib\tdense_over_csr_time\n";
 for(int density:{1,10,100}){
  std::array<double,3> dense_times{},csr_times{};long long dense_peak=0,csr_peak=0;
  for(int pair=1;pair<=3;++pair){
   auto suffix=std::to_string(density)+"-pair"+std::to_string(pair)+".resource";
   auto dense=read(base/("dense-"+suffix)),csr=read(base/("csr-"+suffix));
   dense_times[pair-1]=dense.elapsed;csr_times[pair-1]=csr.elapsed;dense_peak=std::max(dense_peak,dense.rss);csr_peak=std::max(csr_peak,csr.rss);
   std::cout<<density<<'\t'<<pair<<'\t'<<dense.elapsed<<'\t'<<csr.elapsed<<'\t'<<dense.rss<<'\t'<<csr.rss<<'\t'<<dense.elapsed/csr.elapsed<<'\n';
  }
  std::sort(dense_times.begin(),dense_times.end());std::sort(csr_times.begin(),csr_times.end());
  std::cout<<"MEDIAN\t"<<density<<'\t'<<dense_times[1]<<'\t'<<csr_times[1]<<'\t'<<dense_peak<<'\t'<<csr_peak<<'\t'<<dense_times[1]/csr_times[1]<<'\n';
 }
 std::array<double,3> before_times{},after_times{};
 for(int pair=1;pair<=3;++pair){
  auto suffix="-pair"+std::to_string(pair)+".resource";auto before=read(base/("dense-before"+suffix)),after=read(base/("dense-after"+suffix));
  before_times[pair-1]=before.elapsed;after_times[pair-1]=after.elapsed;
  std::cout<<"DENSE_CORE_PAIR\t"<<pair<<'\t'<<before.elapsed<<'\t'<<after.elapsed<<'\t'<<before.rss<<'\t'<<after.rss<<'\t'<<before.elapsed/after.elapsed<<'\n';
 }
 std::sort(before_times.begin(),before_times.end());std::sort(after_times.begin(),after_times.end());
 std::cout<<"DENSE_CORE_MEDIAN\t"<<before_times[1]<<'\t'<<after_times[1]<<'\t'<<before_times[1]/after_times[1]<<'\n';
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
