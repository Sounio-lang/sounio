// Independent selected far-fallback audit; C++23 execution only under Slurm.
// FULL finite-product PMFs are inherited from the canonical independent oracle,
// not from the production local increment or cache implementation.
#define main saved_dispersion_reference_main
#include "count_dispersion_increment_reference.cpp"
#undef main
#include <array>
#include <fstream>
#include <sstream>
struct FarCase{int count;double m0,a0,m1,a1;int status;};
int main(int argc,char** argv){try{
 need(argc==2,"actual far native stdout required");
 const std::array<FarCase,8> cases{{{4,4,.5,40,.75,0},{4,5,.5,50,.75,0},{4,40,.75,4,.5,0},{63,4,.5,40,.75,0},{64,4,.5,40,.75,0},{1,1,1e-200,1,2e-200,5},{0,2,0,3,0,0},{0,3,0,2,0,0}}};
 std::ifstream in(argv[1]);need(bool(in),"native open");std::string line;
 for(std::size_t i=0;i<cases.size();++i){
  need(bool(std::getline(in,line)),"complete far rows");std::istringstream f(line);
  std::string tag,extra;long long index,status,bits,first,second,local,calls;
  f>>tag>>index>>status>>bits>>first>>second>>local>>calls;
  need(bool(f)&&!(f>>extra)&&tag=="CACHED_FAR"&&index==static_cast<long long>(i),"far row format/order");
  auto c=cases[i];need(status==c.status&&first==0&&second==0&&local==0&&calls==2,"far status/full metadata/calls");
  const double native=std::bit_cast<double>(std::bit_cast<std::uint64_t>(bits));
  B want=pmf(c.count,exact(c.m1),exact(c.a1))-pmf(c.count,exact(c.m0),exact(c.a0));
  need(want!=0,"far control discriminates a real contrast");
  if(status==0){need(std::isfinite(native),"finite contrast");B got=exact(native);need(abs(got-want)<=abs(want)*B("1e-11"),"independent full-PMF far contrast");}
  else need(native==0,"unavailable rounded-zero payload");
  std::cout<<i<<'\t'<<std::setprecision(50)<<want<<'\t'<<native<<'\n';
 }
 need(bool(std::getline(in,line))&&line=="CHECKED CACHED RANK FAR PASS","far completion");
 while(std::getline(in,line))need(line.empty(),"no unknown trailing rows");
 std::cout<<"CACHED RANK FULL PMF FAR CONCORDANCE PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
