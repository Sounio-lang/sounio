// Independent C++23/binary128 verification of two actual 256-row pilots on
// a full20k x512 holder. This is NOT the complete20k fit-coverage checker.
#include <quadmath.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
using Q=__float128;
static void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static Q number(const std::string& token){char* end=nullptr;Q v=strtoflt128(token.c_str(),&end);need(end!=token.c_str()&&*end=='\0'&&!isnanq(v)&&!isinfq(v),"finite complete numeric token");return v;}
static Q q(const char* token){return number(token);}
static void emit(const char* name,Q value){char text[160];quadmath_snprintf(text,sizeof text,"%.35Qg",value);std::cout<<name<<'\t'<<text<<'\n';}
struct Reader{
 std::ifstream file;
 explicit Reader(const char* path):file(path){need(bool(file),"actual native report open");}
 std::string line(){std::string s;need(bool(std::getline(file,s)),"complete actual native output");return s;}
 void exact(const std::string& wanted){need(line()==wanted,"native record schema/order");}
};
struct Pilot{std::array<Q,256> beta0{},beta1{};long long evaluations=0;};
static Pilot read_pilot(const char* path){
 Reader r(path);r.exact("SCALE_LAYOUT 20000 512 2 10506992");Pilot result;
 for(int i=0;i<256;++i){
  r.exact("SCALE_POINT "+std::to_string(i)+" 0");
  need(number(r.line())==0,"actual Poisson boundary alpha");
  Q b0=number(r.line()),b1=number(r.line());
  Q expected=logq(Q(i%7+1)*sqrtq(Q(2)));
  need(fabsq(b0-expected)<=q("2e-8")+q("2e-8")*fabsq(expected),"independent normalized intercept");
  need(fabsq(b1)<=q("1e-7"),"independent normalized group contrast");
  result.beta0[static_cast<size_t>(i)]=b0;result.beta1[static_cast<size_t>(i)]=b1;
 }
 auto work=r.line();const std::string prefix="SCALE_EVALUATIONS ";need(work.starts_with(prefix),"native actual work receipt");
 std::string count=work.substr(prefix.size());size_t end=0;result.evaluations=std::stoll(count,&end);need(end==count.size(),"complete work count");
 need(result.evaluations>=2560&&result.evaluations<=16384,"declared per-profile work bounds");
 r.exact("GENOMICS_BATCH_SCALE_PASS");std::string extra;need(!std::getline(r.file,extra),"no trailing native rows");return result;
}
struct Metrics{Q elapsed=0,rss=0,user=0;bool wall=false,peak=false,cpu=false,success=false;};
static Metrics read_metrics(const char* path){
 std::ifstream input(path);need(bool(input),"actual native timing open");Metrics result;std::string line;
 while(std::getline(input,line)){
  const std::string wall="Elapsed (wall clock) time (h:mm:ss or m:ss): ";
  const std::string rss="Maximum resident set size (kbytes): ";
  const std::string cpu="User time (seconds): ";
  if(auto at=line.find(wall);at!=std::string::npos){
   need(!result.wall,"single native elapsed receipt");result.wall=true;
   std::istringstream fields(line.substr(at+wall.size()));std::string value;need(bool(fields>>value),"elapsed value");
   std::replace(value.begin(),value.end(),':',' ');std::istringstream parts(value);std::string token;
   while(parts>>token)result.elapsed=result.elapsed*60+number(token);
  }else if(auto at=line.find(rss);at!=std::string::npos){need(!result.peak,"single native peak receipt");result.peak=true;result.rss=number(line.substr(at+rss.size()));}
  else if(auto at=line.find(cpu);at!=std::string::npos){need(!result.cpu,"single user-time receipt");result.cpu=true;result.user=number(line.substr(at+cpu.size()));}
  else if(line.find("Exit status: 0")!=std::string::npos)result.success=true;
 }
 need(result.wall&&result.peak&&result.cpu&&result.success&&result.elapsed>0&&result.rss>0,"complete successful native metrics");return result;
}
int main(int argc,char**argv){try{
 if(argc!=5)return 64;
 auto before=read_pilot(argv[1]),after=read_pilot(argv[2]);
 auto base=read_metrics(argv[3]),seed=read_metrics(argv[4]);Q delta0=0,delta1=0;
 for(size_t i=0;i<256;++i){delta0=fmaxq(delta0,fabsq(before.beta0[i]-after.beta0[i]));delta1=fmaxq(delta1,fabsq(before.beta1[i]-after.beta1[i]));}
 emit("maximum_observed_beta0_delta",delta0);emit("maximum_observed_beta1_delta",delta1);
 emit("unseeded_actual_objective_evaluations",Q(before.evaluations));emit("seeded_actual_objective_evaluations",Q(after.evaluations));
 emit("unseeded_native_elapsed_seconds",base.elapsed);emit("seeded_native_elapsed_seconds",seed.elapsed);
 emit("seeded_over_unseeded_elapsed_ratio",seed.elapsed/base.elapsed);
 emit("unseeded_peak_KiB",base.rss);emit("seeded_peak_KiB",seed.rss);
 emit("seeded_over_unseeded_peak_ratio",seed.rss/base.rss);
 std::cout<<"SCOPE full20k_holder_actual256_point_profiles_only;two_observations;no_full20k_fit_or_calibration_claim\n";
 std::cout<<"SIO171_PILOT256_NUMERICAL_AND_METRICS_PASS\n";return 0;
}catch(const std::exception& error){std::cerr<<"SIO171_PILOT256_REJECT "<<error.what()<<'\n';return 2;}}
