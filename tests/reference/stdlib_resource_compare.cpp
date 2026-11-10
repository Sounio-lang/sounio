// C++23 analysis only: medians/ratios from GNU time, no numerical model oracle.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
constexpr long long batch_cap=6400;
const std::string header="kernel\tvariant\ttrial\tbatch\twall_s\tuser_s\tsystem_s\tmax_rss_kib\texit_status\tstdout_sha256\telf_sha256\tcompiler_sha256\tfixture_sha256";
std::vector<std::string> split(const std::string&s){std::vector<std::string>v;std::istringstream in(s);
 std::string x;while(std::getline(in,x,'\t'))v.push_back(x);
 return v;}
long long integer(const std::string&s){std::size_t p=0;auto x=std::stoll(s,&p);
  if(p!=s.size()){throw std::runtime_error("invalid integer");}return x;}
double number(const std::string&s){std::size_t p=0;auto x=std::stod(s,&p);
  if(p!=s.size()||!std::isfinite(x)||x<0){throw std::runtime_error("invalid nonnegative finite timing");}return x;}
bool sha(const std::string&s){return s.size()==64&&std::all_of(s.begin(),s.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});}
double median(std::vector<double>v){if(v.size()!=3){throw std::runtime_error("exactly three paired trials required");}std::sort(v.begin(),v.end());
 return v[1];}
struct Row{long long batch;double wall,cpu,rss;
 std::string output,elf,compiler,fixture;};
int report(const char*path){
 std::ifstream in(path);
  if(!in){throw std::runtime_error("metrics missing");}
 std::string s;std::getline(in,s);
  if(s!=header){throw std::runtime_error("metrics header mismatch");}
 std::map<std::string,std::map<std::string,std::map<int,Row>>> rows;
 std::string common_compiler;
 while(std::getline(in,s)){
  const auto f=split(s);
  if(f.size()!=13){throw std::runtime_error("wrong metrics column count");}
  if(f[0]!="qr"&&f[0]!="count_glm"){throw std::runtime_error("unknown kernel");}
  if(f[1]!="before"&&f[1]!="after"){throw std::runtime_error("unknown variant");}
  const auto trial=integer(f[2]),batch=integer(f[3]);
  if(trial<1||trial>3||batch<1||batch>batch_cap){throw std::runtime_error("trial/batch range");}
  if(f[0]=="qr"&&batch!=1){throw std::runtime_error("QR must be batch1");}
  const auto wall=number(f[4]),cpu=number(f[5])+number(f[6]),rss=number(f[7]);
  if(!std::isfinite(cpu)||wall<=0||rss<=0||integer(f[8])!=0){throw std::runtime_error("failed or below-resolution comparative row");}
  if(f[0]=="count_glm"&&wall<1){throw std::runtime_error("CountGLM measured batch below one second");}
  for(int j=9;j<13;++j)if(!sha(f[j])){throw std::runtime_error("invalid SHA256");}
  if(common_compiler.empty())common_compiler=f[11];
  if(common_compiler!=f[11]){throw std::runtime_error("compiler identity changed");}
  auto&dst=rows[f[0]][f[1]];
  if(dst.contains(int(trial))){throw std::runtime_error("duplicate comparative row");}
  dst.emplace(int(trial),Row{batch,wall,cpu,rss,f[9],f[10],f[11],f[12]});
 }
 bool passed=true;std::cout<<std::setprecision(12);
 std::cout<<"scope\tthree paired trials on one worker; descriptive only, no CI or universal speed claim\n";
 std::cout<<"timing\tGNU time centisecond wall/CPU resolution; batch includes shell+exec/check overhead\n";
 std::cout<<"rss\tlargest process/descendant maximum, not aggregate; batch driver can dominate small programs\n";
 std::cout<<"compiler_sha256\t"<<common_compiler<<'\n';
 for(const std::string kernel:{"qr","count_glm"}){
  std::array<std::vector<double>,2>walls,cpus,rsss;long long common_batch=0;
 std::string output,fixture;
  for(int v=0;v<2;++v){const std::string name=v==0?"before":"after";auto&data=rows[kernel][name];
  if(data.size()!=3){throw std::runtime_error("incomplete required rows");}
 std::string elf;
   for(int t=1;t<=3;++t){if(!data.contains(t)){throw std::runtime_error("missing trial");}const auto&r=data.at(t);
    if(common_batch==0)common_batch=r.batch;
  if(common_batch!=r.batch){throw std::runtime_error("batch changed across variants/trials");}
    if(output.empty())output=r.output;
  if(output!=r.output){throw std::runtime_error("stdout concordance differs");}
    if(fixture.empty())fixture=r.fixture;
  if(fixture!=r.fixture){throw std::runtime_error("fixture bytes differ");}
    if(elf.empty())elf=r.elf;
  if(elf!=r.elf){throw std::runtime_error("native ELF changed within variant");}
    walls[v].push_back(r.wall/double(r.batch));cpus[v].push_back(r.cpu/double(r.batch));rsss[v].push_back(r.rss);
   }
  }
  const double wb=median(walls[0]),wa=median(walls[1]),cb=median(cpus[0]),ca=median(cpus[1]),rb=median(rsss[0]),ra=median(rsss[1]);
  const double wr=wa/wb,rr=ra/rb;const bool ok=wr<=1.10&&rr<=1.05;passed=passed&&ok;
  std::cout<<"summary\t"<<kernel<<"\tbatch\t"<<common_batch<<"\twall_before_s_per_exec\t"<<wb<<"\twall_after_s_per_exec\t"<<wa<<"\twall_ratio\t"<<wr<<"\tcpu_before_s_per_exec\t"<<cb<<"\tcpu_after_s_per_exec\t"<<ca<<"\trss_before_kib\t"<<rb<<"\trss_after_kib\t"<<ra<<"\trss_ratio\t"<<rr<<"\tstatus\t"<<(ok?"PASS":"REGRESSION_REVIEW_REQUIRED")<<'\n';
  for(int t=1;t<=3;++t){const auto&b=rows[kernel]["before"].at(t);const auto&a=rows[kernel]["after"].at(t);std::cout<<"pair\t"<<kernel<<'\t'<<t<<"\twall_ratio\t"<<a.wall/b.wall<<"\trss_ratio\t"<<a.rss/b.rss<<"\tcpu_ratio\t";
  if(b.cpu>0)std::cout<<a.cpu/b.cpu;else std::cout<<"below_resolution";std::cout<<'\n';}
 }
 std::cout<<"RESOURCE_COMPARE_"<<(passed?"PASS":"REGRESSION_REVIEW_REQUIRED")<<'\n';
 return passed?0:1;
}
}
int main(int argc,char**argv){try{
 if(argc==5&&std::string(argv[1])=="--next-batch"){
  const auto a=number(argv[2]),b=number(argv[3]);const auto current=integer(argv[4]);
  if(current<25||current>batch_cap){throw std::runtime_error("invalid calibration batch");}
  if(a>=1&&b>=1){std::cout<<current<<'\n';
 return 0;}
  if(current>=batch_cap){throw std::runtime_error("calibration cap exhausted below one second");}
  std::cout<<std::min(batch_cap,current*2)<<'\n';
 return 0;
 }
 if(argc==2)return report(argv[1]);
 std::cerr<<"usage: stdlib_resource_compare METRICS | --next-batch WALL_BEFORE WALL_AFTER CURRENT\n";
 return 2;
 }catch(const std::exception&e){std::cerr<<"RESOURCE_COMPARE_REFUSE "<<e.what()<<'\n';
 return 2;}}
