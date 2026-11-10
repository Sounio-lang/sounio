// C++23 descriptive benchmark analyzer. Numerical work only inside Slurm.
// g++ -std=c++23 -O2 count_profile_benchmark_reference.cpp -o benchmark_reference
// benchmark_reference [--pilot] BEFORE_DRIVER AFTER_DRIVER RUNS.tsv COMPILER_ELF
// TSV header (literal tabs):
// variant trial stdout resource status elf source_sha256 elf_sha256 compiler_sha256 input_manifest input_manifest_sha256
// Pilot: one paired B31 run; full target: three paired B199 runs. No sampling,
// fitting, or replacement of Sounio numerical output occurs in this analyzer.
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
static void need(bool b,const char* m){if(!b)throw std::runtime_error(m);}
static std::string read(const std::string& p){std::ifstream f(p,std::ios::binary);need(bool(f),"input file open");std::ostringstream s;s<<f.rdbuf();need(!f.bad(),"input file read");return s.str();}
static std::vector<std::string> split(const std::string& s,char delimiter){std::vector<std::string> v;std::stringstream f(s);std::string x;while(std::getline(f,x,delimiter))v.push_back(x);return v;}
static long long integer(const std::string& s){size_t n=0;auto v=std::stoll(s,&n);need(n==s.size(),"whole integer field");return v;}
static long double real(const std::string& s){size_t n=0;auto v=std::stold(s,&n);need(n==s.size()&&std::isfinite(v),"finite resource/scientific field");return v;}
static bool digest(const std::string& s){return s.size()==64&&std::all_of(s.begin(),s.end(),[](unsigned char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');});}
// Metadata integrity only. exec has no shell interpolation or path-to-code step.
static std::string sha(const std::string& path){int fd[2];need(pipe(fd)==0,"hash pipe");pid_t pid=fork();need(pid>=0,"hash fork");if(pid==0){close(fd[0]);if(dup2(fd[1],STDOUT_FILENO)<0)_exit(126);close(fd[1]);execl("/usr/bin/sha256sum","sha256sum","--",path.c_str(),static_cast<char*>(nullptr));_exit(127);}close(fd[1]);std::string output;char b[1024];ssize_t n;while((n=::read(fd[0],b,sizeof b))>0){output.append(b,size_t(n));need(output.size()<=65536,"bounded hash output");}close(fd[0]);int status=0;need(n==0&&waitpid(pid,&status,0)==pid&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"hash command success");need(output.size()>=66&&digest(output.substr(0,64)),"hash format");return output.substr(0,64);}
static std::string source_line(const std::string& source,const std::string& token){std::istringstream in(source);std::string line,found;int count=0;while(std::getline(in,line))if(line.find(token)!=std::string::npos){found=line;count++;}need(count==1,"unique source scientific contract");return found;}
struct Scientific{long long index=0,status=0,hits=0,completed=0,failed=0,stage=0,hi=0,lo=0;long double p=0;std::string result_line,p_line;};
static Scientific scientific(const std::string& path,int B){Scientific r;bool result=false,p=false,complete=false;std::istringstream in(read(path));std::string line;
 while(std::getline(in,line)){
  if(line.starts_with("CAL_RESULT ")){need(!result,"unique CAL_RESULT");result=true;r.result_line=line;std::istringstream f(line.substr(11));f>>r.index>>r.status>>r.hits>>r.completed>>r.failed>>r.stage>>r.hi>>r.lo;need(bool(f),"CAL_RESULT field parse");std::string extra;need(!(f>>extra),"exact CAL_RESULT arity");}
  else if(line.starts_with("CAL_P ")){need(!p,"unique CAL_P");p=true;r.p_line=line;std::istringstream f(line.substr(6));long long index;std::string value,extra;f>>index>>value;need(bool(f)&&index==0&&!(f>>extra),"CAL_P fields");r.p=real(value);}
  else if(line=="COUNT PROFILE CALIBRATION CASE COMPLETE"){need(!complete,"unique case completion");complete=true;}
  else need(line.empty(),"unknown native output: counters must be separately declared");
 }
 need(result&&p&&complete,"complete actual scientific output, no successful prefix acceptance");
 need(r.index==0&&r.status==0&&r.completed==B&&r.failed==-1&&r.stage==0&&r.hits>=0&&r.hits<=B,"available complete case0 bootstrap");
 long double want=static_cast<long double>(r.hits+1)/static_cast<long double>(B+1);need(std::fabs(r.p-want)<=2e-14L*want,"plus-one tail fraction");return r;
}
struct Resource{long double wall=0;long long rss=0;long long exit=-1;};
static long double duration(const std::string& x){auto f=split(x,':');need(f.size()>=1&&f.size()<=3,"wall duration fields");long double value=0;for(auto& part:f)value=value*60+real(part);return value;}
static Resource resource(const std::string& path){Resource r;bool wall=false,rss=false,exit=false;std::istringstream in(read(path));std::string line;
 while(std::getline(in,line)){auto last=line.rfind(": ");
  if(line.starts_with("elapsed_seconds=")){need(!wall,"unique wall");r.wall=real(line.substr(16));wall=true;}
  else if(line.starts_with("max_rss_kib=")){need(!rss,"unique RSS");r.rss=integer(line.substr(12));rss=true;}
  else if(line.starts_with("exit_code=")){need(!exit,"unique resource exit");r.exit=integer(line.substr(10));exit=true;}
  else if(line.find("Elapsed (wall clock) time")!=std::string::npos){need(!wall&&last!=std::string::npos,"GNU wall");r.wall=duration(line.substr(last+2));wall=true;}
  else if(line.find("Maximum resident set size (kbytes)")!=std::string::npos){need(!rss&&last!=std::string::npos,"GNU RSS");r.rss=integer(line.substr(last+2));rss=true;}
  else if(line.find("Exit status")!=std::string::npos){need(!exit&&last!=std::string::npos,"GNU exit");r.exit=integer(line.substr(last+2));exit=true;}
 }
 need(wall&&rss&&exit&&r.wall>0&&r.rss>0&&r.exit==0,"accepted finite timed native process");return r;
}
static long double median(std::vector<long double> x){need(!x.empty()&&x.size()%2==1,"odd median sample");std::sort(x.begin(),x.end());return x[x.size()/2];}
int main(int argc,char** argv){try{
 bool pilot=argc>1&&std::string(argv[1])=="--pilot";int first=pilot?2:1;need(argc==first+4,"usage");int B=pilot?31:199,N=pilot?1:3;
 std::array<std::string,2> source_paths{argv[first],argv[first+1]},sources{read(argv[first]),read(argv[first+1])};std::array<std::string,2> source_hash{sha(source_paths[0]),sha(source_paths[1])};
 for(const char* token:{"let data:[i64;4096]=","let options=CountProfileBootstrapOptions{","let initial_rng=count_rng("})need(source_line(sources[0],token)==source_line(sources[1],token),"same dataset, options and seed schedule");
 for(const auto& source:sources){need(source.find("let n:i64=256")!=std::string::npos,"n256 source contract");auto options=source_line(source,"let options=CountProfileBootstrapOptions{");need(options.find("replicates:"+std::to_string(B)+",")!=std::string::npos,"declared pilot/full B");need(options.find("max_profile_attempts:"+std::to_string(pilot?64:400)+",")!=std::string::npos,"declared exact profile budget");}
 auto compiler_sha=sha(argv[first+3]);std::ifstream in(argv[first+2]);need(bool(in),"runs manifest open");std::string line;need(bool(std::getline(in,line))&&line=="variant\ttrial\tstdout\tresource\tstatus\telf\tsource_sha256\telf_sha256\tcompiler_sha256\tinput_manifest\tinput_manifest_sha256","manifest schema");
 Resource rows[2][3]{};Scientific science[2][3]{};bool seen[2][3]{};std::array<std::string,2> closure{},elf_hash{};int records=0;
 while(std::getline(in,line)){auto f=split(line,'\t');need(f.size()==11,"eleven manifest fields");int v=f[0]=="before"?0:f[0]=="after"?1:-1;need(v>=0,"variant");long long trial=integer(f[1]);need(trial>=1&&trial<=N,"bounded trial");int t=int(trial)-1;need(!seen[v][t],"unique trial");need(digest(f[6])&&digest(f[7])&&digest(f[8])&&digest(f[10]),"manifest SHA256 syntax");need(f[6]==source_hash[v]&&f[8]==compiler_sha,"source/compiler binding");need(sha(f[5])==f[7]&&sha(f[9])==f[10],"actual ELF/input-manifest identities");if(closure[v].empty()){closure[v]=f[10];elf_hash[v]=f[7];}else need(closure[v]==f[10]&&elf_hash[v]==f[7],"same compile once per variant");std::istringstream status(read(f[4]));long long rc;std::string extra;status>>rc;need(bool(status)&&rc==0&&!(status>>extra),"native status0");rows[v][t]=resource(f[3]);science[v][t]=scientific(f[2],B);seen[v][t]=true;records++;}
 need(records==2*N,"all paired trials present");for(int v=0;v<2;v++)for(int t=0;t<N;t++){need(seen[v][t],"complete trial grid");need(science[v][t].result_line==science[0][0].result_line&&science[v][t].p_line==science[0][0].p_line,"actual hits, p, completion, failure state and final RNG identical across all runs");}
 std::cout<<std::setprecision(21)<<"mode\t"<<(pilot?"pilot_B31_one_pair":"full_B199_three_pairs")<<"\nscientific_rows_identical\ttrue\n"<<science[0][0].result_line<<'\n'<<science[0][0].p_line<<"\ncounters_available\tfalse (driver emits final RNG/hits/B only; no uniform/proposal totals asserted)\ntrial\tbefore_seconds\tafter_seconds\tbefore_over_after\tbefore_rss_kib\tafter_rss_kib\n";
 std::vector<long double> b,a,ratio,br,ar;
 for(int t=0;t<N;t++){auto before=rows[0][t],after=rows[1][t];b.push_back(before.wall);a.push_back(after.wall);ratio.push_back(before.wall/after.wall);br.push_back(before.rss);ar.push_back(after.rss);std::cout<<t+1<<'\t'<<before.wall<<'\t'<<after.wall<<'\t'<<ratio.back()<<'\t'<<before.rss<<'\t'<<after.rss<<'\n';}
 if(pilot)std::cout<<"full_target_assessment\tunavailable (pilot is one B31 pair, not the B199 target)\n";
 else{auto before=median(b),after=median(a);auto before_max=*std::max_element(br.begin(),br.end()),after_max=*std::max_element(ar.begin(),ar.end());std::cout<<"median_before_seconds\t"<<before<<"\nmedian_after_seconds\t"<<after<<"\nratio_of_medians_before_over_after\t"<<before/after<<"\nmedian_paired_before_over_after\t"<<median(ratio)<<"\nmedian_before_rss_kib\t"<<median(br)<<"\nmedian_after_rss_kib\t"<<median(ar)<<"\nmax_before_rss_kib\t"<<before_max<<"\nmax_after_rss_kib\t"<<after_max<<"\ndescriptive_time_target_at_least2x\t"<<(before/after>=2?"true":"false")<<"\ndescriptive_maxrss_target_at_most5percent\t"<<(after_max<=before_max*1.05L?"true":"false")<<'\n';}
 std::cout<<"COUNT_PROFILE_BENCHMARK_ANALYSIS_COMPLETE\nLIMITS\tsource-bound whole-program observation; no assessed inferential validity, causal superiority, isolated-kernel timing or population performance guarantee\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
