// C++23 descriptive phase summary. Numerical execution belongs on t560.
// No confidence intervals or population speedup claims from three paired trials.
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
using U=std::uint64_t;
using Key=std::tuple<std::string,int,std::string>;
struct Row{
 std::string kind,phase,source,compiler,inputs,runtime,elf;
 int trial;double wall,user,system,rss,cpu=0.0;U entry_bytes,elf_bytes;
};
[[noreturn]] void fail(const std::string&s){throw std::runtime_error(s);}
std::vector<std::string> split(const std::string&s){
 std::vector<std::string> out;std::size_t p=0;
 while(true){auto q=s.find('\t',p);out.push_back(s.substr(p,q==s.npos?q:q-p));if(q==s.npos)return out;p=q+1;}
}
U integer(const std::string&s){
 U n=0;auto [p,e]=std::from_chars(s.data(),s.data()+s.size(),n);
 if(e!=std::errc{}||p!=s.data()+s.size()||s.empty())fail("invalid unsigned integer: "+s);
 return n;
}
double number(const std::string&s){
 std::size_t n=0;double v;
 try{v=std::stod(s,&n);}catch(...){fail("invalid measurement: "+s);}
 if(n!=s.size()||!std::isfinite(v)||v<0)fail("nonfinite/negative measurement: "+s);
 return v;
}
void hash(const std::string&s){
 if(s.size()!=64||!std::all_of(s.begin(),s.end(),[](unsigned char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F');}))fail("invalid SHA256: "+s);
}
struct Summary{double lo,median,hi;};
Summary summarize(std::vector<double> v){
 if(v.empty())fail("empty summary");
 std::sort(v.begin(),v.end());
 return{v.front(),v[v.size()/2],v.back()};
}
void compare(const std::string&scope,const std::string&phase,const std::string&metric,double before,double after,bool time){
 std::cout<<scope<<'\t'<<phase<<'\t'<<metric<<'\t'<<before<<'\t'<<after<<'\t'<<after-before<<'\t';
 if(before<=0||(time&&after==0)){
  std::cout<<"NA\tNA\t"<<(time?"below_time_resolution":"zero_baseline")<<'\n';return;
 }
 std::cout<<after/before<<'\t'<<100*(before-after)/before<<"\tdescriptive_only\n";
}
int main(int argc,char**argv){
 try{
  if(argc!=2)fail("usage: stats_viz_phase_summary METRICS_TSV");
  std::ifstream in(argv[1]);if(!in)fail("cannot open input");
  const std::string expected="kind\ttrial\tphase\twall_s\tuser_s\tsystem_s\tmax_rss_kib\trc\taccepted\tentry_bytes\telf_bytes\tsource_sha256\tcompiler_sha256\tinputs_sha256\truntime_receipt_sha256\tslurm_job_id\telf_sha256\tstdout_sha256\tstderr_sha256";
  std::string line;if(!std::getline(in,line)||line!=expected)fail("header differs from exact19-column schema");
  std::map<Key,Row> rows;std::string compiler,inputs,runtime;
  std::map<std::string,std::pair<U,std::string>> sources;
  std::map<std::pair<std::string,int>,std::pair<U,std::string>> trial_elf;
  std::map<std::string,std::set<U>> elf_sizes;
  std::map<std::string,std::set<std::string>> elf_hashes;
  int lineno=1;
  while(std::getline(in,line)){
   ++lineno;auto v=split(line);if(v.size()!=19)fail("wrong field count at line"+std::to_string(lineno));
   if(v[0]!="before"&&v[0]!="after"&&v[0]!="viz")fail("unknown kind");
   U trial=integer(v[1]);if(trial<1||trial>(v[0]=="viz"?1:3))fail("trial outside required range");
   if(v[2]!="compile"&&v[2]!="run"&&v[2]!="run_batch20")fail("unknown phase");
   if(v[7]!="0"||v[8]!="1")fail("failed/unaccepted measurement at line"+std::to_string(lineno));
   for(int j:{11,12,13,14,16,17,18})hash(v[j]);
   if(integer(v[15])==0)fail("invalid Slurm job identity");
   Row r{v[0],v[2],v[11],v[12],v[13],v[14],v[16],int(trial),number(v[3]),number(v[4]),number(v[5]),number(v[6]),0.0,integer(v[9]),integer(v[10])};
   r.cpu=r.user+r.system;
   if(!std::isfinite(r.cpu))fail("nonfinite combined CPU time");
   if(r.rss<=0)fail("max_rss_kib is zero/unavailable; complete comparative report rejected");
   if(!r.entry_bytes||!r.elf_bytes)fail("empty source or executable");
   if(rows.empty()){compiler=r.compiler;inputs=r.inputs;runtime=r.runtime;}
   if(r.compiler!=compiler||r.inputs!=inputs||r.runtime!=runtime)fail("compiler/input/runtime identity mismatch");
   auto [src,inserted]=sources.emplace(r.kind,std::pair{r.entry_bytes,r.source});
   if(!inserted&&src->second!=std::pair{r.entry_bytes,r.source})fail("source bytes/hash vary within kind");
   auto [elf,fresh]=trial_elf.emplace(std::pair{r.kind,r.trial},std::pair{r.elf_bytes,r.elf});
   if(!fresh&&elf->second!=std::pair{r.elf_bytes,r.elf})fail("ELF identity varies between phases of one trial");
   elf_sizes[r.kind].insert(r.elf_bytes);elf_hashes[r.kind].insert(r.elf);
   if(!rows.emplace(Key{r.kind,r.trial,r.phase},r).second)fail("duplicate kind/trial/phase");
  }
  for(const std::string kind:{"before","after","viz"})
   for(int t=1;t<=(kind=="viz"?1:3);t++)
    for(const std::string phase:{"compile","run","run_batch20"})
     if(!rows.contains(Key{kind,t,phase}))fail("missing "+kind+"/"+std::to_string(t)+"/"+phase);
  if(rows.size()!=21)fail("expected exactly21 accepted rows");
  std::cout<<std::fixed<<std::setprecision(6);
  std::cout<<"# Descriptive paired measurements:3pairs,onehost,no confidence interval or generalized speedup claim.\n"
   <<"# GNU time resolution0.01s; run_batch20 includes20 ELF executions plus helper/guard overhead.\n"
   <<"# Batch per-execution wall is batch total/20, not isolated kernel latency.\n"
   <<"# max_rss_kib is GNU time largest-process/descendant RSS, not aggregate process-tree memory.\n"
   <<"# Compile measurements include canonical compiler integrity work; batch measurements include helper/guard overhead.\n"
   <<"# Static stack ceilings are distinct from measured RSS and are not inferred by this utility.\n";
  std::cout<<"identity\tcompiler_sha256\t"<<compiler<<"\nidentity\tinputs_sha256\t"<<inputs<<"\nidentity\truntime_receipt_sha256\t"<<runtime<<'\n';
  std::cout<<"artifact\tkind\tsource_bytes\tsource_sha256\telf_bytes_min\telf_bytes_max\telf_hash_variants\tflag\n";
  for(const std::string kind:{"before","after","viz"}){
   const auto& sizes=elf_sizes.at(kind);
   bool varies=sizes.size()>1||elf_hashes.at(kind).size()>1;
   std::cout<<"artifact\t"<<kind<<'\t'<<sources.at(kind).first<<'\t'<<sources.at(kind).second<<'\t'<<*sizes.begin()<<'\t'<<*sizes.rbegin()<<'\t'<<elf_hashes.at(kind).size()<<'\t'<<(varies?"executable_variation_between_trials":"stable")<<'\n';
  }
  std::cout<<"summary\tkind\tphase\tmetric\tmin\tmedian\tmax\tunit\tnote\n";
  for(const std::string kind:{"before","after","viz"})for(const std::string phase:{"compile","run","run_batch20"}){
   std::vector<double>w,rss,cpu;for(int t=1;t<=(kind=="viz"?1:3);t++){const auto&r=rows.at(Key{kind,t,phase});w.push_back(r.wall);rss.push_back(r.rss);cpu.push_back(r.cpu);}
   for(bool memory:{false,true}){auto s=summarize(memory?rss:w);
    std::cout<<"summary\t"<<kind<<'\t'<<phase<<'\t'<<(memory?"max_rss":"wall")<<'\t'<<s.lo<<'\t'<<s.median<<'\t'<<s.hi<<'\t'<<(memory?"KiB":"s")<<'\t'<<(!memory&&s.lo==0?"includes_below_resolution":"descriptive_only")<<'\n';
   }
   auto cs=summarize(cpu);
   std::cout<<"summary\t"<<kind<<'\t'<<phase<<"\tcpu_s\t"<<cs.lo<<'\t'<<cs.median<<'\t'<<cs.hi<<"\ts\t"<<(cs.lo==0?"includes_below_resolution":"user_plus_system")<<'\n';
   if(phase=="run_batch20"){auto s=summarize(w);std::cout<<"summary\t"<<kind<<'\t'<<phase<<"\twall_per_exec\t"<<s.lo/20<<'\t'<<s.median/20<<'\t'<<s.hi/20<<"\ts\tbatch_overhead_included\n";}
  }
  std::cout<<"comparison\tphase\tmetric\tbefore\tafter\tafter_minus_before\tafter_over_before\treduction_percent\tnote\n";
  for(const std::string phase:{"compile","run","run_batch20"}){
   std::vector<double> bw,aw,br,ar,bc,ac;
   for(int t=1;t<=3;t++){
    const auto& b=rows.at(Key{"before",t,phase});const auto&a=rows.at(Key{"after",t,phase});
    bw.push_back(b.wall);aw.push_back(a.wall);br.push_back(b.rss);ar.push_back(a.rss);bc.push_back(b.cpu);ac.push_back(a.cpu);
    compare("pair"+std::to_string(t),phase,"wall_s",b.wall,a.wall,true);
    compare("pair"+std::to_string(t),phase,"max_rss_KiB",b.rss,a.rss,false);
    compare("pair"+std::to_string(t),phase,"cpu_s",b.cpu,a.cpu,true);
   }
   compare("median",phase,"wall_s",summarize(bw).median,summarize(aw).median,true);
   compare("median",phase,"max_rss_KiB",summarize(br).median,summarize(ar).median,false);
   compare("median",phase,"cpu_s",summarize(bc).median,summarize(ac).median,true);
  }
  compare("artifact","source","bytes",double(sources.at("before").first),double(sources.at("after").first),false);
  std::vector<double> be,ae;for(int t=1;t<=3;t++){be.push_back(double(rows.at(Key{"before",t,"compile"}).elf_bytes));ae.push_back(double(rows.at(Key{"after",t,"compile"}).elf_bytes));}
  compare("artifact","executable","median_bytes",summarize(be).median,summarize(ae).median,false);
  std::cout<<"# Byte comparisons above use before/after values; ELF variation is flagged, not hidden.\n";
  return 0;
 }catch(const std::exception&e){std::cerr<<"REJECTED: "<<e.what()<<'\n';return 2;}
}
