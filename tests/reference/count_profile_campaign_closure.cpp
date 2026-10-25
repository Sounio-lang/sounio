// Close an exhausted campaign from its original case receipts. Never fit or sample.
// C++23 is the numerical reference; missing executions remain in the planned family.
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
using I=long long;
static void require(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
static std::string read(const fs::path& p){std::ifstream f(p);require(bool(f),"open "+p.string());return {std::istreambuf_iterator<char>(f),{}};}
static std::vector<std::string> lines(const fs::path& p){std::ifstream f(p);require(bool(f),"open "+p.string());std::vector<std::string> v;std::string s;while(std::getline(f,s))v.push_back(s);return v;}
static I integer(const std::string& s){I n=0;auto [p,e]=std::from_chars(s.data(),s.data()+s.size(),n);require(e==std::errc{}&&p==s.data()+s.size(),"integer syntax");return n;}
static I scalar(const fs::path& p){std::istringstream s(read(p));std::string token,extra;s>>token;require(bool(s)&&!(s>>extra),"one integer "+p.string());return integer(token);}
static std::map<std::string,std::string> kv(const fs::path& p){std::map<std::string,std::string> m;for(const auto& l:lines(p)){auto at=l.find('\t');require(at!=std::string::npos,"tab pair "+p.string());require(m.emplace(l.substr(0,at),l.substr(at+1)).second,"duplicate key");}return m;}
template<std::size_t N> static std::array<I,N> numbers(const std::string& s){std::istringstream f(s);std::array<I,N> v{};for(auto& n:v)f>>n;require(bool(f),"numeric row syntax");std::string extra;require(!(f>>extra),"numeric row extra field");return v;}
struct Native{int status=-1;I hits=0,completed=0;std::array<I,5> work{};I trace_status=0;std::array<I,10> trace{};};
static Native native(const fs::path& p,int index){
 auto v=lines(p);require(v.size()==5,"native exact five-line record");
 require(v[0].starts_with("CAL_RESULT ")&&v[1].starts_with("CAL_P ")&&v[2].starts_with("CAL_WORK ")&&v[3].starts_with("CAL_TRACE "),"native row ordering");
 auto r=numbers<8>(v[0].substr(11));require(r[0]==index,"native dataset index");
 require(r[1]>=0&&r[1]<=12&&r[1]!=10,"native declared status");
 require(r[2]>=0&&r[2]<=r[3]&&r[3]<=199,"native bounded hits/completion");
 require(r[4]>=-1&&r[4]<199&&r[5]>=0&&r[5]<=8,"native failure metadata");
 if(r[1]==0)require(r[3]==199&&r[4]==-1&&r[5]==0,"native success metadata");
 std::istringstream pf(v[1].substr(6));I pi=-1;long double probability=-1;pf>>pi>>probability;require(bool(pf),"native p syntax");std::string extra;require(!(pf>>extra)&&pi==index&&std::isfinite(probability)&&probability>=0&&probability<=1,"native p context");
 if(r[1]==0)require(std::fabs(probability-static_cast<long double>(r[2]+1)/200)<=2e-14L,"native plus-one p");
 else require(probability==0,"unavailable p placeholder");
 auto w=numbers<6>(v[2].substr(9));require(w[0]==index,"work dataset index");
 auto t=numbers<12>(v[3].substr(10));require(t[0]==index&&t[1]>=0,"trace dataset/status");
 Native out;out.status=static_cast<int>(r[1]);out.hits=r[2];out.completed=r[3];out.trace_status=t[1];
 for(int j=0;j<5;j++){require(w[j+1]>=0,"nonnegative work");out.work[j]=w[j+1];}
 for(int j=0;j<10;j++){require(t[j+2]>=0,"nonnegative trace");out.trace[j]=t[j+2];}
 require(v[4]=="COUNT PROFILE CALIBRATION CASE COMPLETE","native completion marker");
 return out;
}
struct Cell{
 int n;std::string alpha;int generated=0,attempted=0,completed=0,available=0,rejected=0,unavailable=0,execution_failed=0,not_started=0,ready=0,trace_partial=0;
 std::map<I,int> exits,statuses;std::array<I,5> work{};std::array<I,10> trace{};
};
static void summary_matches(const fs::path& p,int total,int available,int rejected,const std::map<I,int>& statuses){
 const auto s=kv(p);require(integer(s.at("datasets"))==total&&integer(s.at("available"))==available&&integer(s.at("unavailable"))==total-available&&integer(s.at("rejected_available"))==rejected,"summary denominator/count mismatch");
 for(const auto& [st,count]:statuses)require(integer(s.at("status_"+std::to_string(st)))==count,"summary status mismatch");
}
static void report(const fs::path& root,const fs::path& out){
 const std::map<std::string,std::string> expected={{"mode","qualification"},{"entry_policy","nested_traced"},{"B","199"},{"batch_size","16"},{"serial_processes","1"},{"target_per_cell","128"},{"max_cumulative_wall_seconds","85500"}};
 require(kv(root/"PROTOCOL.tsv")==expected,"protocol/denominator mismatch");
 require(scalar(root/"BUDGET_USED_SECONDS")==85500&&scalar(root/"LAST_INVOCATION_STATUS")==124,"exhausted campaign status/budget");
 require(!fs::exists(root/"COMPLETE"),"incomplete campaign must not have COMPLETE marker");
 fs::create_directories(out);
 std::ofstream cases(out/"cases.tsv"),cells(out/"cells.tsv"),md(out/"REPORT.md");
 require(bool(cases)&&bool(cells)&&bool(md),"report output open");
 cases<<"n\talpha\tindex\tgeneration_seed\tbootstrap_seed\tgenerated\tprocess_exit\tclassification\tmodel_status\thits\treplicates_completed\ttrace_status\tfinal_rng_hi\tfinal_rng_lo\n";
 cells<<"n\talpha\tplanned\tgenerated\tattempted\tcompleted\tavailable\tmodel_unavailable\texecution_failed\tnot_started\trejected_available\tconditional_rejection\tconditional_mc_se\tconditional_wilson95_lower\tconditional_wilson95_upper\tplanned_unknown_lower\tplanned_unknown_upper\n"<<std::setprecision(18);
 md<<"# Exhausted NB2 qualification campaign\n\nCampaign job 17048 ended with process status 124 after exhausting the round allocation of 85,500 seconds. The preceding 900-second allocation is separate; no pilot or historical observations are pooled here. Frozen source: 6ffc1a5bd8eaebeef2d15ccced830bfbc832a648. Compiler SHA256: d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6.\n\nEvery original case status, native row and seed schedule is reconciled below, including successful cases from the unfinished batch. Generated but never executed cases and entirely ungenerated cases remain not started. Process failures are distinct from completed model refusals.\n\n| n | alpha | planned | generated | attempted | completed | available | model unavailable | process failures | not started | rejected/available | conditional Wilson 95% |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n"<<std::setprecision(9);
 int ci=0;I all_attempted=0,all_completed=0,all_available=0,all_unavailable=0,all_failed=0,all_not_started=0;
 for(int n:{16,64,256})for(const std::string alpha:{"0",".25","1"}){
  Cell c{};c.n=n;c.alpha=alpha;auto cell=root/("n"+std::to_string(n)+"_alpha"+alpha);require(fs::is_directory(cell),"missing declared cell");
  for(const auto& entry:fs::directory_iterator(cell)){
   auto name=entry.path().filename().string();
   if(!entry.is_directory())continue;
   require(name.starts_with("batch"),"unexpected cell subdirectory");
   auto token=name.substr(5);I first=integer(token);
   require(first>=0&&first<128&&first%16==0&&token==std::to_string(first),"unexpected batch index");
  }
  int ready_total=0,ready_avail=0,ready_reject=0;std::map<I,int> ready_status;
  std::set<int> generated;
  for(int first=0;first<128;first+=16){
   auto batch=cell/("batch"+std::to_string(first));
   if(!fs::exists(batch))continue;
   require(fs::is_directory(batch),"batch directory");
   auto ds=lines(batch/"datasets.tsv");require(ds.size()==17&&ds[0]=="index\tn\talpha\tB\tgeneration_seed\tbootstrap_seed","dataset metadata census");
   for(int k=0;k<16;k++){
    std::istringstream f(ds[k+1]);I idx,ns,b,gs,bs;std::string a,extra;f>>idx>>ns>>a>>b>>gs>>bs;
    require(bool(f)&&!(f>>extra),"dataset metadata syntax");
    require(idx==first+k&&ns==n&&a==(alpha==".25"?"0.25":alpha)&&b==199,"dataset metadata cell/index");
    require(gs==24000000+ci*1000+idx&&bs==34000000+ci*1000+idx,"dataset seed schedule");
    require(generated.insert(static_cast<int>(idx)).second,"duplicate dataset index");
   }
   for(const auto& entry:fs::directory_iterator(batch)){
    auto name=entry.path().filename().string();
    if(!name.starts_with("case-"))continue;
    auto dot=name.find('.',5);require(dot!=std::string::npos,"case filename");
    auto token=name.substr(5,dot-5);I k=integer(token);
    require(k>=0&&k<16&&token==std::to_string(k),"unexpected case index");
   }
   bool ready=fs::exists(batch/"READY");
   if(ready)require(scalar(batch/"native.status")==0&&scalar(batch/"compile.status")==0,"READY requires native/compile success");
   int bt=0,ba=0,br=0;std::map<I,int> bst;
   for(int k=0;k<16;k++){
    int idx=first+k;auto stem=batch/("case-"+std::to_string(k));auto status=fs::path(stem.string()+".status"),stdout_path=fs::path(stem.string()+".stdout");
    if(!fs::exists(status)){require(!ready,"READY omitted execution");continue;}
    I exit=scalar(status);c.attempted++;c.exits[exit]++;
    require(fs::exists(stdout_path),"attempt needs raw stdout");
    if(exit!=0){require(!ready,"READY failed execution");require(read(stdout_path).find("COUNT PROFILE CALIBRATION CASE COMPLETE")==std::string::npos,"failed process has completion marker");c.execution_failed++;continue;}
    Native r=native(stdout_path,idx);c.completed++;bt++;c.statuses[r.status]++;bst[r.status]++;
    if(r.status==0){c.available++;ba++;if(r.hits+1<=10){c.rejected++;br++;}}else c.unavailable++;
    c.trace_partial+=r.trace_status!=0;
    for(int j=0;j<5;j++){require(c.work[j]<=9223372036854775807LL-r.work[j],"work aggregate overflow");c.work[j]+=r.work[j];}
    for(int j=0;j<10;j++){require(c.trace[j]<=9223372036854775807LL-r.trace[j],"trace aggregate overflow");c.trace[j]+=r.trace[j];}
   }
   if(ready){
    require(bt==16,"READY exact sixteen executions");c.ready++;
    summary_matches(batch/"summary.tsv",bt,ba,br,bst);
    auto combined=lines(batch/"native.stdout");require(combined.size()==81&&combined.back()=="COUNT PROFILE CALIBRATION BATCH COMPLETE","READY aggregate completion");
    for(int k=0;k<16;k++){auto v=lines(batch/("case-"+std::to_string(k)+".stdout"));require(std::equal(v.begin(),v.end(),combined.begin()+5*k),"aggregate/case stdout equality");}
    ready_total+=bt;ready_avail+=ba;ready_reject+=br;for(auto [st,count]:bst)ready_status[st]+=count;
   }else require(!fs::exists(batch/"summary.tsv"),"unfinished batch must not claim summary");
  }
  c.generated=static_cast<int>(generated.size());c.not_started=128-c.attempted;
  require(c.attempted==c.completed+c.execution_failed&&c.completed==c.available+c.unavailable,"case partition");
  require(c.generated>=c.attempted&&c.generated<=128,"generated/attempted bounds");
  summary_matches(cell/"summary.tsv",ready_total,ready_avail,ready_reject,ready_status);
  for(int idx=0;idx<128;idx++){
   auto batch=cell/("batch"+std::to_string((idx/16)*16));auto stem=batch/("case-"+std::to_string(idx%16));auto st=fs::path(stem.string()+".status");bool isgenerated=generated.contains(idx);
   cases<<n<<'\t'<<alpha<<'\t'<<idx<<'\t'<<24000000+ci*1000+idx<<'\t'<<34000000+ci*1000+idx<<'\t'<<isgenerated<<'\t';
   if(!fs::exists(st)){cases<<"NA\tnot_started\tNA\tNA\tNA\tNA\tNA\tNA\n";continue;}
   I code=scalar(st);cases<<code<<'\t';
   if(code!=0){cases<<"execution_failed\tNA\tNA\tNA\tNA\tNA\tNA\n";continue;}
   auto p=fs::path(stem.string()+".stdout");auto r=native(p,idx);auto result=numbers<8>(lines(p)[0].substr(11));
   cases<<(r.status==0?"available":"model_unavailable")<<'\t'<<r.status<<'\t'<<r.hits<<'\t'<<r.completed<<'\t'<<r.trace_status<<'\t'<<result[6]<<'\t'<<result[7]<<'\n';
  }
  long double rate=0,se=0,lower=0,upper=1;
  if(c.available){long double a=c.available,z=1.95996398454005423552L,z2=z*z;rate=static_cast<long double>(c.rejected)/a;se=std::sqrt(rate*(1-rate)/a);auto d=1+z2/a,center=(rate+z2/(2*a))/d,half=z*std::sqrt((rate*(1-rate)+z2/(4*a))/a)/d;lower=std::max(0.L,center-half);upper=std::min(1.L,center+half);}
  cells<<n<<'\t'<<alpha<<"\t128\t"<<c.generated<<'\t'<<c.attempted<<'\t'<<c.completed<<'\t'<<c.available<<'\t'<<c.unavailable<<'\t'<<c.execution_failed<<'\t'<<c.not_started<<'\t'<<c.rejected<<'\t';
  if(c.available)cells<<rate<<'\t'<<se<<'\t'<<lower<<'\t'<<upper;else cells<<"NA\tNA\tNA\tNA";
  cells<<'\t'<<static_cast<long double>(c.rejected)/128<<'\t'<<static_cast<long double>(c.rejected+128-c.available)/128<<'\n';
  md<<'|'<<n<<'|'<<alpha<<"|128|"<<c.generated<<'|'<<c.attempted<<'|'<<c.completed<<'|'<<c.available<<'|'<<c.unavailable<<'|'<<c.execution_failed<<'|'<<c.not_started<<'|'<<c.rejected<<'/'<<c.available<<'|'<<lower<<" to "<<upper<<"|\n";
  std::cout<<"CELL "<<n<<' '<<alpha<<" attempted "<<c.attempted<<" completed "<<c.completed<<" available "<<c.available<<" refused "<<c.unavailable<<" process_failed "<<c.execution_failed<<" not_started "<<c.not_started<<'\n';
  std::ofstream counters(out/("n"+std::to_string(n)+"_alpha"+alpha+".work.tsv"));for(int j=0;j<5;j++)counters<<"observed_work_"<<j+1<<'\t'<<c.work[j]<<'\n';for(int j=0;j<10;j++)counters<<"observed_trace_prefix_"<<j+1<<'\t'<<c.trace[j]<<'\n';counters<<"trace_partial_completed_cases\t"<<c.trace_partial<<'\n';for(auto [st,count]:c.statuses)counters<<"model_status_"<<st<<'\t'<<count<<'\n';for(auto [st,count]:c.exits)counters<<"process_exit_"<<st<<'\t'<<count<<'\n';
  all_attempted+=c.attempted;all_completed+=c.completed;all_available+=c.available;all_unavailable+=c.unavailable;all_failed+=c.execution_failed;all_not_started+=c.not_started;ci++;
 }
 md<<"\nConditional fractions and Wilson intervals use available completed fits only, at the prespecified plus-one bootstrap p <= 0.05 rule. They are descriptive, approximate Monte Carlo summaries under an independent Bernoulli model; optional stopping, runtime selection, shared RNG streams, and model availability can invalidate a nominal coverage interpretation. The plug-in MC standard error is zero at zero observed rejections; the Wilson upper endpoint remains positive. Rates are never pooled across cells.\n\nThe planned-family bounds in cells.tsv place every unavailable, failed or unattempted result anywhere in [0,1]. They are deterministic missing-outcome bounds, not confidence intervals. No full target of 128 per cell was attained. No calibration_assessed flag is changed. No claim of universal calibration or DESeq2 equivalence follows.\n\nWork counters sum observed CAL_WORK values from completed rows only. CAL_TRACE sums are explicitly recorded prefixes; partial traces are not reconstructed from replicate counts or RNG. cases.tsv lists all planned identities and scheduled seeds, distinguishes generated data from not-started data, and includes final RNG words only where actually emitted.\n";
 std::cout<<"TOTAL planned 1152 attempted "<<all_attempted<<" completed "<<all_completed<<" available "<<all_available<<" model_unavailable "<<all_unavailable<<" execution_failed "<<all_failed<<" not_started "<<all_not_started<<"\nCOUNT_PROFILE_CAMPAIGN_CLOSURE_PASS\n";
 cases.flush();cells.flush();md.flush();require(bool(cases)&&bool(cells)&&bool(md),"complete report write");
}
int main(int argc,char** argv){try{require(argc==3,"usage: CAMPAIGN_ROOT FRESH_REPORT_ROOT");require(!fs::exists(argv[2]),"fresh output required");report(fs::path(argv[1]),fs::path(argv[2]));return 0;}catch(const std::exception& e){std::cerr<<"COUNT_PROFILE_CAMPAIGN_CLOSURE_REJECT "<<e.what()<<'\n';return 2;}}

