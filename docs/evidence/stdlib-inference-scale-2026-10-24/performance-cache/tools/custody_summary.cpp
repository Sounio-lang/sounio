// C++23 receipt-only summary. Compile/run numerically only under Slurm.
// Does not execute, reconstruct, or authenticate the excluded native ELFs.
#define main preserved_full_benchmark_main
#include "count_profile_benchmark_reference.cpp"
#undef main
#include <filesystem>
#include <limits>
static bool same_number(long double a,long double b){return std::fabs(a-b)<=16*std::numeric_limits<long double>::epsilon()*std::max(std::fabs(a),std::fabs(b));}
static void status_zero(const std::filesystem::path& p){std::istringstream in(read(p.string()));long long rc;std::string extra;in>>rc;need(bool(in)&&rc==0&&!(in>>extra),"whole status0 field");}
int main(int argc,char**argv){try{
 need(argc==2,"bundle root argument");std::filesystem::path root=argv[1];
 const std::array<std::string,4> names{"04","05","06","07"};
 const std::array<std::string,4> roles{"full02_to_strict_warm_fastlocal","isolated_trace_factory","full02_to_objective_cache","full02_to_objective_and_fallback_cache"};
 const std::string compiler="92d2108ef595c5382d582aa363df6e95dab907a554f039d5f8f008073e535c21";
 std::cout<<"scope\treceipt_recalculation_only_no_executable_replay\n";
 std::cout<<"pilot\trole\tB\tbefore_s\tafter_s\tbefore_over_after\tbefore_rss_kib\tafter_rss_kib\tscientific_rows_identical\tcompiler_sha256\n"<<std::setprecision(21);
 for(std::size_t i=0;i<names.size();++i){
  auto p=root/"pilots"/("profile_pilot"+names[i]);
  status_zero(p/"STATUS");
  const auto before=scientific((p/"runs/before.stdout").string(),31),after=scientific((p/"runs/after.stdout").string(),31);
  need(before.result_line==after.result_line&&before.p_line==after.p_line,"whole native scientific rows identical");
  const auto br=resource((p/"runs/before.resource").string()),ar=resource((p/"runs/after.resource").string());
  status_zero(p/"runs/before.status");status_zero(p/"runs/after.status");
  std::array<std::string,2> drivers{read((p/"drivers/before.sio").string()),read((p/"drivers/after.sio").string())};
  for(const char* token:{"let data:[i64;4096]=","let options=CountProfileBootstrapOptions{","let initial_rng=count_rng("})need(source_line(drivers[0],token)==source_line(drivers[1],token),"same copied dataset/options/seeds");
  for(const auto& driver:drivers){need(driver.find("let n:i64=256")!=std::string::npos,"n256");const auto opts=source_line(driver,"let options=CountProfileBootstrapOptions{");need(opts.find("replicates:31,")!=std::string::npos&&opts.find("max_profile_attempts:64,")!=std::string::npos,"B31 maxfits64");}
  std::istringstream runs(read((p/"RUNS.tsv").string()));std::string line;need(bool(std::getline(runs,line))&&split(line,'\t').size()==11,"eleven-column runs header");std::array<bool,2> seen{};int rows=0;
  while(std::getline(runs,line)){auto f=split(line,'\t');need(f.size()==11,"eleven-column row");int v=f[0]=="before"?0:f[0]=="after"?1:-1;need(v>=0&&!seen[v]&&f[1]=="1","one paired trial");seen[v]=true;rows++;
   need(f[8]==compiler&&digest(f[7]),"recorded compiler/ELF identity");need(f[6]==sha((p/(v?"drivers/after.sio":"drivers/before.sio")).string()),"copied driver SHA binding");need(f[10]==sha((p/(v?"INPUTS.after.sha256":"INPUTS.before.sha256")).string()),"copied source-manifest SHA binding");
  }
  need(rows==2&&seen[0]&&seen[1],"complete one-pair manifest");
  const long double ratio=br.wall/ar.wall;bool reported=false;std::istringstream report(read((p/"REPORT.tsv").string()));
  while(std::getline(report,line)){auto f=split(line,'\t');if(f.size()==6&&f[0]=="1"){need(!reported,"unique original numeric report");reported=true;need(same_number(real(f[1]),br.wall)&&same_number(real(f[2]),ar.wall)&&same_number(real(f[3]),ratio)&&integer(f[4])==br.rss&&integer(f[5])==ar.rss,"raw resources match original C++ report");}}
  need(reported,"original numeric C++ report present");
  std::cout<<names[i]<<'\t'<<roles[i]<<"\t31\t"<<br.wall<<'\t'<<ar.wall<<'\t'<<ratio<<'\t'<<br.rss<<'\t'<<ar.rss<<"\ttrue\t"<<compiler<<'\n';
 }
 std::cout<<"limits\tfour_distinct_single_B31_pairs_no_pooling_no_B199_target_or_calibration_claim_no_ELF_verification_from_this_bundle\n";
 std::cout<<"CUSTODY_RESOURCE_SUMMARY_PASS\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
