// Qualification generator/summary, not a substitute for native production fits.
// C++23 binary128 independent NB2/Poisson sampler; generator emits a Sounio
// driver derived from the accepted profile-bootstrap fixture's owner/call shape.
// Slurm: g++ -std=c++23 -O2 tests/reference/count_profile_calibration.cpp -lquadmath -o count_profile_calibration
#define main saved_calibration_sampling_main
#include "deep_count_sampling_reference.cpp"
#undef main
#include <filesystem>
#include <array>
#include <map>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <iomanip>
#include <set>
namespace fs=std::filesystem;
static void cqneed(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static std::string cqread(const fs::path& path){std::ifstream f(path);cqneed(bool(f),"input file open");return {std::istreambuf_iterator<char>(f),{}};}
static std::string cqbetween(const std::string& s,const std::string& left,const std::string& right){auto a=s.find(left);cqneed(a!=std::string::npos,"template left boundary");auto b=s.find(right,a);cqneed(b!=std::string::npos,"template right boundary");return s.substr(a,b-a);}
static void cqgenerate(int argc,char** argv){
 cqneed(argc==11,"--generate fixture outdir n alpha B batch_size first_index generation_seed bootstrap_seed");
 std::string fixture=cqread(argv[2]);fs::path dir=argv[3];int n=std::stoi(argv[4]),B=std::stoi(argv[6]),batch=std::stoi(argv[7]),first=std::stoi(argv[8]);
 double alpha=std::stod(argv[5]);auto seed=std::stoll(argv[9]),bootstrap=std::stoll(argv[10]);
 cqneed((n==16||n==64||n==256)&&(alpha==0||alpha==.25||alpha==1),"declared calibration cell");
 cqneed((B==31||B==199)&&batch>=1&&batch<=16&&first>=0&&first+batch<=128,"bounded protocol");
 cqneed(seed>0&&seed<1000000000&&bootstrap>0&&bootstrap<1000000000,"seed arithmetic range");
 fs::create_directories(dir);std::ofstream out(dir/"driver.sio"),meta(dir/"datasets.tsv");cqneed(bool(out)&&bool(meta),"output open");
 std::vector<std::int64_t> counts;counts.reserve(static_cast<std::size_t>(n*batch));
 meta<<"index\tn\talpha\tB\tgeneration_seed\tbootstrap_seed\n";
 for(int k=0;k<batch;k++){Cursor cursor(seed+first+k);for(int i=0;i<n;i++)counts.push_back(nb(cursor,Q(8),Q(alpha)));meta<<first+k<<'\t'<<n<<'\t'<<alpha<<'\t'<<B<<'\t'<<seed+first+k<<'\t'<<bootstrap+first+k<<'\n';}
 out<<"// Generated qualification driver: full profile-refit bootstrap, no assertions of calibrated validity.\n";

 auto imports=cqbetween(fixture,"use collections::numeric_buffer", "fn bootstrap_case");out<<imports;
 out<<R"SOUNIO(// Linux-only bounded environment selection; never log the environment.
fn qualification_case_from_env()->i64 with IO,Mut,Div,Panic {
 let data=read_file("/proc/self/environ")
 let wanted="COUNT_PROFILE_CALIBRATION_CASE="
 let wanted_len=str_len(wanted)
 var size:i64=0
 var found_end=false
 while size<1048575 {
  if data[size as usize]==0 {
   if data[(size+1) as usize]==0 {found_end=true;break}
  }
  size=size+1
 }
 if !found_end {return -1}
 var selected:i64= -1
 var at:i64=0
 while at<size {
  var end=at
  while end<size {if data[end as usize]==0 {break};end=end+1}
  if end-at>=wanted_len {
   var same=true
   var j:i64=0
   while j<wanted_len {
    if data[(at+j) as usize]!=(str_char_at(wanted,j) as i8) {same=false}
    j=j+1
   }
   if same {
    if selected>=0 {return -1}
    let digits=end-at-wanted_len
    if digits<1 || digits>2 {return -1}
    var value:i64=0
    j=0
    while j<digits {
     let digit=data[(at+wanted_len+j) as usize] as i64
     if digit<48 || digit>57 {return -1}
     if j==0 {if digits>1 {if digit==48 {return -1}}}
     value=value*10+digit-48
     j=j+1
    }
    if value>15 {return -1}
    selected=value
   }
  }
  at=end+1
 }
 selected
}
)SOUNIO";
 out<<"fn qualification_case(dataset:i64,data:&[i64;4096]) with Alloc,IO,Mut,Div,Panic {\n let n:i64="<<n<<"\n let p0:i64=1\n let p1:i64=2\n";
 out<<cqbetween(fixture," let plan=", " let data:");
 out<<" var i:i64=0\n while i<n {\n integer_buffer_set(&!counts,i,data[(dataset*n+i) as usize])\n numeric_buffer_set(&!a0,i,1.0)\n if i<n/2 {numeric_buffer_set(&!a1,i*2,1.0)}else{numeric_buffer_set(&!a1,i*2+1,1.0)}\n i=i+1\n }\n let initial=ln_checked(8.0)\n numeric_buffer_set(&!initial0,0,initial.value)\n numeric_buffer_set(&!initial1,0,initial.value)\n numeric_buffer_set(&!initial1,1,initial.value)\n";
 out<<" let options=CountProfileBootstrapOptions{replicates:"<<B<<",rcond:0.0,log_alpha_lower:-12.0,log_alpha_upper:6.0,outer_tolerance:1e-6,max_outer_evaluations:120,inner_iterations:100,inner_tolerance:1e-9,max_profile_attempts:"<<2*(B+1)<<",max_uniforms_per_draw:1000000,max_proposals_per_draw:1000000,max_total_uniforms:1000000000,max_total_proposals:1000000000,nesting_declared:true,model_declared:true,independence_declared:true}\n";
 out<<" let initial_rng=count_rng("<<bootstrap+first<<"+dataset)\n";
 out<<cqbetween(fixture," let result=dynamic_count_profile_lrt_bootstrap", " let trace_status:");
 out<<" let status:i64=result.status\n let tail:f64=result.p_value\n let hits:i64=result.hits\n let completed:i64=result.replicates_completed\n let failed:i64=result.failure_replica\n let stage:i64=result.failure_stage\n let hi:i64=result.rng.state_hi\n let lo:i64=result.rng.state_lo\n let index:i64="<<first<<"+dataset\n";
 out<<" print(\"CAL_RESULT \");print(index);print(\" \");print(status);print(\" \");print(hits);print(\" \");print(completed);print(\" \");print(failed);print(\" \");print(stage);print(\" \");print(hi);print(\" \");println(lo)\n print(\"CAL_P \");print(index);print(\" \");println_f64_sci(tail,15)\n";
 out<<cqbetween(fixture," numeric_buffer_close(a0)","fn main()");
 out<<"fn main()->i32 with Alloc,IO,Mut,Div,Panic {\n let data:[i64;4096]=[";
 for(int i=0;i<4096;i++){if(i)out<<',';out<<(i<int(counts.size())?counts[static_cast<std::size_t>(i)]:0);}
 out<<"]\n let dataset:i64=qualification_case_from_env()\n if dataset<0 || dataset>="<<batch<<" {return 64}\n qualification_case(dataset,&data)\n println(\"COUNT PROFILE CALIBRATION CASE COMPLETE\")\n 0\n}\n";
 cqneed(bool(out)&&bool(meta),"complete output writes");
}
static void cqreplace(std::string& text,const std::string& from,const std::string& to,int wanted){
 std::size_t at=0;int count=0;while((at=text.find(from,at))!=std::string::npos){text.replace(at,from.size(),to);at+=to.size();count++;}cqneed(count==wanted,"exact nested driver transformation boundary");
}
static void cq_nested_arguments(int argc,char** argv){
 cqneed(argc==11,"nested generation arguments");
 const std::string n=argv[4],alpha=argv[5],B=argv[6];
 cqneed(n=="16"||n=="64"||n=="256","exact grid n token");
 cqneed(alpha=="0"||alpha==".25"||alpha=="0.25"||alpha=="1","exact grid alpha token");
 cqneed(B=="2"||B=="31"||B=="199","exact nested B token");
 std::array<long long,4> values{};const int args[4]={7,8,9,10};
 for(int i=0;i<4;i++){std::string token=argv[args[i]];cqneed(!token.empty()&&token.size()<=10&&std::all_of(token.begin(),token.end(),[](unsigned char c){return c>='0'&&c<='9';}),"bounded decimal argument token");values[i]=std::stoll(token);}
 cqneed(values[0]>=1&&values[0]<=16&&values[1]>=0&&values[1]<=128-values[0],"bounded batch and first before integer addition");
 cqneed(values[2]>0&&values[2]<1000000000&&values[3]>0&&values[3]<1000000000,"bounded exact seed tokens");
}

static void cqgenerate_nested(int argc,char** argv){
 // The original mode remains unmodified, including output bytes. Derive the
 // additive nested driver ONLY after independent data generation has completed.
 cq_nested_arguments(argc,argv);
 const bool smoke=std::string(argv[6])=="2";
 if(smoke){std::array<char*,11> args{};std::copy(argv,argv+argc,args.begin());args[6]=const_cast<char*>("31");cqgenerate(argc,args.data());}
 else cqgenerate(argc,argv);
 fs::path driver=fs::path(argv[3])/"driver.sio";
 std::string text=cqread(driver);
 cqreplace(text,"dynamic_count_profile_bootstrap_workspace","dynamic_count_profile_bootstrap_nested_workspace",2);
 cqreplace(text,"dynamic_count_profile_lrt_bootstrap","dynamic_count_profile_lrt_bootstrap_nested",2);
 cqreplace(text," let p1:i64=2\n"," let p1:i64=2\n var embedding=numeric_buffer_new(2,16)\n var warm_start=numeric_buffer_new(2,16)\n numeric_buffer_set(&!embedding,0,1.0)\n numeric_buffer_set(&!embedding,1,1.0)\n",1);
 cqreplace(text,"&!factor_r1,&!factor_scales1,&!permutation1)","&!factor_r1,&!factor_scales1,&!permutation1,&embedding,&!warm_start)",1);
 cqreplace(text," numeric_buffer_close(a0)"," numeric_buffer_close(embedding)\n numeric_buffer_close(warm_start)\n numeric_buffer_close(a0)",1);
 const std::string work=R"SOUNIO( let work_fits:i64=result.fit_attempts
 let work_draws:i64=result.count_draws
 let work_uniforms:i64=result.uniforms
 let work_proposals:i64=result.proposals
 let work_evaluations:i64=result.profile_evaluations
 print("CAL_WORK ");print(index);print(" ");print(work_fits);print(" ");print(work_draws);print(" ");print(work_uniforms);print(" ");print(work_proposals);print(" ");println(work_evaluations)
)SOUNIO";
 cqreplace(text," numeric_buffer_close(embedding)",work+" numeric_buffer_close(embedding)",1);
 cqreplace(text,"// Generated qualification driver: full profile-refit bootstrap, no assertions of calibrated validity.","// Generated nested qualification driver: explicit embedding, observed null cold; no assessed calibration claim.",1);
 if(smoke){
  cqreplace(text,"replicates:31,","replicates:2,",1);
  cqreplace(text,"max_profile_attempts:64,","max_profile_attempts:6,",1);
  fs::path meta=fs::path(argv[3])/"datasets.tsv";std::string metadata=cqread(meta);
  cqreplace(metadata,"\t31\t","\t2\t",std::stoi(argv[7]));std::ofstream mout(meta,std::ios::trunc);mout<<metadata;cqneed(bool(mout),"smoke metadata write");
 }

 std::ofstream out(driver,std::ios::trunc);out<<text;cqneed(bool(out),"nested driver write");
}

static void cqgenerate_nested_traced(int argc,char** argv){
 cqgenerate_nested(argc,argv);
 fs::path driver=fs::path(argv[3])/"driver.sio";std::string text=cqread(driver);
 cqreplace(text,"dynamic_count_profile_lrt_bootstrap_nested","dynamic_count_profile_lrt_bootstrap_nested_traced",2);
 cqreplace(text,"// Linux-only bounded environment selection; never log the environment.","use stats::regression::count_fit_trace::{count_fit_trace_slot,count_fit_trace_status,count_fit_trace_metric}\n// Linux-only bounded environment selection; never log the environment.",1);
 cqreplace(text," let initial_rng=count_rng("," var fit_trace=count_fit_trace_slot()\n let initial_rng=count_rng(",1);
 cqreplace(text,"&embedding,&!warm_start)","&embedding,&!warm_start,&!fit_trace)",1);
 std::ostringstream trace;
 trace<<" let trace_status:i64=count_fit_trace_status(&fit_trace)\n";
 for(int metric=1;metric<=10;metric++)trace<<" let trace_m"<<metric<<":i64=count_fit_trace_metric(&fit_trace,"<<metric<<")\n";
 trace<<" print(\"CAL_TRACE \");print(index);print(\" \");print(trace_status)\n";
 for(int metric=1;metric<=10;metric++)trace<<" print(\" \");"<<(metric==10?"println":"print")<<"(trace_m"<<metric<<")\n";
 cqreplace(text," numeric_buffer_close(embedding)",trace.str()+" numeric_buffer_close(embedding)",1);
 cqreplace(text,"// Generated nested qualification driver:","// Generated nested traced qualification driver:",1);
 std::ofstream out(driver,std::ios::trunc);out<<text;cqneed(bool(out),"traced driver write");
}
static std::string cq_uint128(U value){if(value==0)return "0";std::string s;while(value>0){s.push_back(char('0'+unsigned(value%10)));value/=10;}std::reverse(s.begin(),s.end());return s;}

struct CQRow{int status=0;long long hits=0,completed=0,failure=0,stage=0,hi=0,lo=0;long double p=0;bool p_seen=false;long long fits=0,draws=0,uniforms=0,proposals=0,evaluations=0;bool work_seen=false;int trace_status=-1;std::array<long long,10> trace{};bool trace_seen=false;};
static void cqsummary(int argc,char** argv){
 const bool traced=std::string(argv[1])=="--summarize-nested-traced";
 const bool nested=traced||std::string(argv[1])=="--summarize-nested";
 cqneed(argc>=6,"--summarize B n_per_batch first_expected native_stdout...");
 int B=std::stoi(argv[2]),batch=std::stoi(argv[3]),first_expected=std::stoi(argv[4]);
 if(nested)cqneed(batch>=1&&batch<=16&&first_expected>=0&&first_expected<=128&&argc-5<=(128-first_expected)/batch,"nested summary bounds before arithmetic");
 cqneed(((B==31||B==199)||(nested&&B==2))&&batch>0&&batch<=16&&first_expected>=0&&first_expected+(argc-5)*batch<=128,"summary protocol");
 if(nested&&B==2)std::cout<<"protocol\tfunctional_B2_smoke_not_qualification\n";
 std::map<int,CQRow> rows;std::map<int,int> status_counts;
 for(int file=5;file<argc;file++){
  std::ifstream in(argv[file]);cqneed(bool(in),"native stdout open");std::string line;
  int count=0,case_markers=0,batch_markers=0;bool waiting=false;
  const int first=first_expected+(file-5)*batch;
  while(std::getline(in,line)){
   if(batch_markers>0){cqneed(line.empty(),"batch completion must be the single final marker");continue;}
   if(line=="COUNT PROFILE CALIBRATION BATCH COMPLETE"){
    cqneed(!waiting&&count==batch&&case_markers==batch,"batch marker after complete isolated cases");batch_markers++;continue;
   }
   if(line=="COUNT PROFILE CALIBRATION CASE COMPLETE"){
    cqneed(waiting&&count>0&&rows.at(first+count-1).p_seen&&(!nested||rows.at(first+count-1).work_seen)&&(!traced||rows.at(first+count-1).trace_seen),"case marker after complete native row");case_markers++;waiting=false;continue;
   }
   if(line.starts_with("CAL_RESULT ")){
    cqneed(!waiting&&count<batch,"isolated case output order");
    std::istringstream fields(line.substr(11));int index;CQRow row;
    fields>>index>>row.status>>row.hits>>row.completed>>row.failure>>row.stage>>row.hi>>row.lo;
    cqneed(bool(fields),"result syntax");std::string extra;cqneed(!(fields>>extra),"unexpected result fields");
    cqneed(index==first+count&&!rows.contains(index),"exact expected dataset index");
    cqneed(row.status>=0&&row.status<=(nested?13:12)&&row.status!=10,"declared profile-bootstrap status");
    cqneed(row.completed>=0&&row.completed<=B&&row.hits>=0&&row.hits<=row.completed,"bounded hits/completion");
    cqneed(row.failure>=-1&&row.failure<B&&row.stage>=0&&row.stage<=(traced?10:(nested?9:8)),"bounded failure metadata");
    if(row.status==0)cqneed(row.completed==B&&row.failure==-1&&row.stage==0,"successful aggregate metadata");
    rows.emplace(index,row);count++;waiting=true;continue;
   }
   if(traced&&line.starts_with("CAL_TRACE ")){
    std::istringstream fields(line.substr(10));int index,status;fields>>index>>status;
    cqneed(bool(fields)&&waiting&&index==first+count-1,"trace context");auto& row=rows.at(index);
    cqneed(row.p_seen&&row.work_seen&&!row.trace_seen&&status>=0&&status<=2,"unique bounded trace status");
    for(auto& value:row.trace){fields>>value;cqneed(bool(fields)&&value>=0,"nonnegative true trace metric");}
    std::string extra;cqneed(!(fields>>extra),"ten-metric trace arity");row.trace_status=status;row.trace_seen=true;
    if(status!=0)cqneed(row.status==4&&row.stage==10,"trace cap/control failure remains unavailable model result");
    if(row.stage==10)cqneed(row.status==4&&status!=0,"trace failure stage metadata");
    continue;
   }

   if(nested&&line.starts_with("CAL_WORK ")){
    std::istringstream fields(line.substr(9));int index;long long fits,draws,uniforms,proposals,evaluations;fields>>index>>fits>>draws>>uniforms>>proposals>>evaluations;
    cqneed(bool(fields)&&waiting&&index==first+count-1,"work counter context");std::string extra;cqneed(!(fields>>extra),"exact work counter schema");
    auto& row=rows.at(index);cqneed(row.p_seen&&!row.work_seen,"unique work counters after p");
    cqneed(fits>=0&&fits<=2*(B+1)&&draws>=0&&draws<=256LL*B&&uniforms>=0&&uniforms<=1000000000LL&&proposals>=0&&proposals<=1000000000LL&&evaluations>=0&&evaluations<=120*fits,"bounded actual work counters");
    row.fits=fits;row.draws=draws;row.uniforms=uniforms;row.proposals=proposals;row.evaluations=evaluations;row.work_seen=true;continue;
   }

   if(line.starts_with("CAL_P ")){
    std::istringstream fields(line.substr(6));int index;long double p;fields>>index>>p;
    cqneed(bool(fields)&&waiting&&index==first+count-1,"p-value context");std::string extra;cqneed(!(fields>>extra),"unexpected p fields");
    auto& row=rows.at(index);cqneed(!row.p_seen&&std::isfinite(p)&&p>=0&&p<=1,"finite unique bounded p");row.p=p;row.p_seen=true;continue;
   }
   if(nested)cqneed(line.empty(),"unknown nested qualification output");
  }
  cqneed(batch_markers==1&&count==batch&&case_markers==batch&&!waiting,"complete isolated native batch, no omitted execution failures");
 }
 long long available=0,rejected=0;for(const auto& [index,row]:rows){(void)index;cqneed(row.p_seen,"missing native p row");status_counts[row.status]++;
  if(row.status==0){long double p=static_cast<long double>(row.hits+1)/(B+1);cqneed(std::fabs(p-row.p)<=2e-14L,"plus-one consistency");available++;if(p<=.05L)rejected++;}
  else cqneed(row.p==0,"unavailable aggregate placeholder");
 }
 if(nested){long long fits=0,draws=0,uniforms=0,proposals=0,evaluations=0;for(const auto& [index,row]:rows){(void)index;cqneed(row.work_seen,"missing actual counters");fits+=row.fits;draws+=row.draws;uniforms+=row.uniforms;proposals+=row.proposals;evaluations+=row.evaluations;}
  std::cout<<"work_fit_attempts_total\t"<<fits<<"\nwork_count_draws_total\t"<<draws<<"\nwork_uniforms_total\t"<<uniforms<<"\nwork_proposals_total\t"<<proposals<<"\nwork_profile_evaluations_total\t"<<evaluations<<"\nWORK_SCOPE\t"<<(traced?"actual bootstrap result counters; callsite trace is reported separately; no counters derived from RNG":"actual bootstrap result fields; IRLS/QR callsites are not yet instrumented; no counters derived from RNG")<<'\n';
 }

 if(traced){std::array<U,10> sums{};long long exact_prefix=0,partial=0;
  for(const auto& [index,row]:rows){(void)index;cqneed(row.trace_seen,"trace required for every dataset including failures");if(row.trace_status==0)exact_prefix++;else partial++;for(int metric=0;metric<10;metric++)sums[metric]+=U(row.trace[metric]);}
  std::cout<<"trace_exact_recorded_prefix_datasets\t"<<exact_prefix<<"\ntrace_partial_prefix_datasets\t"<<partial<<'\n';
  for(int metric=0;metric<10;metric++)std::cout<<"trace_recorded_prefix_sum_metric_"<<metric+1<<'\t'<<cq_uint128(sums[metric])<<'\n';
  std::cout<<"TRACE_SCOPE\tactual metric IDs1..10; default i64MAX limit PER METRIC; nonzero trace status means partial prefix and model unavailability; never work inferred from B or RNG\n";
 }

 const auto total=static_cast<long long>(rows.size()),missing=total-available;
 std::cout<<std::setprecision(18)<<"datasets\t"<<total<<"\navailable\t"<<available<<"\nunavailable\t"<<missing<<"\nrejected_available\t"<<rejected<<'\n';
 for(auto [status,count]:status_counts)std::cout<<"status_"<<status<<'\t'<<count<<'\n';
 if(total>0)std::cout<<"overall_unknown_failure_lower\t"<<static_cast<long double>(rejected)/total<<"\noverall_unknown_failure_upper\t"<<static_cast<long double>(rejected+missing)/total<<'\n';
 if(available>0){long double m=available,r=static_cast<long double>(rejected)/m,z=1.95996398454005423552L,z2=z*z,den=1+z2/m,center=(r+z2/(2*m))/den,half=z*std::sqrt((r*(1-r)+z2/(4*m))/m)/den;
  std::cout<<"conditional_rejection_fraction\t"<<r<<"\nconditional_mc_se\t"<<std::sqrt(r*(1-r)/m)<<"\nconditional_wilson95_lower\t"<<std::max(0.L,center-half)<<"\nconditional_wilson95_upper\t"<<std::min(1.L,center+half)<<'\n';
 }
 std::cout<<"INTERPRETATION\tconditional on available fits; availability selection can bias rejection fractions; Wilson is approximate; no universal validity claim\n";
}
static void cqseeds(int argc,char** argv){
 cqneed(argc==3,"--seeds cell_index");
 const std::string token=argv[2];cqneed(token.size()==1&&token[0]>='0'&&token[0]<='8',"cell index 0..8");
 const std::int64_t cell=token[0]-'0';
 const std::int64_t generation=24000000+cell*1000;
 const std::int64_t bootstrap=34000000+cell*1000;
 std::cout<<generation<<' '<<bootstrap<<'\n';
}
int main(int argc,char** argv){try{cqneed(argc>1,"mode required");std::string mode=argv[1];if(mode=="--seeds")cqseeds(argc,argv);else if(mode=="--generate")cqgenerate(argc,argv);else if(mode=="--generate-nested")cqgenerate_nested(argc,argv);else if(mode=="--generate-nested-traced")cqgenerate_nested_traced(argc,argv);else if(mode=="--summarize"||mode=="--summarize-nested"||mode=="--summarize-nested-traced")cqsummary(argc,argv);else throw std::runtime_error("unknown mode");return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
