// Qualification generator/summary, not a substitute for native production fits.
// C++23 binary128 independent NB2/Poisson sampler; generator emits a Sounio
// driver derived from the accepted profile-bootstrap fixture's owner/call shape.
// Slurm: g++ -std=c++23 -O2 tests/reference/count_profile_calibration.cpp -lquadmath -o count_profile_calibration
#define main saved_calibration_sampling_main
#include "deep_count_sampling_reference.cpp"
#undef main
#include <filesystem>
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
struct CQRow{int status=0;long long hits=0,completed=0,failure=0,stage=0,hi=0,lo=0;long double p=0;bool p_seen=false;};
static void cqsummary(int argc,char** argv){
 cqneed(argc>=6,"--summarize B n_per_batch first_expected native_stdout...");
 int B=std::stoi(argv[2]),batch=std::stoi(argv[3]),first_expected=std::stoi(argv[4]);
 cqneed((B==31||B==199)&&batch>0&&batch<=16&&first_expected>=0&&first_expected+(argc-5)*batch<=128,"summary protocol");
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
    cqneed(waiting&&count>0&&rows.at(first+count-1).p_seen,"case marker after complete native row");case_markers++;waiting=false;continue;
   }
   if(line.starts_with("CAL_RESULT ")){
    cqneed(!waiting&&count<batch,"isolated case output order");
    std::istringstream fields(line.substr(11));int index;CQRow row;
    fields>>index>>row.status>>row.hits>>row.completed>>row.failure>>row.stage>>row.hi>>row.lo;
    cqneed(bool(fields),"result syntax");std::string extra;cqneed(!(fields>>extra),"unexpected result fields");
    cqneed(index==first+count&&!rows.contains(index),"exact expected dataset index");
    cqneed(row.status>=0&&row.status<=12&&row.status!=10,"declared profile-bootstrap status");
    cqneed(row.completed>=0&&row.completed<=B&&row.hits>=0&&row.hits<=row.completed,"bounded hits/completion");
    cqneed(row.failure>=-1&&row.failure<B&&row.stage>=0&&row.stage<=8,"bounded failure metadata");
    if(row.status==0)cqneed(row.completed==B&&row.failure==-1&&row.stage==0,"successful aggregate metadata");
    rows.emplace(index,row);count++;waiting=true;continue;
   }
   if(line.starts_with("CAL_P ")){
    std::istringstream fields(line.substr(6));int index;long double p;fields>>index>>p;
    cqneed(bool(fields)&&waiting&&index==first+count-1,"p-value context");std::string extra;cqneed(!(fields>>extra),"unexpected p fields");
    auto& row=rows.at(index);cqneed(!row.p_seen&&std::isfinite(p)&&p>=0&&p<=1,"finite unique bounded p");row.p=p;row.p_seen=true;
   }
  }
  cqneed(batch_markers==1&&count==batch&&case_markers==batch&&!waiting,"complete isolated native batch, no omitted execution failures");
 }
 long long available=0,rejected=0;for(const auto& [index,row]:rows){(void)index;cqneed(row.p_seen,"missing native p row");status_counts[row.status]++;
  if(row.status==0){long double p=static_cast<long double>(row.hits+1)/(B+1);cqneed(std::fabs(p-row.p)<=2e-14L,"plus-one consistency");available++;if(p<=.05L)rejected++;}
  else cqneed(row.p==0,"unavailable aggregate placeholder");
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
int main(int argc,char** argv){try{cqneed(argc>1,"mode required");std::string mode=argv[1];if(mode=="--seeds")cqseeds(argc,argv);else if(mode=="--generate")cqgenerate(argc,argv);else if(mode=="--summarize")cqsummary(argc,argv);else throw std::runtime_error("unknown mode");return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
