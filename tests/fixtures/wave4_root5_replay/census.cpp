#include <array>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
struct Case { std::string id,family,path,sha; int old_code; std::uintmax_t bytes; };
struct Totals { unsigned seen=0,nonzero=0,exit139=0,timeouts=0,diagnostic_zero=0,compile_ok=0,negative_signals=0; };
static std::string read(const fs::path&p) {
 std::ifstream f(p,std::ios::binary);
 if(!f) throw std::runtime_error("missing "+p.string());
 return {std::istreambuf_iterator<char>(f),{}};
}
static std::string trim(std::string s) {
 while(!s.empty()&&(s.back()=='\n'||s.back()=='\r')) s.pop_back();
 return s;
}
static int status(const fs::path&p) {
 auto s=trim(read(p));int n=-1;auto r=std::from_chars(s.data(),s.data()+s.size(),n);
 if(r.ec!=std::errc{}||r.ptr!=s.data()+s.size()||n<0||n>255) throw std::runtime_error("invalid status "+p.string());
 return n;
}
static std::vector<std::string> nul(const fs::path&p) {
 auto s=read(p);std::vector<std::string> out;std::size_t at=0;
 while(at<s.size()) {auto end=s.find('\0',at);if(end==std::string::npos)throw std::runtime_error("unterminated argv "+p.string());out.push_back(s.substr(at,end-at));at=end+1;}
 return out;
}
static std::string quote(const std::string&s) {
 std::ostringstream o;o<<'"';
 for(unsigned char c:s) {
  if(c=='"'||c=='\\')o<<'\\'<<static_cast<char>(c);
  else if(c=='\n')o<<"\\n";else if(c=='\r')o<<"\\r";else if(c=='\t')o<<"\\t";
  else if(c<32)o<<"\\u00"<<std::hex<<std::setw(2)<<std::setfill('0')<<static_cast<unsigned>(c)<<std::dec;
  else o<<static_cast<char>(c);
 }
 o<<'"';return o.str();
}
static bool hash_shape(const std::string&s) {
 if(s.size()!=64)return false;
 for(char c:s)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
 return true;
}
static bool elf64(const fs::path&p) {
 if(fs::is_symlink(p)||!fs::is_regular_file(p)||fs::file_size(p)<64)return false;
 std::ifstream f(p,std::ios::binary);std::array<unsigned char,64> h{};f.read(reinterpret_cast<char*>(h.data()),h.size());
 return f.gcount()==64&&h[0]==0x7f&&h[1]=='E'&&h[2]=='L'&&h[3]=='F'&&h[4]==2&&h[5]==1&&h[6]==1&&h[18]==62&&h[19]==0;
}
static const std::array<std::string,12> expected_paths={
 "tests/stdlib/chemistry/test_chem_network_ode_sys5.sio",
 "tests/stdlib/chemistry/test_kin_sigma_small_variance.sio",
 "tests/stdlib/chemistry/test_kinetics_deep_stdlib.sio",
 "tests/stdlib/chemistry/test_kinetics_epistemic_ensemble.sio",
 "tests/stdlib/chemistry/test_kinetics_gri_mech.sio",
 "tests/stdlib/graph/test_epistemic_curvature_e2e.sio",
 "tests/stdlib/nn/test_nn_primitives_d3.sio",
 "tests/stdlib/nn/test_pinn_full_integration_d6.sio",
 "tests/stdlib/tensor/test_tensor_autograd_d2_hardening.sio",
 "tests/stdlib/tensor/test_tensor_autograd_d2.sio",
 "tests/stdlib/tensor/test_tensor_ops_d1.sio",
 "tests/stdlib/chemistry/test_pbpk_clearance_ode.sio",
};
static std::vector<Case> load_cases(const fs::path&p) {
 std::istringstream in(read(p));std::string line;std::vector<Case> out;
 while(std::getline(in,line)) {
  std::vector<std::string> f;std::istringstream row(line);std::string value;
  while(std::getline(row,value,'\t'))f.push_back(value);
  if(f.size()!=6)throw std::runtime_error("case shape");
  std::ostringstream id;id<<std::setw(2)<<std::setfill('0')<<out.size();
  if(out.size()>=expected_paths.size()||f[0]!=id.str()||f[3]!=expected_paths[out.size()]||!hash_shape(f[4]))throw std::runtime_error("case identity");
  Case c{f[0],f[1],f[3],f[4],std::stoi(f[2]),std::stoull(f[5])};
  if(c.old_code!=(out.size()==11?1:139))throw std::runtime_error("historical code");
  out.push_back(c);
 }
 if(out.size()!=12)throw std::runtime_error("case census");
 return out;
}
static int time_signal(const std::string&text) {
 const std::string prefix="Command terminated by signal ";
 std::istringstream in(text);std::string line;int result=0;
 while(std::getline(in,line)) {
  if(line.rfind(prefix,0)!=0)continue;
  const auto value=line.substr(prefix.size());int n=0;
  const auto r=std::from_chars(value.data(),value.data()+value.size(),n);
  if(r.ec!=std::errc{}||r.ptr!=value.data()+value.size()||n<1||n>64||result!=0)throw std::runtime_error("invalid GNUtime signal");
  result=-n;
 }
 return result;
}
static bool diagnostics(const std::string&text,std::ostream&out,const std::string&prefix,const std::string&stream) {
 static const std::array<std::string,9> patterns={"error[","Error[","error:","Error:","ERROR:","parse error","internal compiler error","IR_INVALID","<error>"};
 bool found=false;
 for(const auto&pattern:patterns) {
  std::size_t at=0;
  while((at=text.find(pattern,at))!=std::string::npos) {
   found=true;
   out<<prefix<<'\t'<<stream<<'\t'<<at<<'\t'<<quote(pattern)<<'\t'<<quote(text.substr(at,160))<<'\n';
   at+=pattern.size();
  }
 }
 return found;
}
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 const fs::path d=argv[1];unsigned invalid=0;std::array<Totals,2> totals{};
 std::ofstream phases(d/"CENSUS/PHASES.tsv"),diags(d/"CENSUS/DIAGNOSTICS.tsv");
 phases<<"case\tsdk\tloop\tphase\touter_exit_code\ttime_reported_signal\terror_bytes\telf_present\telf_valid\n";
 diags<<"case\tsdk\tloop\tphase\tstream\tbyte_offset\tpattern\tbyte_excerpt\n";
 std::vector<std::string> cfg;std::vector<Case> cases;
 try {
  cfg=nul(d/"binding/config.argv0");
  if(cfg.size()!=14||cfg[10]!=d.string()||!hash_shape(cfg[1]))throw std::runtime_error("binding shape");
  if(trim(read(d/"binding/BUILD_STATUS"))!="0"||trim(read(d/"binding/SOURCE_HEAD"))!=cfg[9])throw std::runtime_error("build identity");
  if(read(d/"binding/GIT_SOURCE_TREE").empty())throw std::runtime_error("source tree absent");
  std::istringstream ready(read(d/"binding/RAW_READY.captured"));std::string hash,path,extra;
  if(!(ready>>hash>>path)||(ready>>extra)||hash!=cfg[1]||path!=cfg[0])throw std::runtime_error("RAW_READY binding");
  std::istringstream terminal(read(d/"binding/BUILD_TERMINAL.captured"));std::string line;bool main_complete=false;
  while(std::getline(terminal,line))if(line==cfg[13]+"|COMPLETED|0:0"||line==cfg[13]+"|COMPLETED|0:0|")main_complete=true;
  if(!main_complete)throw std::runtime_error("main build not completed");
  cases=load_cases(d/"harness/CASES.tsv");
 } catch(const std::exception&e) {std::cerr<<e.what()<<'\n';return 3;}
 for(const auto&c:cases) {
  const fs::path out=d/"results"/c.id,src=d/"input/original"/c.path;
  try {
   if(fs::file_size(src)!=c.bytes)throw std::runtime_error("source byte count");
   if(status(out/"COLLECTOR_STATUS")!=0)throw std::runtime_error("case collector failure");
   if(trim(read(out/"RAW_SHA256"))!=cfg[1]||trim(read(out/"ROOT_HEAD"))!=cfg[9])throw std::runtime_error("case binding");
   if(read(out/"INPUTS.before.verify").empty()||read(out/"INPUTS.final.verify").empty()||read(out/"TOOLS.final.verify").empty())throw std::runtime_error("guard evidence absent");
  } catch(const std::exception&e) {std::cerr<<c.id<<": "<<e.what()<<'\n';++invalid;}
  for(unsigned sdk=0;sdk<2;++sdk)for(unsigned mode=0;mode<2;++mode)for(const auto&phase:{"check","compile"}) {
   const std::string kind=sdk?"current":"historical",m=std::to_string(mode);
   const fs::path dir=out/kind/("loop"+m),prefix=dir/phase;
   const std::string tag=c.id+"\t"+kind+"\t"+m+"\t"+phase;
   try {
    const int rc=status(prefix.string()+".STATUS");
    const auto a=nul(prefix.string()+".argv0");
    const bool compile=std::string(phase)=="compile";
    if(a.size()!=(compile?12u:10u)||a[0]!=cfg[11]||a[1]!="-v"||a[2]!="-o"||a[3]!=prefix.string()+".resource"||a[4]!=cfg[12]||a[5]!="--kill-after=30s"||a[6]!="1200"||a[7]!=cfg[0]||a[8]!=phase||a[9]!=src.string())throw std::runtime_error("command binding");
    if(compile&&(a[10]!="-o"||a[11]!=(dir/"program.elf").string()))throw std::runtime_error("output binding");
    const std::string env=read(dir/"ENVIRONMENT.txt");
    const std::string wanted="SOUNIO_STDLIB_PATH="+cfg[sdk?4:3]+"/stdlib\nSOUNIO_LOOP_REGIONS="+m+"\nSOUNIO_LOOP_CHECKPOINT_ELISION=unset(default)\nSOUNIO_MADAROS_NOCACHE=1\nSOUNIO_MADAROS_CACHE_READONLY=1\n";
    if(env!=wanted)throw std::runtime_error("environment binding");
    const auto stdout_bytes=read(prefix.string()+".stdout"),stderr_bytes=read(prefix.string()+".stderr");
    const bool has_error=diagnostics(stdout_bytes,diags,tag,"stdout")|diagnostics(stderr_bytes,diags,tag,"stderr");
    const auto resource=read(prefix.string()+".resource");
    if(resource.empty())throw std::runtime_error("resource evidence missing");
    const int signal=time_signal(resource);
    if(signal!=0&&rc!=128-signal)throw std::runtime_error("GNUtime signal/status mismatch");
    if(read(prefix.string()+".started").empty()||read(prefix.string()+".ended").empty())throw std::runtime_error("time envelope missing");
    const bool present=compile&&fs::exists(dir/"program.elf"),valid=present&&elf64(dir/"program.elf");
    if(compile&&rc==0&&!valid)throw std::runtime_error("compile zero without valid ELF");
    if(compile&&present&&read(dir/"program.elf.sha256").empty())throw std::runtime_error("ELF hash absent");
    phases<<tag<<'\t'<<rc<<'\t'<<signal<<'\t'<<has_error<<'\t'<<present<<'\t'<<valid<<'\n';
    auto&t=totals[sdk];++t.seen;t.nonzero+=rc!=0;t.exit139+=rc==139;t.timeouts+=rc==124;t.diagnostic_zero+=rc==0&&has_error;t.compile_ok+=compile&&rc==0&&!has_error&&valid;t.negative_signals+=signal!=0;
   } catch(const std::exception&e) {std::cerr<<tag<<": "<<e.what()<<'\n';++invalid;}
  }
 }
 unsigned seen=totals[0].seen+totals[1].seen;
 if(seen!=96)++invalid;
 const bool historical_clean=totals[0].seen==48&&totals[0].nonzero==0&&totals[0].diagnostic_zero==0;
 const bool current_clean=totals[1].seen==48&&totals[1].nonzero==0&&totals[1].diagnostic_zero==0;
 std::ofstream hstatus(d/"CENSUS/HISTORICAL_COMPILER_STATUS"),cstatus(d/"CENSUS/CURRENT_COMPILER_STATUS");
 hstatus<<(historical_clean?0:1)<<'\n';cstatus<<(current_clean?0:1)<<'\n';
 std::ofstream json(d/"CENSUS/RESULT.json");
 json<<"{\n \"schema\":\"sounio.root5.regression-census.v1\",\n \"raw_sha256\":"<<quote(cfg[1])<<",\n \"root_head\":"<<quote(cfg[9])<<",\n \"phase_count\":"<<seen<<",\n \"structural_errors\":"<<invalid<<",\n \"historical_compiler_clean\":"<<(historical_clean?"true":"false")<<",\n \"current_compiler_clean\":"<<(current_clean?"true":"false")<<",\n \"native_programs_executed\":false,\n \"numerical_validation_performed\":false\n}\n";
 for(unsigned sdk=0;sdk<2;++sdk) {
  const auto&t=totals[sdk];
  std::cout<<(sdk?"CURRENT":"HISTORICAL")<<" seen="<<t.seen<<" nonzero="<<t.nonzero<<" exit139="<<t.exit139<<" time_reported_negative_signals="<<t.negative_signals<<" timeout124="<<t.timeouts<<" rc0_error_bytes="<<t.diagnostic_zero<<" valid_compiles="<<t.compile_ok<<'\n';
 }
 std::cout<<"CENSUS expected=96 seen="<<seen<<" structural_errors="<<invalid<<" native_programs=0 numerical_validation=0\n";
 phases.flush();diags.flush();hstatus.flush();cstatus.flush();json.flush();
 if(!phases||!diags||!hstatus||!cstatus||!json)return 1;
 return invalid?1:0;
}
