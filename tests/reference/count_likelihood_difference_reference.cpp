// Canonical C++23 oracle/emitter for paired scalar and aggregate full likelihood.
// Reuses independent FULL finite-product PMF (400 decimal digits), never a fit.
// Compile only in Slurm: g++ -std=c++23 -O2 this.cpp -o paired_reference
// paired_reference --emit case-3.stdout case-11.stdout case-13.stdout > fixture.sio
// paired_reference --native native.stdout case-3.stdout case-11.stdout case-13.stdout
#define main saved_alpha_identity_reference_main
#include "count_dispersion_increment_reference.cpp"
#undef main
#include <sstream>
#include <fstream>
#include <limits>
struct Pair{long long y;double m0,a0,m1,a1;};
struct Dataset{std::vector<Pair> rows;};
static double decode(long long bits){return std::bit_cast<double>(std::bit_cast<std::uint64_t>(bits));}
static Dataset read_dataset(const char* path){std::ifstream in(path);need(bool(in),"original diagnostic open");Dataset d;double a0=0,a1=0;std::string line;bool done=false,has0=false,has1=false;
 while(std::getline(in,line)){std::istringstream f(line);std::string tag;f>>tag;if(tag=="LAST_ALPHA0"||tag=="LAST_ALPHA1"){long long bits;f>>bits;need(bool(f),"alpha rawbits");if(tag=="LAST_ALPHA0"){a0=decode(bits);has0=true;}else{a1=decode(bits);has1=true;}}else if(tag=="DIAG_ROW"){int i,s0,s1;long long observed,y,m0,m1,e0,e1,p0,p1;f>>i>>observed>>y>>m0>>m1>>e0>>e1>>s0>>p0>>s1>>p1;need(bool(f)&&i==int(d.rows.size())&&s0==0&&s1==0,"original ordered row");d.rows.push_back({y,decode(m0),0,decode(m1),0});}else if(tag=="LR_DIAGNOSTIC_COMPLETE")done=true;}
 need(done&&has0&&has1&&d.rows.size()==16,"complete original dataset");for(Pair& p:d.rows){p.a0=a0;p.a1=a1;}return d;}
static std::vector<Pair> extras(){double a=.25;return {
 {0,2,0,3,0},{0,3,0,2,0},{8,8,0,std::nextafter(8.,9.),0},
 {8,3.25,a,3.25,a},{0,3.25,a,3.25,std::nextafter(a,1.)},
 {1,3.25,a,3.25,std::nextafter(a,1.)},{7,3.25,a,3.25,.28125},
 {8,3.25,.28125,3.25,a},{64,3.25,0,3.25,1e-20},
 {1,2.,1e-300,2.,std::nextafter(1e-300,1.)},
 {0,2.,0,2.,1e-300},{0,2.,1e-300,2.,0},
 {8,3.25,a,std::nextafter(3.25,4.),std::nextafter(a,1.)}};}
static B truth(const Pair& p){if(p.m0==p.m1)return full_difference(int(p.y),p.m0,p.a0,p.a1);return B(pmf(int(p.y),exact(p.m1),exact(p.a1))-pmf(int(p.y),exact(p.m0),exact(p.a0)));}
static void lit(double x){std::cout<<std::scientific<<std::setprecision(17)<<x;}
static void emit(const std::vector<Dataset>& ds){
 std::cout<<R"SIO(//@ run-pass
//@ requires: madaros
//@ expect-stdout: PAIRED LIKELIHOOD PASS
use stats::regression::nb2_likelihood::{nb2_likelihood_difference_slot,nb2_log_probability_difference_into,nb2_log_probability_mean_difference_into,nb2_likelihood_difference_value,nb2_likelihood_difference_snapshot}
use collections::numeric_buffer::{numeric_buffer_new,numeric_buffer_set,numeric_buffer_close}
use collections::integer_buffer::{integer_buffer_new,integer_buffer_set,integer_buffer_close}
fn paired_one(index:i64,expected:i64,count:i64,m0:f64,a0:f64,m1:f64,a1:f64) with IO,Mut,Div,Panic {
 var slot=nb2_likelihood_difference_slot()
 let status:i64=nb2_log_probability_difference_into(count,m0,a0,m1,a1,&!slot)
 assert(status==expected)
 let value:f64=nb2_likelihood_difference_value(&slot)
 let bits:i64=f64_to_bits(value)
 print("PAIR ");print(index);print(" ");print(status);print(" ");println(bits)
 let saved=nb2_likelihood_difference_snapshot(&slot)
 assert(nb2_log_probability_difference_into(-1,m0,a0,m1,a1,&!slot)==1)
 assert(f64_to_bits(saved.value)==bits && saved.status==expected)
}
fn main()->i32 with IO,Alloc,Mut,Div,Panic {
)SIO";
 int index=0;
 for(const Dataset& d:ds)for(const Pair& p:d.rows){std::cout<<" paired_one("<<index++<<",0,"<<p.y<<",";lit(p.m0);std::cout<<",";lit(p.a0);std::cout<<",";lit(p.m1);std::cout<<",";lit(p.a1);std::cout<<")\n";}
 int extra_index=0;for(const Pair& p:extras()){int expected=(extra_index>=9&&extra_index<=11)?5:0;extra_index++;std::cout<<" paired_one("<<index++<<","<<expected<<","<<p.y<<",";lit(p.m0);std::cout<<",";lit(p.a0);std::cout<<",";lit(p.m1);std::cout<<",";lit(p.a1);std::cout<<")\n";}
 for(size_t k=0;k<ds.size();k++){const Dataset& d=ds[k];std::cout<<" var c"<<k<<"=integer_buffer_new(16,128)\n var m"<<k<<"=numeric_buffer_new(16,128)\n var t"<<k<<"=numeric_buffer_new(16,128)\n";for(size_t i=0;i<d.rows.size();i++){auto p=d.rows[i];std::cout<<" integer_buffer_set(&!c"<<k<<","<<i<<","<<p.y<<")\n numeric_buffer_set(&!m"<<k<<","<<i<<",";lit(p.m0);std::cout<<")\n numeric_buffer_set(&!t"<<k<<","<<i<<",";lit(p.m1);std::cout<<")\n";}std::cout<<" var s"<<k<<"=nb2_likelihood_difference_slot()\n let status"<<k<<":i64=nb2_log_probability_mean_difference_into(&c"<<k<<",&m"<<k<<",";lit(d.rows[0].a0);std::cout<<",&t"<<k<<",";lit(d.rows[0].a1);std::cout<<",16,&!s"<<k<<")\n assert(status"<<k<<"==0)\n let v"<<k<<":f64=nb2_likelihood_difference_value(&s"<<k<<")\n assert(v"<<k<<"<0.0)\n let b"<<k<<":i64=f64_to_bits(v"<<k<<")\n print(\"PAIR_SUM "<<k<<" \");print(status"<<k<<");print(\" \");println(b"<<k<<")\n integer_buffer_close(c"<<k<<")\n numeric_buffer_close(m"<<k<<")\n numeric_buffer_close(t"<<k<<")\n";}
 std::cout<<" println(\"PAIRED LIKELIHOOD PASS\")\n 0\n}\n";
}
int main(int argc,char** argv){try{need(argc>=5,"mode and three original diagnostics");bool emitting=std::string(argv[1])=="--emit";bool native=std::string(argv[1])=="--native";need(emitting||native,"mode");int first=emitting?2:3;need(argc==first+3,"exact three original diagnostics");std::vector<Dataset> ds;for(int j=first;j<argc;j++)ds.push_back(read_dataset(argv[j]));if(emitting){emit(ds);return 0;}std::ifstream in(argv[2]);need(bool(in),"native open");std::vector<Pair> all;for(const auto& d:ds)all.insert(all.end(),d.rows.begin(),d.rows.end());auto more=extras();all.insert(all.end(),more.begin(),more.end());std::string line;
 for(size_t j=0;j<all.size();j++){need(bool(std::getline(in,line)),"complete paired native rows");std::istringstream f(line);std::string tag;int index,status;long long bits;f>>tag>>index>>status>>bits;need(bool(f)&&tag=="PAIR"&&index==int(j)&&status==((j>=57&&j<=59)?5:0),"native paired metadata");B got=exact(decode(bits)),want=truth(all[j]);if(status==5){need(got==0&&want!=0,"explicit tiny-alpha unavailable payload");}else if(want==0){need(got==0,"exact equal-parameter zero");}else need(abs(got-want)<=abs(want)*B("1e-11")+2*exact(std::numeric_limits<double>::denorm_min()),"native paired full likelihood difference");std::cout<<j<<'\t'<<std::setprecision(50)<<want<<'\t'<<got<<'\n';}
 for(size_t j=0;j<ds.size();j++){need(bool(std::getline(in,line)),"complete native aggregate rows");std::istringstream f(line);std::string tag;int index,status;long long bits;f>>tag>>index>>status>>bits;need(bool(f)&&tag=="PAIR_SUM"&&index==int(j)&&status==0,"native aggregate metadata");B want=0;for(const Pair& p:ds[j].rows)want+=truth(p);want/=16;B got=exact(decode(bits));need(want<0&&got<0,"original fixed-native negative preserved");need(abs(got-want)<=abs(want)*B("1e-6"),"near-zero original aggregate relative accuracy");std::cout<<"aggregate\t"<<j<<'\t'<<want<<'\t'<<got<<'\n';}
 need(bool(std::getline(in,line))&&line=="PAIRED LIKELIHOOD PASS","final paired marker");while(std::getline(in,line))need(line.empty(),"no extra paired output");std::cout<<"PAIRED_LIKELIHOOD_REFERENCE_COMPLETE\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
