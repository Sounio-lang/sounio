// Independent 400-decimal-digit oracle: raw integer logs, geometric means,
// std::sort log-ratio medians, across-sample centering. No production helpers.
// --verify ACTUAL_STDOUT checks scientific15 actual native output.
// Default emits the corresponding Sounio fixture; --report prints references.
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using Big=boost::multiprecision::number<boost::multiprecision::cpp_dec_float<400>>;
struct Case{std::string name;std::int64_t genes,samples,mode;std::vector<std::int64_t> counts;};
static std::vector<Case> cases(){
 constexpr std::int64_t M=9007199254740992LL;
 return {
 {"near_pair",1,2,0,{M,M-1}},
 {"near_pair_reverse",1,2,0,{M-1,M}},
 {"near_three",1,3,0,{M,M-1,M-2}},
 {"near_three_tie",1,3,0,{M,M-1,M-1}},
 {"near_even_duplicates",4,3,0,{M,M-1,M-2,M-3,M-1,M-4,M-7,M-9,M-8,M-7,M-9,M-8}},
 {"strict_zero_exclusion",3,3,0,{0,1,2,3,6,12,2,4,8}},
 {"positive_zero_ratios",3,3,1,{0,1,2,3,6,12,2,4,8}},
 {"positive_empty_sample",2,3,1,{0,1,0,0,2,0}},
 {"strict_no_genes",2,3,0,{0,1,2,3,0,4}},
 {"positive_crossed_zeros",2,3,1,{0,1,2,3,0,4}},
 {"far_extreme_quotients",1,3,0,{M,1,2}},
 {"reciprocal_even_zero",2,3,0,{1,M,1,M,1,M}},
 {"moderate_even",4,3,0,{1,5,11,2,9,21,4,17,41,4,17,41}},
 {"moderate_odd",5,3,0,{1,5,11,2,9,21,4,17,41,4,17,41,10,21,41}},
 {"all_ties",7,4,0,{17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17}},
 {"one_sample",4,1,0,{1,M,7,0}},
 {"over_binary64_limit",1,2,0,{M+1,1}},
 {"negative_preflight",1,2,0,{-1,0}},
 {"invalid_mode",1,2,2,{1,2}},
 {"local_half_boundary",3,3,0,{2,3,4,3,1,2,7,4,11}},
 {"far_two_samples",2,2,0,{1,M,2,M}},
 {"duplicate_even_ties",4,2,0,{2,3,2,3,4,6,8,12}},
 {"strict_all_zero",2,3,0,{0,0,0,0,0,0}},
 {"positive_all_zero",2,3,1,{0,0,0,0,0,0}},
 {"range_before_exclusion",1,3,0,{0,M+1,2}},
 {"negative_before_exclusion",1,3,0,{0,-1,2}}
 };
}
struct Reference{
 int status=0;std::int64_t used=0,excluded=0;
 std::vector<Big> logs,factors;
};
static Reference evaluate(const Case&c){
 Reference out;out.logs.assign(c.samples,Big(109));out.factors.assign(c.samples,Big(107));
 if(c.genes<1||c.samples<1||c.mode<0||c.mode>1){out.status=1;return out;}
 if(c.counts.size()!=static_cast<std::size_t>(c.genes*c.samples))throw std::logic_error("fixture shape");
 for(auto x:c.counts){if(x<0){out.status=2;return out;}if(x>9007199254740992LL){out.status=5;return out;}}
 std::vector<bool> eligible(c.genes);
 std::vector<Big> means(c.genes);
 for(std::int64_t g=0;g<c.genes;++g){
  std::int64_t positive=0;Big total=0;
  for(std::int64_t s=0;s<c.samples;++s){auto x=c.counts[g*c.samples+s];if(x>0){++positive;total+=log(Big(x));}}
  eligible[g]=positive>0&&(c.mode==1||positive==c.samples);
  if(eligible[g]){means[g]=total/Big(positive);++out.used;}
 }
 out.excluded=c.genes-out.used;
 if(out.used==0){out.status=3;return out;}
 std::vector<Big> medians;
 for(std::int64_t s=0;s<c.samples;++s){
  std::vector<Big> ratios;
  for(std::int64_t g=0;g<c.genes;++g){
   auto x=c.counts[g*c.samples+s];
   if(eligible[g]&&x>0)ratios.push_back(log(Big(x))-means[g]);
  }
  if(ratios.empty()){out.status=3;return out;}
  std::sort(ratios.begin(),ratios.end());auto n=ratios.size();
  Big median=ratios[n/2];if(n%2==0)median=(ratios[n/2-1]+ratios[n/2])/2;
  medians.push_back(median);
 }
 Big center=0;for(const auto&m:medians)center+=m;center/=Big(c.samples);
 for(std::int64_t s=0;s<c.samples;++s){out.logs[s]=medians[s]-center;out.factors[s]=exp(out.logs[s]);}
 return out;
}
static bool zero_target(const Big&x){
 // These finite fixtures use at most five nonidentical genes and exact
 // reciprocal/all-equal symmetries. The threshold is far below their
 // intended nonzero contrasts (~1e-16 or ~1e-32), at 400-digit precision.
 return abs(x)<Big("1e-300");
}
static void verify(const std::string&path,const std::vector<Case>&cs,const std::vector<Reference>&rs){
 std::ifstream in(path);if(!in)throw std::runtime_error("actual stdout unavailable");
 std::set<int> headers;std::set<std::pair<int,int>> values;std::string line;bool marker=false;
 while(std::getline(in,line)){
  if(line.empty())continue;
  if(line=="STDLIB_DEEP_NORMALIZATION_PRECISION_PASS"){if(marker)throw std::runtime_error("duplicate marker");marker=true;continue;}
  std::istringstream fields(line);std::string tag;fields>>tag;
  if(tag=="CASE"){
   int id,status;std::int64_t used,excluded;
   if(!(fields>>id>>status>>used>>excluded)||id<0||id>=static_cast<int>(cs.size()))throw std::runtime_error("malformed CASE");
   std::string extra;if(fields>>extra)throw std::runtime_error("extra CASE field");
   if(!headers.insert(id).second)throw std::runtime_error("duplicate CASE");
   if(status!=rs[id].status||used!=rs[id].used||excluded!=rs[id].excluded)throw std::runtime_error("status/eligibility mismatch "+cs[id].name);
  }else if(tag=="VALUE"){
   int id,s;std::string lt,ft;
   if(!(fields>>id>>s>>lt>>ft)||id<0||id>=static_cast<int>(cs.size())||s<0||s>=cs[id].samples)throw std::runtime_error("malformed VALUE");
   std::string extra;if(fields>>extra)throw std::runtime_error("extra VALUE field");
   if(!values.emplace(id,s).second)throw std::runtime_error("duplicate VALUE");
   Big actual_log(lt),actual_factor(ft);
   if(!boost::multiprecision::isfinite(actual_log)||!boost::multiprecision::isfinite(actual_factor))throw std::runtime_error("nonfinite actual");
   const auto&r=rs[id];
   if(r.status!=0){
    if(actual_log!=109||actual_factor!=107)throw std::runtime_error("nontransactional failure "+cs[id].name);
   }else{
    const Big log_error=abs(actual_log-r.logs[s]);
    const Big log_bound=zero_target(r.logs[s])?Big("2e-14"):Big("1e-29")+abs(r.logs[s])*Big("1e-11");
    if(log_error>log_bound){
     std::cerr<<std::setprecision(45)<<"log mismatch "<<cs[id].name<<" sample "<<s<<" actual "<<actual_log<<" expected "<<r.logs[s]<<" error "<<log_error<<" bound "<<log_bound<<"\n";
     throw std::runtime_error("log-factor precision");
    }
    if(actual_factor<=0||abs(actual_factor-r.factors[s])>abs(r.factors[s])*Big("1e-12"))throw std::runtime_error("factor mismatch "+cs[id].name);
   }
  }else throw std::runtime_error("unrecognized actual output "+tag);
 }
 if(!marker||headers.size()!=cs.size())throw std::runtime_error("incomplete actual CASE output");
 std::size_t expected=0;for(const auto&c:cs)expected+=static_cast<std::size_t>(c.samples);
 if(values.size()!=expected)throw std::runtime_error("incomplete actual VALUE output");
 std::cout<<"DEEP_NORMALIZATION_PRECISION_ACTUAL_PASS cases="<<cs.size()<<" values="<<values.size()<<"\n";
}
static void emit(const std::vector<Case>&cs){
 std::cout<<R"SIO(//@ run-pass
//@ requires: madaros
//@ expect-stdout: STDLIB_DEEP_NORMALIZATION_PRECISION_PASS
use collections::numeric_buffer::{numeric_buffer_new,numeric_buffer_status,numeric_buffer_at,numeric_buffer_set,numeric_buffer_close}
use collections::integer_buffer::{IntegerBuffer,integer_buffer_new,integer_buffer_status,integer_buffer_set,integer_buffer_close}
use genomics::dynamic_normalization::{dynamic_normalization_workspace,dynamic_median_ratio_size_factors}
use io::sci_print::print_f64_sci
fn precision_case(id:i64,counts:&IntegerBuffer,genes:i64,samples:i64,mode:i64) with IO,Alloc,Mut,Div,Panic {
 let need=dynamic_normalization_workspace(genes,samples)
 assert(need>0)
 var factors=numeric_buffer_new(samples,samples*8)
 var logs=numeric_buffer_new(samples,samples*8)
 var work=numeric_buffer_new(need,need*8)
 var flags=integer_buffer_new(2*genes,2*genes*8)
 assert(numeric_buffer_status(&factors)==0 && numeric_buffer_status(&logs)==0 && numeric_buffer_status(&work)==0 && integer_buffer_status(&flags)==0)
 var s:i64=0
 while s<samples {numeric_buffer_set(&!factors,s,107.0);numeric_buffer_set(&!logs,s,109.0);s=s+1}
 let result=dynamic_median_ratio_size_factors(counts,genes,samples,mode,&!factors,&!logs,&!work,&!flags)
 assert(result.genes==genes && result.samples==samples && result.mode==mode)
 print("CASE ");print(id);print(" ");print(result.status);print(" ");print(result.genes_used);print(" ");print(result.genes_excluded);println("")
 s=0
 while s<samples {
  let logged=numeric_buffer_at(&logs,s)
  let factor=numeric_buffer_at(&factors,s)
  if result.status!=0 {assert(logged==109.0 && factor==107.0)}
  print("VALUE ");print(id);print(" ");print(s);print(" ")
  print_f64_sci(logged,15);print(" ");print_f64_sci(factor,15);println("")
  s=s+1
 }
 numeric_buffer_close(factors);numeric_buffer_close(logs);numeric_buffer_close(work);integer_buffer_close(flags)
}
)SIO";
 for(std::size_t id=0;id<cs.size();++id){
  const auto&c=cs[id];
  std::cout<<"fn case_"<<id<<"() with IO,Alloc,Mut,Div,Panic {\n var counts=integer_buffer_new("<<c.counts.size()<<","<<c.counts.size()*8<<")\n assert(integer_buffer_status(&counts)==0)\n";
  for(std::size_t i=0;i<c.counts.size();++i)std::cout<<" integer_buffer_set(&!counts,"<<i<<","<<c.counts[i]<<")\n";
  std::cout<<" precision_case("<<id<<",&counts,"<<c.genes<<","<<c.samples<<","<<c.mode<<")\n integer_buffer_close(counts)\n}\n";
 }
 std::cout<<"fn main()->i32 with IO,Alloc,Mut,Div,Panic {\n";
 for(std::size_t id=0;id<cs.size();++id)std::cout<<" case_"<<id<<"()\n";
 std::cout<<" println(\"STDLIB_DEEP_NORMALIZATION_PRECISION_PASS\")\n 0\n}\n";
}
int main(int argc,char**argv){
 try{
  const auto cs=cases();std::vector<Reference>rs;for(const auto&c:cs)rs.push_back(evaluate(c));
  if(argc==3&&std::string(argv[1])=="--verify"){verify(argv[2],cs,rs);return 0;}
  if(argc==2&&std::string(argv[1])=="--report"){
   std::cout<<std::setprecision(80);
   for(std::size_t id=0;id<cs.size();++id){const auto&r=rs[id];std::cout<<"CASE "<<id<<" "<<cs[id].name<<" status "<<r.status<<" used "<<r.used<<" excluded "<<r.excluded<<"\n";if(r.status==0)for(std::size_t s=0;s<r.logs.size();++s)std::cout<<"REFERENCE "<<id<<" "<<s<<" log "<<r.logs[s]<<" factor "<<r.factors[s]<<" zero "<<zero_target(r.logs[s])<<"\n";}
   return 0;
  }
  if(argc!=1)throw std::runtime_error("usage: oracle [--report | --verify ACTUAL_STDOUT]");
  emit(cs);return std::cout.good()?0:1;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
