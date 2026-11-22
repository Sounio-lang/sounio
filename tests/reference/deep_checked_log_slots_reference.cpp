// Independent C++23 high-precision checked-log slot concordance.
// The 800-digit direct identities resolve cancellation down to binary64's
// minimum-subnormal squared. No production series/preprocessing is included.
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
using Q=boost::multiprecision::number<boost::multiprecision::cpp_dec_float<800>>;
static void need(bool ok,const std::string& text){if(!ok)throw std::runtime_error(text);}
static Q absolute(const Q& x){return x<0?-x:x;}
// Determine nearest-even zero in the reference domain. cpp_dec_float's double
// conversion can flush representable subnormals; it is not a rounding oracle.
static bool rounds_to_zero(const Q& value){
 const Q half_quantum=Q(std::numeric_limits<double>::denorm_min())/Q(2);
 return absolute(value)<=half_quantum;
}
static void conversion_control(){
 const Q q(std::numeric_limits<double>::denorm_min());
 const std::array<Q,6> probes{Q(0),Q(q/4),Q(q/2),Q(q*3/4),q,Q(q*3/2)};
 const std::array<bool,6> zero{true,true,true,false,false,false};
 for(std::size_t i=0;i<probes.size();++i){
  need(rounds_to_zero(probes[i])==zero[i],"high-precision zero-rounding control");
  need(rounds_to_zero(Q(-probes[i]))==zero[i],"negative zero-rounding control");
  std::cout<<"CHECKED_LOG_CONVERSION_CONTROL probe="<<i
           <<" high_precision_zero="<<zero[i]
           <<" backend_double_bits="<<std::bit_cast<std::uint64_t>(static_cast<double>(probes[i]))<<"\n";
 }
 std::cout<<"CHECKED_LOG_CONVERSION_CONTROL_PASS half-tie=even-zero\n";
}
static Q direct(int kind,double x){
 const Q value(x);
 if(kind==0)return log(value);
 const Q l=log(Q(1)+value);
 if(kind==1)return l;
 if(kind==2)return Q(l-value);
 return Q((Q(1)+value)*l-value);
}
static void one(int kind,std::uint64_t input,int status,std::uint64_t result){
 const double x=std::bit_cast<double>(input),actual=std::bit_cast<double>(result);
 int expected=0;
 if(!std::isfinite(x))expected=2;
 else if(kind==0?x<=0:x<=-1)expected=1;
 if(expected!=0){need(status==expected&&result==0,"invalid status/stale error payload");return;}
 const Q value=direct(kind,x);
 const Q maximum(std::numeric_limits<double>::max());
 if(absolute(value)>maximum){need(status==2&&result==0,"outer overflow leaked nested value");return;}
 need(status==0&&std::isfinite(actual),"missing finite result");
 if(kind==1&&std::abs(x)<1e-16){need(result==input,"tiny/signed-zero increment lost");return;}
 const bool zero=rounds_to_zero(value);
 // Diagnostic only: the backend conversion must never decide availability.
 if(static_cast<double>(value)==0 && !zero){
  std::cerr<<std::setprecision(40)<<"CHECKED_LOG_BACKEND_FLUSH kind="<<kind
           <<" input_bits="<<input<<" actual_bits="<<result
           <<" Qvalue="<<value<<" Qcast_bits="
           <<std::bit_cast<std::uint64_t>(static_cast<double>(value))<<"\n";
 }
 if(zero){need(result==0,"represented zero must be positive normalized zero");return;}
 need(actual!=0,"represented nonzero lost");
 const Q quantum(std::numeric_limits<double>::denorm_min());
 // Additional direct-identity grid. Existing consumer concordance budgets remain
 // untouched. No uniform-ULP/correct-rounding claim follows from this tolerance.
 const Q tolerance=absolute(value)*Q("1e-11")+Q(2)*quantum;
 need(absolute(Q(actual)-value)<=tolerance,"direct high-precision value mismatch");
}
static void emit_legacy(){
 std::cout<<R"SIO(//@ run-pass
//@ requires: madaros
//@ expect-stdout: CHECKED_LOG_LEGACY_MILLION_PASS
use math::pure::{ln_checked}
use math::logarithmic::{log1p_checked,log1pmx_checked,log1p_phi_checked}
use io::sci_print::{println_f64_sci}
fn main()->i32 with IO,Mut,Div,Panic {
 var i:i64=0;var sum:f64=0.0;var correction:f64=0.0
 while i<1000000 {
  let x:f64=if i%2==0 {2.0}else{-0.75}
  let a=ln_checked(1.0+x);let av:f64=a.value
  let b=log1p_checked(x);let bv:f64=b.value
  let c=log1pmx_checked(x);let cv:f64=c.value
  let d=log1p_phi_checked(x);let dv:f64=d.value
  assert(a.status==0 && b.status==0 && c.status==0 && d.status==0)
  let contribution=((av+bv)+cv)+dv
  let adjusted=contribution-correction;let next=sum+adjusted
  correction=(next-sum)-adjusted;sum=next;i=i+1
 }
 println("CHECKED_LOG_MILLION");println(i);println_f64_sci(sum,15)
 println("CHECKED_LOG_LEGACY_MILLION_PASS");0
}
)SIO";
}
int main(int argc,char** argv){
 try{
  if(argc==2&&std::string(argv[1])=="--emit-legacy-million"){emit_legacy();return 0;}
  if(argc==2&&std::string(argv[1])=="--conversion-selftest"){conversion_control();return 0;}
  need(argc==2,"usage: ORACLE actual.stdout | --emit-legacy-million | --conversion-selftest");
  std::ifstream in(argv[1]);need(bool(in),"open actual stdout");
  std::string token;int rows=0;std::uint64_t group_input=0;bool million=false,complete=false;
  while(in>>token){
   if(token=="CHECKED_LOG_ROW"){
    int kind=0,status=0;std::int64_t input=0,result=0;
    need(bool(in>>kind>>input>>status>>result),"incomplete checked row");
    need(kind==rows%4&&rows<64,"row kind/order/count");
    const auto bits=std::bit_cast<std::uint64_t>(input);
    if(kind==0)group_input=bits;else need(bits==group_input,"row input changed by kind");
    one(kind,bits,status,std::bit_cast<std::uint64_t>(result));++rows;
   }else if(token=="CHECKED_LOG_MILLION"){
    long long n=0;std::string scalar;need(bool(in>>n>>scalar)&&!million&&n==1000000,"million record");
    const Q actual(scalar);
    auto contribution=[](const Q& x)->Q{const Q l=log(Q(1)+x);return Q(((l+l)+(l-x))+((Q(1)+x)*l-x));};
    const Q expected=(contribution(Q(2))+contribution(Q("-0.75")))*Q(500000);
    need(absolute(actual-expected)<=absolute(expected)*Q("1e-11"),"million checksum");
    million=true;
   }else if(token=="STDLIB_DEEP_CHECKED_LOG_SLOTS_PASS")complete=true;
  }
  need(rows==64&&million&&complete,"incomplete checked slots witness");
  std::cout<<"CHECKED_LOG_SLOTS_REFERENCE_PASS rows="<<rows<<" calls=4000000 precision=800 direct-identities\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
