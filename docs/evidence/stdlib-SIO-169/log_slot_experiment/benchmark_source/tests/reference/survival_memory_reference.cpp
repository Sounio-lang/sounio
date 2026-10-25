// C++23 generator/reader for paired standalone dense-in-time memory witnesses.
// The arithmetic digests cover every raw f64 field bit pattern and integer
// output. They are change detectors, not a cryptographic equality proof.
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
void generate(std::int64_t n){
 std::cout<<R"SIO(use collections::numeric_buffer::{numeric_buffer_new,numeric_buffer_at,numeric_buffer_set,numeric_buffer_close}
use collections::integer_buffer::{integer_buffer_new,integer_buffer_at,integer_buffer_set,integer_buffer_close}
use stats::checked_survival::{survival_workspace,survival_checked}
fn yes(x:bool)->i64 {if x {1}else{0}}
fn main()->i32 with IO,Alloc,Mut,Div,Panic {
)SIO";
 std::cout<<" let n:i64="<<n<<"\n";
 std::cout<<R"SIO( let q=survival_workspace(n,2000000000)
 assert(q.status==0)
 var times=numeric_buffer_new(n,n*8);var event=integer_buffer_new(n,n*8);var group=integer_buffer_new(n,n*8)
 var order=integer_buffer_new(n,n*8);var curves=numeric_buffer_new(q.curve_len,q.curve_len*8);var counts=integer_buffer_new(q.count_len,q.count_len*8)
 var i:i64=0
 while i<n {numeric_buffer_set(&!times,i,(n-1-i) as f64);integer_buffer_set(&!event,i,1);integer_buffer_set(&!group,i,i%2);i=i+1}
 let r=survival_checked(&times,&event,&group,n,0.05,&!order,&!curves,&!counts,2000000000,80000000,100000,100000,1.0e-13)
 assert(r.status==0 && r.rows==n && r.risk_visits==n)
 var a:i64=0;var b:i64=0;var c:i64=0;var d:i64=0
 i=0
 while i<q.curve_len {
  let bits:i64=f64_to_bits(numeric_buffer_at(&curves,i))
  var x=bits%1000000007;var y=bits%1000000009
  if x<0 {x=x+1000000007};if y<0 {y=y+1000000009}
  a=(a*131+x)%1000000007;b=(b*257+y)%1000000009;i=i+1
 }
 i=0
 while i<q.count_len {
  let value=integer_buffer_at(&counts,i)
  c=(c*131+value)%1000000007;d=(d*257+value)%1000000009;i=i+1
 }
 let median:i64=f64_to_bits(r.median)
 let median0:i64=f64_to_bits(r.median0);let median1:i64=f64_to_bits(r.median1)
 println("MEMORY {} {} {} {} {} {} {} {} {}",n,r.status,r.rows,r.risk_visits,r.payload_bytes,yes(r.median_available),r.median_status,median,r.critical_evaluations)
 println("RAW_BUFFER_BITS {} {} {} {}",a,b,c,d)
 println("MEDIANS {} {} {} {} {} {}",yes(r.median0_available),r.median0_status,median0,yes(r.median1_available),r.median1_status,median1)
 let l=r.logrank
 let e:i64=f64_to_bits(l.expected1);let score:i64=f64_to_bits(l.score);let v:i64=f64_to_bits(l.variance)
 let z:i64=f64_to_bits(l.z);let chi:i64=f64_to_bits(l.chi2);let p:i64=f64_to_bits(l.p);let lp:i64=f64_to_bits(l.log_p);let oe:i64=f64_to_bits(l.oe_ratio)
 println("LOGRANK {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",l.status,yes(l.available),l.observed1,e,score,v,z,chi,yes(l.p_available),p,yes(l.log_p_available),lp,yes(l.oe_ratio_available),oe,l.kernel_iterations)
 numeric_buffer_close(times);integer_buffer_close(event);integer_buffer_close(group);integer_buffer_close(order);numeric_buffer_close(curves);integer_buffer_close(counts)
 println("SURVIVAL_MEMORY_COMPLETE")
 0
}
)SIO";
}
int verify(const char*file){
 std::ifstream in(file);std::string line;bool complete=false;int headers=0,digests=0,medians=0,logranks=0;
 while(std::getline(in,line)){std::istringstream s(line);std::string kind;s>>kind;
  if(kind=="MEMORY"){std::array<std::int64_t,9>a{};for(auto&x:a)s>>x;if(!s||(a[0]!=100000&&a[0]!=1000000)||a[1]!=0||a[2]!=a[0]||a[3]!=a[0]||a[4]!=296*a[0]||a[5]!=1||a[6]!=0||std::bit_cast<double>(a[7])!=double(a[0]/2-1)||a[8]<=0){std::cerr<<"SURVIVAL_MEMORY_METADATA_FAILURE\n";return 1;}++headers;
  }else if(kind=="RAW_BUFFER_BITS"){std::array<std::int64_t,4>a{};for(auto&x:a)s>>x;if(!s||a[0]<0||a[0]>=1000000007||a[1]<0||a[1]>=1000000009||a[2]<0||a[2]>=1000000007||a[3]<0||a[3]>=1000000009){std::cerr<<"SURVIVAL_MEMORY_DIGEST_FAILURE\n";return 1;}++digests;
  }else if(kind=="MEDIANS"){++medians;
  }else if(kind=="LOGRANK"){++logranks;
  }else if(kind=="SURVIVAL_MEMORY_COMPLETE"){complete=true;
  }else{std::cerr<<"SURVIVAL_MEMORY_UNEXPECTED_OUTPUT\n";return 1;}
 }
 if(!complete||headers!=1||digests!=1||medians!=1||logranks!=1){std::cerr<<"SURVIVAL_MEMORY_INCOMPLETE_OUTPUT\n";return 10;}
 std::cout<<"SURVIVAL_MEMORY_REFERENCE_PASS\n";return 0;
}
int main(int argc,char**argv){
 if(argc==3&&std::string(argv[1])=="--generate"){std::string arg=argv[2];if(arg!="100000"&&arg!="1000000")return 64;generate(std::stoll(arg));return 0;}
 if(argc==2)return verify(argv[1]);
 std::cerr<<"usage: reference --generate 100000|1000000 | ACTUAL\n";return 64;
}
