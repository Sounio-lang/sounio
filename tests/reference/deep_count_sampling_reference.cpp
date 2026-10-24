// Independent unsigned-128 PCG route + binary128 rejection arithmetic.
// Fixed LEGACY increment, NOT PCG stream-selector zero/increment one.
// Seed: step from zero, add seed's unsigned64 bit pattern, step again.
// Build C++23 with -O2 -Wall -Wextra -lquadmath. Stdout emits Sounio fixture.
// --moments prints deterministic sample moments and rejects gross calibration failures.
#include <quadmath.h>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
using Q=__float128;
using U=unsigned __int128;
static Q q(const char* s){return strtoflt128(s,nullptr);}
static constexpr std::uint64_t ih=6364136223846793005ULL,il=1442695040888963407ULL;
static U increment(){return (U(ih)<<64)|il;}
static U multiplier(){return (U(2549297995355413924ULL)<<64)|4865540595714422341ULL;}
struct Rng{
 U state=0,inc=increment();
 explicit Rng(std::int64_t seed){step();state+=std::bit_cast<std::uint64_t>(seed);step();}
 std::uint64_t step(){
  state=state*multiplier()+inc;
  const auto hi=std::uint64_t(state>>64),lo=std::uint64_t(state);
  return std::rotr(hi^lo,int(hi>>58));
 }
 Q uniform(){return Q(step()>>11)/Q(9007199254740992ULL);}
};
struct Cursor{
 Rng rng;std::int64_t uniforms=0,proposals=0;
 explicit Cursor(std::int64_t seed):rng(seed){}
 Q open(){for(;;){++uniforms;Q u=rng.uniform();if(u>0)return u;}}
};
static std::int64_t poisson(Cursor& c,Q mean){
 if(mean==0)return 0;
 if(mean<30){
  Q sum=0;std::int64_t n=0;
  for(;;){++c.proposals;sum+=logq(c.open());if(sum<=-mean)return n;++n;}
 }
 Q b=q(".931")+q("2.53")*sqrtq(mean),a=-q(".059")+q(".02483")*b;
 Q ai=q("1.1239")+q("1.1328")/(b-q("3.4"));
 Q vr=q(".9277")-q("3.6224")/(b-2);
 // Mean split preserves integer part rather than rounding mean+offset.
 const auto whole=std::int64_t(mean);Q fraction=mean-Q(whole);
 for(;;){
  ++c.proposals;Q u=c.open()-q(".5"),v=c.open(),us=q(".5")-fabsq(u);
  Q off=floorq((2*a/us+b)*u+fraction+q(".43"));
  if(off < -Q(9007199254740992ULL)||off>Q(9007199254740992ULL))throw std::runtime_error("proposal range");
  auto k=whole+std::int64_t(off);
  if(k<0)continue;
  if(k>9007199254740992LL)throw std::runtime_error("count range");
  if(us>=q(".07")&&v<=vr)return k;
  if(us<q(".013")&&v>us)continue;
  // Binary128 direct log PMF is independent of Sounio's deviance/Stirling route.
  Q lp=-mean+Q(k)*logq(mean)-lgammaq(Q(k)+1);
  if(logq(v)+logq(ai)-logq(a/(us*us)+b)<=lp)return k;
 }
}
static Q gamma_log(Cursor& c,Q shape){
 bool boost=shape<1;Q a=boost?shape+1:shape,d=a-Q(1)/3,z=Q(1)/3/sqrtq(d);
 for(;;){
  ++c.proposals;Q first=c.open(),second=c.open();
  Q x=sqrtq(-2*logq(first))*cosq(2*acosq(Q(-1))*second),t=z*x;
  if(t<=-1)continue;
  Q u=c.open(),x2=x*x;
  // Independent high-precision original MT expression, not the factored polynomial.
  Q v=(1+t)*(1+t)*(1+t);
  Q bound=x2/2+d*(1-v+logq(v));
  if(u>=1-q(".0331")*x2*x2 && logq(u)>=bound)continue;
  Q result=logq(d)+logq(v);
  if(boost)result+=logq(c.open())/shape;
  return result;
 }
}
static std::int64_t nb(Cursor&c,Q mean,Q alpha){
 if(mean==0)return 0;
 if(alpha==0)return poisson(c,mean);
 Q lr=gamma_log(c,1/alpha)+logq(mean)+logq(alpha);
 // Target is binary64-rate sampler; round once at the public mixture boundary.
 double rate=double(expq(lr));
 if(rate==0)return 0;
 if(!std::isfinite(rate)||rate>4503599627370496.0)throw std::runtime_error("rate range");
 return poisson(c,Q(rate));
}
static std::int64_t signed_bits(std::uint64_t x){return std::bit_cast<std::int64_t>(x);}
static void state_checks(const std::string&obj,const Rng&r){
 std::cout<<" assert("<<obj<<".state_hi=="<<signed_bits(std::uint64_t(r.state>>64))
 <<" && "<<obj<<".state_lo=="<<signed_bits(std::uint64_t(r.state))<<")\n"
 <<" assert("<<obj<<".inc_hi=="<<ih<<" && "<<obj<<".inc_lo=="<<il<<")\n";
}
static void progress(const std::string&obj,const Cursor&c){
 std::cout<<" assert("<<obj<<".uniforms=="<<c.uniforms<<" && "<<obj<<".proposals=="<<c.proposals<<")\n";
 state_checks(obj+".rng",c.rng);
}
static void moments(){
 for(int kind=0;kind<3;++kind){
  Cursor c(913+kind);constexpr int n=20000;Q sum=0,sum2=0;
  for(int i=0;i<n;++i){
   Q v=kind==0?Q(poisson(c,40)):kind==1?expq(gamma_log(c,q("2.5"))):Q(nb(c,12,q(".4")));
   sum+=v;sum2+=v*v;
  }
  Q mean=sum/n,var=sum2/n-mean*mean;
  Q target=kind==0?Q(40):kind==1?q("2.5"):Q(12);
  Q variance=kind==0?Q(40):kind==1?q("2.5"):q("69.6");
  if(fabsq(mean-target)>8*sqrtq(variance/n))throw std::runtime_error("moment mean");
  if(fabsq(var-variance)>variance*q(".15"))throw std::runtime_error("moment variance");
  std::cout<<"kind\t"<<kind<<"\tn\t"<<n<<"\tmean\t"<<double(mean)<<"\tvariance\t"<<double(var)<<"\n";
 }
}
struct MomentResult {Q mean,variance;Cursor cursor;};
static MomentResult moment_result(int kind){
 Cursor c(913+kind);constexpr int n=20000;Q sum=0,sum2=0;
 for(int i=0;i<n;++i){Q v=kind==0?Q(poisson(c,40)):kind==1?expq(gamma_log(c,Q(2.5))):Q(nb(c,12,Q(double(.4))));sum+=v;sum2+=v*v;}
 const Q mean=sum/n;return {mean,sum2/n-mean*mean,c};
}
static void emit_moments(){
 std::cout<<R"SIO(//@ run-pass
//@ expect-stdout: STDLIB_DEEP_COUNT_SAMPLING_MOMENTS_PASS
use stats::checked_count_sampling::{count_rng,poisson_count_draw,gamma_unit_draw,nb2_count_draw}
use math::pure::{sqrt}
use io::sci_print::{print_f64_sci}
fn abs(x:f64)->f64 {if x<0.0 {-x}else{x}}
fn main()->i32 with IO,Mut,Div,Panic {
 var kind:i64=0
 while kind<3 {
  var rng=count_rng(913+kind)
  var sum:f64=0.0;var sum2:f64=0.0;var correction:f64=0.0;var correction2:f64=0.0
  var uniforms:i64=0;var proposals:i64=0;var i:i64=0
  while i<20000 {
   var value:f64=0.0
   if kind==0 {
    let draw=poisson_count_draw(rng,40.0,10000,10000)
    assert(draw.status==0)
    value=draw.count as f64;rng=draw.rng;uniforms=uniforms+draw.uniforms;proposals=proposals+draw.proposals
   }else{if kind==1 {
    let draw=gamma_unit_draw(rng,2.5,10000,10000)
    assert(draw.status==0 && draw.value_available && draw.log_available)
    value=draw.value;rng=draw.rng;uniforms=uniforms+draw.uniforms;proposals=proposals+draw.proposals
   }else{
    let draw=nb2_count_draw(rng,12.0,0.4,10000,10000)
    assert(draw.status==0)
    value=draw.count as f64;rng=draw.rng;uniforms=uniforms+draw.uniforms;proposals=proposals+draw.proposals
   }}
   let adjusted=value-correction;let next=sum+adjusted;correction=(next-sum)-adjusted;sum=next
   let square=value*value;let adjusted2=square-correction2;let next2=sum2+adjusted2;correction2=(next2-sum2)-adjusted2;sum2=next2
   i=i+1
  }
  let average:f64=sum/20000.0;let variance:f64=sum2/20000.0-average*average
  let target:f64=if kind==0 {40.0}else{if kind==1 {2.5}else{12.0}}
  let target_variance:f64=if kind==0 {40.0}else{if kind==1 {2.5}else{69.6}}
  assert(abs(average-target)<8.0*sqrt(target_variance/20000.0))
  assert(abs(variance-target_variance)<0.15*target_variance)
  let hi:i64=rng.state_hi;let lo:i64=rng.state_lo
  print("COUNT_MOMENT ");print(kind);print(" 20000 ");print_f64_sci(average,15);print(" ");print_f64_sci(variance,15)
  print(" ");print(uniforms);print(" ");print(proposals);print(" ");print(hi);print(" ");println(lo)
  kind=kind+1
 }
 println("STDLIB_DEEP_COUNT_SAMPLING_MOMENTS_PASS")
 0
}
)SIO";
}
static void check_moments(const char*path){
 std::ifstream input(path);if(!input)throw std::runtime_error("moment stdout open");
 std::string line;int cases=0;bool completed=false;
 while(std::getline(input,line)){
  std::istringstream row(line);std::string tag;row>>tag;
  if(tag=="STDLIB_DEEP_COUNT_SAMPLING_MOMENTS_PASS")completed=true;
  if(tag!="COUNT_MOMENT")continue;
  int kind,n;double mean,variance;std::int64_t uniforms,proposals,hi,lo;
  row>>kind>>n>>mean>>variance>>uniforms>>proposals>>hi>>lo;
  if(!row||kind!=cases||n!=20000||kind>=3)throw std::runtime_error("moment record");
  const auto expected=moment_result(kind);
  if(fabsq(Q(mean)-expected.mean)>q("1e-10")*fabsq(expected.mean)+q("1e-12")||fabsq(Q(variance)-expected.variance)>q("1e-10")*fabsq(expected.variance)+q("1e-12"))throw std::runtime_error("native moment concordance");
  if(uniforms!=expected.cursor.uniforms||proposals!=expected.cursor.proposals||hi!=signed_bits(std::uint64_t(expected.cursor.rng.state>>64))||lo!=signed_bits(std::uint64_t(expected.cursor.rng.state)))throw std::runtime_error("moment PCG/acceptance trace");
  ++cases;
 }
 if(cases!=3||!completed)throw std::runtime_error("moment count/completion");
 std::cout<<"DEEP_COUNT_SAMPLING_MOMENTS_CONCORDANCE_PASS kinds=3 n_each=20000\n";
}
static void status_checks(){
 Cursor first(17);const Q u=first.open();
 if(!(u>0&&logq(u)>-20))throw std::runtime_error("seed17 one-uniform budget witness");
 Cursor tiny(17);const double shape=std::numeric_limits<double>::min()/1e12;
 const Q l=gamma_log(tiny,Q(shape));
 if(!(l<0&&!std::isfinite(double(l))))throw std::runtime_error("tinyshape log overflow witness");
 std::cout<<"DEEP_COUNT_SAMPLING_STATUS_WITNESSES_PASS seed=17 positive_subnormal_shape\n";
}
static void emit_rescaled_rate(){
 std::int64_t seed=-1;Q reference_log=0;Cursor gamma_cursor(0);
 for(std::int64_t candidate=0;candidate<10000;++candidate){
  Cursor trial(candidate);const Q l=gamma_log(trial,Q(double(.001)));
  if(l> -770&&l< -750){seed=candidate;reference_log=l;gamma_cursor=trial;break;}
 }
 if(seed<0)throw std::runtime_error("bounded rescaled-rate witness search exhausted");
 if(double(expq(reference_log))!=0)throw std::runtime_error("gamma witness not linearly underflowed");
 Cursor mixed(seed);const auto count=nb(mixed,Q(4503599627370496.0),Q(1000));
 std::cout<<" let rescued_gamma=gamma_unit_draw(count_rng("<<seed<<"),0.001,10000,10000)\n"
  " assert(rescued_gamma.status==0 && rescued_gamma.value_available && rescued_gamma.log_available && rescued_gamma.rounded_underflow)\n"
  " assert(rescued_gamma.value==0.0)\n"
  " assert(near(rescued_gamma.log_value,"<<double(reference_log)<<"))\n";
 progress("rescued_gamma",gamma_cursor);
 std::cout<<" let rescued_nb=nb2_count_draw(count_rng("<<seed<<"),4503599627370496.0,1000.0,10000,10000)\n"
  " assert(rescued_nb.status==0 && rescued_nb.count=="<<count<<" && !rescued_nb.rate_underflow && rescued_nb.resolution_limited)\n";
 progress("rescued_nb",mixed);
}
struct MillionPcg {Rng rng;std::uint64_t checksum;};
static MillionPcg million_pcg(){Rng rng(77);std::uint64_t checksum=0;for(int i=0;i<1000000;++i)checksum^=rng.step()>>11;return {rng,checksum};}
static void emit_pcg_into(bool legacy){
 std::cout<<"//@ run-pass\n//@ requires: madaros\n//@ expect-stdout: STDLIB_DEEP_PCG_INTO_PASS\n"
  "use stats::checked_count_sampling::{count_rng}\n"
  "use random::pcg64_core::{pcg_step,pcg_next_f64,pcg_step_into,pcg_next_f64_into}\n"
  "fn main()->i32 with IO,Mut,Div,Panic {\n";
 int index=0;
 for(auto seed:std::vector<std::int64_t>{0,1,-1,std::numeric_limits<std::int64_t>::min(),std::numeric_limits<std::int64_t>::max()}){
  Rng rng(seed);const std::string a="old"+std::to_string(index),b="borrowed"+std::to_string(index++);
  std::cout<<" var "<<a<<"=count_rng("<<seed<<")\n var "<<b<<"=count_rng("<<seed<<")\n";
  for(int j=0;j<8;++j){
   const auto bits=rng.step();const std::string tag=std::to_string(index++);
   std::cout<<" let oldstep"<<tag<<"=pcg_step("<<a<<")\n let newstep"<<tag<<"=pcg_step_into(&!"<<b<<")\n"
    " assert(newstep"<<tag<<"=="<<signed_bits(bits)<<" && oldstep"<<tag<<".1==newstep"<<tag<<")\n "<<a<<"=oldstep"<<tag<<".0\n";
   state_checks(a,rng);state_checks(b,rng);
   const auto mantissa=rng.step()>>11;
   std::cout<<" let olduniform"<<tag<<"=pcg_next_f64("<<a<<")\n let newuniform"<<tag<<"=pcg_next_f64_into(&!"<<b<<")\n"
    " assert(newuniform"<<tag<<"==olduniform"<<tag<<".1 && newuniform"<<tag<<"*9007199254740992.0=="<<mantissa<<".0)\n "<<a<<"=olduniform"<<tag<<".0\n";
   state_checks(a,rng);state_checks(b,rng);
  }
 }
 std::cout<<" var longrng=count_rng(77)\n var i:i64=0\n var checksum:i64=0\n while i<1000000 {\n";
 if(legacy)std::cout<<"  let next=pcg_next_f64(longrng)\n  longrng=next.0\n  let u:f64=next.1\n";
 else std::cout<<"  let u:f64=pcg_next_f64_into(&!longrng)\n";
 std::cout<<"  assert(u>=0.0 && u<1.0)\n  checksum=checksum ^ ((u*9007199254740992.0) as i64)\n  i=i+1\n }\n";
 const auto expected=million_pcg();state_checks("longrng",expected.rng);
 std::cout<<" assert(checksum=="<<expected.checksum<<")\n"
  " let hi:i64=longrng.state_hi;let lo:i64=longrng.state_lo\n"
  " print(\"PCG_MILLION 1000000 \");print(hi);print(\" \");print(lo);print(\" \");println(checksum)\n"
  " println(\"STDLIB_DEEP_PCG_INTO_PASS\")\n 0\n}\n";
}
static void check_pcg_into(const char*path){
 std::ifstream f(path);if(!f)throw std::runtime_error("PCG stdout open");std::string row;bool seen=false,complete=false;const auto expected=million_pcg();
 while(std::getline(f,row)){std::istringstream in(row);std::string tag;in>>tag;
  if(tag=="STDLIB_DEEP_PCG_INTO_PASS")complete=true;
  if(tag=="PCG_MILLION"){std::int64_t n,hi,lo;std::uint64_t checksum;in>>n>>hi>>lo>>checksum;if(!in||seen||n!=1000000||hi!=signed_bits(std::uint64_t(expected.rng.state>>64))||lo!=signed_bits(std::uint64_t(expected.rng.state))||checksum!=expected.checksum)throw std::runtime_error("million PCG state/checksum");seen=true;}
 }
 if(!seen||!complete)throw std::runtime_error("PCG completion");
 std::cout<<"DEEP_PCG_INTO_REFERENCE_PASS uniforms=1000000 exact_unsigned128\n";
}
struct MillionCounts {Cursor cursor;std::int64_t sum;};
static MillionCounts million_counts(){
 Cursor cursor(901);std::int64_t sum=0;const Q threshold=expq(-Q(double(.01)));
 // Independent product-form Knuth route, not the production compensated log sum.
 for(int i=0;i<1000000;++i){Q product=1;std::int64_t count=0;const auto before=cursor.uniforms;
  for(;;){++cursor.proposals;product*=cursor.open();if(product<=threshold)break;++count;}
  if(cursor.uniforms-before>100)throw std::runtime_error("million draw budget");sum+=count;
 }
 return {cursor,sum};
}
// Independent binary128 direct-PMF PTRS route, exercising one million high-mean
// draws rather than the product-form small-mean witness above.
static MillionCounts million_ptrs(){
 Cursor cursor(901);std::int64_t sum=0;
 for(int i=0;i<1000000;++i){
  const auto before_u=cursor.uniforms,before_p=cursor.proposals;
  sum+=poisson(cursor,Q(40));
  if(cursor.uniforms-before_u>100 || cursor.proposals-before_p>100)
   throw std::runtime_error("million PTRS draw budget");
 }
 return {cursor,sum};
}
static void emit_ptrs_million(bool legacy){
 std::cout<<"//@ run-pass\n//@ requires: madaros\n//@ expect-stdout: STDLIB_DEEP_PTRS_MILLION_PASS\n"
  "use stats::checked_count_sampling::{count_rng,count_cursor_new,count_cursor_rng,count_draw_value,poisson_count_draw_into,poisson_count_draw}\n"
  "fn main()->i32 with IO,Mut,Div,Panic {\n var cursor=count_cursor_new(count_rng(901))\n var slot=count_draw_value()\n var old=count_rng(901)\n"
  " var i:i64=0;var total:i64=0;var uniforms:i64=0;var proposals:i64=0\n while i<1000000 {\n";
 if(legacy)std::cout<<"  let result=poisson_count_draw(old,40.0,100,100)\n  assert(result.status==0)\n  old=result.rng\n  total=total+result.count;uniforms=uniforms+result.uniforms;proposals=proposals+result.proposals\n";
 else std::cout<<"  let status=poisson_count_draw_into(&!cursor,40.0,100,100,&!slot)\n  assert(status==0 && !slot.resolution_limited && !slot.rate_underflow)\n  total=total+slot.count;uniforms=uniforms+slot.uniforms;proposals=proposals+slot.proposals\n";
 std::cout<<"  i=i+1\n }\n";
 if(legacy)std::cout<<" let final=old\n";else std::cout<<" let final=count_cursor_rng(&cursor)\n";
 // The checker computes the independent reference. Do not calculate it while
 // emitting the legacy diagnostic, which is expected to expose handle pressure.
 std::cout<<" let hi:i64=final.state_hi;let lo:i64=final.state_lo\n"
  " print(\"PTRS_MILLION 1000000 \");print(total);print(\" \");print(uniforms);print(\" \");print(proposals);print(\" \");print(hi);print(\" \");println(lo)\n"
  " println(\"STDLIB_DEEP_PTRS_MILLION_PASS\")\n 0\n}\n";
}
static void check_ptrs_million(const char*path){
 std::ifstream f(path);if(!f)throw std::runtime_error("PTRS stdout open");
 std::string row;bool seen=false,complete=false;const auto expected=million_ptrs();
 while(std::getline(f,row)){
  std::istringstream in(row);std::string tag;in>>tag;
  if(tag=="STDLIB_DEEP_PTRS_MILLION_PASS")complete=true;
  if(tag=="PTRS_MILLION"){
   std::int64_t n,sum,uniforms,proposals,hi,lo;in>>n>>sum>>uniforms>>proposals>>hi>>lo;
   if(!in || seen || n!=1000000 || sum!=expected.sum || uniforms!=expected.cursor.uniforms || proposals!=expected.cursor.proposals || hi!=signed_bits(std::uint64_t(expected.cursor.rng.state>>64)) || lo!=signed_bits(std::uint64_t(expected.cursor.rng.state)))
    throw std::runtime_error("million PTRS trace");
   seen=true;
  }
 }
 if(!seen||!complete)throw std::runtime_error("million PTRS completion");
 std::cout<<"DEEP_PTRS_MILLION_REFERENCE_PASS draws=1000000 mean=40 direct_binary128_PMF exact_state_counters\n";
}
static void emit_count_into(bool legacy){
 std::cout<<"//@ run-pass\n//@ requires: madaros\n//@ expect-stdout: STDLIB_DEEP_COUNT_INTO_MILLION_PASS\n"
  "use stats::checked_count_sampling::{count_rng,count_cursor_new,count_cursor_rng,count_draw_value,poisson_count_draw_into,poisson_count_draw}\n"
  "fn main()->i32 with IO,Mut,Div,Panic {\n var cursor=count_cursor_new(count_rng(901))\n var slot=count_draw_value()\n var old=count_rng(901)\n"
  " var i:i64=0;var total:i64=0;var uniforms:i64=0;var proposals:i64=0\n while i<1000000 {\n";
 if(legacy)std::cout<<"  let result=poisson_count_draw(old,0.01,100,100)\n  assert(result.status==0)\n  old=result.rng\n  total=total+result.count;uniforms=uniforms+result.uniforms;proposals=proposals+result.proposals\n";
 else std::cout<<"  let status=poisson_count_draw_into(&!cursor,0.01,100,100,&!slot)\n  assert(status==0 && !slot.resolution_limited && !slot.rate_underflow)\n  total=total+slot.count;uniforms=uniforms+slot.uniforms;proposals=proposals+slot.proposals\n";
 std::cout<<"  i=i+1\n }\n";
 if(legacy)std::cout<<" let final=old\n";else std::cout<<" let final=count_cursor_rng(&cursor)\n";
 const auto expected=million_counts();state_checks("final",expected.cursor.rng);
 std::cout<<" assert(total=="<<expected.sum<<" && uniforms=="<<expected.cursor.uniforms<<" && proposals=="<<expected.cursor.proposals<<")\n"
  " let hi:i64=final.state_hi;let lo:i64=final.state_lo\n"
  " print(\"COUNT_INTO_MILLION 1000000 \");print(total);print(\" \");print(uniforms);print(\" \");print(proposals);print(\" \");print(hi);print(\" \");println(lo)\n"
  " println(\"STDLIB_DEEP_COUNT_INTO_MILLION_PASS\")\n 0\n}\n";
}
static void check_count_into(const char*path){
 std::ifstream f(path);if(!f)throw std::runtime_error("count stdout open");std::string row;bool seen=false,complete=false;const auto expected=million_counts();
 while(std::getline(f,row)){std::istringstream in(row);std::string tag;in>>tag;
  if(tag=="STDLIB_DEEP_COUNT_INTO_MILLION_PASS")complete=true;
  if(tag=="COUNT_INTO_MILLION"){std::int64_t n,sum,uniforms,proposals,hi,lo;in>>n>>sum>>uniforms>>proposals>>hi>>lo;if(!in||seen||n!=1000000||sum!=expected.sum||uniforms!=expected.cursor.uniforms||proposals!=expected.cursor.proposals||hi!=signed_bits(std::uint64_t(expected.cursor.rng.state>>64))||lo!=signed_bits(std::uint64_t(expected.cursor.rng.state)))throw std::runtime_error("million count trace");seen=true;}
 }
 if(!seen||!complete)throw std::runtime_error("million count completion");
 std::cout<<"DEEP_COUNT_INTO_REFERENCE_PASS draws=1000000 independent_product_route\n";
}
int main(int argc,char**argv){
 try{
  std::cout<<std::setprecision(17)<<std::showpoint;
  if(argc==2&&std::string(argv[1])=="--emit-ptrs-into-million"){emit_ptrs_million(false);return 0;}
  if(argc==2&&std::string(argv[1])=="--emit-ptrs-legacy-million"){emit_ptrs_million(true);return 0;}
  if(argc==3&&std::string(argv[1])=="--check-ptrs-million"){check_ptrs_million(argv[2]);return 0;}
  if(argc==2&&std::string(argv[1])=="--emit-pcg-into"){emit_pcg_into(false);return 0;}
  if(argc==2&&std::string(argv[1])=="--emit-pcg-legacy-million"){emit_pcg_into(true);return 0;}
  if(argc==3&&std::string(argv[1])=="--check-pcg-into"){check_pcg_into(argv[2]);return 0;}
  if(argc==2&&std::string(argv[1])=="--emit-count-into-million"){emit_count_into(false);return 0;}
  if(argc==2&&std::string(argv[1])=="--emit-count-legacy-million"){emit_count_into(true);return 0;}
  if(argc==3&&std::string(argv[1])=="--check-count-into"){check_count_into(argv[2]);return 0;}
  if(argc==2&&std::string(argv[1])=="--moments"){moments();return 0;}
  if(argc==2&&std::string(argv[1])=="--emit-moments"){emit_moments();return 0;}
  if(argc==3&&std::string(argv[1])=="--check-moments"){check_moments(argv[2]);return 0;}
  if(argc==2&&std::string(argv[1])=="--status-checks"){status_checks();return 0;}
  std::cout<<"//@ run-pass\n//@ expect-stdout: STDLIB_DEEP_COUNT_SAMPLING_REFERENCE_PASS\n"
   "use stats::checked_count_sampling::{count_rng,poisson_count_draw,gamma_unit_draw,nb2_count_draw}\n"
   "use random::pcg64_core::{PcgState,pcg_step,pcg_next_f64}\n"
   "fn abs(x:f64)->f64 {if x<0.0 {-x} else {x}}\n"
   "fn near(a:f64,b:f64)->bool with Div {if b==0.0 {return a==0.0} abs(a-b)<=abs(b)*1.0e-10}\n"
   "fn main()->i32 with IO,Mut,Div,Panic {\n";
  int id=0;
  for(auto seed:std::vector<std::int64_t>{0,1,-1,std::numeric_limits<std::int64_t>::min(),std::numeric_limits<std::int64_t>::max()}){
   Rng r(seed);std::string name="r"+std::to_string(id++);
   std::cout<<" var "<<name<<"=count_rng("<<seed<<")\n";state_checks(name,r);
   for(int j=0;j<8;++j){
    auto out=r.step();std::string step="s"+std::to_string(id++);
    std::cout<<" let "<<step<<"=pcg_step("<<name<<")\n assert("<<step<<".1=="<<signed_bits(out)<<")\n "<<name<<"="<<step<<".0\n";state_checks(name,r);
   }
  }
  // A constructed odd-increment state whose NEXT output is exactly zero.
  U inv=1;for(int i=0;i<8;++i)inv*=2-multiplier()*inv;
  U pre=(U(0)-increment())*inv;
  if(pre*multiplier()+increment()!=0)throw std::runtime_error("inverse");
  std::cout<<" let z=PcgState{state_hi:"<<signed_bits(std::uint64_t(pre>>64))
  <<",state_lo:"<<signed_bits(std::uint64_t(pre))<<",inc_hi:"<<ih<<",inc_lo:"<<il<<"}\n"
   " let rejected=poisson_count_draw(z,2.0,1,10)\n"
   " assert(rejected.status==3 && rejected.uniforms==1 && rejected.proposals==1)\n"
   " assert(rejected.rng.state_hi==0 && rejected.rng.state_lo==0)\n";
  int cidx=0;
  for(double mean:std::vector<double>{0,1e-30,.1,3,29.9,30,40,1000,1e8,4503599627370496.0}){
   Cursor c(123+cidx);auto count=poisson(c,Q(mean));std::string obj="p"+std::to_string(cidx++);
   std::cout<<" let "<<obj<<"=poisson_count_draw(count_rng("<<123+cidx-1<<"),"<<mean<<",10000,10000)\n"
    <<" assert("<<obj<<".status==0 && "<<obj<<".count=="<<count<<")\n";progress(obj,c);
  }
  cidx=0;
  for(double shape:std::vector<double>{.01,.5,1,2,10,1e8,9007199254740992.0}){
   Cursor c(301+cidx);Q l=gamma_log(c,Q(shape));std::string obj="g"+std::to_string(cidx++);
   std::cout<<" let "<<obj<<"=gamma_unit_draw(count_rng("<<300+cidx<<"),"<<shape<<",10000,10000)\n"
    <<" assert("<<obj<<".status==0 && "<<obj<<".log_available && "<<obj<<".value_available)\n"
    <<" assert(near("<<obj<<".log_value,"<<double(l)<<") && near("<<obj<<".value,"<<double(expq(l))<<"))\n";progress(obj,c);
  }
  cidx=0;
  for(auto pair:std::vector<std::pair<double,double>>{{0,.5},{3,0},{12,.4},{100,2},{1e8,.01}}){
   Cursor c(501+cidx);auto count=nb(c,Q(pair.first),Q(pair.second));std::string obj="nb"+std::to_string(cidx++);
   std::cout<<" let "<<obj<<"=nb2_count_draw(count_rng("<<500+cidx<<"),"<<pair.first<<","<<pair.second<<",10000,10000)\n"
    <<" assert("<<obj<<".status==0 && "<<obj<<".count=="<<count<<")\n";progress(obj,c);
  }
  emit_rescaled_rate();
  std::cout<<" println(\"STDLIB_DEEP_COUNT_SAMPLING_REFERENCE_PASS\")\n 0\n}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
