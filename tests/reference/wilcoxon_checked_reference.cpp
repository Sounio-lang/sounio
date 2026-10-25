// C++23 exact conditional sign enumeration / binomial control, not a copied DP.
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <algorithm>
#include <array>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/constants/constants.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
using I=boost::multiprecision::cpp_int;
using Q=boost::multiprecision::number<boost::multiprecision::cpp_bin_float<4096,boost::multiprecision::digit_base_2,void,std::int32_t,-10000000,10000000>>;
using N=boost::multiprecision::number<boost::multiprecision::cpp_bin_float<113,boost::multiprecision::digit_base_2,void,std::int32_t,-100000000,100000000>>;
struct Cohort{std::vector<double>d;bool all_tied=false;};
std::vector<Cohort> cases(){
 std::vector<Cohort> c={{{}},{{0.,-0.,0.}},{{1.}},{{-1.}},{{1.5,2.5,-3.5,4.,-1.,5.,2.,3.}},{{1.,1.,-2.,2.,3.}},{{0.,0.,2.,-3.,4.}},{{1e-15,-2e-15,3e-15}},{{1.,1.0000000000000002,-1.0000000000000004,-2.}}};
 for(int n=2;n<=12;++n){std::vector<double>d;for(int i=0;i<n;i++)d.push_back(i%7==0?0.:(i%3==0?-1.:1.)*(i%5+1));c.push_back({d});}
 for(auto [n,positive,zeros]:std::vector<std::array<int,3>>{{62,62,0},{63,62,0},{124,1,0},{512,256,3},{1074,1074,0},{1075,1,0},{1080,1080,0},{2048,2047,0}}){std::vector<double>d(zeros,0.);for(int i=0;i<n;++i)d.push_back(i<positive?1.:-1.);c.push_back({d,true});}
 return c;
}
struct Ranked{std::vector<std::int64_t>w;std::int64_t total=0,observed=0,zeros=0,ties=0,gcd=0;Q variance=0,z=0,rb=0;};
Ranked ranked(const Cohort&c,int method){
 Ranked r;std::vector<std::size_t> ix(c.d.size());std::iota(ix.begin(),ix.end(),0);
 std::sort(ix.begin(),ix.end(),[&](auto a,auto b){auto x=std::abs(c.d[a]),y=std::abs(c.d[b]);return x<y||(x==y&&a<b);});
 while(r.zeros<static_cast<std::int64_t>(ix.size())&&c.d[ix[r.zeros]]==0)++r.zeros;
 for(std::size_t g=static_cast<std::size_t>(r.zeros);g<ix.size();){std::size_t h=g+1;while(h<ix.size()&&std::abs(c.d[ix[h]])==std::abs(c.d[ix[g]]))++h;if(h-g>1)++r.ties;
  auto rank2=static_cast<std::int64_t>(g+h+1)-(method==0?2*r.zeros:0);
  for(auto k=g;k<h;k++){r.w.push_back(c.d[ix[k]]>0?rank2:-rank2);r.total+=rank2;if(c.d[ix[k]]>0)r.observed+=rank2;r.gcd=std::gcd(r.gcd,rank2);r.variance+=Q(rank2)*rank2/16;}
  g=h;
 }
 if(r.gcd==0)r.gcd=1;
 if(!r.w.empty()){Q signed_sum=2*r.observed-r.total;r.rb=signed_sum/r.total;Q raw=signed_sum/4;Q magnitude=abs(raw);magnitude=magnitude>Q(.5)?Q(magnitude-Q(.5)):Q(0);r.z=(raw<0?-magnitude:magnitude)/sqrt(r.variance);}
 return r;
}
bool include(std::int64_t at,std::int64_t total,std::int64_t observed,int alternative){if(alternative==1)return at<=observed;if(alternative==2)return at>=observed;auto lo=std::min(observed,total-observed);return at<=lo||at>=total-lo;}
I numerator(const Cohort&c,const Ranked&r,int alternative){
 I count=0;auto n=r.w.size();
 if(c.all_tied){I coefficient=1;auto positive=std::count_if(r.w.begin(),r.w.end(),[](auto x){return x>0;});for(std::size_t k=0;k<=n;k++){if(include(static_cast<std::int64_t>(k),static_cast<std::int64_t>(n),positive,alternative))count+=coefficient;if(k<n)coefficient=coefficient*(n-k)/(k+1);}return count;}
 for(std::uint64_t mask=0;mask<(std::uint64_t(1)<<n);++mask){std::int64_t at=0;for(std::size_t k=0;k<n;k++)if(mask&(std::uint64_t(1)<<k))at+=std::abs(r.w[k]);if(include(at,r.total,r.observed,alternative))++count;}
 return count;
}
double val(std::int64_t bits){return std::bit_cast<double>(bits);}
std::string lit(double x){std::ostringstream s;s<<std::scientific<<std::setprecision(17)<<x;return s.str();}
void generate(const std::vector<Cohort>&c){
 std::cout<<R"SIO(use collections::numeric_buffer::{numeric_buffer_new,numeric_buffer_set,numeric_buffer_close}
use collections::integer_buffer::{IntegerBuffer,integer_buffer_new,integer_buffer_at,integer_buffer_set,integer_buffer_close}
use stats::checked_wilcoxon::{CheckedWilcoxonRanks,wilcoxon_rank_workspace,wilcoxon_rank_checked,wilcoxon_exact_workspace,wilcoxon_exact_ranked_checked,wilcoxon_normal_approx_checked}
fn flag(x:bool)->i64 {if x {1}else{0}}
fn bits(x:f64)->i64 {f64_to_bits(x)}
fn row(id:i64,zero:i64) with IO,Alloc,Mut,Div,Panic {
 var n:i64=0
)SIO";
 for(std::size_t id=0;id<c.size();id++)std::cout<<" if id=="<<id<<" {n="<<c[id].d.size()<<"}\n";
 std::cout<<R"SIO( let layout=wilcoxon_rank_workspace(n,64000000)
 assert(layout.status==0)
 var d=numeric_buffer_new(n,n*8)
 var magnitude=numeric_buffer_new(n,n*8)
 var order=integer_buffer_new(n,n*8)
 var ranks=integer_buffer_new(n,n*8)
)SIO";
 for(std::size_t id=0;id<c.size();id++){std::cout<<" if id=="<<id<<" {\n";if(c[id].all_tied){auto z=std::count(c[id].d.begin(),c[id].d.end(),0.);auto pos=std::count(c[id].d.begin(),c[id].d.end(),1.);std::cout<<"  var i:i64=0\n  while i<n {numeric_buffer_set(&!d,i,if i<"<<z<<" {0.0}else{if i<"<<z+pos<<" {1.0}else{-1.0}});i=i+1}\n";}else for(std::size_t i=0;i<c[id].d.size();i++)std::cout<<"  numeric_buffer_set(&!d,"<<i<<","<<lit(c[id].d[i])<<")\n";std::cout<<" }\n";}
 std::cout<<R"SIO( let r=wilcoxon_rank_checked(&d,n,zero,true,&!magnitude,&!order,&!ranks,80000000)
 print("RANK ");print(id);print(" ");print(zero);print(" ");print(r.status);print(" ");print(r.n_eff);print(" ");print(r.n_zero);print(" ");print(r.tie_groups);print(" ");print(r.rank2_total);print(" ");print(r.rank2_positive);print(" ");print(bits(r.w_plus));print(" ");print(bits(r.w_minus));print(" ");print(flag(r.rank_biserial_available));print(" ");print(bits(r.rank_biserial));print(" ");print(flag(r.z_available));print(" ");print(bits(r.z));print(" ");println(bits(r.z_over_sqrt_n))
 assert(r.status==0)
 var alt:i64=0
 while alt<3 {approx_row(id,zero,alt,&r);exact_row(id,zero,alt,r.n_eff,&ranks);alt=alt+1}
 numeric_buffer_close(d);numeric_buffer_close(magnitude);integer_buffer_close(order);integer_buffer_close(ranks)
}
fn approx_row(id:i64,zero:i64,alt:i64,r:&CheckedWilcoxonRanks) with IO,Mut,Div,Panic {
 let p=wilcoxon_normal_approx_checked(r,alt,4096,1.0e-11)
 assert(p.status==0 && p.method==2 && !p.numerator_available && p.count_operations==0)
 print("APPROX ");print(id);print(" ");print(zero);print(" ");print(alt);print(" ");print(p.status);print(" ");print(flag(p.value_available));print(" ");print(flag(p.log_available));print(" ");print(bits(p.value));print(" ");print(bits(p.log_value));print(" ");println(p.normal_iterations)
}
fn exact_row(id:i64,zero:i64,alt:i64,n_eff:i64,ranks:&IntegerBuffer) with IO,Alloc,Mut,Div,Panic {
 let q=wilcoxon_exact_workspace(ranks,n_eff,alt,64000000,1000000000)
 assert(q.status==0 && q.operation_bound_available)
 var counts=integer_buffer_new(q.count_len,q.count_len*8)
 var num=integer_buffer_new(q.numerator_len,q.numerator_len*8)
 let p=wilcoxon_exact_ranked_checked(ranks,n_eff,alt,&!counts,&!num,64000000,1000000000)
 print("EXACT ");print(id);print(" ");print(zero);print(" ");print(alt);print(" ");print(p.status);print(" ");print(flag(p.value_available));print(" ");print(flag(p.log_available));print(" ");print(flag(p.numerator_available));print(" ");print(bits(p.value));print(" ");print(bits(p.log_value));print(" ");print(p.numerator_limbs);print(" ");print(p.denominator_exponent);print(" ");print(p.support);print(" ");print(p.gcd_rank2);print(" ");print(p.count_operations);print(" ");print(p.payload_bytes);print(" ");print(q.operation_bound);print(" ");println(p.method)
 var limb:i64=0
 while limb<q.limbs {let digit:i64=integer_buffer_at(&num,limb);println("DIGIT {} {} {} {} {}",id,zero,alt,limb,digit);limb=limb+1}
 integer_buffer_close(counts);integer_buffer_close(num)
}
fn main()->i32 with IO,Alloc,Mut,Div,Panic {
)SIO";
 for(std::size_t id=0;id<c.size();id++)for(int zero=0;zero<2;zero++)std::cout<<" row("<<id<<","<<zero<<")\n";
 std::cout<<" println(\"WILCOXON_ROWS_COMPLETE "<<c.size()<<"\")\n 0\n}\n";
}
void generate_legacy(const std::vector<Cohort>&c){
 std::cout<<"use stats::wilcoxon::{wilcoxon_signed_rank,wilcoxon_one_sample}\nfn row(id:i64,paired:bool) with IO,Alloc,Mut,Div,Panic {\n var x:[f64;256]=[0.0;256]\n var y:[f64;256]=[0.0;256]\n var n:i32=0\n";
 std::size_t count=0;
 for(std::size_t id=0;id<c.size();id++)if(!c[id].all_tied){++count;std::cout<<" if id=="<<id<<" {n="<<c[id].d.size()<<"\n";for(std::size_t k=0;k<c[id].d.size();k++)std::cout<<" x["<<k<<"]="<<lit(c[id].d[k])<<"\n";std::cout<<" }\n";}
 std::cout<<R"SIO( let r=if paired {wilcoxon_signed_rank(&x,&y,n)}else{wilcoxon_one_sample(&x,n,0.0)}
 let kind:i64=if paired {0}else{1}
 let neff:i64=r.n_eff;let wp:i64=f64_to_bits(r.w_plus);let wm:i64=f64_to_bits(r.w_minus)
 let pb:i64=f64_to_bits(r.p_value);let zb:i64=f64_to_bits(r.z);let eb:i64=f64_to_bits(r.effect_r)
 println("LEGACY {} {} {} {} {} {} {} {}",id,kind,neff,wp,wm,pb,zb,eb)
}
fn main()->i32 with IO,Alloc,Mut,Div,Panic {
)SIO";
 for(std::size_t id=0;id<c.size();id++)if(!c[id].all_tied)std::cout<<" row("<<id<<",true)\n row("<<id<<",false)\n";
 std::cout<<" println(\"LEGACY_WILCOXON_COMPLETE "<<count<<"\")\n 0\n}\n";
}
int verify_legacy(const std::vector<Cohort>&c,const char*path){
 std::ifstream in(path);if(!in)return 3;std::string line;int errors=0;std::size_t count=0;
 for(std::size_t id=0;id<c.size();id++)if(!c[id].all_tied){++count;auto r=ranked(c[id],0);I num=numerator(c[id],r,0);Q exact=Q(num)/Q(I(1)<<r.w.size());for(int kind=0;kind<2;kind++){
  if(!std::getline(in,line))return 4;
  std::istringstream s(line);std::string tag,extra;std::size_t actual;int k;std::int64_t n,wp,wm,pb,zb,eb;
  if(!(s>>tag>>actual>>k>>n>>wp>>wm>>pb>>zb>>eb)||tag!="LEGACY"||actual!=id||k!=kind||s>>extra)return 5;
  bool ok=n==static_cast<std::int64_t>(r.w.size())&&val(wp)==r.observed/2.0&&val(wm)==(r.total-r.observed)/2.0&&std::isfinite(val(pb))&&abs(Q(val(pb))-exact)<Q("1e-14");
  if(!r.w.empty())ok=ok&&abs(Q(val(zb))-r.z)<Q("1e-12")&&abs(Q(val(eb))-r.z/sqrt(Q(r.w.size())))<Q("1e-12");
  if(!ok){std::cerr<<std::setprecision(17)<<"FAIL legacy id="<<id<<" kind="<<kind<<" n="<<n<<" expected_n="<<r.w.size()<<" p="<<val(pb)<<" expected_p="<<exact<<'\n';++errors;}
 }}
 if(!std::getline(in,line)||line!="LEGACY_WILCOXON_COMPLETE "+std::to_string(count)||std::getline(in,line))return 6;
 std::cout<<"LEGACY_WILCOXON_ROWS "<<2*count<<" ERRORS "<<errors<<'\n';return errors?1:0;
}
// Positive scaled Normal-tail integral, independent of Sounio Gamma CF.
// SF(z)=phi(z)/z * integral_0^infinity exp(-u-u^2/(2*z^2)) du.
N mills_two_log(N z){
 if(z<=0)throw std::runtime_error("Mills oracle requires positive z");
 auto integrand=[&](const N&u)->N{return exp(-u-u*u/(2*z*z));};
 N e1=0,e2=0,l1=0,l2=0;
 N first=boost::math::quadrature::gauss_kronrod<N,61>::integrate(integrand,N(0),N(128),12,N("1e-24"),&e1,&l1);
 N second=boost::math::quadrature::gauss_kronrod<N,61>::integrate(integrand,N(0),N(128),15,N("1e-29"),&e2,&l2);
 if(second<=0||second>1||abs(first-second)>second*N("1e-24")||e2>second*N("1e-24")||exp(N(-128))>second*N("1e-25"))throw std::runtime_error("Mills oracle refinement/truncation refused");
 return log(N(2))-z*z/2-log(z)-log(N(2)*boost::math::constants::pi<N>())/2+log(second);
}
void mills_controls(){
 boost::math::normal_distribution<N> normal;
 for(int z:{1,3,8,30,100}){
  N expected=log(N(2)*cdf(complement(normal,N(z))));
  if(!boost::multiprecision::isfinite(expected)||abs(mills_two_log(N(z))-expected)>N("1e-23"))throw std::runtime_error("Mills oracle independent overlap control failed");
 }
}
void generate_scale(){
 std::cout<<R"SIO(use collections::numeric_buffer::{numeric_buffer_new,numeric_buffer_set,numeric_buffer_close}
use collections::integer_buffer::{integer_buffer_new,integer_buffer_close}
use stats::checked_wilcoxon::{wilcoxon_rank_workspace,wilcoxon_rank_checked,wilcoxon_exact_workspace,wilcoxon_normal_approx_checked}
fn bits(x:f64)->i64 {f64_to_bits(x)}
fn flag(x:bool)->i64 {if x {1}else{0}}
fn main()->i32 with IO,Alloc,Mut,Div,Panic {
 let n:i64=1000000
 let layout=wilcoxon_rank_workspace(n,64000000)
 assert(layout.status==0 && layout.payload_bytes==24000000)
 var d=numeric_buffer_new(n,n*8)
 var magnitude=numeric_buffer_new(n,n*8)
 var order=integer_buffer_new(n,n*8)
 var ranks=integer_buffer_new(n,n*8)
 var i:i64=0
 while i<n {numeric_buffer_set(&!d,i,if i<600000 {1.0}else{-1.0});i=i+1}
 let r=wilcoxon_rank_checked(&d,n,0,true,&!magnitude,&!order,&!ranks,80000000)
 assert(r.status==0 && r.n_eff==n && r.n_zero==0 && r.tie_groups==1)
 print("SCALE ");print(r.status);print(" ");print(r.n_eff);print(" ");print(r.rank2_total);print(" ");print(r.rank2_positive);print(" ");print(bits(r.w_plus));print(" ");print(bits(r.w_minus));print(" ");print(bits(r.rank_biserial));print(" ");print(bits(r.z));print(" ");print(bits(r.z_over_sqrt_n));print(" ");print(r.sort_comparisons);print(" ");println(r.sort_swaps)
 let q=wilcoxon_exact_workspace(&ranks,n,0,64000000,1000000000)
 assert(q.status==6 && !q.operation_bound_available)
 print("CAP ");print(q.status);print(" ");print(q.support);print(" ");print(q.limbs);print(" ");print(q.count_len);print(" ");println(q.payload_bytes)
 let p=wilcoxon_normal_approx_checked(&r,0,4096,1.0e-11)
 assert(p.status==0 && p.method==2 && p.count_operations==0 && !p.numerator_available)
 print("APPROX_SCALE ");print(p.status);print(" ");print(flag(p.value_available));print(" ");print(flag(p.log_available));print(" ");print(bits(p.value));print(" ");print(bits(p.log_value));print(" ");println(p.normal_iterations)
 numeric_buffer_close(d);numeric_buffer_close(magnitude);integer_buffer_close(order);integer_buffer_close(ranks)
 println("WILCOXON_SCALE_COMPLETE")
 0
}
)SIO";
}
int verify_scale(const char*path){
 std::ifstream in(path);if(!in)return 3;std::string line,tag,extra;
 constexpr std::int64_t n=1000000,positive=600000,rank2=n+1,total=n*rank2,observed=positive*rank2,limbs=n/62+1;
 if(!std::getline(in,line))return 4;
 std::istringstream row(line);std::int64_t status,count,actual_total,actual_observed,wp,wm,rb,zb,eb,comparisons,swaps;
 if(!(row>>tag>>status>>count>>actual_total>>actual_observed>>wp>>wm>>rb>>zb>>eb>>comparisons>>swaps)||tag!="SCALE"||row>>extra)return 5;
 Q variance=Q(n)*rank2*rank2/16;
 Q expected_z=(Q(2*observed-total)/4-Q(.5))/sqrt(variance);
 if(status||count!=n||actual_total!=total||actual_observed!=observed||val(wp)!=observed/2.0||val(wm)!=(total-observed)/2.0||abs(Q(val(rb))-Q(2*observed-total)/total)>Q("1e-14")||abs(Q(val(zb))-expected_z)>Q("1e-10")||abs(Q(val(eb))-expected_z/sqrt(Q(n)))>Q("1e-12")||comparisons<=0||comparisons>80000000||swaps<0)return 6;
 if(!std::getline(in,line))return 7;
 std::istringstream cap(line);std::int64_t support,actual_limbs,len,bytes;
 if(!(cap>>tag>>status>>support>>actual_limbs>>len>>bytes)||tag!="CAP"||status!=6||support!=n||actual_limbs!=limbs||len!=(n+3)*limbs||bytes!=(n+4)*limbs*8||cap>>extra)return 8;
 if(!std::getline(in,line))return 9;
 std::istringstream approximation(line);int va,la;std::int64_t vb,lb,iterations;
 if(!(approximation>>tag>>status>>va>>la>>vb>>lb>>iterations)||tag!="APPROX_SCALE"||approximation>>extra)return 10;
 mills_controls();N logged=mills_two_log(N(expected_z));N p=exp(logged);
 if(status||va!=(p.convert_to<double>()>0)||!la||!std::isfinite(val(lb))||abs(N(val(lb))-logged)>N("3e-10")||iterations<0||iterations>4096){std::cerr<<std::setprecision(40)<<"SCALE_NORMAL_DIAGNOSTIC status="<<status<<" value_flag="<<va<<" expected_value="<<p.convert_to<double>()<<" log_flag="<<la<<" z="<<val(zb)<<" expected_z="<<expected_z<<" log="<<val(lb)<<" expected_log="<<logged<<" abs_log_error="<<abs(N(val(lb))-logged)<<" iterations="<<iterations<<"\n";return 11;}
 if(!std::getline(in,line)||line!="WILCOXON_SCALE_COMPLETE"||std::getline(in,line))return 12;
 std::cout<<std::setprecision(17)<<"WILCOXON_SCALE_REFERENCE_PASS n="<<n<<" scratch_payload="<<n*24<<" input_payload="<<n*8<<" exact_refused_payload="<<bytes<<" normal_log_error="<<abs(N(val(lb))-logged)<<'\n';return 0;
}
int main(int argc,char**argv){auto c=cases();if(argc==2&&std::string(argv[1])=="--generate-scale"){generate_scale();return 0;}if(argc==3&&std::string(argv[1])=="--scale")return verify_scale(argv[2]);if(argc==2&&std::string(argv[1])=="--generate-legacy"){generate_legacy(c);return 0;}if(argc==3&&std::string(argv[1])=="--legacy")return verify_legacy(c,argv[2]);if(argc==2&&std::string(argv[1])=="--generate"){generate(c);return 0;}if(argc!=2)return 2;std::ifstream in(argv[1]);if(!in)return 3;int errors=0;std::string line;
 auto fail=[&](std::string text){std::cerr<<text<<'\n';++errors;};
 for(std::size_t id=0;id<c.size();id++)for(int zero=0;zero<2;zero++){
  if(!std::getline(in,line))return 4;std::istringstream row(line);std::string tag,extra;std::size_t actual;int policy,status,rb_available,z_available;std::int64_t n,z,t,total,observed,wp,wm,rb,zb,eb;
  if(!(row>>tag>>actual>>policy>>status>>n>>z>>t>>total>>observed>>wp>>wm>>rb_available>>rb>>z_available>>zb>>eb)||tag!="RANK"||actual!=id||policy!=zero||row>>extra)return 5;
  auto r=ranked(c[id],zero);auto context="cohort="+std::to_string(id)+" zero="+std::to_string(zero);
  if(status||n!=static_cast<std::int64_t>(r.w.size())||z!=r.zeros||t!=r.ties||total!=r.total||observed!=r.observed||val(wp)!=Q(r.observed/2.0).convert_to<double>()||val(wm)!=Q((r.total-r.observed)/2.0).convert_to<double>())fail("rank "+context);
  if(rb_available!=!r.w.empty()||z_available!=!r.w.empty())fail("effect availability "+context);
  if(!r.w.empty()&&(!std::isfinite(val(rb))||!std::isfinite(val(zb))||abs(Q(val(rb))-r.rb)>Q("1e-13")||abs(Q(val(zb))-r.z)>Q("1e-12")||abs(Q(val(eb))-r.z/sqrt(Q(r.w.size())))>Q("1e-12")))fail("effect value "+context);
  for(int alt=0;alt<3;alt++){
   if(!std::getline(in,line))return 11;
   {std::istringstream arow(line);int actual_alt,av,al;std::int64_t ab,alb,iterations;
    if(!(arow>>tag>>actual>>policy>>actual_alt>>status>>av>>al>>ab>>alb>>iterations)||tag!="APPROX"||actual!=id||policy!=zero||actual_alt!=alt||arow>>extra)return 12;
    N expected=1,logged=0;
    if(!r.w.empty()){
     Q raw=Q(2*r.observed-r.total)/4;
     Q normal_z=alt==0?r.z:Q((raw+(alt==1?Q(.5):Q(-.5)))/sqrt(r.variance));
     N nz(normal_z);boost::math::normal_distribution<N> normal;
     if(alt==0){expected=N(2)*cdf(complement(normal,N(abs(nz))));logged=log(expected);}
     else if(alt==1){expected=cdf(normal,nz);logged=nz>0?boost::math::log1p(-cdf(complement(normal,nz))):log(expected);}
     else{expected=cdf(complement(normal,nz));logged=nz<0?boost::math::log1p(-cdf(normal,nz)):log(expected);}
    }
    double rounded=expected.convert_to<double>(),rounded_log=logged.convert_to<double>();
    bool log_available=logged==0||rounded_log!=0;
    N magnitude=abs(logged);N log_limit=magnitude<N(1)?magnitude*N("3e-10"):N("3e-10");
    N quantization=N(std::numeric_limits<double>::denorm_min())/2;if(log_limit<quantization)log_limit=quantization;
    if((av&&(val(ab)<0||val(ab)>1))||(al&&val(alb)>0))fail("normal probability/log bounds "+context);
    if(alt==0&&!r.w.empty()&&r.z==0&&(!av||!al||val(ab)!=1||val(alb)!=0))fail("nonempty balanced normal endpoint "+context);
    if(status||al!=log_available||av!=(rounded>0)||iterations<0||iterations>4096||(av&&(!std::isfinite(val(ab))||abs(N(val(ab))-expected)>expected*N("3e-10")))||(al&&(!std::isfinite(val(alb))||abs(N(val(alb))-logged)>log_limit))||(al&&val(alb)==0&&logged!=0))fail("explicit normal approximation "+context+" alt="+std::to_string(alt));
   }
   if(!std::getline(in,line))return 6;
   std::istringstream out(line);int a,va,la,na,method;std::int64_t vb,lb,limbs,exponent,support,gcd,ops,bytes,bound;
   if(!(out>>tag>>actual>>policy>>a>>status>>va>>la>>na>>vb>>lb>>limbs>>exponent>>support>>gcd>>ops>>bytes>>bound>>method)||tag!="EXACT"||actual!=id||policy!=zero||a!=alt||out>>extra)return 7;
   I expected=numerator(c[id],r,alt),denominator=I(1)<<r.w.size(),native=0;
   if(status||method!=1||!na||limbs!=static_cast<std::int64_t>(r.w.size()/62+1)||exponent!=static_cast<std::int64_t>(r.w.size())||support!=r.total/r.gcd||gcd!=r.gcd||ops<1||ops>bound||bound>1000000000||bytes!=(support+4)*limbs*8)fail("exact metadata "+context+" alt="+std::to_string(alt));
   for(std::int64_t k=0;k<limbs;k++){if(!std::getline(in,line))return 8;std::istringstream digit(line);std::int64_t at,value;if(!(digit>>tag>>actual>>policy>>a>>at>>value)||tag!="DIGIT"||actual!=id||policy!=zero||a!=alt||at!=k||value<0||value>=4611686018427387904LL||digit>>extra)return 9;native+=I(value)<<(62*k);}
   if(native!=expected)fail("exact integer numerator "+context+" alt="+std::to_string(alt));
   Q probability=Q(expected)/Q(denominator);double expected_value=probability.convert_to<double>();
   Q logged=expected==denominator?Q(0):(probability<Q(.5)?log(probability):boost::math::log1p(-Q(denominator-expected)/Q(denominator)));
   double expected_log=logged.convert_to<double>();bool log_available=expected==denominator||expected_log!=0;
   if(va!=(expected_value>0)||la!=log_available||(va&&std::bit_cast<std::int64_t>(expected_value)!=vb))fail("dyadic binary64/availability "+context+" alt="+std::to_string(alt));
   if(la&&val(lb)==0&&logged!=0)fail("representable log erased to zero "+context+" alt="+std::to_string(alt));
   if(la&&(!std::isfinite(val(lb))||abs(Q(val(lb))-logged)>std::max(Q("3e-12"),Q(abs(logged)*Q("2e-15")))))fail("log probability "+context+" alt="+std::to_string(alt));
  }
 }
 if(!std::getline(in,line)||line!="WILCOXON_ROWS_COMPLETE "+std::to_string(c.size())||std::getline(in,line)){std::cerr<<"WILCOXON_REFERENCE_INCOMPLETE_OUTPUT\n";return 10;}
 std::cout<<"WILCOXON_REFERENCE_ROWS "<<c.size()<<" ZERO_POLICIES 2 ALTERNATIVES 3 ERRORS "<<errors<<'\n';return errors?1:0;
}
