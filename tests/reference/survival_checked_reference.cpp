// C++23 independent exact small risk tables + extended-exponent numerical oracle.
// No R/Python execution. Exact binary64 times define distinct event times.
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
using N=boost::multiprecision::number<boost::multiprecision::cpp_bin_float<113,boost::multiprecision::digit_base_2,void,std::int32_t,-100000000,100000000>>;
using R=boost::multiprecision::cpp_rational;
struct Input {double t;int e,g;};
using Cohort=std::vector<Input>;
using MedianSpec=std::array<std::array<int,2>,5>;
std::array<MedianSpec,2> median_specs(){return {{{{{1000000,3872},{900001,2623},{800011,3986},{700001,2569},{500830,246511}}},{{{1000000,3620},{900001,3027},{800011,2678},{700001,1597},{609150,300703}}}}};}
std::vector<Cohort> cases(bool scale){
 if(scale){std::vector<Cohort> r(2);for(int i=0;i<100000;i++){r[0].push_back({double((i*37)%997),i%5!=0,i%2});r[1].push_back({double((std::int64_t(i)*7919)%100000),1,i%2});}for(auto spec:median_specs()){
 Cohort c;std::size_t index=0;
 for(int j=0;j<5;j++){int risk=spec[j][0],deaths=spec[j][1],next=j<4?spec[j+1][0]:0;
  for(int k=0;k<deaths;k++){c.push_back({double(j+1),1,int(index%2)});++index;}
  int censored=risk-deaths-next;
  for(int k=0;k<censored;k++){c.push_back({double(j==4?6:j+1),0,int(index%2)});++index;}
 }
 r.push_back(std::move(c));
 }return r;}
 std::vector<Cohort> r={ {},{{0,1,0}},{{1,0,1}},{{1,1,0},{1,1,1}},{{1,0,0},{1,0,1}},
 {{1,1,0},{3,1,0},{5,1,0},{2,1,1},{4,1,1},{6,1,1}},
 {{1,1,0},{2,1,0},{3,1,0},{1,1,1},{2,1,1},{3,1,1}},
 {{1,1,0},{1,0,1},{1,1,1},{2,0,0},{3,1,1},{4,0,1}},
 {{1,0,0},{2,0,0},{3,0,1},{4,0,1}},{{1,1,0},{2,1,0},{3,0,0}},
 {{1,1,0},{1.0000000000000002,1,1},{1.0000000000000004,0,0},{2,1,1}},
 {{1e-15,1,0},{2e-15,0,1},{3e-15,1,1},{1,0,0}},
 {{1e200,1,0},{2e200,0,1},{3e200,1,1}},
 {{-0.0,1,0},{0.0,0,1},{1,1,1}},
 {{1,1,0},{2,0,0},{3,0,0},{4,1,1},{5,0,1}},
 {{1,1,0},{2,1,0},{3,1,1},{4,0,1},{5,1,1},{5,0,0}}};
 for(int seed=0;seed<10;seed++){Cohort c;for(int i=0;i<12+seed;i++)c.push_back({double((i*7+seed)%9),((i+seed)%4)!=0,(i*3+seed)%2});r.push_back(c);}
 Cohort separated;for(int i=0;i<256;i++)separated.push_back({double(i+1),1,i<128?0:1});r.push_back(separated);
 for(int deaths:{500,501}){Cohort c;for(int i=0;i<1001;i++)c.push_back({i<deaths?1.:2.,i<deaths,i%2});r.push_back(c);}
 return r;
}
N mills_log_sf(N z){
 if(z<40)return log(boost::math::cdf(boost::math::complement(boost::math::normal_distribution<N>(),z)));
 auto f=[&](const N&u)->N{return exp(-u-u*u/(2*z*z));};
 N e1,e2,l1,l2;
 N a=boost::math::quadrature::gauss_kronrod<N,61>::integrate(f,N(0),N(128),12,N("1e-24"),&e1,&l1);
 N b=boost::math::quadrature::gauss_kronrod<N,61>::integrate(f,N(0),N(128),15,N("1e-29"),&e2,&l2);
 if(abs(a-b)>N("1e-24")||e1>N("1e-24")||e2>N("1e-24")||exp(N(-128))/b>N("1e-25"))throw std::runtime_error("positive Mills oracle qualification");
 return -z*z/2-log(z)-log(2*boost::math::constants::pi<N>())/2+log(b);
}
N two_logtail(N z){
 z=abs(z);if(z==0)return 0;
 N result=z<1?boost::math::log1p(-boost::math::erf(z/sqrt(N(2)))):N(mills_log_sf(z)+log(N(2)));
 if(result>=0||!boost::multiprecision::isfinite(result))throw std::runtime_error("independent near-null logtail erased");
 return result;
}
struct Record {std::array<std::int64_t,4> counts{};std::array<N,8> values{};};
struct Reference {std::vector<std::array<Record,3>> rows;std::array<double,3> med{};std::array<bool,3> median{};std::int64_t deaths=0,censored=0,group1=0,observed1=0;N expected=0,score=0,variance=0,z=0,chi=0,p=0,lp=0,oe=0;bool lr=false,hr=false;};
Reference reference(const Cohort&in){
 Reference out;std::map<double,std::array<int,4>> times;
 for(auto x:in){times[x.t][x.g*2+(x.e?0:1)]++;out.group1+=x.g;out.deaths+=x.e;out.censored+=1-x.e;}
 std::array<std::int64_t,3> risk{std::int64_t(in.size()),std::int64_t(in.size())-out.group1,out.group1};
 std::array<R,3> exact_s{R(1),R(1),R(1)},exact_g{};R exact_e=0,exact_v=0;
 std::array<N,3> surv{N(1),N(1),N(1)},green{},hazard{};std::array<bool,3>terminal{};
 bool small=in.size()<=256;
 const N critical=boost::math::quantile(boost::math::normal_distribution<N>(),N(1)-N(".025"));
 for(auto [time,c]:times){std::array<int,3>d{c[0]+c[2],c[0],c[2]},cen{c[1]+c[3],c[1],c[3]};
  N e=N(d[0])*risk[2]/risk[0];out.expected+=e;out.score+=N(d[2])-e;
  if(risk[0]>1)out.variance+=N(d[0])*risk[2]*risk[1]*(risk[0]-d[0])/(N(risk[0])*risk[0]*(risk[0]-1));
  if(small){exact_e+=R(d[0])*risk[2]/risk[0];if(risk[0]>1)exact_v+=R(d[0])*risk[2]*risk[1]*(risk[0]-d[0])/(R(risk[0])*risk[0]*(risk[0]-1));}
  out.observed1+=d[2];std::array<Record,3>row;
  for(int g=0;g<3;g++){auto&r=row[g];r.counts={risk[g],d[g],cen[g],0};r.values[0]=time;
   if(risk[g]>0){
    if(d[g]){hazard[g]+=N(d[g])/risk[g];if(d[g]==risk[g]){surv[g]=0;terminal[g]=true;}else{surv[g]*=N(risk[g]-d[g])/risk[g];green[g]+=N(d[g])/(N(risk[g])*(risk[g]-d[g]));}
     exact_s[g]*=R(risk[g]-d[g])/risk[g];if(small&&d[g]!=risk[g])exact_g[g]+=R(d[g])/(R(risk[g])*(risk[g]-d[g]));
    }
    if(small){surv[g]=N(exact_s[g]);green[g]=N(exact_g[g]);}
    r.values[1]=surv[g];r.values[2]=terminal[g]?-std::numeric_limits<N>::infinity():log(surv[g]);r.values[7]=hazard[g];r.counts[3]=35;
    if(!terminal[g]){
     r.values[3]=green[g];r.values[4]=surv[g]*sqrt(green[g]);r.counts[3]+=12;
     if(surv[g]<1&&green[g]>0){N a=log(-log(surv[g])),width=critical*sqrt(green[g])/(-log(surv[g]));r.values[5]=exp(-exp(a+width));r.values[6]=exp(-exp(a-width));r.counts[3]+=16;}
    }
    if(!out.median[g]&&exact_s[g]<=R(1)/2){out.med[g]=time;out.median[g]=true;}
   }
   risk[g]-=d[g]+cen[g];
  }
  out.rows.push_back(row);
 }
 if(small){out.expected=N(exact_e);out.variance=N(exact_v);out.score=N(R(out.observed1)-exact_e);}
 out.lr=out.variance>0&&out.group1>0&&out.group1<std::int64_t(in.size());
 if(out.lr){out.z=out.score/sqrt(out.variance);out.chi=out.z*out.z;out.lp=two_logtail(out.z);out.p=exp(out.lp);}
 auto obs0=out.deaths-out.observed1;N e0=N(out.deaths)-out.expected;
 if(out.expected>0&&e0>0&&obs0>0){out.oe=(N(out.observed1)/out.expected)/(N(obs0)/e0);out.hr=true;}
 return out;
}
std::string lit(double x){std::ostringstream s;s<<std::scientific<<std::setprecision(17)<<x;return s.str();}
void generate(const std::vector<Cohort>&cs,bool scale){
 std::cout<<R"SIO(use collections::numeric_buffer::{numeric_buffer_new,numeric_buffer_at,numeric_buffer_set,numeric_buffer_close}
use collections::integer_buffer::{integer_buffer_new,integer_buffer_at,integer_buffer_set,integer_buffer_close}
use stats::checked_survival::{survival_workspace,survival_checked}
fn bit(x:f64)->i64 {f64_to_bits(x)}
fn yes(x:bool)->i64 {if x {1}else{0}}
fn row(id:i64) with IO,Alloc,Mut,Div,Panic {
 var n:i64=0
)SIO";
 for(std::size_t id=0;id<cs.size();id++)std::cout<<" if id=="<<id<<" {n="<<cs[id].size()<<"}\n";
 std::cout<<R"SIO( let q=survival_workspace(n,320000000)
 assert(q.status==0)
 var times=numeric_buffer_new(n,n*8);var event=integer_buffer_new(n,n*8);var group=integer_buffer_new(n,n*8)
 var order=integer_buffer_new(n,n*8);var curves=numeric_buffer_new(q.curve_len,q.curve_len*8);var counts=integer_buffer_new(q.count_len,q.count_len*8)
)SIO";
 if(scale)std::cout<<R"SIO( var at:i64=0
 while at<n {
  numeric_buffer_set(&!times,at,if id==0 {((at*37)%997) as f64}else{((at*7919)%100000) as f64})
  integer_buffer_set(&!event,at,if id==0 && at%5==0 {0}else{1});integer_buffer_set(&!group,at,at%2)
  at=at+1
 }
)SIO";
 if(scale){int id=2;for(auto spec:median_specs()){
 std::cout<<" if id=="<<id++<<" {\n  var i:i64=0\n  while i<n {\n";int cursor=0;
 for(int j=0;j<5;j++){int risk=spec[j][0],deaths=spec[j][1],next=j<4?spec[j+1][0]:0;cursor+=deaths;
 std::cout<<"   "<<(j==0?"if":"else if")<<" i<"<<cursor<<" {numeric_buffer_set(&!times,i,"<<lit(j+1)<<");integer_buffer_set(&!event,i,1)}\n";
 int censored=risk-deaths-next;cursor+=censored;
 std::cout<<"   else if i<"<<cursor<<" {numeric_buffer_set(&!times,i,"<<lit(j==4?6:j+1)<<");integer_buffer_set(&!event,i,0)}\n";
 }
 std::cout<<"   else {integer_buffer_set(&!event,i,0)}\n   i=i+1\n  }\n }\n";
 }}
 else for(std::size_t id=0;id<cs.size();id++){std::cout<<" if id=="<<id<<" {\n";if(cs[id].size()==1001){std::cout<<" var i:i64=0\n while i<n {numeric_buffer_set(&!times,i,if i<"<<(id==27?500:501)<<" {1.0}else{2.0});integer_buffer_set(&!event,i,if i<"<<(id==27?500:501)<<" {1}else{0});integer_buffer_set(&!group,i,i%2);i=i+1}\n";}else for(std::size_t i=0;i<cs[id].size();i++){auto x=cs[id][i];std::cout<<" numeric_buffer_set(&!times,"<<i<<","<<lit(x.t)<<");integer_buffer_set(&!event,"<<i<<","<<x.e<<");integer_buffer_set(&!group,"<<i<<","<<x.g<<")\n";}std::cout<<" }\n";}
 std::cout<<R"SIO( let r=survival_checked(&times,&event,&group,n,0.05,&!order,&!curves,&!counts,320000000,80000000,100000,100000,1.0e-13)
 assert(r.status==0)
 println("SUMMARY {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",id,r.status,r.n,r.rows,r.deaths,r.censored,r.group0,r.group1,yes(r.median_available),bit(r.median),yes(r.median0_available),bit(r.median0),yes(r.median1_available),bit(r.median1),r.sort_comparisons,r.risk_visits,r.critical_evaluations)
 let l=r.logrank
 println("LOGRANK {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",id,l.status,yes(l.available),l.observed1,bit(l.expected1),bit(l.score),bit(l.variance),bit(l.z),bit(l.chi2),yes(l.p_available),bit(l.p),yes(l.log_p_available),bit(l.log_p),yes(l.oe_ratio_available),bit(l.oe_ratio),l.kernel_iterations)
 println("MEDIAN_STATUS {} {} {} {}",id,r.median_status,r.median0_status,r.median1_status)
 var ri:i64=0
 var digest:i64=0
 var mass:f64=0.0;var mass_c:f64=0.0
 while ri<r.rows {
  var g:i64=0
  while g<3 {
   let rec=ri*3+g;let ib=rec*4;let nb=rec*8
   var k:i64=0
   while k<4 {digest=(digest*131+integer_buffer_at(&counts,ib+k))%1000000007;k=k+1}
   let term=numeric_buffer_at(&curves,nb+1)-mass_c;let next=mass+term;mass_c=(next-mass)-term;mass=next
)SIO";
 if(scale)std::cout<<"   if id!=1 || ri%7919==0 || ri==r.rows-1 {\n";
 std::cout<<R"SIO(   println("CURVE {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",id,ri,g,integer_buffer_at(&counts,ib),integer_buffer_at(&counts,ib+1),integer_buffer_at(&counts,ib+2),integer_buffer_at(&counts,ib+3),bit(numeric_buffer_at(&curves,nb)),bit(numeric_buffer_at(&curves,nb+1)),bit(numeric_buffer_at(&curves,nb+2)),bit(numeric_buffer_at(&curves,nb+3)),bit(numeric_buffer_at(&curves,nb+4)),bit(numeric_buffer_at(&curves,nb+5)),bit(numeric_buffer_at(&curves,nb+6)),bit(numeric_buffer_at(&curves,nb+7)))
)SIO";
 if(scale)std::cout<<"   }\n";
 std::cout<<R"SIO(   g=g+1
  }
  ri=ri+1
 }
 println("AGGREGATE {} {} {}",id,digest,bit(mass))
 numeric_buffer_close(times);integer_buffer_close(event);integer_buffer_close(group);integer_buffer_close(order);numeric_buffer_close(curves);integer_buffer_close(counts)
}
fn main()->i32 with IO,Alloc,Mut,Div,Panic {
)SIO";
 for(std::size_t id=0;id<cs.size();id++)std::cout<<" row("<<id<<")\n";
 std::cout<<" println(\"SURVIVAL_ROWS_COMPLETE "<<cs.size()<<"\")\n 0\n}\n";
}
double dbl(std::int64_t bits){return std::bit_cast<double>(bits);}
int verify(const std::vector<Cohort>&cs,const char*file,bool scale){
 std::ifstream in(file);if(!in){std::cerr<<"SURVIVAL_INPUT_OPEN_FAILURE\n";return 11;}
 std::vector<Reference>refs;for(auto&c:cs)refs.push_back(reference(c));
 int errors=0;std::vector<int>summary(cs.size()),lr(cs.size()),agg(cs.size()),curves(cs.size()),median_status(cs.size());bool complete=false;
 N maximum=0,logerror=0;
 auto fail=[&](std::string s){++errors;if(errors<25)std::cerr<<s<<'\n';};
 auto number=[&](std::int64_t b,N expected,std::string label,N absolute=N("5e-12"),N relative=N("3e-13")){
  double actual=dbl(b);if(!boost::multiprecision::isfinite(expected)){if(!std::isinf(actual)||(actual<0)!=(expected<0))fail(label+" endpoint");return;}
  if(!std::isfinite(actual)){fail(label+" nonfinite");return;}
  N error=abs(N(actual)-expected);if(error>maximum)maximum=error;
  if(error>absolute+relative*abs(expected))fail(label+" numerical");
 };
 std::string line;while(std::getline(in,line)){std::istringstream s(line);std::string kind;s>>kind;
  if(kind=="SURVIVAL_ROWS_COMPLETE"){int n=-1;s>>n;if(n==int(cs.size()))complete=true;else fail("completion count");continue;}
  if(kind!="SUMMARY"&&kind!="LOGRANK"&&kind!="CURVE"&&kind!="AGGREGATE"&&kind!="MEDIAN_STATUS"){fail("unexpected protocol line");continue;}
  int id=-1;s>>id;if(id<0||id>=int(cs.size())){fail("cohort id");continue;}auto&r=refs[id];std::string label=kind+" cohort="+std::to_string(id);
  if(kind=="SUMMARY"){
   std::array<std::int64_t,16>a{};for(auto&x:a)s>>x;
   if(!s){fail(label+" parse");continue;}++summary[id];
   if(a[0]!=0||a[1]!=std::int64_t(cs[id].size())||a[2]!=std::int64_t(r.rows.size())||a[3]!=r.deaths||a[4]!=r.censored||a[5]!=std::int64_t(cs[id].size())-r.group1||a[6]!=r.group1||a[14]!=std::int64_t(cs[id].size())||(cs[id].size()>1&&a[13]<=0))fail(label+" metadata");
   for(int g=0;g<3;g++){bool ambiguous=scale&&id>=2&&g==0;
   if(a[7+2*g]!=(ambiguous?false:r.median[g])){fail(label+" median availability");}
   if(!ambiguous&&r.median[g]&&dbl(a[8+2*g])!=r.med[g]){fail(label+" median time");}}
   if(!cs[id].empty()&&a[15]<=0)fail(label+" kernel work");
  }else if(kind=="MEDIAN_STATUS"){
   for(int g=0;g<3;g++){int code=-1;s>>code;if(code!=((scale&&id>=2&&g==0)?5:(r.median[g]?0:1)))fail(label+" boundary classification");}++median_status[id];
  }else if(kind=="LOGRANK"){
   std::array<std::int64_t,15>a{};for(auto&x:a)s>>x;if(!s){fail(label+" parse");continue;}++lr[id];
   if(a[0]!=(r.lr?0:5)||a[1]!=r.lr||a[2]!=r.observed1||a[12]!=r.hr)fail(label+" metadata");
   number(a[3],r.expected,label+" expected");number(a[4],r.score,label+" score");number(a[5],r.variance,label+" variance");if(r.hr)number(a[13],r.oe,label+" OE diagnostic");
   if(r.lr){number(a[6],r.z,label+" z");number(a[7],r.chi,label+" chi");double p=r.p.convert_to<double>();if(a[8]!=(p>0)||a[10]!=1)fail(label+" tail availability");if(p>0)number(a[9],r.p,label+" p",N(0),N("1e-10"));number(a[11],r.lp,label+" logp",N("3e-11"),N("2e-15"));logerror=std::max(logerror,N(abs(N(dbl(a[11]))-r.lp)));if(dbl(a[9])<0||dbl(a[9])>1||dbl(a[11])>0)fail(label+" bounds");}
   else if(a[8]||a[10])fail(label+" unavailable fabricated tail");
  }else if(kind=="CURVE"){
   std::int64_t row=-1,g=-1;s>>row>>g;std::array<std::int64_t,12>a{};for(auto&x:a)s>>x;if(!s||row<0||row>=std::int64_t(r.rows.size())||g<0||g>2){fail(label+" parse/index");continue;}++curves[id];auto&expected=r.rows[row][g];
   for(int k=0;k<4;k++)if(a[k]!=expected.counts[k])fail(label+" risk/availability row="+std::to_string(row)+" group="+std::to_string(g));
   for(int k=0;k<8;k++){bool available=k==0||((expected.counts[3]&(k==1?1:k==2?2:k==3?4:k==4?8:k==7?32:16))!=0);if(available)number(a[k+4],expected.values[k],label+" field="+std::to_string(k));else if(a[k+4]!=0)fail(label+" unavailable payload not zero");}
  }else{
   std::int64_t hash=0,bits=0;s>>hash>>bits;if(!s){fail(label+" parse");continue;}++agg[id];std::int64_t expected=0;N mass=0;
   for(auto&row:r.rows)for(auto&rec:row){for(auto c:rec.counts)expected=(expected*131+c)%1000000007;mass+=rec.values[1];}
   if(hash!=expected){fail(label+" full integer risk-table digest");}
   number(bits,mass,label+" all-record survival sum",N("3e-10"),N("3e-13"));
  }
  std::string extra;if(s>>extra)fail(label+" trailing tokens");
 }
 if(!complete){std::cerr<<"SURVIVAL_REFERENCE_INCOMPLETE_OUTPUT\n";return 10;}
 for(std::size_t id=0;id<cs.size();id++){int want=0;for(std::size_t row=0;row<refs[id].rows.size();row++)if(!scale||id!=1||row%7919==0||row+1==refs[id].rows.size())want+=3;if(summary[id]!=1||lr[id]!=1||agg[id]!=1||median_status[id]!=1||curves[id]!=want)fail("cohort="+std::to_string(id)+" record selection/count");}
 std::cout<<std::setprecision(17)<<"SURVIVAL_REFERENCE_COHORTS "<<cs.size()<<" EXACT_SMALL_RISK "<<(!scale)<<" ERRORS "<<errors<<" MAX_FIELD_ABSOLUTE "<<maximum<<" MAX_LOGTAIL_ABSOLUTE "<<logerror<<'\n';return errors?1:0;
}
void legacy_generate(){
 std::cout<<R"SIO(use stats::logrank::{logrank}
fn main()->i32 with IO,Alloc,Mut,Div,Panic {
 var t:[f64;256]=[0.0;256];var e:[i64;256]=[1;256];var g:[i64;256]=[0;256]
 var i:i64=0
 while i<256 {t[i as usize]=(i+1) as f64;g[i as usize]=if i<128 {0}else{1};i=i+1}
 let r=logrank(&t,&e,&g,256)
 let chi:i64=f64_to_bits(r.chi2);let p:i64=f64_to_bits(r.p);let o:i64=f64_to_bits(r.o1);let ex:i64=f64_to_bits(r.e1);let hr:i64=f64_to_bits(r.hr)
 println("LEGACY_LOGRANK {} {} {} {} {}",chi,p,o,ex,hr)
 println("LEGACY_LOGRANK_COMPLETE")
 0
}
)SIO";
}
int legacy_verify(const char*file){std::ifstream in(file);std::string kind,end;std::array<std::int64_t,5>a{};in>>kind;for(auto&x:a)in>>x;in>>end;if(!in||kind!="LEGACY_LOGRANK"||end!="LEGACY_LOGRANK_COMPLETE"){std::cerr<<"LEGACY_LOGRANK_PROTOCOL_FAILURE\n";return 10;}auto r=reference(cases(false).at(26));double p=dbl(a[1]);if(p<=0||abs(N(p)-r.p)>abs(r.p)*N("1e-10")){std::cerr<<std::setprecision(17)<<"LEGACY_LOGRANK_DIRECT_TAIL_MISMATCH actual="<<p<<" reference="<<r.p<<" log_reference="<<r.lp<<'\n';return 1;}std::cout<<std::setprecision(17)<<"LEGACY_LOGRANK_REFERENCE_PASS p="<<p<<" reference="<<r.p<<'\n';return 0;}
struct TailCase {double z;int budget=100000;double tol=1e-13;int status=0;};
std::vector<TailCase>tail_cases(){
 std::vector<TailCase>r;
 for(double z:{0.,.1,1.,2.,10.,30.,40.,128.,400.,1e-10,1e-20,1e-100,1e-150}){r.push_back({z});if(z!=0)r.push_back({-z});}
 for(double z:{1e-160,-1e-160,1e-200,-1e-200,1e200})r.push_back({z,100000,1e-13,5});
 r.push_back({std::numeric_limits<double>::infinity(),100000,1e-13,2});r.push_back({std::numeric_limits<double>::quiet_NaN(),100000,1e-13,2});
 r.push_back({2.,1,1e-13,3});r.push_back({2.,100000,-1.,1});return r;
}
void tail_generate(){
 std::cout<<R"SIO(use stats::checked_survival::{logrank_tail_checked}
fn yes(x:bool)->i64 {if x {1}else{0}}
fn row(id:i64,z:f64,budget:i64,tol:f64) with IO,Mut,Div,Panic {
 let r=logrank_tail_checked(z,budget,tol)
 let pb:i64=f64_to_bits(r.p);let lb:i64=f64_to_bits(r.log_p)
 println("TAIL {} {} {} {} {} {} {}",id,r.status,yes(r.p_available),yes(r.log_p_available),pb,lb,r.iterations)
}
fn main()->i32 with IO,Mut,Div,Panic {
)SIO";
 auto cases=tail_cases();for(std::size_t i=0;i<cases.size();i++){auto x=cases[i];std::string z=std::isnan(x.z)?"0.0/0.0":std::isinf(x.z)?"1.0e308*10.0":lit(x.z);std::cout<<" row("<<i<<","<<z<<","<<x.budget<<","<<lit(x.tol)<<")\n";}
 std::cout<<" println(\"SURVIVAL_TAILS_COMPLETE\")\n 0\n}\n";
}
int tail_verify(const char*file){
 std::ifstream in(file);auto cases=tail_cases();std::vector<bool>seen(cases.size());std::string line;bool complete=false;int errors=0;N maxrelative=0,maxabsolute=0;
 auto fail=[&](std::string why){++errors;std::cerr<<why<<'\n';};
 while(std::getline(in,line)){if(line=="SURVIVAL_TAILS_COMPLETE"){complete=true;continue;}std::istringstream s(line);std::string kind;int id=-1;s>>kind>>id;std::array<std::int64_t,6>a{};for(auto&x:a)s>>x;
  if(!s||kind!="TAIL"||id<0||id>=int(cases.size())||seen[id]){fail("TAIL protocol/index");continue;}seen[id]=true;auto x=cases[id];
  if(a[0]!=x.status){fail("TAIL status id="+std::to_string(id));continue;}
  if(x.status){if(a[1]||a[2]||(x.status==3&&a[5]<=0))fail("TAIL refusal metadata id="+std::to_string(id));continue;}
  N logp=two_logtail(N(x.z));N p=exp(logp);double expected=p.convert_to<double>();
  if(a[1]!=(expected>0)||a[2]!=1){fail("TAIL availability id="+std::to_string(id));continue;}
  double actual=dbl(a[3]),logged=dbl(a[4]);
  if(expected>0&&(!std::isfinite(actual)||actual<=0||abs(N(actual)-p)>p*N("1e-10")))fail("TAIL scalar id="+std::to_string(id));
  N error=abs(N(logged)-logp);maxabsolute=std::max(maxabsolute,error);
  if(logp!=0){N relative=error/abs(logp);maxrelative=std::max(maxrelative,relative);if(logged>=0||!std::isfinite(logged)||error>(abs(logp)<1?abs(logp)*N("2e-12"):N("3e-11")+abs(logp)*N("2e-15")))fail("TAIL log id="+std::to_string(id));}
  else if(logged!=0||actual!=1)fail("TAIL exact null endpoint");
 }
 if(!complete||std::count(seen.begin(),seen.end(),true)!=int(cases.size())){std::cerr<<"SURVIVAL_TAIL_INCOMPLETE_OUTPUT\n";return 10;}
 std::cout<<std::setprecision(17)<<"SURVIVAL_TAIL_REFERENCE_ROWS "<<cases.size()<<" ERRORS "<<errors<<" MAX_RELATIVE_LOG "<<maxrelative<<" MAX_ABSOLUTE_LOG "<<maxabsolute<<'\n';return errors?1:0;
}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--generate-tail"){tail_generate();return 0;}
 if(argc==3&&std::string(argv[1])=="--tail")return tail_verify(argv[2]);
 if(argc==2&&std::string(argv[1])=="--generate"){generate(cases(false),false);return 0;}
 if(argc==2&&std::string(argv[1])=="--generate-scale"){generate(cases(true),true);return 0;}
 if(argc==2&&std::string(argv[1])=="--generate-legacy"){legacy_generate();return 0;}
 if(argc==3&&std::string(argv[1])=="--legacy")return legacy_verify(argv[2]);
 if(argc==3&&std::string(argv[1])=="--scale")return verify(cases(true),argv[2],true);
 if(argc==2)return verify(cases(false),argv[1],false);
 std::cerr<<"usage: reference --generate|--generate-scale|--generate-legacy|[--scale|--legacy] ACTUAL\n";return 64;
 }catch(const std::exception&e){std::cerr<<"SURVIVAL_REFERENCE_EXCEPTION "<<e.what()<<'\n';return 12;}}
