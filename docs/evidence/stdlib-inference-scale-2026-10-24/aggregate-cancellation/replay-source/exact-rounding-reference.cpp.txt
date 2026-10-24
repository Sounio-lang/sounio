// C++23 exact-binary-rational oracle. Compile/execute only in coordinated Slurm.
// reference NATIVE_STDOUT; --before NATIVE_STDOUT accepts ONLY preserved diagnostic schema.
// Expected unavailable on mixed-row exact cancellation is a conservative API policy,
// not a claim that its exact rational sum is nonzero.
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <bit>
#include <cmath>
#include <limits>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <sstream>
#include <stdexcept>
using R=boost::multiprecision::cpp_rational;
using I=boost::multiprecision::cpp_int;
using D=boost::multiprecision::cpp_dec_float_100;
static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
static R exact(std::int64_t bits){auto u=std::bit_cast<std::uint64_t>(bits);auto e=(u>>52)&2047;need(e!=2047,"finite native bits");I m=u&((std::uint64_t(1)<<52)-1);if(e)m+=I(1)<<52;int power=e?int(e)-1023-52:-1074;R v=m;if(power>=0)v*=I(1)<<power;else v/=I(1)<<(-power);if(u>>63)v=-v;return v;}
static bool rounds_to(const R&truth,std::int64_t bits){
 const double value=std::bit_cast<double>(std::bit_cast<std::uint64_t>(bits));
 if(!std::isfinite(value))return false;
 const double previous=std::nextafter(value,-std::numeric_limits<double>::infinity());
 const double next=std::nextafter(value,std::numeric_limits<double>::infinity());
 need(std::isfinite(previous)&&std::isfinite(next),"finite rounding neighbors for this fixture");
 R center=exact(bits),lo=(exact(std::bit_cast<std::int64_t>(previous))+center)/2;
 R hi=(center+exact(std::bit_cast<std::int64_t>(next)))/2;
 if(truth<lo||truth>hi)return false;
 const bool even=(std::bit_cast<std::uint64_t>(bits)&1)==0;
 if(truth==lo||truth==hi)return even;
 return true;
}
int main(int argc,char**argv){try{
 bool before=argc==3&&std::string(argv[1])=="--before";need(argc==(before?3:2),"actual stdout required");std::ifstream f(argv[before?2:1]);need(bool(f),"native stdout open");
 std::map<int,R>sums;std::map<int,int>counts;std::map<std::pair<int,int>,bool>seen;
 std::string line;bool marker=false;int outputs=0;std::cout<<std::setprecision(60);
 while(std::getline(f,line)){
  std::istringstream in(line);std::string tag;in>>tag;
  if(tag=="INPUT"){
   int k,i;std::int64_t a,b;in>>k>>i>>a>>b;std::string extra;need(bool(in)&&!(in>>extra)&&k>=0&&k<(before?3:4)&&i==counts[k],"input order/arity");
   // Independent analytic Poisson count0: logP=-mu, so second-first=mu0-mu1.
   sums[k]+=exact(a)-exact(b);counts[k]++;
  }else if(tag=="AGG"||tag=="NATIVE"){
   int k,mode,status;std::int64_t bits;long long trace_status=0,calls=-1;
   if(before){std::string name;in>>k>>name>>status>>bits;mode=name=="legacy"?0:name=="rank"?1:name=="cached"?2:-1;need(tag=="NATIVE","before schema");}
   else{in>>k>>mode>>status>>bits>>trace_status>>calls;need(tag=="AGG","after schema");}
   std::string extra;need(bool(in)&&!(in>>extra)&&mode>=0&&mode<(before?3:6),"output arity/mode");
   need(k>=0&&k<(before?3:4)&&counts[k]==(k==2?2:3)&&!seen[{k,mode}],"output context/unique");seen[{k,mode}]=true;
   R truth=sums[k]/counts[k];const bool truth_nonzero=boost::multiprecision::numerator(truth)!=0;
   const bool native_zero=bits==0||bits==INT64_MIN;R native=exact(bits);
   // Freeze predicates before decimal presentation: never derive classification
   // from a rounded decimal rendering or from the program's aggregate.
   const bool false_zero=status==0&&native_zero&&truth_nonzero;
   if(before){need(status==0,"preserved before status");if(k==0)need(false_zero,"discriminating failing-before tiny-first");if(k==1)need(rounds_to(truth,bits),"reversed exact-rational rounding cell");if(k==2)need(!truth_nonzero&&native_zero,"exact cancellation before control");}
   else{
    const int expected=(k==0||k==2)?5:0;need(status==expected&&trace_status==0,"conservative status and trace availability");
    if(status!=0)need(bits==0,"unavailable cleared positive-zero payload");
    else if(truth_nonzero)need(rounds_to(truth,bits),"reversed exact-rational rounding cell");else need(bits==0,"pointwise-equal positive zero");
    const long long wanted_calls=mode<3?-1:((mode==3||k!=3)?2*counts[k]:0);need(calls==wanted_calls,"actual full PMF callsites retained");
   }
   std::cout<<"CASE\t"<<k<<'\t'<<mode<<"\tstatus\t"<<status<<"\texact_mean\t"<<truth.convert_to<D>()<<"\tnative\t"<<native.convert_to<D>()<<"\tfalse_available_zero\t"<<false_zero<<'\n';outputs++;
  }else if(line==(before?"AGGREGATE DIAGNOSTIC COMPLETE":"CHECKED AGGREGATE CANCELLATION PASS")){need(!marker,"unique final marker");marker=true;}
  else need(line.empty(),"unknown native line");
 }
 need(marker&&outputs==(before?9:24)&&counts.size()==(before?3:4),"complete actual coverage");
 std::cout<<(before?"PRESERVED BEFORE FALSE-ZERO WITNESS CONFIRMED":"EXACT RATIONAL CONSERVATIVE AGGREGATE CONCORDANCE PASS")<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
