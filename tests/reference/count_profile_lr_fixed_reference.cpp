// Fixed-native-parameter LR diagnosis. C++23/Big400, no production fit reused.
// Slurm: g++ -std=c++23 -O2 tests/reference/count_profile_lr_fixed_reference.cpp -lquadmath -o lr_fixed_reference
// Usage: lr_fixed_reference case-3.stdout [case-11.stdout case-13.stdout]
// --mutant-sign reverses only the reference orientation; supplementary controls
// must reject it. Original input datasets/means/alphas are never replaced/refit.
#define main saved_nb2_likelihood_reference_main
#include "deep_stats_nb2_likelihood_reference.cpp"
#undef main
#include <map>
#include <cmath>
#include <sstream>
#include <limits>
static bool reverse_lr=false;
static void lfneed(bool b,const char* w){if(!b)throw std::runtime_error(w);}
static double lfdecode(std::int64_t value){return std::bit_cast<double>(std::bit_cast<std::uint64_t>(value));}
static Big lfdifference(const Big& alt,const Big& null){return reverse_lr?null-alt:alt-null;}
static void lfrow(const char* key,const Big& v){std::cout<<key<<'\t'<<std::setprecision(65)<<v<<'\n';}
static int lfsign(const Big& x){return x>0?1:x<0?-1:0;}
static void lfcontrols(){
 Big null=direct(0,Big(2),Big(0)),alt=direct(0,Big(3),Big(0));
 lfneed(2*lfdifference(alt,null)==-2,"signed Poisson-zero orientation control");
 lfneed(2*lfdifference(null,alt)==2,"signed reversed point control");
 lfneed(lfdifference(null,null)==0,"identical point exact-zero control");
 double mean=8,next=std::nextafter(mean,std::numeric_limits<double>::infinity());
 Big peaked=direct(8,exact(mean),Big(0)),perturbed=direct(8,exact(next),Big(0));
 Big tiny=2*lfdifference(perturbed,peaked);
 lfneed(tiny<0&&tiny>Big("-1e-28"),"near-zero Poisson peak signed contrast");
 lfrow("supplementary_near_zero_fixed_lr",tiny);
 Big tiny_alpha=exact(1e-300),one=1,after=exact(std::nextafter(1.0,2.0));
 Big ntiny=2*lfdifference(direct(1,after,tiny_alpha),direct(1,one,tiny_alpha));
 lfneed(ntiny<0&&ntiny>Big("-1e-28"),"near-Poisson normalizer cancellation control");
 lfrow("supplementary_near_zero_nb2_lr",ntiny);
 std::cout<<"SUPPLEMENTARY_SIGN_NEAR_ZERO_CONTROLS_PASS\n";
}
struct LFRow{std::int64_t observed=0,count=0;double mean0=0,mean1=0,eta0=0,eta1=0,pmf0=0,pmf1=0;};
static void lfreplay(const std::string& path){
 std::ifstream in(path);lfneed(bool(in),"native diagnostic open");std::string line;
 std::array<LFRow,16> rows{};std::map<std::string,double> scalar;int count=0,b0=0,b1=0;bool done=false,cal=false,fits=false;std::int64_t dataset=-1;
 while(std::getline(in,line)){
  std::istringstream fields(line);std::string key;fields>>key;
  if(key=="LR_DIAGNOSTIC_COMPLETE"){lfneed(!done,"unique completion");done=true;continue;}
  if(key=="CAL_RESULT"){int status;std::int64_t hits,complete,failed,stage,hi,lo;fields>>dataset>>status>>hits>>complete>>failed>>stage>>hi>>lo;lfneed(bool(fields)&&status==6&&stage==5&&complete==failed&&hits<=complete,"original negative-LR checkpoint");cal=true;}
  else if(key=="FIT_STATUS"){int n,a,np,ap;fields>>n>>a>>np>>ap;lfneed(bool(fields)&&n==0&&a==0,"converged native fixed fits");fits=true;}
  else if(key=="DIAG_ROW"){
   int index,s0,s1;std::int64_t m0,m1,e0,e1,l0,l1;LFRow row;fields>>index>>row.observed>>row.count>>m0>>m1>>e0>>e1>>s0>>l0>>s1>>l1;
   lfneed(bool(fields)&&index==count&&count<16&&s0==0&&s1==0,"complete ordered native row");
   lfneed(row.observed>=0&&row.count>=0&&row.observed<=9007199254740992LL&&row.count<=9007199254740992LL,"exact count domain");
   row.mean0=lfdecode(m0);row.mean1=lfdecode(m1);row.eta0=lfdecode(e0);row.eta1=lfdecode(e1);row.pmf0=lfdecode(l0);row.pmf1=lfdecode(l1);
   lfneed(std::isfinite(row.mean0)&&row.mean0>0&&std::isfinite(row.mean1)&&row.mean1>0&&std::isfinite(row.eta0)&&std::isfinite(row.eta1)&&std::isfinite(row.pmf0)&&std::isfinite(row.pmf1),"finite captured native parameters");rows[count++]=row;
  }else if(key=="BETA0"||key=="BETA1"){int index;std::int64_t bits;fields>>index>>bits;lfneed(bool(fields)&&std::isfinite(lfdecode(bits)),"finite native coefficient");if(key=="BETA0"){lfneed(index==b0&&b0<1,"null coefficient order");b0++;}else{lfneed(index==b1&&b1<2,"alternative coefficient order");b1++;}}
  else if(key=="LAST_ALPHA0"||key=="LAST_ALPHA1"||key=="OBS_ALPHA0"||key=="OBS_ALPHA1"||key=="OBS_LR"||key=="OBS_DIFF"||key=="FAIL_LR"||key=="FAIL_DIFF"){
   std::int64_t bits;fields>>bits;lfneed(bool(fields)&&!scalar.contains(key),"unique rawbit scalar");double value=lfdecode(bits);lfneed(std::isfinite(value),"finite rawbit scalar");scalar.emplace(key,value);
  }
 }
 lfneed(done&&cal&&fits&&count==16&&b0==1&&b1==2&&scalar.size()==8,"complete original diagnostic, no successful prefix acceptance");
 double a0=scalar.at("LAST_ALPHA0"),a1=scalar.at("LAST_ALPHA1");lfneed(a0>=0&&a1>=0,"native dispersion domain");
 Big sum0=0,sum1=0,paired=0,rounded0=0,rounded1=0,max_error0=0,max_error1=0,max_mean_eta_error=0;std::int64_t group_sum[2]={0,0};
 for(int i=0;i<16;i++){
  const auto& r=rows[i];Big l0=direct(r.count,exact(r.mean0),exact(a0)),l1=direct(r.count,exact(r.mean1),exact(a1));
  sum0+=l0;sum1+=l1;paired+=lfdifference(l1,l0);rounded0+=exact(r.pmf0);rounded1+=exact(r.pmf1);
  Big error0=abs(exact(r.pmf0)-l0),error1=abs(exact(r.pmf1)-l1);if(error0>max_error0)max_error0=error0;if(error1>max_error1)max_error1=error1;
  Big me0=abs(exact(r.mean0)-exp(exact(r.eta0))),me1=abs(exact(r.mean1)-exp(exact(r.eta1)));if(me0>max_mean_eta_error)max_mean_eta_error=me0;if(me1>max_mean_eta_error)max_mean_eta_error=me1;
  lfneed(r.count<=std::numeric_limits<std::int64_t>::max()-group_sum[i/8],"group total overflow");group_sum[i/8]+=r.count;
 }
 Big lr=2*paired,independent=2*(sum1-sum0),rounded_lr=2*(rounded1-rounded0),native_lr=exact(scalar.at("FAIL_LR"));
 lfneed(abs(lr-independent)<Big("1e-340"),"paired-vs-total high precision orientation");
 lfneed(native_lr<0&&native_lr==32*exact(scalar.at("FAIL_DIFF")),"original signed native LR scaling");
 std::cout<<"ORIGINAL_FIXED_PARAMETER_REPLAY dataset="<<dataset<<" file="<<path<<'\n';
 std::cout<<"replicate_group_totals\t"<<group_sum[0]<<'\t'<<group_sum[1]<<'\n';
 lfrow("fixed_parameter_direct_lr",lr);lfrow("native_rounded_pmf_exact_sum_lr",rounded_lr);lfrow("native_failed_lr",native_lr);
 lfrow("native_lr_minus_fixed_truth",native_lr-lr);lfrow("native_aggregate_minus_rounded_row_sum",native_lr-rounded_lr);
 lfrow("max_native_pmf0_absolute_error",max_error0);lfrow("max_native_pmf1_absolute_error",max_error1);lfrow("max_mean_vs_exp_eta_error",max_mean_eta_error);
 std::cout<<"signs_fixed_rounded_native\t"<<lfsign(lr)<<'\t'<<lfsign(rounded_lr)<<'\t'<<lfsign(native_lr)<<'\n';
 std::cout<<"INTERPRETATION\tfixed captured parameters only; no refit, no assertion that either point is a global MLE; original datasets retained\n";
}
int main(int argc,char** argv){try{int first=1;if(argc>1&&std::string(argv[1])=="--mutant-sign"){reverse_lr=true;first=2;}lfcontrols();lfneed(argc>first,"original diagnostic paths required");for(int i=first;i<argc;i++)lfreplay(argv[i]);std::cout<<"FIXED_NATIVE_LR_REFERENCE_COMPLETE\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
