// Independent direct-full-PMF checks of the local fixed-alpha increment.
// Central Slurm C++23 only; Boost400 direct log-Gamma PMF is authoritative.
// The series identity is checked in high precision, not claimed as an IEEE proof.
// Actual Sounio end-to-end regressions are checked separately from fixture stdout.
// g++ -std=c++23 -O2 tests/reference/deep_stats_count_increment_reference.cpp -lquadmath -o increment_reference
#define main saved_nb2_likelihood_reference_main
#include "deep_stats_nb2_likelihood_reference.cpp"
#undef main
static Big integrated(long long y,const Big& mu,const Big& trial,const Big& alpha){
 Big d=(trial-mu)/mu,w=mu/(1+alpha*mu),q=alpha*mu/(1+alpha*mu);
 Big a=1,b=Big("0.5"),power=1,geometric=1;
 for(unsigned k=1;k<32;++k){power*=-d;geometric=1+q*geometric;a+=power*geometric/Big(k+1);b+=power*geometric/Big(k+2);}
 return ((Big(y)-mu)/mu)*w*d*a-w*d*d*b;
}
int main(int argc,char** argv){try{
 std::cout<<std::setprecision(50);
 unsigned checked=0;
 for(double alpha:{0.,.25,1e6,1e300})for(long long count:{0LL,1LL,8LL,100LL})
 for(double mu:{.125,1.,7.,100.})for(double factor:{.75,.875,1.125,1.25}){
  Big m=exact(mu),a=exact(alpha),trial=exact(mu*factor);
  Big expected=direct(count,trial,a)-direct(count,m,a);
  Big value=integrated(count,m,trial,a);
  require(expected!=0,"selected nonzero increment");
  require(ab(value-expected)<=ab(expected)*Big("1e-15"),"integral vs independent full gamma log PMF");
  ++checked;
 }
 // Extreme q nearly one: the gain is hundreds of orders smaller than each log PMF.
 Big a=exact(1e300),m=7,trial=Big("7.125");
 Big tiny=direct(8,trial,a)-direct(8,m,a);
 require(tiny>0,"tiny positive extreme-weight gain");
 std::cout<<"extreme_weight_direct_gain\t"<<tiny<<"\n";
 std::cout<<"direct_gamma_increment_cases\t"<<checked<<"\n";
 const std::array<long long,16> raw{8,7,8,10,5,11,6,3,6,7,13,8,9,9,5,8};
 Big recorded_total=0;for(auto count:raw)recorded_total+=count;
 std::array<Big,3> beta{};
 for(unsigned group=0;group<2;++group){
  Big sum=0;for(unsigned i=group*8;i<(group+1)*8;++i)sum+=raw[i];
  beta[group]=log(sum/8);
 }
 beta[2]=log(Big(8));
 if(argc==2){
  std::ifstream in(argv[1]);require(bool(in),"CountGLM native transcript open");
  std::string line;int mode=-1;unsigned cases=0,index=0;bool status=false,complete=false;
  while(std::getline(in,line)){
   if(line=="DEEP COUNT GLM PASS")complete=true;
   if(line.starts_with("increment case: ")){
    if(mode>=0)require(status&&index==(mode==0?2u:1u),"previous increment regression complete");
    mode=std::stoi(line.substr(16));require(mode==int(cases)&&mode<2,"increment case order");
    ++cases;index=0;status=false;
   }else if(line.starts_with("increment status: ")){
    require(mode>=0,"increment status context");require(std::stoi(line.substr(18))==0,"increment native convergence");status=true;
   }else if(line.starts_with("increment coefficient: ")){
    require(mode>=0&&index<(mode==0?2u:1u),"increment beta row");
    Big actual(line.substr(23));Big expected=beta[mode==0?index:2];
    // Verify the declared convergence against the analytic MLE, not an
    // unjustified tighter coefficient tolerance. Account only for 15-digit
    // scientific serialization in this independent coefficient-based check.
    Big fitted_mu=exp(actual),target_mu=exp(expected);
    Big relative_step=ab(target_mu/fitted_mu-1)/(1+ab(actual));
    Big alpha=mode==0?Big(0):exact(1e300);
    Big total=mode==0?recorded_total:Big(8)*8;
    Big score=8*ab(target_mu-fitted_mu)/((1+alpha*fitted_mu)*(1+total));
    const Big budget=Big("1e-9")+Big("2e-14");
    require(relative_step<=budget,"native analytic normalized full step");
    require(score<=budget,"native analytic normalized score");
    ++index;
   }
  }
  require(cases==2&&status&&index==1&&complete,"full CountGLM completion and two regressions");
  std::cout<<"actual native analytic increment-regression concordance PASS\n";
 }
 std::cout<<"DEEP COUNT INCREMENT CPP PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}}
