// C++23/binary128 closed-form grouped-refit bootstrap reference.
// Includes the independent unsigned128 PCG/high-precision sampler oracle ONLY;
// no production Sounio fit, likelihood, QR or bootstrap implementation reused.
// Slurm: g++ -std=c++23 -O2 tests/reference/deep_stats_count_bootstrap_reference.cpp -lquadmath -o bootstrap_reference
#define main saved_count_sampling_reference_main
#include "deep_count_sampling_reference.cpp"
#undef main
#include <array>
static void bneed(bool value,const char* why){if(!value)throw std::runtime_error(why);}
static void brow(const char* key,Q x){char b[150];quadmath_snprintf(b,sizeof(b),"%.35Qg",x);std::cout<<key<<"\t"<<b<<"\n";}
static Q bll(const std::array<std::int64_t,16>& counts,Q mean0,Q mean1,Q alpha){
 Q sum=0;
 for(unsigned i=0;i<16;++i){
  Q mu=i<8?mean0:mean1,y=counts[i];
  if(alpha==0)sum+=y*logq(mu)-mu-lgammaq(y+1);
  else {Q r=1/alpha;sum+=lgammaq(y+r)-lgammaq(r)-lgammaq(y+1)+y*logq(mu/r)-(y+r)*log1pq(mu/r);}
 }
 return sum;
}
static Q blr(const std::array<std::int64_t,16>& counts,Q alpha,bool same){
 Q s0=0,s1=0;for(unsigned i=0;i<8;++i)s0+=counts[i];for(unsigned i=8;i<16;++i)s1+=counts[i];
 bneed(s0>0&&s1>0,"selected sample finite group means");
 Q all=(s0+s1)/16,mu0=s0/8,mu1=s1/8;
 return 2*(bll(counts,same?all:mu0,same?all:mu1,alpha)-bll(counts,all,all,alpha));
}
struct BRef {
 Q lr=0,p=0,raw=0,se=0;
 std::int64_t hits=0,uniforms=0,proposals=0,hi=0,lo=0;
 std::array<std::int64_t,16> last{};
};
static void bclose(Q x,Q y,const char* key){
 bneed(!isnanq(x)&&!isinfq(x),key);
 if(y==0){bneed(x==0,key);return;}
 if(fabsq(x-y)>Q(1e-7)*fabsq(y)){brow("actual",x);brow("expected",y);throw std::runtime_error(key);}
}
int main(int argc,char** argv){try{
 constexpr std::array<std::int64_t,16> data={0,0,1,2,5,8,13,19,1,1,2,3,7,12,18,28};
 constexpr std::array<std::int64_t,3> seeds={7,19,0};
 constexpr std::int64_t draws=7;
 if(argc>=2&&std::string(argv[1])=="--repro"){
  Cursor cursor(7);std::array<std::int64_t,16> sample{};
  for(int replica=0;replica<2;++replica)for(auto& count:sample)count=poisson(cursor,Q(7.5));
  std::cout<<"reproducer replica: 1\n";
  for(auto count:sample)std::cout<<"reproducer count: "<<count<<"\n";
  Q totals[2]={0,0};for(unsigned i=0;i<16;++i)totals[i/8]+=sample[i];
  for(unsigned g=0;g<2;++g){brow("analytic mean",totals[g]/8);brow("analytic beta",logq(totals[g]/8));}
  if(argc==3){
   std::ifstream input(argv[2]);bneed(bool(input),"failure transcript open");
   std::string line;std::array<Q,2> beta{};unsigned count_rows=0,beta_rows=0;
   while(std::getline(input,line)){
    if(line.starts_with("bootstrap failure count: ")){bneed(count_rows<16,"failure count rows");bneed(std::stoll(line.substr(25))==sample[count_rows],"failure sample vs independent PCG");++count_rows;}
    if(line.starts_with("bootstrap failure staged beta: ")){bneed(beta_rows<2,"failure beta rows");beta[beta_rows++]=strtoflt128(line.c_str()+31,nullptr);}
   }
   bneed(count_rows==16&&beta_rows==2,"complete native failure transcript");
   Q denominator=1+totals[0]+totals[1],score=0,step=0,after[2];
   for(unsigned g=0;g<2;++g){
    Q mu=expq(beta[g]),s=(totals[g]-8*mu)/denominator;
    Q increment=(totals[g]/8-mu)/mu;
    if(fabsq(s)>score)score=fabsq(s);
    Q scaled=fabsq(increment)/(1+fabsq(beta[g]));
    if(scaled>step)step=scaled;
    after[g]=expq(beta[g]+increment);
    brow("staged beta error",beta[g]-logq(totals[g]/8));
    brow("analytic full step",increment);
   }
   brow("analytic normalized score",score);brow("analytic normalized full step",step);
   Q gain=bll(sample,after[0],after[1],0)-bll(sample,expq(beta[0]),expq(beta[1]),0);
   brow("high_precision_full_step_likelihood_gain",gain);
  }
  std::cout<<"REPRODUCER CPP PASS\n";return 0;
 }
 std::array<BRef,3> results{};
 Q observed_sum=0;for(auto y:data)observed_sum+=y;
 Q generating_mean=observed_sum/16;
 for(unsigned model=0;model<3;++model){
  Q alpha=model==1?Q(0.25):Q(0);bool same=model==2;Cursor cursor(seeds[model]);auto& out=results[model];
  out.lr=blr(data,alpha,same);
  bneed(out.lr>=0,"selected observed LR nonnegative");
  for(std::int64_t b=0;b<draws;++b){
   for(auto& count:out.last)count=nb(cursor,generating_mean,alpha);
   Q statistic=blr(out.last,alpha,same);
   // An exact zero is possible. Tiny high-precision negative cancellation is
   // NOT converted into a nonnegative LR: fail and inspect the selected sample.
   bneed(statistic>=0,"independent sample LR nonnegative");
   if(statistic>=out.lr)++out.hits;
  }
  out.p=Q(out.hits+1)/Q(draws+1);out.raw=Q(out.hits)/draws;
  out.se=sqrtq(out.raw*(1-out.raw)/draws);
  out.uniforms=cursor.uniforms;out.proposals=cursor.proposals;
  out.hi=signed_bits(std::uint64_t(cursor.rng.state>>64));out.lo=signed_bits(std::uint64_t(cursor.rng.state));
  if(same)bneed(out.lr==0&&out.hits==draws&&out.p==1,"identical-model analytic zero LR");
  std::cout<<"reference bootstrap: "<<model<<"\n";
  brow("observed_lr",out.lr);brow("p_plus_one",out.p);brow("mc_raw",out.raw);brow("mc_se",out.se);
  std::cout<<"hits\t"<<out.hits<<"\nuniforms\t"<<out.uniforms<<"\nproposals\t"<<out.proposals
   <<"\nstate_hi\t"<<out.hi<<"\nstate_lo\t"<<out.lo<<"\n";
 }
 // Deliberately nonnested, equal-dimensional models: wrong declarations must
 // not cause production to clamp a negative LR to zero.
 Q common=observed_sum/(8+8*Q(10));
 Q wrong_lr=2*(bll(data,common,10*common,0)-bll(data,generating_mean,generating_mean,0));
 bneed(wrong_lr<0,"nonnested negative LR discriminator");brow("nonnested_signed_lr",wrong_lr);
 if(argc==2){
  std::ifstream in(argv[1]);bneed(bool(in),"native report open");std::string line;
  int model=-1;unsigned cases=0,seen=0,last=0;bool complete=false;
  while(std::getline(in,line)){
   if(line=="DEEP COUNT BOOTSTRAP PASS")complete=true;
   if(line.starts_with("bootstrap case: ")){
    if(model>=0)bneed(seen==1023&&last==16,"previous bootstrap fields");
    model=std::stoi(line.substr(16));bneed(model==int(cases)&&model<3,"native model order");++cases;seen=last=0;
   }else if(line.starts_with("bootstrap status: ")){bneed(model>=0,"status context");bneed(std::stoll(line.substr(18))==0,"native bootstrap status");seen|=1;}
   else if(line.starts_with("bootstrap lr: ")){bneed(model>=0,"LR context");bclose(strtoflt128(line.c_str()+14,nullptr),results[model].lr,"native observed LR");seen|=2;}
   else if(line.starts_with("bootstrap p: ")){bneed(model>=0,"p context");bclose(strtoflt128(line.c_str()+13,nullptr),results[model].p,"native plus-one p");seen|=4;}
   else if(line.starts_with("bootstrap mc raw: ")){bneed(model>=0,"raw context");bclose(strtoflt128(line.c_str()+18,nullptr),results[model].raw,"native MC raw");seen|=8;}
   else if(line.starts_with("bootstrap mc se: ")){bneed(model>=0,"MC SE context");bclose(strtoflt128(line.c_str()+17,nullptr),results[model].se,"native MC SE");seen|=16;}
   else if(line.starts_with("bootstrap hits: ")){bneed(model>=0,"hits context");bneed(std::stoll(line.substr(16))==results[model].hits,"native hits");seen|=32;}
   else if(line.starts_with("bootstrap uniforms: ")){bneed(model>=0,"uniform context");bneed(std::stoll(line.substr(20))==results[model].uniforms,"native uniforms");seen|=64;}
   else if(line.starts_with("bootstrap proposals: ")){bneed(model>=0,"proposal context");bneed(std::stoll(line.substr(21))==results[model].proposals,"native proposals");seen|=128;}
   else if(line.starts_with("bootstrap state hi: ")){bneed(model>=0,"state context");bneed(std::stoll(line.substr(20))==results[model].hi,"native RNG high");seen|=256;}
   else if(line.starts_with("bootstrap state lo: ")){bneed(model>=0,"state context");bneed(std::stoll(line.substr(20))==results[model].lo,"native RNG low");seen|=512;}
   else if(line.starts_with("bootstrap last count: ")){bneed(model>=0&&last<16,"count row");bneed(std::stoll(line.substr(22))==results[model].last[last],"native last counts");++last;}
  }
  bneed(cases==3&&seen==1023&&last==16&&complete,"full native bootstrap completion");
  std::cout<<"actual native independent bootstrap concordance PASS\n";
 }
 std::cout<<"DEEP COUNT BOOTSTRAP CPP PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}}
