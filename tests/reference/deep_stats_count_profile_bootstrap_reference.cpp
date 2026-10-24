// Independent C++23 binary128 grouped ML profile-refit bootstrap reference.
// Slurm only: g++ -std=c++23 -O2 tests/reference/deep_stats_count_profile_bootstrap_reference.cpp -lquadmath -o profile_bootstrap_reference
// Reuses ONLY independent C++ sampler/PCG; means are analytic and dispersion
// uses harmonic-score bisection, not production IRLS/QR/grid/golden search.
#define main saved_count_sampling_profile_reference_main
#include "deep_count_sampling_reference.cpp"
#undef main
#include <array>
using Data=std::array<std::int64_t,16>;
static void pbneed(bool b,const char* w){if(!b)throw std::runtime_error(w);}
static void pbrow(const char* key,Q v){char b[160];quadmath_snprintf(b,sizeof b,"%.35Qg",v);std::cout<<key<<": "<<b<<'\n';}
struct Profile {Q alpha=0,ll=0;};
static std::array<Q,2> pbmeans(const Data& y,bool grouped){
 std::array<Q,2> m{};for(int i=0;i<16;i++)m[i/8]+=y[i];
 if(grouped){m[0]/=8;m[1]/=8;}else{m[0]=(m[0]+m[1])/16;m[1]=m[0];}
 pbneed(m[0]>0&&m[1]>0,"finite selected group means");return m;
}
static Q pbll(const Data& y,const std::array<Q,2>& m,Q a){Q sum=0;
 for(int i=0;i<16;i++){Q mu=m[i/8],v=y[i];
  if(a==0)sum+=v*logq(mu)-mu-lgammaq(v+1);
  else{Q r=1/a;sum+=lgammaq(v+r)-lgammaq(r)-lgammaq(v+1)+v*logq(mu/r)-(v+r)*log1pq(mu/r);}
 }return sum;
}
static Q pbscore(const Data& y,const std::array<Q,2>& m,Q t){Q r=expq(-t),sum=0;
 for(int i=0;i<16;i++){Q h=0;for(std::int64_t k=0;k<y[i];k++)h+=1/(r+k);
 sum-=r*(h-log1pq(m[i/8]/r)+(m[i/8]-Q(y[i]))/(r+m[i/8]));}return sum;
}
static Profile pbprofile(const Data& y,bool grouped){
 auto m=pbmeans(y,grouped);Profile out{0,pbll(y,m,0)};
 // The selected data's score is bracketed or favors exact Poisson. Scan 128
 // cells independently to detect every positive-to-negative score crossing.
 Q lo=-8,sl=pbscore(y,m,lo);bool stationary=false;
 for(int cell=1;cell<=128;cell++){
  Q hi=Q(-8)+Q(13)*cell/128,sh=pbscore(y,m,hi);
  if(sl>0&&sh<0){Q left=lo,right=hi;
   for(int j=0;j<150;j++){Q mid=(left+right)/2;if(mid==left||mid==right)break;
    if(pbscore(y,m,mid)>0)left=mid;else right=mid;}
   Q a=expq((left+right)/2),ll=pbll(y,m,a);if(ll>out.ll)out={a,ll};stationary=true;
  }lo=hi;sl=sh;
 }
 pbneed(out.alpha==0||stationary,"resolved selected profile");
 pbneed(out.ll>=pbll(y,m,expq(Q(-8)))&&out.ll>=pbll(y,m,expq(Q(5))),"selected maximum not range edge");
 return out;
}
struct PBReference{Q lr=0,a0=0,a1=0,last_a0=0,last_a1=0,tail=0;std::int64_t hits=0,hi=0,lo=0;Data last{};};
static void pbclose(Q got,Q want,Q relative,Q absolute,const char* w){
 pbneed(!isnanq(got)&&!isinfq(got),w);
 if(fabsq(got-want)>absolute+relative*fabsq(want)){pbrow("actual",got);pbrow("expected",want);throw std::runtime_error(w);}
}
int main(int argc,char** argv){try{
 bool mutant=argc==3&&std::string(argv[1])=="--mutant-fixed-alpha";
 pbneed(argc<=2||mutant,"oracle arguments");
 const Data data={0,0,1,2,5,8,13,19,1,1,2,3,7,12,18,28};
 std::array<PBReference,2> refs{};
 for(int model=0;model<2;model++){
  auto& o=refs[model];auto f0=pbprofile(data,false),f1=pbprofile(data,model==0);
  o.a0=f0.alpha;o.a1=f1.alpha;o.lr=2*(f1.ll-f0.ll);pbneed(o.lr>=0,"nonnegative observed LR");
  auto mu=pbmeans(data,false);Cursor rng(model==0?7:19);
  for(int rep=0;rep<2;rep++){
   for(auto& y:o.last)y=nb(rng,mu[0],Q(double(o.a0)));
   auto r0=pbprofile(o.last,false),r1=pbprofile(o.last,model==0);
   if(mutant){r0={f0.alpha,pbll(o.last,pbmeans(o.last,false),f0.alpha)};r1={f0.alpha,pbll(o.last,pbmeans(o.last,model==0),f0.alpha)};}
   o.last_a0=r0.alpha;o.last_a1=r1.alpha;
   Q lr=2*(r1.ll-r0.ll);pbneed(lr>=0,"nonnegative replicate LR");if(lr>=o.lr)o.hits++;
   std::cout<<"reference replicate "<<model<<' '<<rep<<'\n';pbrow("null alpha",r0.alpha);pbrow("alternative alpha",r1.alpha);pbrow("LR",lr);
  }
  o.tail=Q(o.hits+1)/3;o.hi=std::bit_cast<std::int64_t>(std::uint64_t(rng.rng.state>>64));o.lo=std::bit_cast<std::int64_t>(std::uint64_t(rng.rng.state));
  std::cout<<"reference case "<<model<<'\n';pbrow("observed LR",o.lr);pbrow("null alpha",o.a0);pbrow("alternative alpha",o.a1);pbrow("tail",o.tail);
 }
 if(!mutant)pbneed(refs[0].last_a0!=refs[0].a0||refs[0].last_a1!=refs[0].a0,"replicate dispersion refit discriminator");
 if(argc>=2){std::ifstream in(argv[mutant?2:1]);pbneed(bool(in),"native stdout open");std::string line;int current=-1,cases=0,seen=0,counts=0;bool done=false;
  const std::string prefix="profile bootstrap ";
  while(std::getline(in,line)){
   if(line=="DEEP COUNT PROFILE BOOTSTRAP PASS")done=true;
   if(!line.starts_with(prefix))continue;auto sep=line.find(": ");pbneed(sep!=std::string::npos,"native row syntax");
   auto key=line.substr(prefix.size(),sep-prefix.size()),value=line.substr(sep+2);
   if(key=="case"){if(current>=0)pbneed(seen==511&&counts==16,"previous complete native case");current=std::stoi(value);pbneed(current==cases&&current<2,"case order");cases++;seen=counts=0;continue;}
   pbneed(current>=0,"native case context");auto& e=refs[current];Q v=strtoflt128(value.c_str(),nullptr);
   if(key=="lr"){pbclose(v,e.lr,Q(2e-7),Q(2e-9),"native full profiled LR");seen|=1;}
   else if(key=="alpha0"){pbclose(v,e.a0,Q(2e-5),0,"native null alpha");seen|=2;}
   else if(key=="alpha1"){pbclose(v,e.a1,Q(2e-5),0,"native alternative alpha");seen|=4;}
   else if(key=="last alpha0"){pbclose(v,e.last_a0,Q(2e-5),0,"native last refitted null alpha");seen|=128;}
   else if(key=="last alpha1"){pbclose(v,e.last_a1,Q(2e-5),0,"native last refitted alternative alpha");seen|=256;}
   else if(key=="tail"){pbclose(v,e.tail,Q(2e-14),0,"native tail");seen|=8;}
   else if(key=="hits"){pbneed(std::stoll(value)==e.hits,"native hits");seen|=16;}
   else if(key=="state hi"){pbneed(std::stoll(value)==e.hi,"native RNG high");seen|=32;}
   else if(key=="state lo"){pbneed(std::stoll(value)==e.lo,"native RNG low");seen|=64;}
   else if(key=="count"){pbneed(counts<16&&std::stoll(value)==e.last[counts],"native last sample");counts++;}
  }
  pbneed(done&&cases==2&&seen==511&&counts==16,"full native profile-bootstrap completion");
  std::cout<<"ACTUAL PROFILE BOOTSTRAP CONCORDANCE PASS\n";
 }
 std::cout<<"DEEP COUNT PROFILE BOOTSTRAP CPP PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
