// Independent C++23 quad113 Poisson reference. Compile with -lquadmath.
// Uses raw quad score/Hessian and dense pivoted Gaussian elimination, not CPQR.
#include <quadmath.h>
#include <array>
#include <bit>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using Q=__float128; using I=std::int64_t;
struct Case {std::string name;I n,p;std::array<double,2048>a{};std::array<I,256>y{};std::array<double,256>off{};};
struct Fit{std::array<Q,8>b{};std::array<Q,256>mu{},eta{};Q dev=0;};
Q deviance(const Case&c,Fit&f){
 Q total=0;for(I i=0;i<c.n;i++){Q eta=c.off[i];for(I j=0;j<c.p;j++)eta+=Q(c.a[i*8+j])*f.b[j];Q m=expq(eta),y=c.y[i];f.mu[i]=m;f.eta[i]=eta;
 Q v=0;if(y==0)v=2*m;else{Q d=(y-m)/m;if(fabsq(d)<Q(.25)){Q pow=d*d,phi=pow/2;for(int k=3;k<80;k++){pow*=d;Q term=pow/(k*(k-1));phi+=(k%2?-term:term);}v=2*m*phi;}else v=2*(y*(logq(y)-eta)-y+m);}total+=v;}f.dev=total;return total;
}
bool solve(Q h[8][8],Q g[8],Q d[8],int p){
 for(int k=0;k<p;k++){int pivot=k;for(int i=k+1;i<p;i++)if(fabsq(h[i][k])>fabsq(h[pivot][k]))pivot=i;if(h[pivot][k]==0)return false;for(int j=k;j<p;j++)std::swap(h[k][j],h[pivot][j]);std::swap(g[k],g[pivot]);for(int i=k+1;i<p;i++){Q r=h[i][k]/h[k][k];for(int j=k+1;j<p;j++)h[i][j]-=r*h[k][j];g[i]-=r*g[k];}}
 for(int i=p-1;i>=0;i--){Q v=g[i];for(int j=i+1;j<p;j++)v-=h[i][j]*d[j];d[i]=v/h[i][i];}return true;
}
Fit reference(const Case&c){
 Fit f;deviance(c,f);
 if(c.p==1){bool ones=true;for(I i=0;i<c.n;i++)if(c.a[i*8]!=1)ones=false;if(ones){Q sy=0,se=0;for(I i=0;i<c.n;i++){sy+=Q(c.y[i]);se+=expq(Q(c.off[i]));}f.b[0]=logq(sy/se);deviance(c,f);return f;}}
 bool onehot=true;for(I i=0;i<c.n;i++){int active=0;for(I j=0;j<c.p;j++){if(c.a[i*8+j]==1)active++;else if(c.a[i*8+j]!=0)onehot=false;}if(active!=1)onehot=false;}
 if(onehot){for(I j=0;j<c.p;j++){Q sy=0,se=0;for(I i=0;i<c.n;i++)if(c.a[i*8+j]==1){sy+=Q(c.y[i]);se+=expq(Q(c.off[i]));}f.b[j]=logq(sy/se);}deviance(c,f);return f;}
 for(int it=0;it<200;it++){
  Q h[8][8]{},g[8]{},d[8]{};for(I i=0;i<c.n;i++)for(I j=0;j<c.p;j++){Q x=c.a[i*8+j];g[j]+=x*(Q(c.y[i])-f.mu[i]);for(I k=0;k<c.p;k++)h[j][k]+=x*Q(c.a[i*8+k])*f.mu[i];}
  if(!solve(h,g,d,int(c.p))){std::cerr<<"reference singular "<<c.name<<"\n";std::exit(2);}
  Q step=0;for(I j=0;j<c.p;j++)step=fmaxq(step,fabsq(d[j])/(1+fabsq(f.b[j])));
  if(step<Q(1e-14))return f;
  Q alpha=1;bool accepted=false;
  for(int z=0;z<100;z++){Fit trial=f;for(I j=0;j<c.p;j++)trial.b[j]+=alpha*d[j];deviance(c,trial);if(trial.dev<=f.dev){f=trial;accepted=true;break;}alpha/=2;}
  if(!accepted){std::cerr<<"reference line search "<<c.name<<"\n";std::exit(3);}
 }
 std::cerr<<"reference iterations "<<c.name<<"\n";std::exit(4);
}
int main(){
 std::vector<Case> all;
 auto add=[&](std::string name,I n,I p,auto fn){Case c;c.name=name;c.n=n;c.p=p;for(I i=0;i<n;i++)fn(c,i);all.push_back(c);};
 add("intercept",6,1,[](Case&c,I i){c.a[i*8]=1;c.y[i]=i+1;});
 add("exposure",6,1,[](Case&c,I i){c.a[i*8]=1;c.y[i]=2*(i+1);c.off[i]=std::log(double(i+1));});
 add("some_zero",6,1,[](Case&c,I i){c.a[i*8]=1;c.y[i]=i%3==0?0:i;});
 add("one_zero",1,1,[](Case&c,I){c.a[0]=1;c.y[0]=7;});
 add("perfect_intercept",5,1,[](Case&c,I i){c.a[i*8]=1;c.y[i]=4;});
 add("grouped",8,2,[](Case&c,I i){c.a[i*8]=1;c.a[i*8+1]=i>=4?1:0;c.y[i]=i<4?i+1:2*(i-3);});
 add("group_exposure",8,2,[](Case&c,I i){c.a[i*8]=1;c.a[i*8+1]=i>=4?1:0;c.y[i]=(i%4+1)*(i<4?2:5);c.off[i]=std::log(double(i%4+1));});
 add("slope",10,2,[](Case&c,I i){c.a[i*8]=1;c.a[i*8+1]=(double(i)-4.5)/4;c.y[i]=(i*i+3*i+7)%19;});
 add("offset_slope",12,2,[](Case&c,I i){c.a[i*8]=1;c.a[i*8+1]=(double(i)-5.5)/5;c.y[i]=(i*i+2*i+13)%23;c.off[i]=std::log(double(i%3+1));});
 add("no_intercept",6,1,[](Case&c,I i){c.a[i*8]=(double(i)-2.5)/3;c.y[i]=i+1;});
 add("balanced_allzero",2,1,[](Case&c,I i){c.a[i*8]=i==0?-1:1;c.y[i]=0;});
 add("offset_shift",6,1,[](Case&c,I i){c.a[i*8]=1;c.y[i]=i+1;c.off[i]=50;});
 add("offset_negative",6,1,[](Case&c,I i){c.a[i*8]=1;c.y[i]=i+1;c.off[i]=-50;});
 add("huge_count",4,1,[](Case&c,I i){c.a[i*8]=1;c.y[i]=I(9007199254740992LL)+i*2;});
 add("max_count",3,1,[](Case&c,I i){c.a[i*8]=1;c.y[i]=I(9223372036854775807LL)-i;});
 add("max_shape",256,8,[](Case&c,I i){for(int j=0;j<8;j++)c.a[i*8+j]=(i%8==j)?1:0;c.y[i]=(i*13)%31+1;c.off[i]=std::log(double(i%3+1));});
 add("column_scale",10,2,[](Case&c,I i){c.a[i*8]=1e-100;c.a[i*8+1]=((double(i)-4.5)/4)*1e100;c.y[i]=(i*i+3*i+7)%19;});
 for(int k=0;k<9;k++)add("deterministic_"+std::to_string(k),12+k,3,[k](Case&c,I i){c.a[i*8]=1;c.a[i*8+1]=(double(i)-double(11+k)/2)/double(12+k);c.a[i*8+2]=double((i*7)%5)/5;c.y[i]=(i*17+k*11)%29+1;c.off[i]=std::log(double(i%4+1));});
 std::cout<<std::scientific<<std::setprecision(17);
 std::cout<<"//@ run-pass\n//@ expect-stdout: STATS_POISSON_REFERENCE_PASS\n// Generated independent C++23 quad score/Hessian oracle.\nuse stats::regression::poisson::{PoissonFit,poisson_fit_offset}\nfn close(a:f64,b:f64) with Mut,Div,Panic {let d=if a>b {a-b}else{b-a} let m=if b<0.0 {-b}else{b} assert(d<=2.0e-8*(1.0+m))}\n";
 for(size_t k=0;k<all.size();k++){auto c=all[k];auto r=reference(c);std::cerr<<k<<" "<<c.name<<" deviance="<<double(r.dev)<<"\n";
  if(c.name=="max_shape"){
   std::array<double,3> offsets{};
   std::array<double,24> means{},etas{};
   std::array<bool,24> seen{};
   for(I phase=0;phase<3;phase++)offsets[phase]=c.off[phase];
   for(I i=0;i<c.n;i++){
    I cell=(i%8)*3+i%3;
    if(!seen[cell]){means[cell]=double(r.mu[i]);etas[cell]=double(r.eta[i]);seen[cell]=true;}
    if(std::bit_cast<std::uint64_t>(means[cell])!=std::bit_cast<std::uint64_t>(double(r.mu[i])) ||
       std::bit_cast<std::uint64_t>(etas[cell])!=std::bit_cast<std::uint64_t>(double(r.eta[i])) ||
       std::bit_cast<std::uint64_t>(offsets[i%3])!=std::bit_cast<std::uint64_t>(c.off[i]) ||
       c.y[i]!=(i*13)%31+1){std::cerr<<"compact identity failure row "<<i<<"\n";return 5;}
    for(I j=0;j<c.p;j++){double design=(i%8==j)?1.0:0.0;
     if(std::bit_cast<std::uint64_t>(design)!=std::bit_cast<std::uint64_t>(c.a[i*8+j])){std::cerr<<"compact design identity failure\n";return 6;}
    }
   }
   for(bool v:seen)if(!v){std::cerr<<"compact missing table cell\n";return 7;}
   std::cerr<<"compact identity PASS:256meanbits,256etabits,256offsetbits,256counts,2048designbits;24/24tablecells\n";
   std::cout<<"// Compact max-shape case; raw quad expected bits checked for every row by C++ generator.\n";
   std::cout<<"fn case_"<<k<<"() with Mut,Div,Panic {\n var a:[f64;2048]=[0.0;2048]\n var y:[i64;256]=[-1;256]\n var o:[f64;256]=[0.0;256]\n var offsets:[f64;3]=[0.0;3]\n";
   for(I phase=0;phase<3;phase++)std::cout<<" offsets["<<phase<<"]="<<offsets[phase]<<"\n";
   std::cout<<" var i:i64=0\n while i<256 {\n  a[(i*8+i%8) as usize]=1.0\n  y[i as usize]=(i*13)%31+1\n  o[i as usize]=offsets[(i%3) as usize]\n  i=i+1\n }\n";
   std::cout<<" let r=poisson_fit_offset(&a,&y,&o,256,8,0.0,100,1.0e-10)\n assert(r.status==0)\n assert(r.rank==8)\n assert(r.df==248)\n assert(!r.poisson_assumptions_assessed)\n assert(r.score_norm<=1.0e-10 && r.step_norm<=1.0e-10)\n assert(r.iterations>=1 && r.iterations<=100)\n close(r.deviance,"<<double(r.dev)<<")\n";
   std::cout<<" var expected_beta:[f64;8]=[0.0;8]\n";
   for(I j=0;j<8;j++)std::cout<<" expected_beta["<<j<<"]="<<double(r.b[j])<<"\n";
   std::cout<<" var expected_means:[f64;24]=[0.0;24]\n var expected_eta:[f64;24]=[0.0;24]\n";
   for(I cell=0;cell<24;cell++)std::cout<<" expected_means["<<cell<<"]="<<means[cell]<<"\n expected_eta["<<cell<<"]="<<etas[cell]<<"\n";
   std::cout<<" var j:i64=0\n while j<8 { let actual:f64=r.coefficients[j as usize] let expected:f64=expected_beta[j as usize] close(actual,expected) j=j+1 }\n i=0\n while i<256 {\n  let cell=(i%8)*3+i%3\n  let actual_mean:f64=r.means[i as usize]\n  let expected_mean:f64=expected_means[cell as usize]\n  close(actual_mean,expected_mean)\n  let actual_eta:f64=r.linear_predictors[i as usize]\n  let target_eta:f64=expected_eta[cell as usize]\n  close(actual_eta,target_eta)\n  i=i+1\n }\n}\n";
   continue;
  }
  auto setup=[&](I first,I end){for(I i=first;i<end;i++){for(I j=0;j<c.p;j++)std::cout<<" a["<<i*8+j<<"]="<<c.a[i*8+j]<<"\n";std::cout<<" y["<<i<<"]="<<c.y[i]<<"\n o["<<i<<"]="<<c.off[i]<<"\n";}};
  auto check=[&](I first,I end){for(I i=first;i<end;i++)std::cout<<" close(r.means["<<i<<"],"<<double(r.mu[i])<<")\n close(r.linear_predictors["<<i<<"],"<<double(r.eta[i])<<")\n";};
  bool chunks=c.n>64;
  if(chunks)for(I first=0;first<c.n;first+=32){
   std::cout<<"fn setup_"<<k<<"_"<<first<<"(a:&![f64;2048],y:&![i64;256],o:&![f64;256]) with Mut,Div,Panic {\n";
   setup(first,std::min(c.n,first+32));std::cout<<"}\n";
   std::cout<<"fn check_"<<k<<"_"<<first<<"(r:&PoissonFit) with Mut,Div,Panic {\n";
   check(first,std::min(c.n,first+32));std::cout<<"}\n";
  }
  std::cout<<"fn case_"<<k<<"() with Mut,Div,Panic {\n var a:[f64;2048]=[0.0;2048]\n var y:[i64;256]=[-1;256]\n var o:[f64;256]=[0.0;256]\n";
  if(chunks){for(I first=0;first<c.n;first+=32)std::cout<<" setup_"<<k<<"_"<<first<<"(&!a,&!y,&!o)\n";}else setup(0,c.n);
  std::cout<<" let r=poisson_fit_offset(&a,&y,&o,"<<c.n<<","<<c.p<<",0.0,100,1.0e-10)\n assert(r.status==0)\n assert(r.rank=="<<c.p<<")\n assert(r.df=="<<c.n-c.p<<")\n assert(!r.poisson_assumptions_assessed)\n assert(r.score_norm<=1.0e-10 && r.step_norm<=1.0e-10)\n assert(r.iterations>=1 && r.iterations<=100)\n close(r.deviance,"<<double(r.dev)<<")\n";
  for(I j=0;j<c.p;j++){
   std::cout<<" close(r.coefficients["<<j<<"],"<<double(r.b[j])<<")\n";
   if(r.b[j]!=0 && fabsq(r.b[j])<Q(1e-50)){
    std::cout<<" assert(r.coefficients["<<j<<"]!=0.0)\n";
    std::cout<<" let e"<<j<<"=r.coefficients["<<j<<"]-("<<double(r.b[j])<<")\n";
    std::cout<<" let d"<<j<<"=if e"<<j<<"<0.0 { -e"<<j<<" } else { e"<<j<<" }\n";
    std::cout<<" assert(d"<<j<<"<="<<double(fabsq(r.b[j])*Q(1e-7))<<")\n";
   }
  }
  if(chunks){for(I first=0;first<c.n;first+=32)std::cout<<" check_"<<k<<"_"<<first<<"(&r)\n";}else check(0,c.n);
  std::cout<<"}\n";
 }
 std::cout<<"fn main()->i32 with IO,Mut,Div,Panic {\n";for(size_t k=0;k<all.size();k++)std::cout<<" case_"<<k<<"()\n";std::cout<<" println(\"STATS_POISSON_REFERENCE_PASS\")\n 0\n}\n";std::cerr<<"cases="<<all.size()<<"\n";
}
