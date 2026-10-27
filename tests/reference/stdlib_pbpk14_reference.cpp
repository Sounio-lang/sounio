#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using Vec=std::array<long double,14>;
using Matrix=std::array<Vec,14>;
struct Params {long double ka=1,rb=1,fu=.3L,cl=50,gfr=7.5L;};
struct Case {std::string name;Vec initial{};Params p;long double time=0;std::int64_t intervals=0;int status=0;};
static void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static constexpr Vec volumes{1,1,3.9L,.5L,.33L,1.4L,28,14,2.6L,4,.15L,1,1.5L,.28L};
static constexpr Vec partition{1,1,1,.8L,1,.5L,.8L,2,1,.5L,1,1,1.2L,1.5L};
static constexpr Vec flows{0,0,0,175,7,21,31.5L,8.75L,8.75L,8.75L,5.25L,31.5L,10.5L,38.5L};
static Vec deriv(const Vec& y,const Params&p){
 Vec c{},d{};for(std::size_t i=0;i<14;++i)c[i]=y[i]/volumes[i];
 c[1]/=p.rb;c[2]/=p.rb;
 long double arterial_out=0;for(std::size_t i=4;i<14;++i)arterial_out+=flows[i];
 d[0]=-p.ka*y[0];d[1]=175*c[3]/partition[3]-arterial_out*c[1];d[2]=-175*c[2];d[3]=175*(c[2]-c[3]/partition[3]);
 for(std::size_t i=4;i<14;++i){
  d[i]=flows[i]*(c[1]-c[i]/partition[i]);
  if(i!=10&&i!=11&&i!=12)d[2]+=flows[i]*c[i]/partition[i];
 }
 const auto portal=flows[10]+flows[11],hepatic=portal+flows[12];
 d[11]+=p.ka*y[0];
 d[12]=flows[12]*c[1]+flows[10]*c[10]/partition[10]+flows[11]*c[11]/partition[11]-hepatic*c[12]/partition[12]-p.cl*p.fu*c[12];
 d[2]+=hepatic*c[12]/partition[12];
 d[13]-=p.gfr*p.fu*c[1]*p.rb;
 return d;
}
static Matrix matrix(const Params&p){
 Matrix a{};for(std::size_t j=0;j<14;++j){Vec unit{};unit[j]=1;const auto col=deriv(unit,p);for(std::size_t i=0;i<14;++i)a[i][j]=col[i];}return a;
}
static long double mass(const Vec&v){long double s=0;for(auto x:v)s+=x;return s;}
static long double rate(const Matrix&a){
 long double r=0;
 for(std::size_t j=0;j<14;++j){
  long double col=0;for(std::size_t i=0;i<14;++i){require(std::isfinite(a[i][j]),"matrix finite");if(i!=j)require(a[i][j]>=0,"negative off-diagonal");col+=a[i][j];}
  require(col<=1e-15L,"positive model mass source");r=std::max(r,-a[j][j]);
 }require(r>0&&std::isfinite(r),"removal rate");return r;
}
static Vec multiply(const Matrix&a,const Vec&x){
 Vec out{};for(std::size_t i=0;i<14;++i)for(std::size_t j=0;j<14;++j)out[i]+=a[i][j]*x[j];return out;
}
// Independent matrix-exponential action through positive uniformization.
// Small Poisson chunks avoid exp(-lambda*t) underflow at long horizons.
static Vec exact_action(const Case& c){
 if(c.time==0)return c.initial;
 const auto a=matrix(c.p);const auto lambda=rate(a);
 const auto count=static_cast<std::uint64_t>(std::ceil(2*lambda*c.time));
 require(count>0&&count<=10000000,"reference chunk budget");
 const long double h=c.time/count,mu=lambda*h;
 Matrix p{};for(std::size_t i=0;i<14;++i)for(std::size_t j=0;j<14;++j)p[i][j]=a[i][j]/lambda+(i==j?1:0);
 Vec y=c.initial;
 for(std::uint64_t chunk=0;chunk<count;++chunk){
  Vec power=y,sum{};long double weight=std::exp(-mu),accumulated=weight;
  for(std::size_t i=0;i<14;++i)sum[i]=weight*power[i];
  bool closed=false;
  for(unsigned k=1;k<128;++k){
   power=multiply(p,power);weight*=mu/k;accumulated+=weight;
   for(std::size_t i=0;i<14;++i)sum[i]+=weight*power[i];
   const long double next=weight*mu/(k+1),tail=next/(1-mu/(k+2));
   if(tail<=1e-25L){closed=true;break;}
  }
  require(closed&&std::fabs(accumulated-1)<1e-17L,"Poisson truncation");y=sum;
 }
 return y;
}
static Vec rk4(const Vec&y,long double h,const Params&p){
 const auto k1=deriv(y,p);Vec v{};for(std::size_t i=0;i<14;++i)v[i]=y[i]+h*k1[i]/2;
 const auto k2=deriv(v,p);for(std::size_t i=0;i<14;++i)v[i]=y[i]+h*k2[i]/2;
 const auto k3=deriv(v,p);for(std::size_t i=0;i<14;++i)v[i]=y[i]+h*k3[i];
 const auto k4=deriv(v,p);for(std::size_t i=0;i<14;++i)v[i]=y[i]+h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;return v;
}
static std::vector<Case> cases(){
 std::vector<Case> out;auto add=[&](std::string name,long double t,std::int64_t n,long double dose){Case c;c.name=name;c.time=t;c.intervals=n;c.initial[0]=dose;out.push_back(c);};
 add("oral24",24,100,100);add("oral2",2,50,100);add("short",.001L,1,100);add("empty",24,100,0);add("zero",0,100,100);
 add("custom",3,80,75);out.back().p={2,.75L,.1L,60,9};
 add("arterial",1,40,0);out.back().initial[1]=40;
 add("invalid-rb",1,10,100);out.back().p.rb=0;out.back().status=3;
 add("invalid-fu",1,10,100);out.back().p.fu=2;out.back().status=3;
 add("invalid-filtration",1,10,100);out.back().p.gfr=1e6L;out.back().status=3;
 add("invalid-steps",1,0,100);out.back().status=3;
 add("invalid-negative",1,10,-1);out.back().status=3;
 add("budget",1e9L,1,100);out.back().status=1;
 return out;
}
static std::string literal(long double x){std::ostringstream s;s<<std::setprecision(17)<<std::showpoint<<static_cast<double>(x);return s.str();}
static const std::array<std::string,14> names{"gut","art","ven","lung","heart","brain","muscle","adipose","skin","bone","spleen","gut_tis","liver","kidney"};
static void fixture(){
 std::cout<<"//@ run-pass\n//@ expect-stdout: PBPK14_CHECKED_OK\nuse ode::pbpk14_rk4::{PBPK14State,PBPK14Params,PBPK14Solution,solve_pbpk14_rk4}\n";
 std::cout<<"fn emit(name:string,s:&PBPK14Solution) with IO {\nprint(\"PBPK14 \");print(name);print(\" \");print_int(s.status);print(\" \");print_int(s.nsteps);print(\" \");print_int(s.internal_steps);print(\" \");print_int(s.nfeval);print(\" \");print_int(f64_to_bits(s.t_final));";
 for(const auto& n:names)std::cout<<"print(\" \");print_int(f64_to_bits(s."<<n<<"));";
 std::cout<<"print(\"\\n\")\n}\nfn main()->i32 with IO,Mut,Div,Panic {\n";
 for(const auto& c:cases()){
  std::cout<<"{\nlet state=PBPK14State {";
  for(std::size_t i=0;i<14;++i){if(i)std::cout<<',';std::cout<<names[i]<<':'<<literal(c.initial[i]);}
  std::cout<<"}\nlet params=PBPK14Params {ka:"<<literal(c.p.ka)<<",rb:"<<literal(c.p.rb)<<",fu:"<<literal(c.p.fu)<<",cl_int:"<<literal(c.p.cl)<<",gfr:"<<literal(c.p.gfr)<<"}\n";
  std::cout<<"let sol=solve_pbpk14_rk4(state,params,"<<literal(c.time)<<','<<c.intervals<<")\n";
  std::cout<<"assert(sol.status=="<<c.status<<")\nemit(\""<<c.name<<"\",&sol)\n}\n";
 }
 std::cout<<"println(\"PBPK14_CHECKED_OK\")\n0\n}\n";
}
static void selftest(){
 const auto all=cases();
 for(const auto& c:all)if(c.status==0){
  const auto a=matrix(c.p);const auto lambda=rate(a);const auto want=exact_action(c);
  Vec candidate=c.initial;
  const auto sub=c.time==0?0:static_cast<std::int64_t>(std::max(1.0L,std::ceil((c.time/c.intervals)*4*lambda)));
  const auto total=c.intervals*sub;
  if(total>0){const long double h=c.time/total;for(std::int64_t i=0;i<total;++i)candidate=rk4(candidate,h,c.p);}
  long double error=0;for(std::size_t i=0;i<14;++i){require(std::isfinite(want[i])&&want[i]>=0,"positive exact action");error=std::max(error,std::fabs(candidate[i]-want[i]));}
  require(error<1e-8L,"substep/matrix-exponential discordance");
  require(mass(want)<=mass(c.initial)+1e-10L,"mass creation");
  std::cout<<std::setprecision(21)<<c.name<<" internal="<<total<<" max_error="<<error<<" final_mass="<<mass(want)<<"\n";
 }
 Vec old=all[0].initial;const auto dt=all[0].time/all[0].intervals;
 for(std::int64_t i=0;i<all[0].intervals;++i)old=rk4(old,dt,all[0].p);
 const auto old_total=old[0]+old[1]+old[2]+old[3]+old[12]+old[13];
 bool invalid=false;for(auto value:old)invalid=invalid||!std::isfinite(value)||value<0;
 invalid=invalid||mass(old)>mass(all[0].initial)+1e-8L;
 std::cout<<std::setprecision(21)<<"old_coarse_six_compartment_total="<<old_total<<" old_coarse_mass="<<mass(old)<<"\n";
 require(invalid,"old coarse physical witness unexpectedly valid");
 std::cout<<"PBPK14_OLD_COARSE_REJECTED\nPBPK14_CPP_REFERENCE_OK\n";
}
static void check(const std::string& path){
 std::ifstream in(path);require(bool(in),"open output");std::string line;const auto all=cases();std::size_t cursor=0;bool marker=false;
 while(std::getline(in,line)){
  if(line=="PBPK14_CHECKED_OK"){require(!marker,"duplicate marker");marker=true;continue;}
  std::istringstream row(line);std::string tag,name,extra;std::int64_t status,n,internal,evals,tbits;
  require(bool(row>>tag>>name>>status>>n>>internal>>evals>>tbits)&&tag=="PBPK14"&&cursor<all.size(),"result row");
  const auto&c=all[cursor++];require(name==c.name&&status==c.status,"case/status");
  Vec got{};for(auto& x:got){std::int64_t bits;require(bool(row>>bits),"state bits");x=std::bit_cast<double>(bits);}
  require(!(row>>extra),"extra output");
  if(status==0){
   require(n==(c.time==0?0:c.intervals)&&internal>=n&&evals==4*internal,"step accounting");
   const auto time=std::bit_cast<double>(tbits);require(std::fabs(time-c.time)<1e-12L*std::max(1.0L,c.time),"final time");
   const auto want=exact_action(c);
   for(std::size_t i=0;i<14;++i)if(!std::isfinite(got[i])||got[i]<0||std::fabs(got[i]-want[i])>1e-8L){std::cerr<<std::setprecision(21)<<name<<' '<<names[i]<<" got="<<got[i]<<" expected="<<want[i]<<"\n";throw std::runtime_error("numerical mismatch");}
   require(mass(got)<=mass(c.initial)+1e-8L,"mass creation");
  }else require(n==0&&internal==0&&evals==0&&tbits==0,"refusal performed integration");
 }
 require(marker&&cursor==all.size(),"incomplete output");
 std::cout<<"PBPK14_NATIVE_CPP_OK\n";
}

static void solver_fixture(){
 std::cout<<"//@ run-pass\n//@ expect-stdout: ODE_OPTION_FACTORIES_OK\nuse ode::solver::{SolverODESolution,default_options,fast_options,high_accuracy_options,solve_rk45_exp_decay,exp_decay_analytical}\n";
 std::cout<<"fn emit(name:string,s:&SolverODESolution) with IO {\nprint(\"SOLVER \");print(name);print(\" \");print_int(s.status);print(\" \");print_int(s.nsteps);print(\" \");print_int(s.nreject);print(\" \");print_int(f64_to_bits(s.u_final));print(\" \");print_int(f64_to_bits(s.t_final));print(\"\\n\")\n}\n";
 std::cout<<"fn main()->i32 with IO,Mut,Div,Panic {\n";
 for(const auto& pair:std::array<std::pair<std::string,std::string>,3>{{{"default","default_options"},{"fast","fast_options"},{"high","high_accuracy_options"}}}){
  std::cout<<"{\nlet opts="<<pair.second<<"()\nlet sol=solve_rk45_exp_decay(1.0,0.0,10.0,opts)\nassert(sol.success)\nassert(sol.status==0)\nemit(\""<<pair.first<<"\",&sol)\n}\n";
 }
 std::cout<<"{\nvar bad=default_options()\nbad.max_steps=1000001\nlet refused=solve_rk45_exp_decay(1.0,0.0,10.0,bad)\nassert(refused.status==3)\nassert(refused.nsteps==0)\nassert(refused.nfeval==0)\n}\nprintln(\"ODE_OPTION_FACTORIES_OK\")\n0\n}\n";
}
static void check_solver(const std::string& path){
 std::ifstream in(path);require(bool(in),"solver output open");std::string line;unsigned rows=0;bool marker=false;
 const std::array<std::string,3> names{"default","fast","high"};const long double exact=std::exp(-1.0L);
 for(;std::getline(in,line);){
  if(line=="ODE_OPTION_FACTORIES_OK"){require(!marker,"duplicate solver marker");marker=true;continue;}
  std::istringstream row(line);std::string tag,name,extra;std::int64_t status,steps,rejects,u,t;
  require(bool(row>>tag>>name>>status>>steps>>rejects>>u>>t)&&!(row>>extra)&&tag=="SOLVER"&&rows<3&&name==names[rows],"solver row");
  const auto value=std::bit_cast<double>(u),time=std::bit_cast<double>(t);
  const long double tolerance=rows==2?1e-8L:rows==1?1e-3L:1e-4L;
  require(status==0&&steps>0&&steps<10000&&rejects>=0&&std::isfinite(value)&&std::fabs(value-exact)<tolerance&&std::fabs(time-10)<1e-12L,"solver numerical mismatch");
  std::cout<<std::setprecision(21)<<name<<" error="<<std::fabs(value-exact)<<" steps="<<steps<<"\n";++rows;
 }
 require(marker&&rows==3,"incomplete solver output");std::cout<<"ODE_OPTIONS_NATIVE_CPP_OK\n";
}
static void wrong_output(const std::string& path,const std::string& target){
 std::ifstream in(path);std::ofstream out(target);require(bool(in)&&bool(out),"wrong-output files");std::string line;bool changed=false;
 while(std::getline(in,line)){
  if(!changed&&line.rfind("PBPK14 oral24 ",0)==0){
   std::istringstream row(line);std::vector<std::string> words;std::string word;while(row>>word)words.push_back(word);
   require(words.size()==21,"mutation row");words[7]=std::to_string(std::bit_cast<std::int64_t>(1000000.0));line.clear();
   for(const auto& w:words){if(!line.empty())line+=' ';line+=w;}changed=true;
  }out<<line<<"\n";
 }require(changed&&bool(out),"mutation write");
}

int main(int argc,char**argv){
 try{
  require(argc>=2,"mode");const std::string mode=argv[1];
  if(mode=="--fixture"){require(argc==2,"fixture args");fixture();}
  else if(mode=="--selftest"){require(argc==2,"selftest args");selftest();}
  else if(mode=="--check"){require(argc==3,"check args");check(argv[2]);}
  else if(mode=="--solver-fixture"){require(argc==2,"solver fixture args");solver_fixture();}
  else if(mode=="--check-solver"){require(argc==3,"solver check args");check_solver(argv[2]);}
  else if(mode=="--wrong-output"){require(argc==4,"mutation args");wrong_output(argv[2],argv[3]);}
  else throw std::runtime_error("unknown mode");
  return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
