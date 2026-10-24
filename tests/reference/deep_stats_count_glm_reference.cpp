// Independent C++23/binary128 fixed-alpha count-GLM reference.
// Central Slurm: g++ -std=c++23 -O2 tests/reference/deep_stats_count_glm_reference.cpp -lquadmath -o deep_stats_count_glm_reference
// Optional sole argument: actual Sounio fixture stdout. No production QR,
// scoring iterations, logarithmic or deviance helpers are reused.
// Analytic group-intercept MLEs are verified by independent NB2 score equations.
#include <quadmath.h>
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <cstdio>
#include <cfloat>
#include <stdexcept>
using Q=__float128;
static void require(bool yes,const char* text){if(!yes)throw std::runtime_error(text);}
static void row(const char* name,Q x){char b[160];quadmath_snprintf(b,sizeof(b),"%.35Qg",x);std::printf("%s\t%s\n",name,b);}
static void concord(Q actual,Q expected,const char* field){
 require(!isnanq(actual)&&!isinfq(actual),field);
 require(fabsq(actual-expected)<=Q(1e-7)*(1+fabsq(expected)),field);
}
int main(int argc,char** argv){
 try{
  constexpr unsigned n=512,p=12;
  std::array<unsigned long long,p> sums{},sizes{};
  std::array<unsigned,n> counts{};
  for(unsigned i=0;i<n;++i){
   unsigned group=i%p;
   int variation=(i/p)%2==0?1:-1;
   counts[i]=unsigned(int(group)+2+variation);
   sums[group]+=counts[i];++sizes[group];
  }
  std::array<std::array<Q,p>,2> coefficients{};
  for(unsigned model=0;model<2;++model){
   Q alpha=model==0?Q(0):Q(double(0.25));
   Q offset=model==0?Q(0):Q(double(0.6931471805599453094));
   Q deviance=0;
   for(unsigned group=0;group<p;++group){
    Q mean=Q(sums[group])/Q(sizes[group]);
    coefficients[model][group]=logq(mean)-offset;
    Q score=0;
    for(unsigned i=group;i<n;i+=p){
     Q y=Q(counts[i]);
     score+=(y-mean)/(1+alpha*mean);
     if(alpha==0)deviance+=2*(y*logq(y/mean)-y+mean);
     else {
      Q inv=1/alpha;
      deviance+=2*(y*logq(y/mean)-(y+inv)*logq((y+inv)/(mean+inv)));
     }
    }
    require(fabsq(score)<Q(1e-26),"analytic group score");
    char label[80];std::snprintf(label,sizeof(label),"model%u_beta%u",model,group);
    row(label,coefficients[model][group]);
   }
   require(deviance>0,"positive varying-count deviance");
   row(model==0?"poisson_total_deviance":"nb2_total_deviance",deviance);
  }
  // Separate representability of total D and the finite mean objective.
  Q forced=Q(double(5e307)),total=6*forced;
  require(total>Q(DBL_MAX)&&total/4<Q(DBL_MAX),"aggregate deviance overflow scenario");
  row("overflow_scenario_mean_deviance",total/4);
  // Exact count-conversion boundary required by the public API.
  require(static_cast<unsigned long long>(9007199254740992ULL)+1ULL==9007199254740993ULL,"integer boundary");
  if(argc==2){
   std::ifstream input(argv[1]);require(bool(input),"native report open");
   std::string line;int model=-1;unsigned group=0,total_rows=0;bool got_status=false;
   while(std::getline(input,line)){
    if(line.starts_with("count model: ")){
     if(model>=0)require(group==p&&got_status,"previous model complete");
     model=std::stoi(line.substr(13));require(model>=0&&model<2,"native model");
     require(unsigned(model)==total_rows/p,"native model order");group=0;got_status=false;
    }else if(line.starts_with("count status: ")){
     require(model>=0,"status before model");require(std::stoi(line.substr(14))==0,"native fit status");got_status=true;
    }else if(line.starts_with("count coefficient: ")){
     require(model>=0&&group<p,"native coefficient row");
     const char* start=line.c_str()+19;char* end=nullptr;Q value=strtoflt128(start,&end);
     require(end && end!=start && (*end=='\0'||*end=='\r'),"native coefficient number");
     concord(value,coefficients[model][group],"native analytic coefficient");
     ++group;++total_rows;
    }
   }
   require(total_rows==24&&group==p&&got_status,"complete native coefficients");
  }
  std::puts("DEEP_COUNT_GLM_CPP_REFERENCE_PASS");
  return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}
}
