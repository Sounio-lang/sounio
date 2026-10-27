// C++23 binary128 active-prefix concordance. Reuses the independent empirical
// model oracle, not production Sounio QR or shrinkage implementation.
#define main dispersion_original_reference_main
#include "deep_stats_dispersion_reference.cpp"
#undef main
#include <iostream>
#include <sstream>
#include <set>
#include <tuple>
#include <numeric>

int main(int argc,char** argv){
 try{
  if(argc!=2)return 64;
  std::ifstream in(argv[1]);if(!in)return 2;
  std::array<Fit,2> expected;
  for(int c=0;c<2;++c){
   std::vector<Q> means,estimates,variances;
   constexpr std::array<double,8> observed{80,70,15,20,4,5,1,4};
   const int n=c==0?8:513;
   for(int i=0;i<n;++i){
    const double mu=c==0?double(1ULL<<i):double(i+1);
    const double e=c==0?observed[static_cast<std::size_t>(i)]:2.0+3.0/mu;
    means.push_back(Q(mu));estimates.push_back(Q(e));variances.push_back(Q(double(0.01)));
   }
   expected[static_cast<std::size_t>(c)]=fit(means,estimates,variances,c==0?Q(double(0.1)):Q(2),c==0?Q(100):Q(3),c==0?Q(1):Q(2));
  }
  std::set<int> cases,metas,tails,guards;
  std::set<std::tuple<int,int,int>> values;
  bool qr=false,complete=false;
  std::string tag;
  while(in>>tag){
   if(complete)return 9;
   if(tag=="PREFIX_CASE"){
    int c,n,cap;if(!(in>>c>>n>>cap)||c<0||c>1||!cases.insert(c).second)return 9;
    require(n==(c==0?8:513)&&cap==(c==0?13:1024),"active prefix declared shape");
   }else if(tag=="PREFIX_META"){
    int c,status,genes,rank,outliers,boundary;
    if(!(in>>c>>status>>genes>>rank>>outliers>>boundary)||c<0||c>1||!metas.insert(c).second)return 9;
    const auto& e=expected[static_cast<std::size_t>(c)];
    require(status==0&&genes==(c==0?8:513)&&rank==2&&outliers==std::accumulate(e.flags.begin(),e.flags.end(),0)&&boundary==int(e.boundary),"active prefix fit metadata");
   }else if(tag=="PREFIX_VALUE"){
    int c,kind,index;std::string token;
    if(!(in>>c>>kind>>index>>token)||!values.insert({c,kind,index}).second)return 9;
    char* end=nullptr;const Q actual=strtoflt128(token.c_str(),&end);if(end==token.c_str()||*end!='\0')return 9;
    Q value;
    if(c==2){
     if(kind!=0||index<0||index>1)return 9;
     value=Q(index+1);
    }else{
     if(c<0||c>1)return 9;
     const auto& e=expected[static_cast<std::size_t>(c)];
     if(kind==0){
      const std::array<Q,5> params{e.a0,e.a1,e.prior,e.ms,e.corr};
      if(index<0||index>=5)return 9;
      value=params[static_cast<std::size_t>(index)];
     }else{
      if(index<0||static_cast<std::size_t>(index)>=e.mapped.size())return 9;
      if(kind==1)value=e.mapped[static_cast<std::size_t>(index)];
      else if(kind==2)value=e.variance[static_cast<std::size_t>(index)];
      else if(kind==3)value=Q(e.flags[static_cast<std::size_t>(index)]);
      else return 9;
     }
    }
    concord(actual,value,"active prefix numerical disagreement");
   }else if(tag=="PREFIX_TAIL"){
    int c,n;if(!(in>>c>>n)||c<0||c>1||!tails.insert(c).second)return 9;
    require(n==(c==0?13-8:1024-513),"tail preserved");
   }else if(tag=="PREFIX_GUARDS"){
    int c,a,b,d,e;if(!(in>>c>>a>>b>>d>>e)||c<0||c>1||!guards.insert(c).second)return 9;
    require(a==1&&b==1&&d==1&&e==2,"active prefix refusal/legacy shape disagreement");
   }else if(tag=="PREFIX_QR"){
    int a,b,rank;if(!(in>>a>>b>>rank)||qr)return 9;qr=true;
    require(a==1&&b==0&&rank==2,"QR active prefix metadata");
   }else if(tag=="DISPERSION_PREFIX_PASS")complete=true;
   else return 9;
  }
  const std::size_t required=2+5+3*8+5+3*513;
  if(!in.eof()||!complete||!qr||cases.size()!=2||metas.size()!=2||tails.size()!=2||guards.size()!=2||values.size()!=required)return 9;
  std::cout<<"DISPERSION_ACTIVE_CPP_PASS cases=2 values="<<values.size()<<" QR=2 active_counts=8,513 capacities=13,1024\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 8;}
}
