#include <bit>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>
#include <array>
struct Draw {double value;std::uint64_t state;};
static Draw normal(std::uint64_t state){
 double sum=0;
 for(int i=0;i<12;++i){
  state=state*6364136223846793005ULL+1442695040888963407ULL;
  std::uint64_t magnitude=state;
  if(std::bit_cast<std::int64_t>(state)<0){magnitude=0ULL-state;}
  const auto signed_magnitude=std::bit_cast<std::int64_t>(magnitude);
  const std::int64_t bucket=signed_magnitude%1073741824LL;
  sum+=static_cast<double>(bucket)/1073741824.0;
 }
 return {sum-6.0,state};
}
struct Result {double mean,variance;std::int64_t confidence;};
static Result mc(bool square,bool sum_two){
 double total=0,squares=0;
 std::uint64_t seed=sum_two?67890:12345;
 for(int i=0;i<64;++i){
  auto a=normal(seed);seed=a.state;
  double value=3.0+a.value;
  if(sum_two){auto b=normal(seed);seed=b.state;value+=1.0+b.value;}
  if(square){value*=value;}
  total+=value;squares+=value*value;
 }
 const double mean=total/64.0;
 return {mean,std::fmax(squares/64.0-mean*mean,0.0),57};
}
static void require(bool b,const char* message){if(!b){throw std::runtime_error(message);}}
static double decode(std::int64_t value){return std::bit_cast<double>(value);}
static void close(double got,double expected){
 require(std::isfinite(got),"nonfinite result");
 if(std::fabs(got-expected)>1e-11*std::fmax(1.0,std::fabs(expected))){std::cerr<<std::setprecision(17)<<"got="<<got<<" expected="<<expected<<"\n";throw std::runtime_error("numerical mismatch");}
}
static std::string negative(const std::string& mode){
 const std::string imports="use epistemic::knowledge::{epistemic_new}\nuse epistemic::propagate::{monte_carlo_identity,monte_carlo}\n";
 if(mode=="--missing-prob"){
  return imports+"fn main()->i32 with Mut,Div,Panic {let x=epistemic_new(3.0,1.0,850);let y=monte_carlo_identity(x,64);0}\n";
 }
 if(mode=="--io-callback"){
  return imports+"fn cb(x:f64)->f64 with IO {println(\"BAD_IO\");x}\nfn main()->i32 with IO,Prob,Mut,Div,Panic {let x=epistemic_new(3.0,1.0,850);let y=monte_carlo(x,cb,64);0}\n";
 }
 return imports+"fn main()->i32 with IO,Prob,Mut,Div,Panic {let x=epistemic_new(3.0,1.0,850);let y=monte_carlo_identity(x,0);println(\"BAD_EMPTY_SAMPLE\");0}\n";
}
int main(int argc,char**argv){
 try{
  require(argc>=2,"input required");
  const std::string mode=argv[1];
  if(mode=="--missing-prob"||mode=="--io-callback"||mode=="--empty-samples"){std::cout<<negative(mode);return 0;}
  std::ifstream in(argv[1]);require(bool(in),"input unavailable");
  if(argc==4&&std::string(argv[2])=="--mutate"){
   std::ofstream out(argv[3]);std::string line;bool changed=false;
   while(std::getline(in,line)){
    if(line.rfind("MC identity ",0)==0){std::istringstream row(line);std::string tag,name;std::int64_t v,u,c;row>>tag>>name>>v>>u>>c;line="MC identity "+std::to_string(v)+" 0 "+std::to_string(c);changed=true;}
    out<<line<<'\n';
   }
   require(changed&&bool(out),"mutation failed");return 0;
  }
  int constructors=0,mc_rows=0,bridges=0;bool marker=false;std::string line;
  const std::array<double,4> values{3.25,-1.5,0.125,0.0},variances{0.25,0.5,0.0,1.0};
  const std::array<std::int64_t,4> confidence{850,0,1000,500};
  const std::array<std::string,5> names{"identity","generic","square","generic_square","sum"};
  while(std::getline(in,line)){
   if(line=="EPISTEMIC_BRIDGE_OK"){require(!marker,"duplicate marker");marker=true;continue;}
   std::istringstream row(line);std::string tag;row>>tag;
   if(tag=="CTOR"){
    std::int64_t v=0,u=0,c=0;require(bool(row>>v>>u>>c),"bad constructor row");
    require(constructors<4,"extra constructor");
    require(v==std::bit_cast<std::int64_t>(values[constructors])&&u==std::bit_cast<std::int64_t>(variances[constructors])&&c==confidence[constructors],"constructor field mismatch");
    ++constructors;
   }else if(tag=="MC"){
    std::string name;std::int64_t v=0,u=0,c=0;require(bool(row>>name>>v>>u>>c),"bad MC row");require(mc_rows<5&&name==names[mc_rows],"MC order mismatch");
    const Result expected=mc(name=="square"||name=="generic_square",name=="sum");
    close(decode(v),expected.mean);close(decode(u),expected.variance);require(c==expected.confidence,"MC confidence mismatch");++mc_rows;
   }else if(tag=="BRIDGE"){
    std::string name;std::int64_t e=0,p=0,t=0;require(bool(row>>name>>e>>p>>t),"bad bridge row");
    require(bridges<2&&name==(bridges==0?"risk":"precise"),"bridge order mismatch");
    const double reward=bridges==0?-std::log(0.75*std::exp(-20.0)+0.25*std::exp(-12.0))/2.0:9.0;
    const double info=0.5*std::log(4.0),prag=1.5*reward;
    close(decode(e),info);close(decode(p),prag);close(decode(t),-info-prag);++bridges;
   }else{throw std::runtime_error("unknown output row");}
   std::string extra;require(!(row>>extra),"trailing row fields");
  }
  require(constructors==4&&mc_rows==5&&bridges==2&&marker,"missing rows or marker");
  std::cout<<"EPISTEMIC_BRIDGE_CPP_OK\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
