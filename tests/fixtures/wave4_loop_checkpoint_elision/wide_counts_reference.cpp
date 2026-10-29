#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>
using U=std::uint64_t;using I=std::int64_t;using P=std::array<double,4>;using N=std::array<U,4>;
constexpr U A=6364136223846793005ULL,C=1442695040888963407ULL;
struct Case {std::string name;P p;U n;I seed;bool reverse;};
static U inverse(){U x=1;for(int i=0;i<6;++i){x*=2-A*x;}return x;}
static I special_seed(U hi){return std::bit_cast<I>((((hi<<32)|0x1234ULL)-C)*inverse());}
static std::vector<Case> cases(){
 const double q=std::numeric_limits<double>::quiet_NaN();const P quarters{.25,.25,.25,.25};
 return {
 {"zero",{0,0,0,1},1000,0,false},{"negative_zero",{-0.0,0,0,1},1,special_seed(0),true},
 {"one",{1,0,0,0},1000,17,false},
 {"threshold25",quarters,1,special_seed(0x40000000),false},
 {"threshold50",quarters,1,special_seed(0x80000000),true},
 {"threshold75",quarters,1,special_seed(0xc0000000),false},
 {"below50",{std::nextafter(.5,0.),0,0,1-std::nextafter(.5,0.)},1,special_seed(0x80000000),false},
 {"above50",{std::nextafter(.5,1.),0,0,1-std::nextafter(.5,1.)},1,special_seed(0x80000000),true},
 {"upper32",{1-std::ldexp(1.,-32),0,0,std::ldexp(1.,-32)},1,special_seed(0xffffffff),false},
 {"nan00",{q,.5,0,.5},31,987,false},{"nan01",{.25,q,.25,.5},31,987,true},
 {"bench0",{.17,.23,.31,.29},10000000,200001,false},
 {"bench1",{.17,.23,.31,.29},10000000,208001,true},
 {"bench2",{.17,.23,.31,.29},10000000,216001,false}
 };
}
static N count(const Case&c){N out{};U s=std::bit_cast<U>(c.seed);if(s==0){s=1;}const double b01=c.p[0]+c.p[1],b10=b01+c.p[2];
 for(U k=0;k<c.n;++k){s=s*A+C;double u=double((s>>32)&0xffffffffULL)/4294967296.0;std::size_t ix=u<c.p[0]?0:u<b01?1:u<b10?2:3;++out[ix];}return out;
}
static std::string sio_int(I x){if(x==std::numeric_limits<I>::min()){return "(0 - 9223372036854775807 - 1)";}return std::to_string(x);}
static int generate(const std::string&dir){
 std::filesystem::create_directories(dir);std::ofstream cfg(dir+"/CASES.tsv"),calls(dir+"/CALLS.sio");
 cfg<<"name\tn\tseed\treverse\tp00_bits\tp01_bits\tp10_bits\tp11_bits\n";
 for(const auto&c:cases()){
  cfg<<c.name<<'\t'<<c.n<<'\t'<<c.seed<<'\t'<<c.reverse;
  calls<<" run_case(&p,\""<<c.name<<"\",BasisProb{";
  const char* names[]={"p00","p01","p10","p11"};
  for(unsigned i=0;i<4;++i){auto bits=std::bit_cast<I>(c.p[i]);cfg<<'\t'<<bits;calls<<(i?",":"")<<names[i]<<":bits_to_f64("<<sio_int(bits)<<")";}
  cfg<<'\n';calls<<"},"<<c.n<<","<<sio_int(c.seed)<<","<<(c.reverse?"true":"false")<<")\n";
 }
 return cfg&&calls?0:1;
}
struct Row {N n{};U ns=0;};
using Rows=std::map<std::string,std::map<std::string,Row>>;
static bool parse(std::istream&in,Rows&rows){
 std::string s,name,kind;Row row;unsigned at=0;bool ended=false;
 while(std::getline(in,s)){
  if(s=="WIDE_COUNTS_DIAGNOSTIC_END"){if(at!=0||ended){return false;}ended=true;continue;}
  if(ended){return false;}auto eq=s.find('=');if(eq==std::string::npos){return false;}auto key=s.substr(0,eq),val=s.substr(eq+1);
  if(at==0){if(key!="CASE"||val.empty()){return false;}name=val;}
  else if(at==1){if(key!="VARIANT"||(val!="literal"&&val!="hoist")){return false;}kind=val;}
  else {U v=0;auto r=std::from_chars(val.data(),val.data()+val.size(),v);if(r.ec!=std::errc{}||r.ptr!=val.data()+val.size()){return false;}
   if(at==2){if(key!="ELAPSED_NS"){return false;}row.ns=v;}else{const char* keys[]={"N00","N01","N10","N11"};if(key!=keys[at-3]){return false;}row.n[at-3]=v;}
  }
  ++at;if(at==7){if(rows[name].contains(kind)){return false;}rows[name][kind]=row;row={};at=0;}
 }
 return ended&&at==0;
}
static bool verify(const Rows&rows,bool loud,bool small=false){
 const auto cs=cases();std::size_t wanted=0;long double oldns=0,newns=0;
 for(const auto&c:cs){
  if(small&&c.n>1000){continue;}++wanted;
  auto it=rows.find(c.name);if(it==rows.end()||it->second.size()!=2){return false;}auto n=count(c);
  for(const std::string kind:{"literal","hoist"}){auto r=it->second.find(kind);if(r==it->second.end()||r->second.n!=n){if(loud){std::cerr<<"COUNT_REFUSED "<<c.name<<" "<<kind<<"\n";}return false;}}
  if(c.n==10000000){auto a=it->second.at("literal").ns,b=it->second.at("hoist").ns;if(a==0||b==0){return false;}oldns+=a;newns+=b;if(loud){std::cout<<"BENCH "<<c.name<<" literal_ns="<<a<<" hoist_ns="<<b<<" ratio="<<std::setprecision(9)<<static_cast<long double>(a)/b<<"\n";}}
 }
 if(rows.size()!=wanted){return false;}if(loud){std::cout<<"WIDE_CPP_COUNTS_OK cases="<<wanted<<" aggregate_ratio="<<std::setprecision(9)<<oldns/newns<<"\n";}return true;
}
int main(int argc,char**argv){
 if(argc==3&&std::string(argv[1])=="--generate"){return generate(argv[2]);}
 if(argc==2&&std::string(argv[1])=="--self-test"){
  if(A*inverse()!=1){return 1;}for(U h:{U(0),U(0x40000000),U(0x80000000),U(0xc0000000),U(0xffffffff)}){if((std::bit_cast<U>(special_seed(h))*A+C)>>32!=h){return 1;}}
  Rows rows;for(const auto&c:cases()){if(c.n>1000){continue;}auto n=count(c);rows[c.name]["literal"]={n,1};rows[c.name]["hoist"]={n,1};}
  if(!verify(rows,false,true)){return 1;}auto bad=rows;++bad.begin()->second.begin()->second.n[0];if(verify(bad,false,true)){return 1;}bad=rows;bad.erase(bad.begin());if(verify(bad,false,true)){return 1;}
  std::istringstream truncated("CASE=x\n");Rows p;if(parse(truncated,p)){return 1;}
  std::cout<<"WIDE_CPP_ORACLE_CONTROLS_OK\n";return 0;
 }
 if(argc!=2){return 64;}std::ifstream f(argv[1]);Rows rows;if(!f||!parse(f,rows)||!verify(rows,true)){return 1;}return 0;
}
