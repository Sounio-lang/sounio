// ADR009 C++23 independent exact trust oracle and native-fixture generator.
#include <boost/multiprecision/cpp_int.hpp>
#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using Z=boost::multiprecision::cpp_int;
using T=std::optional<Z>;
using A=std::array<T,3>;
constexpr int NS=36936;
static void need(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
static T mn(const T&a,const T&b){if(!a)return b;if(!b)return a;return *a<*b?a:b;}
static A add(const A&a,const A&b){A c;for(int j=0;j<3;++j)c[j]=mn(a[j],b[j]);return c;}
static A mul(const A&a,const A&b){A c;for(int j=0;j<3;++j)if(a[j]&&b[j])c[j]=*a[j]+*b[j];return c;}
static A ann(long long s,long long i,long long a){A q;std::array<long long,3> v{s,i,a};for(int j=0;j<3;++j)if(v[j]>=0)q[j]=Z(v[j]);return q;}
static A one(){return ann(0,0,0);}static A cost(){return ann(-1,1,1);}
static bool present(const A&a){return bool(a[2]);}
static long long val(const T&x){if(!x)return -1;need(*x<=std::numeric_limits<long long>::max(),"unexpected reference overflow");return x->convert_to<long long>();}
static int S(int a,int b){return a*64+b;}static int R(int r,int a,int f){return 4096+(r*64+a)*64+f;}static int H(int r,int s){return 36864+r*8+s;}
struct Seed {int slot; A a;};
struct Model {int nc,nb,nr,nchain,top;std::array<int,704> p{};std::array<int,24> ch{};std::vector<Seed> seed;};
static Model model(int id){
 Model m{};for(int j=192;j<704;++j)m.p[j]=-1;
 if(id==0){
  m.nc=16;m.nb=8;m.nr=3;m.nchain=1;m.top=15;m.p[15]=1;
  m.p[8]=2;m.p[64+8]=1;m.p[128+8]=2;
  for(int j=0;j<6;++j){int d=9+j,r=j%3,f=j<3?1:2;m.p[d]=3;m.p[64+d]=r;m.p[128+d]=f;m.p[192+r*64+f]=d;}
  m.ch[0]=1;m.ch[8]=0;m.ch[16]=2;
  m.seed={{S(0,8),one()},{S(0,9),ann(-1,0,0)},{S(0,1),ann(7,7,7)},{S(0,6),ann(-1,-1,20)},{S(1,2),one()},{S(1,3),ann(-1,-1,0)},{S(2,4),one()},{S(3,5),ann(-1,0,0)},{S(4,5),one()},{S(5,6),one()},{S(6,0),one()},{S(7,0),one()},{R(0,1,2),one()},{H(0,1),one()},{H(1,2),ann(-1,0,0)},{36928,ann(-1,-1,0)}};
 }else{
  m.nc=64;m.nb=54;m.nr=8;m.nchain=8;m.top=63;m.p[63]=1;
  m.p[54]=2;m.p[64+54]=1;m.p[128+54]=2;
  for(int r=0;r<8;++r){int d=55+r;m.p[d]=3;m.p[64+d]=r;m.p[128+d]=1;m.p[192+r*64+1]=d;m.ch[r]=r;m.ch[8+r]=(r+1)%8;m.ch[16+r]=(r+2)%8;m.seed.push_back({H(r,(r+1)%8),r%2?ann(-1,0,0):one()});m.seed.push_back({36928+r,r%2?ann(-1,-1,0):one()});m.seed.push_back({R(r,(r+1)%54,1),r%3==0?one():(r%3==1?ann(-1,0,0):ann(-1,-1,0))});}
  for(int c=0;c<54;++c)m.seed.push_back({S(c,(c+1)%54),c%3==0?one():(c%3==1?ann(-1,0,0):ann(-1,-1,0))});
  m.seed.push_back({S(0,54),one()});m.seed.push_back({S(0,55),ann(-1,0,0)});m.seed.push_back({S(0,1),ann(30,30,30)});
 }
 for(int c=0;c<m.nb;++c)m.p[64+c]=c; // Distinct interned atom identifiers.
 return m;
}
static std::vector<int> active(const Model&m){std::vector<int> v;for(int a=0;a<m.nc;++a)for(int b=0;b<m.nc;++b)v.push_back(S(a,b));for(int r=0;r<m.nr;++r)for(int a=0;a<m.nc;++a)for(int f=0;f<m.nb;++f)v.push_back(R(r,a,f));for(int r=0;r<m.nr;++r)for(int s=0;s<m.nr;++s)v.push_back(H(r,s));for(int k=0;k<m.nchain;++k)v.push_back(36928+k);return v;}
// Synchronous snapshots avoid reproducing native in-place rule scheduling.
static std::vector<A> reference(const Model&m){
 std::vector<A> x(NS);for(int a=0;a<m.nc;++a){x[S(a,a)]=one();x[S(a,m.top)]=one();}for(int r=0;r<m.nr;++r)x[H(r,r)]=one();for(auto&s:m.seed)x[s.slot]=add(x[s.slot],s.a);
 for(int round=0;round<256;++round){auto y=x;auto rule=[&](int dst,std::initializer_list<int> premises){A z=cost();for(int slot:premises){if(!present(x[slot]))return;z=mul(z,x[slot]);}y[dst]=add(y[dst],z);};
  for(int r=0;r<m.nr;++r)for(int s=0;s<m.nr;++s)if(present(x[H(r,s)]))for(int t=0;t<m.nr;++t)rule(H(r,t),{H(r,s),H(s,t)});
  for(int a=0;a<m.nc;++a)for(int k=0;k<m.nc;++k)if(present(x[S(a,k)]))for(int d=0;d<m.nc;++d)rule(S(a,d),{S(a,k),S(k,d)});
  for(int d=0;d<m.nc;++d)if(m.p[d]==2)for(int a=0;a<m.nc;++a){int left=m.p[64+d],right=m.p[128+d];rule(S(a,left),{S(a,d)});rule(S(a,right),{S(a,d)});rule(S(a,d),{S(a,left),S(a,right)});}
  for(int d=0;d<m.nc;++d)if(m.p[d]==3)for(int a=0;a<m.nc;++a)rule(R(m.p[64+d],a,m.p[128+d]),{S(a,d)});
  for(int r=0;r<m.nr;++r)for(int a=0;a<m.nc;++a)for(int f=0;f<m.nb;++f)if(present(x[R(r,a,f)])){
   for(int g=0;g<m.nb;++g)if(m.p[192+r*64+g]>=0)rule(R(r,a,g),{R(r,a,f),S(f,g)});
   for(int s=0;s<m.nr;++s)if(m.p[192+s*64+f]>=0)rule(R(s,a,f),{R(r,a,f),H(r,s)});
   int e=m.p[192+r*64+f];if(e>=0)rule(S(a,e),{R(r,a,f)});
  }
  for(int k=0;k<m.nchain;++k){int r1=m.ch[k],r2=m.ch[8+k],r3=m.ch[16+k];for(int a=0;a<m.nc;++a)for(int f=0;f<m.nb;++f)if(present(x[R(r1,a,f)]))for(int g=0;g<m.nb;++g)if(m.p[192+r3*64+g]>=0)rule(R(r3,a,g),{R(r1,a,f),R(r2,f,g),36928+k});}
  if(y==x){std::cerr<<"CPP_SYNC_ROUNDS scenario="<<(m.nc==16?0:1)<<" rounds="<<round+1<<'\n';return x;}x=std::move(y);
 }
 throw std::runtime_error("reference did not stabilize within declared bound");
}
static std::vector<A> domain(){std::vector<A> d;auto le=[](int a,int b){return b==-1||(a>=0&&a<=b);};for(int s=-1;s<4;++s)for(int i=-1;i<4;++i)for(int a=-1;a<4;++a)if(le(i,s)&&le(a,i))d.push_back(ann(s,i,a));return d;}
static void emit(std::ostream&o){auto d=domain();
 o<<"//@ run-pass\n//@ expect-stdout: TRUST_SEMIRING_ELPLUS_PASS\n//@ timeout: 300\n// Generated by C++23 ontology_trust_reference.cpp --generate, inspectable deterministic fixture.\nuse ontology::trust_semiring::{TrustAnn,trust_ann,trust_zero,trust_one,trust_plus,trust_times,trust_status,trust_signed,trust_inferred,trust_assumed}\nuse ontology::elplus_trust::{trust_elplus_init,trust_elplus_seed,trust_elplus_fixpoint}\n\n";
 o<<"fn same(a:&TrustAnn,b:&TrustAnn)->bool { trust_status(a)==trust_status(b) && trust_signed(a)==trust_signed(b) && trust_inferred(a)==trust_inferred(b) && trust_assumed(a)==trust_assumed(b) }\nfn ann_at(n:i64)->TrustAnn {\n";
 for(unsigned j=0;j<d.size();++j)o<<" if n=="<<j<<" { return trust_ann("<<val(d[j][0])<<","<<val(d[j][1])<<","<<val(d[j][2])<<") }\n";
 o<<" trust_ann(-2,-2,-2)\n}\nfn scalar_laws()->i32 with IO,Mut,Div,Panic {\n var count:i64=0\n var i:i64=0\n let z=trust_zero()\n let one=trust_one()\n while i<"<<d.size()<<" {\n let a=ann_at(i)\n let az=trust_plus(&a,&z)\n let aa=trust_plus(&a,&a)\n let absorb=trust_plus(&one,&a)\n let ao=trust_times(&a,&one)\n let annih=trust_times(&a,&z)\n if !same(&az,&a)||!same(&aa,&a)||!same(&absorb,&one)||!same(&ao,&a)||!same(&annih,&z) { return 11 }\n var j:i64=0\n while j<"<<d.size()<<" {\n let b=ann_at(j)\n let ab=trust_plus(&a,&b)\n let ba=trust_plus(&b,&a)\n let tab=trust_times(&a,&b)\n let tba=trust_times(&b,&a)\n if !same(&ab,&ba)||!same(&tab,&tba) { return 12 }\n var k:i64=0\n while k<"<<d.size()<<" {\n let c=ann_at(k)\n let bc=trust_plus(&b,&c)\n let lhs=trust_times(&a,&bc)\n let ac=trust_times(&a,&c)\n let rhs=trust_plus(&tab,&ac)\n let abc=trust_plus(&ab,&c)\n let abcr=trust_plus(&a,&bc)\n let tbc=trust_times(&b,&c)\n let tl=trust_times(&tab,&c)\n let tr=trust_times(&a,&tbc)\n if !same(&lhs,&rhs)||!same(&abc,&abcr)||!same(&tl,&tr) { return 13 }\n count=count+1\n k=k+1\n }\n j=j+1\n }\n i=i+1\n }\n let big=trust_ann(9223372036854775807,9223372036854775807,9223372036854775807)\n let step=trust_ann(1,1,1)\n let over=trust_times(&big,&step)\n let inf=trust_times(&big,&z)\n let invalid=trust_ann(0,1,1)\n let malformed=trust_ann(-2,-1,-1)\n if trust_status(&over)!=2 || trust_status(&invalid)!=1 || trust_status(&malformed)!=1 || !same(&inf,&z) { return 14 }\n let a=trust_ann(-1,-1,0)\n let b=trust_ann(100,100,100)\n let c=trust_ann(-1,0,0)\n let bc=trust_plus(&b,&c)\n let product=trust_times(&a,&bc)\n if trust_assumed(&product)!=0 { return 15 }\n print(\"TRUST_SCALAR "<<d.size()<<" \")\n print_int(count)\n println(\"\")\n return 0\n}\n";
 o<<"fn emit_fact(state:&[i64;110808],scenario:i64,slot:i64) with IO,Mut,Div,Panic { print(\"TRUST_FACT \")\n print_int(scenario)\n print(\" \")\n print_int(slot)\n print(\" \")\n print_int(state[slot*3])\n print(\" \")\n print_int(state[slot*3+1])\n print(\" \")\n print_int(state[slot*3+2])\n println(\"\")\n}\n";
 for(int id=0;id<2;++id){auto m=model(id);o<<"fn scenario_"<<id<<"()->i32 with IO,Mut,Div,Panic {\n var state:[i64;110808]=[-1;110808]\n var pack:[i64;704]=[0;704]\n var chains:[i64;24]=[0;24]\n var dims:[i64;4]=[0;4]\n var ctl:[i64;4]=[0;4]\n var i:i64=192\n while i<704 { pack[i]=-1\n i=i+1 }\n";
  for(int j=0;j<704;++j)if((j<192&&m.p[j]!=0)||(j>=192&&m.p[j]!=-1))o<<" pack["<<j<<"]="<<m.p[j]<<"\n";
  for(int j=0;j<24;++j)if(m.ch[j]!=0)o<<" chains["<<j<<"]="<<m.ch[j]<<"\n";
  o<<" dims[0]="<<m.nchain<<"\n dims[1]="<<m.nc<<"\n dims[2]="<<m.nb<<"\n dims[3]="<<m.nr<<"\n let init=trust_elplus_init(&!state,"<<m.nc<<","<<m.top<<","<<m.nr<<")\n if init!=0 { return 21 }\n var seed_status:i64=0\n";
  for(auto&s:m.seed)o<<" seed_status=trust_elplus_seed(&!state,"<<s.slot<<","<<val(s.a[0])<<","<<val(s.a[1])<<","<<val(s.a[2])<<")\n if seed_status!=0 { return 22 }\n";
  o<<" let status=trust_elplus_fixpoint(&!state,&pack,&chains,&dims,&!ctl,256)\n print(\"TRUST_RUN "<<id<<" \")\n print_int(status)\n print(\" \")\n print_int(ctl[1])\n print(\" \")\n print_int(ctl[2])\n print(\" \")\n print_int(ctl[3])\n println(\"\")\n if status!=0 || ctl[0]!=0 { return 23 }\n";
  o<<" var a:i64=0\n while a<"<<m.nc<<" { var b:i64=0\n while b<"<<m.nc<<" { emit_fact(&state,"<<id<<",a*64+b)\n b=b+1 }\n a=a+1 }\n var r:i64=0\n while r<"<<m.nr<<" { a=0\n while a<"<<m.nc<<" { var f:i64=0\n while f<"<<m.nb<<" { emit_fact(&state,"<<id<<",4096+(r*64+a)*64+f)\n f=f+1 }\n a=a+1 }\n r=r+1 }\n r=0\n while r<"<<m.nr<<" { var s:i64=0\n while s<"<<m.nr<<" { emit_fact(&state,"<<id<<",36864+r*8+s)\n s=s+1 }\n r=r+1 }\n i=0\n while i<"<<m.nchain<<" { emit_fact(&state,"<<id<<",36928+i)\n i=i+1 }\n return 0\n}\n";
 }
 o<<"fn main()->i32 with IO,Mut,Div,Panic { let scalar=scalar_laws()\n if scalar!=0 { return scalar }\n let small=scenario_0()\n if small!=0 { return small }\n let scale=scenario_1()\n if scale!=0 { return scale }\n println(\"TRUST_SEMIRING_ELPLUS_PASS\")\n return 0\n}\n";
}
static int compare(const std::string&path){std::ifstream f(path);need(bool(f),"cannot read native stdout");std::array<std::vector<A>,2> expected={reference(model(0)),reference(model(1))};std::array<std::set<int>,2> want,seen;for(int id=0;id<2;++id)for(int slot:active(model(id)))want[id].insert(slot);std::set<int> runs;bool scalar=false,done=false;std::string line;std::uint64_t rows=0;
 while(std::getline(f,line)){std::istringstream s(line);std::string tag,extra;s>>tag;if(tag=="TRUST_FACT"){int id,slot;long long a,b,c;need(bool(s>>id>>slot>>a>>b>>c)&&!(s>>extra),"bad fact row");need(id>=0&&id<2&&want[id].contains(slot)&&seen[id].insert(slot).second,"unexpected/duplicate fact");need(a>=-1&&b>=-1&&c>=-1,"invalid negative count");need(ann(a,b,c)==expected[id][slot],"fact mismatch scenario="+std::to_string(id)+" slot="+std::to_string(slot));++rows;}else if(tag=="TRUST_RUN"){int id,status;long long rounds,updates,candidates;need(bool(s>>id>>status>>rounds>>updates>>candidates)&&!(s>>extra),"bad run row");need(id>=0&&id<2&&runs.insert(id).second&&status==0&&rounds>0&&rounds<=256&&updates>0&&candidates>=updates,"bad/incomplete run");}else if(tag=="TRUST_SCALAR"){unsigned n;std::uint64_t tests;auto d=domain();need(!scalar&&bool(s>>n>>tests)&&!(s>>extra)&&n==d.size()&&tests==d.size()*d.size()*d.size(),"incomplete scalar laws");scalar=true;}else if(line=="TRUST_SEMIRING_ELPLUS_PASS"){need(!done,"duplicate marker");done=true;}else if(!line.empty())throw std::runtime_error("unexpected output line");}
 need(scalar&&done&&runs.size()==2&&seen==want,"incomplete output");std::cout<<"TRUST_REFERENCE_PASS scenarios=2 facts="<<rows<<" scalar_domain="<<domain().size()<<" exact_cpp_int=true\n";return 0;}
int main(int argc,char**argv){try{if(argc==3&&(std::string(argv[1])=="--emit"||std::string(argv[1])=="--generate")){std::ofstream o(argv[2]);need(bool(o),"cannot create fixture");emit(o);return 0;}if(argc==3&&std::string(argv[1])=="--compare")return compare(argv[2]);if(argc==2)return compare(argv[1]);std::cerr<<"usage: oracle --emit fixture.sio | oracle native.stdout\n";return 64;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
