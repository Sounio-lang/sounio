// C++23 independent analytic Jacobian control for imported Sounio outputs.
// No production implementation is linked. Decimal-400 algebra starts from the
// exact binary64 inputs and compares native result bits, including covariance.
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using Q=boost::multiprecision::number<boost::multiprecision::cpp_dec_float<400>>;
Q exact(double x){auto u=std::bit_cast<std::uint64_t>(x); auto e=int((u>>52)&2047);auto f=u&((1ull<<52)-1); if(e==2047)throw std::runtime_error("nonfinite reference input"); Q q=e?Q(f|(1ull<<52)):Q(f);q=ldexp(q,e?e-1075:-1074);return u>>63?-q:q;}
const Q c=exact(299792458.0),c2=c*c;
Q g(Q v){return c/sqrt((c-v)*(c+v));}
Q norm(const std::vector<Q>&h){Q s=0;for(const auto&x:h)s+=x*x;return sqrt(s);}
struct Scalar{Q x,u;};
struct Pair{std::array<Q,2>x,u;std::array<Q,4>cov;};
Scalar scalar(Q x,std::vector<Q>h){return {x,norm(h)};}
Pair paired(Q x,Q y,std::vector<Q>a,std::vector<Q>b){Q aa=0,ab=0,bb=0;for(std::size_t i=0;i<a.size();++i){aa+=a[i]*a[i];ab+=a[i]*b[i];bb+=b[i]*b[i];}return {{x,y},{sqrt(aa),sqrt(bb)},{aa,ab,ab,bb}};}
Scalar gamma_ref(double vd,double ud){Q v=exact(vd),u=exact(ud),G=g(v);return scalar(G,{G*G*G*v*u/c2});}
Scalar force(double md,double ud,double ad,double ua){Q m=exact(md),um=exact(ud),a=exact(ad),u=exact(ua);return scalar(m*a,{a*um,m*u});}
Scalar ke(double md,double ud,double vd,double uv){Q m=exact(md),um=exact(ud),v=exact(vd),u=exact(uv);return scalar(m*v*v/2,{v*v*um/2,m*v*u});}
Scalar pos(double sd,double us,double vd,double uv,double ad,double ua,double td){Q s=exact(sd),v=exact(vd),a=exact(ad),t=exact(td);return scalar(s+v*t+a*t*t/2,{exact(us),t*exact(uv),t*t*exact(ua)/2});}
Scalar energy(double md,double ud,double vd,double uv){Q m=exact(md),um=exact(ud),v=exact(vd),u=exact(uv),G=g(v);return scalar(G*m*c2,{G*c2*um,m*G*G*G*v*u});}
Scalar momentum(double md,double ud,double vd,double uv){Q m=exact(md),um=exact(ud),v=exact(vd),u=exact(uv),G=g(v);return scalar(G*m*v,{G*v*um,m*G*G*G*u});}
Scalar mass_scaled(Q z,Q uz,Q p,Q up){Q r=sqrt((z-p)*(z+p));return scalar(r/c,{z/r*uz/c,p/r*up/c});}
Scalar four_force(double md,double ud,double ad,double ua,double vd,double uv){Q m=exact(md),um=exact(ud),a=exact(ad),u=exact(ua),v=exact(vd),sv=exact(uv),G=g(v);return scalar(G*G*G*m*a,{G*G*G*a*um,G*G*G*m*u,3*G*G*G*G*G*v*m*a*sv/c2});}
Pair boost_ref(double ed,double ud,double pd,double up,double vd,double uv){Q e=exact(ed),ue=exact(ud),p=exact(pd),sp=exact(up),v=exact(vd),sv=exact(uv),G=g(v),E=G*(e-v*p),P=G*(p-v*e/c2);return paired(E,P,{G*ue,-G*v*sp,-G*G*P*sv},{-G*v/c2*ue,G*sp,-G*G/c2*E*sv});}
Pair collision(double m1d,double u1d,double v1d,double w1d,double m2d,double u2d,double v2d,double w2d){Q m1=exact(m1d),um1=exact(u1d),v1=exact(v1d),uv1=exact(w1d),m2=exact(m2d),um2=exact(u2d),v2=exact(v2d),uv2=exact(w2d),g1=g(v1),g2=g(v2);return paired((g1*m1+g2*m2)*c2,g1*m1*v1+g2*m2*v2,{g1*c2*um1,m1*g1*g1*g1*v1*uv1,g2*c2*um2,m2*g2*g2*g2*v2*uv2},{g1*v1*um1,m1*g1*g1*g1*uv1,g2*v2*um2,m2*g2*g2*g2*uv2});}
std::map<int,Scalar> scalars(){const double C=299792458.0;std::map<int,Scalar>s;
s[0]=force(0,1,2,0);s[1]=ke(2,.1,3,.2);s[2]=pos(1,.1,2,.2,3,.3,2);s[3]=gamma_ref(-.6*C,.001*C);s[4]=gamma_ref(0,1);s[5]=gamma_ref(C-0x1p-24,1e-15);
{Q G=g(exact(-.6*C)),t=exact(2),ut=exact(.1),v=exact(-.6*C),uv=exact(.001*C);s[6]=scalar(G*t,{G*ut,t*G*G*G*v*uv/c2});s[7]=scalar(exact(3)/G,{exact(.2)/G,exact(3)*G*v*uv/c2});}
s[8]=energy(0,.001,.6*C,.001*C);s[9]=momentum(2,.1,0,.1*C);
{Q u=exact(.5*C),v=u,d=1+u*v/c2;s[10]=scalar((u+v)/d,{(1-v*v/c2)/(d*d)*exact(.001*C),(1-u*u/c2)/(d*d)*exact(.002*C)});}
s[11]=mass_scaled(exact(2*C*C)/c,exact(.1*C*C)/c,exact(.6*C),exact(.01*C));
// FourMomentum stores the already rounded conversion E/c and uE/c.
s[12]=mass_scaled(exact(C*C/C),exact(1.0/C),0,0);
s[13]=four_force(2,.1,-3,.2,-.6*C,.001*C);s[14]=energy(1e280,1e279,0,0);s[15]=force(1e200,1e190,1e100,0);s[16]=pos(1e300,0,-1e300,0,2,0,1);s[17]=s[13];{Q u=exact(C-0x1p-24),v=exact(-C+0x1p-23),d=1+u*v/c2;s[18]=scalar((u+v)/d,{(1-v*v/c2)/(d*d)*exact(1e-15),(1-u*u/c2)/(d*d)*exact(2e-15)});}s[19]=mass_scaled(exact(1e308),exact(1e290),exact(5e307),exact(2e290));return s;}
std::map<int,Pair> pairs(){double C=299792458.0;return {{0,boost_ref(10*C,.1*C,2,.2,-.3*C,.001*C)},{1,collision(2,.1,.3*C,.001*C,3,.2,-.4*C,.002*C)},{2,paired(exact(10*C/C),exact(2),{exact(.1*C/C),0},{0,exact(.2)})},{3,boost_ref(10*C,.1*C,2,.2,0,.001*C)},{4,collision(2,.1,0,.001*C,3,.2,0,.002*C)}};}
double from_bits(const std::string&s){std::size_t n=0;auto i=std::stoll(s,&n);if(n!=s.size())throw std::runtime_error("invalid bits");return std::bit_cast<double>(std::bit_cast<std::uint64_t>(static_cast<std::int64_t>(i)));}
void equal(const std::string&field,const std::string&bits,const Q&expected){double x=from_bits(bits);if(!std::isfinite(x))throw std::runtime_error(field+" nonfinite native output");Q d=abs(exact(x)-expected);Q tolerance=abs(expected)*Q("5e-13")+exact(std::numeric_limits<double>::denorm_min())*8;if(d>tolerance){std::ostringstream s;s<<field<<" actual="<<x<<" expected="<<expected<<" abs_error="<<d;throw std::runtime_error(s.str());}}
std::vector<std::string> words(const std::string&s){std::istringstream in(s);std::vector<std::string>w;for(std::string x;in>>x;)w.push_back(x);return w;}
int main(int argc,char**argv){try{if(argc!=3)throw std::runtime_error("usage: reference checked|legacy stdout");bool legacy=std::string(argv[1])=="legacy";if(!legacy&&std::string(argv[1])!="checked")throw std::runtime_error("bad mode");std::ifstream in(argv[2]);if(!in)throw std::runtime_error("missing native stdout");auto s=scalars();auto p=pairs();std::map<int,int>seenS,seenP,seenU,seenPU;int marker=0;for(std::string line;std::getline(in,line);){auto w=words(line);if(w.empty())continue;auto tag=w[0];if(tag=="PHYSICS_UNCERTAINTY_CHECKED_PASS"||tag=="PHYSICS_UNCERTAINTY_LEGACY_PASS"){if(w.size()!=1||tag!=(legacy?"PHYSICS_UNCERTAINTY_LEGACY_PASS":"PHYSICS_UNCERTAINTY_CHECKED_PASS"))throw std::runtime_error("bad marker");++marker;continue;}if(tag.rfind("PHYSICS_",0)!=0)continue;
if(tag=="PHYSICS_LEGACY"&&legacy){if(w.size()!=4)throw std::runtime_error("bad legacy row");int id=std::stoi(w[1]);if(++seenS[id]!=1)throw std::runtime_error("duplicate legacy id");Scalar r;if(id<=5){int target=id==4?10:id==5?12:id;r=s.at(target);}else if(id==6)r={p.at(0).x[0],p.at(0).u[0]};else if(id==7)r={p.at(0).x[1],p.at(0).u[1]};else throw std::runtime_error("unknown legacy id");equal("legacy value "+w[1],w[2],r.x);equal("legacy u "+w[1],w[3],r.u);continue;}
if(tag=="PHYSICS_SCALAR"&&!legacy){if(w.size()!=7||w[2]!="0"||w[5]!="1"||w[6]!="1")throw std::runtime_error("scalar contract");int id=std::stoi(w[1]);if(++seenS[id]!=1)throw std::runtime_error("duplicate scalar");auto r=s.at(id);equal("scalar value "+w[1],w[3],r.x);equal("scalar u "+w[1],w[4],r.u);continue;}
if(tag=="PHYSICS_PAIR"&&!legacy){if(w.size()!=13||w[2]!="0"||w[11]!="1"||w[12]!="1")throw std::runtime_error("pair contract");int id=std::stoi(w[1]);if(++seenP[id]!=1)throw std::runtime_error("duplicate pair");auto r=p.at(id);for(int k=0;k<2;++k){equal("pair value "+w[1],w[3+k],r.x[k]);equal("pair u "+w[1],w[5+k],r.u[k]);}for(int k=0;k<4;++k)equal("pair cov "+w[1],w[7+k],r.cov[k]);continue;}
if(tag=="PHYSICS_UNAVAILABLE"&&!legacy){const std::array<int,10>st={1,1,2,2,3,2,2,3,5,4};if(w.size()!=3)throw std::runtime_error("bad unavailable row");int id=std::stoi(w[1]);if(++seenU[id]!=1||std::stoi(w[2])!=st.at(id))throw std::runtime_error("unavailable contract");continue;}
if(tag=="PHYSICS_PAIR_UNAVAILABLE"&&!legacy){const std::array<int,3>st={3,2,1};if(w.size()!=3)throw std::runtime_error("bad pair unavailable row");int id=std::stoi(w[1]);if(++seenPU[id]!=1||std::stoi(w[2])!=st.at(id))throw std::runtime_error("pair unavailable contract");continue;}
throw std::runtime_error("unexpected physics row: "+line);
}if(marker!=1||seenS.size()!=(legacy?8:20)||(!legacy&&(seenP.size()!=5||seenU.size()!=10||seenPU.size()!=3)))throw std::runtime_error("missing rows or marker");std::cout<<"PHYSICS_CPP23_REFERENCE_PASS mode="<<argv[1]<<" scalar="<<seenS.size()<<" pairs="<<seenP.size()<<" unavailable="<<seenU.size()+seenPU.size()<<" precision=decimal400 relative_tolerance=5e-13\n";return 0;}catch(const std::exception&e){std::cerr<<"PHYSICS_CPP23_REFERENCE_FAIL "<<e.what()<<'\n';return 1;}}
