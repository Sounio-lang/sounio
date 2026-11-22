#pragma once
// sII water-oxygen lattice (Fd-3m origin 2) and the H2 guest potential (one-site LJ,
// Lorentz-Berthelot, quadratic Feynman-Hibbs, point-centred cutoff + uniform-density tail).
#include <array>
#include <cmath>
#include <vector>
#include <algorithm>
using V=std::array<double,3>;
struct Op{int r[3][3];double t[3];};
inline double fr(double v){v-=std::floor(v);if(v>1-1e-9)v=0;return v;}
inline V act(const Op&o,const V&p){V q;for(int i=0;i<3;++i){q[i]=o.t[i];for(int j=0;j<3;++j)q[i]+=o.r[i][j]*p[j];q[i]=fr(q[i]);}return q;}
inline Op comp(const Op&a,const Op&b){Op c{};for(int i=0;i<3;++i){c.t[i]=a.t[i];for(int j=0;j<3;++j){c.t[i]+=a.r[i][j]*b.t[j];c.r[i][j]=0;for(int k=0;k<3;++k)c.r[i][j]+=a.r[i][k]*b.r[k][j];}c.t[i]=fr(c.t[i]);}return c;}
inline bool same(const Op&a,const Op&b){for(int i=0;i<3;++i){for(int j=0;j<3;++j)if(a.r[i][j]!=b.r[i][j])return false;double d=std::fabs(a.t[i]-b.t[i]);if(d>1e-9&&d<1-1e-9)return false;}return true;}
inline std::vector<Op> group(){std::vector<Op> g{{{{-1,0,0},{0,-1,0},{0,0,1}},{0.75,0.25,0.5}},{{{-1,0,0},{0,1,0},{0,0,-1}},{0.25,0.5,0.75}},{{{0,0,1},{1,0,0},{0,1,0}},{0,0,0}},{{{0,1,0},{1,0,0},{0,0,-1}},{0.75,0.25,0.5}},{{{-1,0,0},{0,-1,0},{0,0,-1}},{0,0,0}},{{{1,0,0},{0,1,0},{0,0,1}},{0,0.5,0.5}},{{{1,0,0},{0,1,0},{0,0,1}},{0.5,0,0.5}},{{{1,0,0},{0,1,0},{0,0,1}},{0.5,0.5,0}}};
 std::vector<Op> G{{{{1,0,0},{0,1,0},{0,0,1}},{0,0,0}}};for(bool gr=true;gr;){gr=false;size_t n=G.size();for(size_t i=0;i<n;++i)for(auto&h:g){Op c=comp(h,G[i]);bool f=false;for(auto&e:G)if(same(e,c)){f=true;break;}if(!f){G.push_back(c);gr=true;}}}return G;}
inline std::vector<V> orbit(const std::vector<Op>&G,V p){std::vector<V> o;for(auto&op:G){V q=act(op,p);bool f=false;for(auto&e:o){double d=0;for(int i=0;i<3;++i){double x=std::fabs(e[i]-q[i]);x=std::min(x,1-x);d+=x*x;}if(d<1e-12){f=true;break;}}if(!f)o.push_back(q);}return o;}
inline void gl(int n,std::vector<double>&x,std::vector<double>&w){x.resize(n);w.resize(n);for(int i=0;i<n;++i){double z=std::cos(M_PI*(i+0.75)/(n+0.5)),pp=1;for(int it=0;it<100;++it){double p1=1,p2=0;for(int j=0;j<n;++j){double p3=p2;p2=p1;p1=((2*j+1)*z*p2-j*p3)/(j+1);}pp=n*(z*p1-p2)/(z*z-1);double z1=z;z=z1-p1/pp;if(std::fabs(z-z1)<1e-15)break;}x[i]=z;w[i]=2/((1-z*z)*pp*pp);}}
constexpr double kB=1.380649e-23, hbar=1.054571817e-34, amu=1.66053906660e-27, mH2=2.01588*amu;
struct Model{
  double A=17.31,T=250,cut=12, eO=78.20,sO=3.166, eH=34.2,sH=2.96;
  double x32=.21796, x96=.18198, z96=.37081; // Falenty et al. 2014 (ice XVI), via GenIce; origin choice 2
  double eps,sig,tail,lamO,lamHH, cutHH=12, tailHH;
  std::vector<V> O;      // Cartesian, replicated over [-1,1]^3 cells
  std::vector<V> small_c, large_c; // fractional, one cell
  void init(){
    auto G=group(); std::vector<V> f; for(V p:{V{.125,.125,.125},V{x32,x32,x32},V{x96,x96,z96}}){auto o=orbit(G,p);f.insert(f.end(),o.begin(),o.end());}
    O.clear(); int n=int(std::ceil(cut/A));
    for(auto&p:f)for(int a=-n;a<=n;++a)for(int b=-n;b<=n;++b)for(int c=-n;c<=n;++c)O.push_back(V{A*(p[0]+a),A*(p[1]+b),A*(p[2]+c)});
    small_c=orbit(G,{0,0,0}); large_c=orbit(G,{.375,.375,.375});
    eps=std::sqrt(eO*eH); sig=0.5*(sO+sH);
    double rho=136.0/(A*A*A), s3=sig*sig*sig, c3=cut*cut*cut;
    tail=16*M_PI*rho*eps*s3*(s3*s3*s3/(c3*c3*c3)/9.0 - s3/(3.0*c3));
    lamO=hbar*hbar/(24*mH2*kB*T)*1e20; lamHH=2*lamO; // reduced mass m/2 for H2-H2
    double sh3=sH*sH*sH, ch3=cutHH*cutHH*cutHH; tailHH=0; (void)sh3; (void)ch3;
  }
  // guest-framework effective potential (K) at Cartesian x, x inside the home cell [0,A)^3
  double wO(const V&x) const {double U=0,L=0,s6=std::pow(sig,6),s12=s6*s6,c2=cut*cut;
    for(auto&s:O){double dx=x[0]-s[0],dy=x[1]-s[1],dz=x[2]-s[2],r2=dx*dx+dy*dy+dz*dz; if(r2>c2) continue; double i2=1/r2,i6=i2*i2*i2;U+=4*eps*(s12*i6*i6-s6*i6);L+=4*eps*(132*s12*i6*i6*i2-30*s6*i6*i2);}
    return U+tail+lamO*L;}
  // guest-guest FH pair potential (K), minimum image, truncated (no shift) at cutHH < A/2? A/2=8.655
  double wHH(const V&a,const V&b) const {double r2=0;for(int i=0;i<3;++i){double d=a[i]-b[i];d-=A*std::round(d/A);r2+=d*d;} double rc=std::min(cutHH,0.5*A); if(r2>rc*rc) return 0;
    double s6=std::pow(sH,6),s12=s6*s6,i2=1/r2,i6=i2*i2*i2; return 4*eH*(s12*i6*i6-s6*i6)+lamHH*4*eH*(132*s12*i6*i6*i2-30*s6*i6*i2);}
  // 0 = small cage, 1 = large cage (nearest centre, periodic)
  int cage_id(const V&x) const {double best=1e99;int k=-1,n=0; for(int t=0;t<2;++t) for(auto&c:(t?large_c:small_c)){double r2=0;for(int i=0;i<3;++i){double d=x[i]-A*c[i];d-=A*std::round(d/A);r2+=d*d;} if(r2<best){best=r2;k=n;} ++n;} return k;}
  int cage(const V&x) const {double best=1e99;int k=-1; for(int t=0;t<2;++t) for(auto&c:(t?large_c:small_c)){double r2=0;for(int i=0;i<3;++i){double d=x[i]-A*c[i];d-=A*std::round(d/A);r2+=d*d;} if(r2<best){best=r2;k=t;}} return k;}
  double toMPa() const {return 1e-30/(kB*T)*1e6;}
};

// ---- parameter gradients w.r.t. the H2 site (eH, sH), same truncation as wO / wHH ----
struct Grad { double e=0, s=0; };
inline double wO_grad(const Model& M, const V& x, Grad& g){
  double U=0,L=0, Ue=0, Us=0, s6=std::pow(M.sig,6), s12=s6*s6, c2=M.cut*M.cut;
  for(auto& s:M.O){ double dx=x[0]-s[0],dy=x[1]-s[1],dz=x[2]-s[2],r2=dx*dx+dy*dy+dz*dz; if(r2>c2) continue;
    double i2=1/r2,i6=i2*i2*i2, A=s12*i6*i6, B=s6*i6;
    U+=4*M.eps*(A-B); L+=4*M.eps*(132*A-30*B)*i2;
    Us+=4*M.eps*(12*A-6*B)/M.sig + M.lamO*4*M.eps*(132*12*A-30*6*B)*i2/M.sig; }
  double rho=136.0/(M.A*M.A*M.A), c3=M.cut*M.cut*M.cut, c9=c3*c3*c3, sg=M.sig;
  double tail_s=16*M_PI*rho*M.eps*(12*std::pow(sg,11)/(9*c9) - 6*std::pow(sg,5)/(3*c3));
  double w=U+M.tail+M.lamO*L;
  g.e = w/(2*M.eH);            // eps = sqrt(eO eH): every term is linear in eps
  g.s = 0.5*(Us+tail_s);       // sig = (sO+sH)/2
  return w; }
inline double wHH_grad(const Model& M, const V& a, const V& b, Grad& g){
  double r2=0; for(int i=0;i<3;++i){ double d=a[i]-b[i]; d-=M.A*std::round(d/M.A); r2+=d*d; }
  double rc=std::min(M.cutHH,0.5*M.A); if(r2>rc*rc){ g.e=g.s=0; return 0; }
  double s6=std::pow(M.sH,6),s12=s6*s6,i2=1/r2,i6=i2*i2*i2,A=s12*i6*i6,B=s6*i6;
  double u=4*M.eH*(A-B)+M.lamHH*4*M.eH*(132*A-30*B)*i2;
  g.e=u/M.eH; g.s=(4*M.eH*(12*A-6*B)+M.lamHH*4*M.eH*(132*12*A-30*6*B)*i2)/M.sH; return u; }
