// Oracle for tests/run-pass/langmuir_h2_sii_lattice_uq.sio.
// Explicit sII lattice (Fd-3m origin 2, Falenty 2014 coordinates via sII.hpp), one-site H2
// (Buch), Lorentz-Berthelot, quadratic Feynman-Hibbs, point-centred 12 A cutoff + uniform tail,
// Voronoi domain of one cage. A deliberately small product rule (10 radial panels x 4 GL,
// 8 GL in cos(theta), 16 midpoint in phi) so the witness runs in seconds; this oracle evaluates
// exactly that discrete sum in long double, with its exact gradient in (eH, sH).
//   mode "sites <cage>" : prints the Sounio array literals (site offsets, cage-centre offsets)
//   mode "expect"       : prints ln C and d ln C/d z1, d ln C/d z2 for the cases of the witness
#include "sII.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
using LD = long double;
struct Case { const char* name; double eO, sO, T; int cage; };
static const double CH_A=2.23364425696364341, CH_B=-2.52912550826065453e-2, CH_D=1.12653584638716957e-3;
static std::vector<V> sites_near(const Model& M, const V& c, double R){ std::vector<V> s; for(auto& o: M.O){ V d{o[0]-c[0],o[1]-c[1],o[2]-c[2]}; double r=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]); if(r<R) s.push_back(d);} return s; }
static std::vector<V> centres_rel(const Model& M, const V& c){ std::vector<V> s; for(int t=0;t<2;++t) for(auto& q:(t?M.large_c:M.small_c)) for(int a=-1;a<=1;++a)for(int b=-1;b<=1;++b)for(int e=-1;e<=1;++e){ V d{M.A*(q[0]+a)-c[0],M.A*(q[1]+b)-c[1],M.A*(q[2]+e)-c[2]}; double r2=d[0]*d[0]+d[1]*d[1]+d[2]*d[2]; if(r2>1e-9 && r2<14.0*14.0) s.push_back(d);} return s; }
static void rule(std::vector<LD>& gx, std::vector<LD>& gw, int n){ std::vector<double> x,w; gl(n,x,w); gx.assign(x.begin(),x.end()); gw.assign(w.begin(),w.end()); }
// returns ln C (C in MPa^-1) and gradient wrt (eH, sH)
static void lnC(const Case& K, double eH, double sH, LD& lc, LD& ge, LD& gs){
  Model M; M.T=K.T; M.eO=K.eO; M.sO=K.sO; M.eH=eH; M.sH=sH; M.init();
  V c=K.cage?M.large_c[0]:M.small_c[0]; for(auto& v:c) v*=M.A; double rmax=K.cage?4.6:3.2;
  auto S=sites_near(M,c,M.cut+rmax+0.1); auto Cn=centres_rel(M,c);
  LD eps=std::sqrt((LD)K.eO*eH), sig=0.5L*((LD)K.sO+sH), s6=std::pow(sig,6), s12=s6*s6, lam=(LD)M.lamO, c2=(LD)M.cut*M.cut;
  LD rho=136.0L/((LD)M.A*M.A*M.A), c3=(LD)M.cut*M.cut*M.cut, c9=c3*c3*c3;
  LD tail=16*M_PI*rho*eps*(s12/(9*c9)-s6/(3*c3)), tail_s=16*M_PI*rho*eps*(12*s12/sig/(9*c9)-6*s6/sig/(3*c3));
  std::vector<LD> rx,rw,tx,tw; rule(rx,rw,4); rule(tx,tw,8); const int np=10, nph=16;
  LD I=0, Ie=0, Is=0;
  for(int p=0;p<np;++p) for(int i=0;i<4;++i){ LD r=((LD)p+0.5L+0.5L*rx[i])*rmax/np, wr=0.5L*rw[i]*rmax/np;
    for(int j=0;j<8;++j){ LD ct=tx[j], st=std::sqrt(1-ct*ct); for(int k=0;k<nph;++k){ LD ph=2*M_PI*((LD)k+0.5L)/nph;
      LD x0=r*st*std::cos(ph), x1=r*st*std::sin(ph), x2=r*ct, d0=x0*x0+x1*x1+x2*x2; bool own=true;
      for(auto& q:Cn){ LD a=x0-q[0],b=x1-q[1],e=x2-q[2]; if(a*a+b*b+e*e<d0){ own=false; break; } } if(!own) continue;
      LD U=0,L=0,Us=0; for(auto& s:S){ LD a=x0-s[0],b=x1-s[1],e=x2-s[2],r2=a*a+b*b+e*e; if(r2>c2) continue; LD i2=1/r2,i6=i2*i2*i2,A=s12*i6*i6,B=s6*i6;
        U+=4*eps*(A-B); L+=4*eps*(132*A-30*B)*i2; Us+=4*eps*(12*A-6*B)/sig+lam*4*eps*(132*12*A-30*6*B)*i2/sig; }
      LD w=U+tail+lam*L, we=w/(2*eH), ws=0.5L*(Us+tail_s), f=std::exp(-w/(LD)K.T), q=wr*tw[j]*(2*M_PI/nph)*r*r*f;
      I+=q; Ie+=q*(-we/(LD)K.T); Is+=q*(-ws/(LD)K.T); } } }
  LD toMPa=1e-30L/((LD)kB*K.T)*1e6L; lc=std::log(I*toMPa); ge=Ie/I; gs=Is/I; }
int main(int argc,char**argv){
  if(argc>2 && !std::strcmp(argv[1],"sites")){ Model M; M.init(); int cg=std::atoi(argv[2]); V c=cg?M.large_c[0]:M.small_c[0]; for(auto& v:c) v*=M.A; double rmax=cg?4.6:3.2;
    auto S=sites_near(M,c,M.cut+rmax+0.1); auto Cn=centres_rel(M,c);
    std::printf("// %zu sites, %zu neighbouring cage centres\n",S.size(),Cn.size());
    auto dump=[](const char* nm,const std::vector<V>& v){ std::printf("fn %s(i: i64) -> f64 with Mut, Div {\n    let t = [",nm); for(size_t i=0;i<v.size();++i) for(int a=0;a<3;++a) std::printf("%s%.17g",(i||a)?", ":"",v[i][a]); std::printf("]\n    t[i as usize]\n}\n\n"); };
    dump(cg?"site_l":"site_s",S); dump(cg?"cen_l":"cen_s",Cn); return 0; }
  Case cases[]={{"spce_small_200",78.20,3.166,200,0},{"tip4pice_small_200",106.1,3.1668,200,0},{"spce_large_200",78.20,3.166,200,1},{"tip4pice_large_200",106.1,3.1668,200,1}};
  LD g[4][2], v[4];
  for(int n=0;n<4;++n){ LD lc,ge,gs; lnC(cases[n],34.2,2.96,lc,ge,gs); v[n]=lc; g[n][0]=ge*CH_A+gs*CH_B; g[n][1]=gs*CH_D;
    std::printf("%s lnC %.17Lg dz1 %.17Lg dz2 %.17Lg var %.17Lg\n",cases[n].name,lc,g[n][0],g[n][1],g[n][0]*g[n][0]+g[n][1]*g[n][1]); }
  for(int t=0;t<2;++t){ int a=2*t+1,b=2*t; LD d1=g[a][0]-g[b][0], d2=g[a][1]-g[b][1]; std::printf("diff_%s value %.17Lg var %.17Lg (indep %.17Lg)\n",t?"large":"small",v[a]-v[b],d1*d1+d2*d2, g[a][0]*g[a][0]+g[a][1]*g[a][1]+g[b][0]*g[b][0]+g[b][1]*g[b][1]); }
  // finite-difference check of the gradient (central, relative step 1e-6) on the first case
  { LD lp,lm,x,y; lnC(cases[0],34.2*(1+1e-6),2.96,lp,x,y); lnC(cases[0],34.2*(1-1e-6),2.96,lm,x,y); LD fe=(lp-lm)/(2*34.2e-6);
    lnC(cases[0],34.2,2.96*(1+1e-6),lp,x,y); lnC(cases[0],34.2,2.96*(1-1e-6),lm,x,y); LD fs=(lp-lm)/(2*2.96e-6); LD c,ge,gs; lnC(cases[0],34.2,2.96,c,ge,gs);
    std::printf("fdcheck dlnC/deH %.12Lg fd %.12Lg ; dlnC/dsH %.12Lg fd %.12Lg\n",ge,fe,gs,fs); }
}
