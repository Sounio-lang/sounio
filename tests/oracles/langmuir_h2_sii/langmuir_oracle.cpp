// Oracle: Langmuir constant of H2 in sII cages, Lennard-Jones–Devonshire (LJD) spherical cell,
// single-site LJ guest (Buch H2), O-site LJ of each water model, Lorentz–Berthelot,
// optional quadratic Feynman–Hibbs. Exact 1st/2nd derivatives via jets over
// (eps_O [K], sigma_O [A], R [A], T [K]).
#include "jet.hpp"
#include <cstdio>
#include <vector>
constexpr int NV=4;
using J=Jet<NV>;
constexpr long double kB=1.380649e-23L, hbar=1.054571817e-34L, amu=1.66053906660e-27L, PI=3.14159265358979323846264338327950288L;
constexpr long double mH2=2.01588L*amu, epsH=34.2L, sigH=2.96L;   // Buch H2 (eps/k K, sigma A)
// Gauss-Legendre 32 nodes on [-1,1]
static std::vector<long double> gx,gw;
static void gl_init(int n){gx.resize(n);gw.resize(n);for(int i=0;i<n;i++){long double z=cosl(PI*(i+0.75L)/(n+0.5L)),pp;for(int it=0;it<100;it++){long double p1=1,p2=0;for(int j=1;j<=n;j++){long double p3=p2;p2=p1;p1=((2*j-1)*z*p2-(j-1)*p3)/j;}pp=n*(z*p1-p2)/(z*z-1);long double z1=z;z=z1-p1/pp;if(fabsl(z-z1)<1e-21L)break;}gx[i]=z;gw[i]=2/((1-z*z)*pp*pp);}}
// (1-x)^-k - (1+x)^-k  as jet in x
static J dk(const J&x,long double k){return jpow(J(1)-x,-k)-jpow(J(1)+x,-k);}
struct Out{J lnC;};
// returns ln C  (C in 1/MPa)
static J lnC(J epsO,J sigO,J R,J T,int z,bool fh,int panels){
  J eps=jsqrt(J(epsH)*epsO);                 // K
  J sig=0.5L*(J(sigH)+sigO);                 // A
  J A=jpow(sig,12)/jpow(R,11), B=jpow(sig,6)/jpow(R,5);
  J lam= fh ? J(hbar*hbar/(24*mH2*kB)*1e20L)/T : J(0); // A^2 * K  -> lam*lap(w[K])/? ; lam/T has units A^2
  // w(r)/T, r=R u.  w(r)=2 z eps [A d10/r /10 - B d4/r /4]  (d_N=(1-x)^-N-(1+x)^-N)
  J integ(0);
  const long double umax=0.97L; long double h=umax/panels;
  for(int p=0;p<panels;p++) for(size_t q=0;q<gx.size();q++){
    long double u=p*h+0.5L*h*(gx[q]+1), wq=0.5L*h*gw[q];
    J x(u); J r=R*x;
    J w=2.0L*z*eps*(A*dk(x,10)/(10.0L*r)-B*dk(x,4)/(4.0L*r));
    if(fh){ // lap w = (1/r) d2/dr2 [r w];  r w = 2 z eps [A d10/10 - B d4/4];  d2/dr2 dN = N(N+1)/R^2 [(1-x)^-(N+2)-(1+x)^-(N+2)]
      J lap=2.0L*z*eps/r*(A*(11.0L/(R*R))*dk(x,12)-B*(5.0L/(R*R))*dk(x,6));
      w=w+lam*lap;
    }
    integ=integ+J(wq*u*u)*jexp(J(0)-w/T);
  }
  // C = 4 pi R^3 /(kB T) * integ ; R in A -> m^3 1e-30 ; 1/Pa -> 1/MPa *1e6
  J C=J(4*PI*1e-30L*1e6L/kB)*jpow(R,3)/T*integ;
  return jlog(C);
}
int main_tab(){
  gl_init(32);
  struct M{const char*n;long double e,s;} mods[]={{"SPC/E",78.20L,3.166L},{"TIP4P/Ice",106.1L,3.1668L},{"TIP5P",80.51L,3.12L}};
  struct Cg{const char*n;long double R;int z;} cages[]={{"5^12",3.91L,20},{"5^12 6^4",4.73L,28}};
  long double Ts[]={200,250,274};
  // convergence check
  for(int fh=0;fh<2;fh++){J a=lnC(J::var(78.2L,0),J::var(3.166L,1),J::var(3.91L,2),J::var(250,3),20,fh,200),b=lnC(J::var(78.2L,0),J::var(3.166L,1),J::var(3.91L,2),J::var(250,3),20,fh,400);
    std::printf("# conv fh=%d dlnC=%.3Le dg3=%.3Le dh00=%.3Le\n",fh,a.v-b.v,a.g[3]-b.g[3],a.h[0][0]-b.h[0][0]);}
  std::printf("# model cage T FH  C[1/MPa]  dlnC/depsO[1/K] dlnC/dsigO[1/A] dlnC/dR[1/A] dlnC/dT[1/K]  Q=-R*dlnC/d(1/T)=R T^2 dlnC/dT [kJ/mol]  d2lnC/depsO2 d2lnC/dR2\n");
  for(auto&c:cages)for(long double T:Ts)for(int fh=0;fh<2;fh++)for(auto&m:mods){
    J L=lnC(J::var(m.e,0),J::var(m.s,1),J::var(c.R,2),J::var(T,3),c.z,fh,400);
    long double Q=8.314462618e-3L*T*T*L.g[3];
    std::printf("%-9s %-8s %3.0Lf %d  %.6Le  %.6Le %.6Le %.6Le %.6Le  %.5Lf  %.6Le %.6Le\n",m.n,c.n,T,fh,expl(L.v),L.g[0],L.g[1],L.g[2],L.g[3],Q,L.h[0][0],L.h[2][2]);
  }
  return 0;
}
int main_fd(){ gl_init(32);
  long double p0[4]={106.1L,3.1668L,3.91L,250};
  auto f=[&](long double*p){return lnC(J(p[0]),J(p[1]),J(p[2]),J(p[3]),20,true,400).v;};
  J L=lnC(J::var(p0[0],0),J::var(p0[1],1),J::var(p0[2],2),J::var(p0[3],3),20,true,400);
  long double hs[4]={1e-3L,1e-5L,1e-5L,1e-3L};
  for(int i=0;i<4;i++){long double a[4],b[4];for(int k=0;k<4;k++){a[k]=b[k]=p0[k];}a[i]+=hs[i];b[i]-=hs[i];
    long double fd=(f(a)-f(b))/(2*hs[i]); long double fd2=(f(a)-2*f(p0)+f(b))/(hs[i]*hs[i]);
    std::printf("i=%d jet g=%.12Le fd=%.12Le | jet h=%.8Le fd=%.8Le\n",i,L.g[i],fd,L.h[i][i],fd2);}
  {long double a[4],b[4],c[4],d[4];for(int k=0;k<4;k++)a[k]=b[k]=c[k]=d[k]=p0[k];a[0]+=hs[0];a[2]+=hs[2];b[0]+=hs[0];b[2]-=hs[2];c[0]-=hs[0];c[2]+=hs[2];d[0]-=hs[0];d[2]-=hs[2];
   std::printf("cross h02 jet=%.8Le fd=%.8Le\n",L.h[0][2],(f(a)-f(b)-f(c)+f(d))/(4*hs[0]*hs[2]));}
  return main_tab()*0;}
