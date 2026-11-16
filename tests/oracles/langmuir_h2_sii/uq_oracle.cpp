// UQ oracle: ln C of H2 in sII cages (LJD cell, Lorentz-Berthelot, quadratic FH)
// with jets over (epsH, sigH, epsO, sigO, R, T). Exact first/second derivatives;
// GUM first order, second-order mean shift, and differences between water models
// under the SHARED H2-parameter covariance.
#include "jet.hpp"
#include <cstdio>
#include <vector>
#include <cmath>
constexpr int NV=6; using J=Jet<NV>;
constexpr long double kB=1.380649e-23L, hbar=1.054571817e-34L, amu=1.66053906660e-27L, PI=3.14159265358979323846264338327950288L, Rg=8.314462618e-3L;
constexpr long double mH2=2.01588L*amu;
static long double gx[8],gw[8];
static void gl8(){int n=8;for(int i=0;i<n;i++){long double z=cosl(PI*(i+0.75L)/(n+0.5L)),pp;for(int it=0;it<100;it++){long double p1=1,p2=0;for(int j=1;j<=n;j++){long double p3=p2;p2=p1;p1=((2*j-1)*z*p2-(j-1)*p3)/j;}pp=n*(z*p1-p2)/(z*z-1);long double z1=z;z=z1-p1/pp;if(fabsl(z-z1)<1e-21L)break;}gx[i]=z;gw[i]=2/((1-z*z)*pp*pp);}}
static J dk(const J&x,long double k){return jpow(J(1)-x,-k)-jpow(J(1)+x,-k);}
static J lnC(J eH,J sH,J eO,J sO,J R,J T,int z){
  J eps=jsqrt(eH*eO), sig=0.5L*(sH+sO);
  J A=jpow(sig,12)/jpow(R,11), B=jpow(sig,6)/jpow(R,5);
  J lam=J(hbar*hbar/(24*mH2*kB)*1e20L)/T;
  J integ(0); const int P=100; long double h=0.97L/P;
  for(int p=0;p<P;p++)for(int q=0;q<8;q++){long double u=p*h+0.5L*h*(gx[q]+1),wq=0.5L*h*gw[q]; J x(u),r=R*x;
    J w=2.0L*z*eps*(A*dk(x,10)/(10.0L*r)-B*dk(x,4)/(4.0L*r));
    J lap=2.0L*z*eps/r*(A*(11.0L/(R*R))*dk(x,12)-B*(5.0L/(R*R))*dk(x,6));
    integ=integ+J(wq*u*u)*jexp(J(0)-(w+lam*lap)/T);}
  return jlog(J(4*PI*1e-30L*1e6L/kB)*jpow(R,3)/T*integ);
}
int main(){ gl8();
  // One-site H2 LJ sets (eps/k K, sigma A): Buch; Hirschfelder viscosity (a), (b); 2nd virial (c).
  long double He[4]={34.2L,33.3L,38.0L,37.0L}, Hs[4]={2.96L,2.968L,2.915L,2.928L};
  long double me=0,ms=0; for(int i=0;i<4;i++){me+=He[i];ms+=Hs[i];} me/=4; ms/=4;
  long double cee=0,css=0,ces=0; for(int i=0;i<4;i++){cee+=(He[i]-me)*(He[i]-me);css+=(Hs[i]-ms)*(Hs[i]-ms);ces+=(He[i]-me)*(Hs[i]-ms);} cee/=3;css/=3;ces/=3;
  std::printf("# H2 sets: mean eps=%.6Lf sig=%.6Lf  u_eps=%.6Lf u_sig=%.6Lf rho=%.6Lf (centre = Buch 34.2/2.96)\n",me,ms,sqrtl(cee),sqrtl(css),ces/sqrtl(cee*css));
  // Cholesky of H2 covariance -> latent factors z1,z2 (unit, independent)
  long double L11=sqrtl(cee), L21=ces/L11, L22=sqrtl(css-L21*L21);
  std::printf("# chol L11=%.17Le L21=%.17Le L22=%.17Le\n",L11,L21,L22);
  struct M{const char*n;long double e,s;} mods[]={{"SPC/E",78.20L,3.166L},{"TIP4P/Ice",106.1L,3.1668L},{"TIP5P",80.51L,3.12L}};
  struct Cg{const char*n;long double R;int z;} cages[]={{"5^12",3.91L,20},{"5^12 6^4",4.73L,28}};
  const long double T0=250; const long double rw[3]={0.01L,0.02L,0.05L};
  for(auto&c:cages){
    J Ls[3];
    for(int m=0;m<3;m++) Ls[m]=lnC(J::var(34.2L,0),J::var(2.96L,1),J::var(mods[m].e,2),J::var(mods[m].s,3),J::var(c.R,4),J::var(T0,5),c.z);
    std::printf("\n## cage %s, T=%.0Lf K\n",c.n,T0);
    for(int m=0;m<3;m++){ J&L=Ls[m];
      // first-order variances from H2 (shared covariance)
      auto qH=[&](long double a0,long double a1){return a0*a0*cee+a1*a1*css+2*a0*a1*ces;};
      long double vH_lnC=qH(L.g[0],L.g[1]);
      long double vH_Q=qH(L.h[5][0],L.h[5][1])*powl(Rg*T0*T0,2);   // Q = R T^2 dlnC/dT
      long double vH_dR=qH(L.h[4][0],L.h[4][1]);
      long double Q=Rg*T0*T0*L.g[5];
      // second-order mean shift of ln C and of C from H2
      long double sh_ln=0.5L*(L.h[0][0]*cee+L.h[1][1]*css+2*L.h[0][1]*ces);
      long double sh_C=0.5L*((L.h[0][0]+L.g[0]*L.g[0])*cee+(L.h[1][1]+L.g[1]*L.g[1])*css+2*(L.h[0][1]+L.g[0]*L.g[1])*ces);
      std::printf("%-9s lnC=%.10Lf u_H(lnC)=%.6Lf | Q=%.5Lf u_H(Q)=%.5Lf kJ/mol | dlnC/dR=%.6Lf u_H=%.6Lf | E[lnC]-lnC=%.3Le  E[C]/C-1=%.3Le\n",
        mods[m].n,L.v,sqrtl(vH_lnC),Q,sqrtl(vH_Q),L.g[4],sqrtl(vH_dR),sh_ln,sh_C);
      for(long double r: rw){ long double ue=r*mods[m].e, us=r*mods[m].s;
        long double vW_lnC=L.g[2]*L.g[2]*ue*ue+L.g[3]*L.g[3]*us*us;
        long double vW_Q=(L.h[5][2]*L.h[5][2]*ue*ue+L.h[5][3]*L.h[5][3]*us*us)*powl(Rg*T0*T0,2);
        long double vW_dR=L.h[4][2]*L.h[4][2]*ue*ue+L.h[4][3]*L.h[4][3]*us*us;
        std::printf("    water %2.0Lf%%: u_W(lnC)=%.6Lf u_W(Q)=%.5Lf u_W(dlnC/dR)=%.6Lf\n",100*r,sqrtl(vW_lnC),sqrtl(vW_Q),sqrtl(vW_dR)); }
    }
    // differences vs SPC/E under shared H2 + independent water uncertainty
    for(int m=1;m<3;m++){ J&A=Ls[m]; J&B=Ls[0];
      auto qH=[&](long double a0,long double a1){return a0*a0*cee+a1*a1*css+2*a0*a1*ces;};
      long double d_ln=A.v-B.v, uH_ln=sqrtl(qH(A.g[0]-B.g[0],A.g[1]-B.g[1]));
      long double s=Rg*T0*T0;
      long double d_Q=s*(A.g[5]-B.g[5]), uH_Q=s*sqrtl(qH(A.h[5][0]-B.h[5][0],A.h[5][1]-B.h[5][1]));
      long double d_dR=A.g[4]-B.g[4], uH_dR=sqrtl(qH(A.h[4][0]-B.h[4][0],A.h[4][1]-B.h[4][1]));
      std::printf("  %s - SPC/E: dlnC=%.5Lf u_H=%.6Lf (no-cancel %.6Lf) | dQ=%.5Lf u_H=%.6Lf | d(dlnC/dR)=%.5Lf u_H=%.6Lf\n",mods[m].n,d_ln,uH_ln,sqrtl(qH(A.g[0],A.g[1])+qH(B.g[0],B.g[1])),d_Q,uH_Q,d_dR,uH_dR);
      for(long double r: rw){ auto vW=[&](J&L,int m2,int row){long double ue=r*mods[m2].e,us=r*mods[m2].s; if(row<0) return L.g[2]*L.g[2]*ue*ue+L.g[3]*L.g[3]*us*us; return L.h[row][2]*L.h[row][2]*ue*ue+L.h[row][3]*L.h[row][3]*us*us;};
        long double u_ln=sqrtl(uH_ln*uH_ln+vW(A,m,-1)+vW(B,0,-1));
        long double u_Q=sqrtl(uH_Q*uH_Q+s*s*(vW(A,m,5)+vW(B,0,5)));
        long double u_dR=sqrtl(uH_dR*uH_dR+vW(A,m,4)+vW(B,0,4));
        std::printf("      water %2.0Lf%%: z(dlnC)=%.2Lf z(dQ)=%.2Lf z(d dlnC/dR)=%.2Lf\n",100*r,d_ln/u_ln,d_Q/u_Q,d_dR/u_dR);}
    }
  }
}
