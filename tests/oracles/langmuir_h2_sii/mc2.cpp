// Monte Carlo check of the first-order GUM (plain long double, same LJD/FH model,
// same 8-point x 100-panel quadrature as uq_oracle.cpp and the Sounio witness).
#include <cstdio>
#include <cmath>
#include <random>
typedef long double R_;
constexpr R_ kB=1.380649e-23L, hbar=1.054571817e-34L, amu=1.66053906660e-27L, PI=3.14159265358979323846264338327950288L, mH2=2.01588L*amu;
static R_ gx[8],gw[8];
static void gl8(){int n=8;for(int i=0;i<n;i++){R_ z=cosl(PI*(i+0.75L)/(n+0.5L)),pp;for(int it=0;it<100;it++){R_ p1=1,p2=0;for(int j=1;j<=n;j++){R_ p3=p2;p2=p1;p1=((2*j-1)*z*p2-(j-1)*p3)/j;}pp=n*(z*p1-p2)/(z*z-1);R_ z1=z;z=z1-p1/pp;if(fabsl(z-z1)<1e-21L)break;}gx[i]=z;gw[i]=2/((1-z*z)*pp*pp);}}
static R_ dk(R_ x,int k){return powl(1-x,-k)-powl(1+x,-k);}
static R_ lnC(R_ eH,R_ sH,R_ eO,R_ sO,R_ R,R_ T,int z){
  R_ eps=sqrtl(eH*eO), sig=0.5L*(sH+sO), A=powl(sig,12)/powl(R,11), B=powl(sig,6)/powl(R,5), lam=hbar*hbar/(24*mH2*kB)*1e20L/T, in=0, h=0.97L/100;
  for(int p=0;p<100;p++)for(int q=0;q<8;q++){R_ u=p*h+0.5L*h*(gx[q]+1),wq=0.5L*h*gw[q],r=R*u;
    R_ w=2.0L*z*eps*(A*dk(u,10)/(10*r)-B*dk(u,4)/(4*r)), lap=2.0L*z*eps/r*(A*11/(R*R)*dk(u,12)-B*5/(R*R)*dk(u,6));
    in+=wq*u*u*expl(-(w+lam*lap)/T);}
  return logl(4*PI*1e-30L*1e6L/kB*R*R*R/T*in);}
int main(){ gl8();
  R_ He[4]={34.2L,33.3L,38.0L,37.0L}, Hs[4]={2.96L,2.968L,2.915L,2.928L}, me=0,ms=0; for(int i=0;i<4;i++){me+=He[i];ms+=Hs[i];} me/=4; ms/=4;
  R_ cee=0,css=0,ces=0; for(int i=0;i<4;i++){cee+=(He[i]-me)*(He[i]-me);css+=(Hs[i]-ms)*(Hs[i]-ms);ces+=(He[i]-me)*(Hs[i]-ms);} cee/=3;css/=3;ces/=3;
  R_ L11=sqrtl(cee), L21=ces/L11, L22=sqrtl(css-L21*L21);
  R_ E0[3]={78.20L,106.1L,80.51L}, S0[3]={3.166L,3.1668L,3.12L}; const char*nm[3]={"SPC/E","TIP4P/Ice","TIP5P"};
  struct Cg{const char*n;R_ R;int z;} cages[]={{"5^12",3.91L,20},{"5^12 6^4",4.73L,28}};
  for(int water=0;water<2;water++) for(auto&c:cages){
    std::mt19937_64 g(20261115); std::normal_distribution<R_> N(0,1); const int NS=40000;
    R_ s1[3]={0},s2[3]={0},d1[2]={0},d2[2]={0},base[3]; for(int m=0;m<3;m++) base[m]=lnC(34.2L,2.96L,E0[m],S0[m],c.R,250,c.z);
    for(int k=0;k<NS;k++){R_ z1=N(g),z2=N(g),eH=34.2L+L11*z1,sH=2.96L+L21*z1+L22*z2,v[3];
      for(int m=0;m<3;m++){R_ fe=1,fs=1; if(water){fe=1+0.02L*N(g);fs=1+0.02L*N(g);} v[m]=lnC(eH,sH,E0[m]*fe,S0[m]*fs,c.R,250,c.z); s1[m]+=v[m]; s2[m]+=v[m]*v[m];}
      for(int m=1;m<3;m++){R_ d=v[m]-v[0]; d1[m-1]+=d; d2[m-1]+=d*d;}}
    std::printf("## %s, water %s, N=%d\n",c.n,water?"+-2%":"exact",NS);
    for(int m=0;m<3;m++){R_ mu=s1[m]/NS, sd=sqrtl(s2[m]/NS-mu*mu); std::printf("  %-9s MC E[lnC]-lnC0=%+.4Le (se %.1Le) sd=%.6Lf\n",nm[m],mu-base[m],sd/sqrtl(NS),sd);}
    for(int m=1;m<3;m++){R_ mu=d1[m-1]/NS, sd=sqrtl(d2[m-1]/NS-mu*mu); std::printf("  %s-SPC/E MC mean=%.5Lf sd=%.6Lf\n",nm[m],mu,sd);}
    fflush(stdout);}
}
