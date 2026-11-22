#define main uq_main
#include "uq_oracle.cpp"
#undef main
// Expected values in the latent coordinates used by the Sounio witness:
// eH = 34.2 + L11 z1, sH = 2.96 + L21 z1 + L22 z2, z1, z2 ~ unit, independent.
// Channels: z1=0 z2=1 eO=2 sO=3 R=4 T=5 (water, R, T carry zero uncertainty).
int main(){ gl8();
  long double He[4]={34.2L,33.3L,38.0L,37.0L}, Hs[4]={2.96L,2.968L,2.915L,2.928L};
  long double me=0,ms=0; for(int i=0;i<4;i++){me+=He[i];ms+=Hs[i];} me/=4; ms/=4;
  long double cee=0,css=0,ces=0; for(int i=0;i<4;i++){cee+=(He[i]-me)*(He[i]-me);css+=(Hs[i]-ms)*(Hs[i]-ms);ces+=(He[i]-me)*(Hs[i]-ms);} cee/=3;css/=3;ces/=3;
  long double L11=sqrtl(cee), L21=ces/L11, L22=sqrtl(css-L21*L21);
  std::printf("L11=%.17Le\nL21=%.17Le\nL22=%.17Le\n",L11,L21,L22);
  struct M{const char*n;long double e,s;} mods[]={{"spce",78.20L,3.166L},{"tip4pice",106.1L,3.1668L},{"tip5p",80.51L,3.12L}};
  struct Cg{const char*n;long double R;int z;} cages[]={{"small",3.91L,20},{"large",4.73L,28}};
  for(auto&c:cages){ J Ls[3];
    for(int m=0;m<3;m++){
      // jets directly in latent coordinates: var 0 = z1, var 1 = z2
      J z1=J::var(0,0), z2=J::var(0,1);
      J eH=J(34.2L)+L11*z1, sH=J(2.96L)+L21*z1+L22*z2;
      J L=lnC(eH,sH,J::var(mods[m].e,2),J::var(mods[m].s,3),J::var(c.R,4),J::var(250,5),c.z); Ls[m]=L;
      long double var=L.g[0]*L.g[0]+L.g[1]*L.g[1];
      long double som=L.v+0.5L*(L.h[0][0]+L.h[1][1]);
      std::printf("%s_%s v=%.17Le var=%.17Le som=%.17Le h50=%.17Le h51=%.17Le h40=%.17Le h41=%.17Le\n",mods[m].n,c.n,L.v,var,som,L.h[5][0],L.h[5][1],L.h[4][0],L.h[4][1]);
    }
    for(int m=1;m<3;m++){ long double a=Ls[m].g[0]-Ls[0].g[0], b=Ls[m].g[1]-Ls[0].g[1];
      std::printf("diff_%s_spce_%s d=%.17Le var=%.17Le\n",mods[m].n,c.n,Ls[m].v-Ls[0].v,a*a+b*b);}
  }
}
