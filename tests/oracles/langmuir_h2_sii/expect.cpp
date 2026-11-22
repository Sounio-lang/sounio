#define main main_orig
#include "langmuir_oracle.cpp"
#undef main
int main(){ gl_init(8);
  std::printf("LAM=%.17Le\nPREF=%.17Le\n",hbar*hbar/(24*mH2*kB)*1e20L,4*PI*1e-30L*1e6L/kB);
  struct M{const char*n;long double e,s;} mods[]={{"spce",78.20L,3.166L},{"tip4pice",106.1L,3.1668L},{"tip5p",80.51L,3.12L}};
  struct Cg{const char*n;long double R;int z;} cages[]={{"small",3.91L,20},{"large",4.73L,28}};
  for(auto&c:cages)for(auto&m:mods){ J L=lnC(J::var(m.e,0),J::var(m.s,1),J::var(c.R,2),J::var(250,3),c.z,true,100);
    std::printf("%s_%s v=%.17Le g0=%.17Le g1=%.17Le g2=%.17Le g3=%.17Le h00=%.17Le h22=%.17Le h02=%.17Le h33=%.17Le\n",m.n,c.n,L.v,L.g[0],L.g[1],L.g[2],L.g[3],L.h[0][0],L.h[2][2],L.h[0][2],L.h[3][3]);}}
