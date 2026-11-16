#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
static uint64_t b(double d){uint64_t u;memcpy(&u,&d,8);return u;}
static double f(uint64_t u){double d;memcpy(&d,&u,8);return d;}
int main(){
  double mx=f(0x40862E42FEFA39EFull), mn=f(0xC0874910D52D3052ull);
  double xs[]={0.0,1.0,-1.0,10.0,-700.0,700.0,709.0,709.43,709.5,709.7,mx,nextafter(mx,1e9),1000.0,1e19,
    -708.3,-708.39,-708.4,-710.0,-720.0,-730.5,-740.0,-744.0,-744.44,-745.0,-745.1,nextafter(mn,0),mn,nextafter(mn,-1e9),-800.0,-1e6,-1e19,
    f(0x7FF0000000000000ull),f(0xFFF0000000000000ull)};
  int n=sizeof(xs)/sizeof(xs[0]);
  std::printf("    let xs: [i64; %d] = [",n); for(int i=0;i<n;i++) std::printf("%s%lld",i?", ":"",(long long)b(xs[i])); std::printf("]\n");
  std::printf("    let want: [i64; %d] = [",n); for(int i=0;i<n;i++) std::printf("%s%lld",i?", ":"",(long long)b((double)expl((long double)xs[i]))); std::printf("]\n");
  std::printf("    let n: i64 = %d\n",n);
}
