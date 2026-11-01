// C++23 oracle for tests/stdlib/darwin/test_atlas_fixed_buffer.sio: the dihedral
// symmetry statistics and d_min of stdlib/darwin/atlas.sio, by brute force over
// std::vector sequences (A=0 C=1 G=2 T=3).
// Build: g++ -std=c++23 -O2 darwin_atlas.cpp && ./a.out
#include <cstdio>
#include <vector>
#include <cmath>
using S=std::vector<int>;
S shift(const S&s,int k){int n=s.size();S o(n);for(int i=0;i<n;i++)o[i]=s[((i+k)%n+n)%n];return o;}
S rev(const S&s){return S(s.rbegin(),s.rend());}
S comp(const S&s){S o=s;for(int&b:o)b^=3;return o;}
int ham(const S&a,const S&b){int c=0;for(size_t i=0;i<a.size();i++)c+=a[i]!=b[i];return c;}
int period(const S&s){int n=s.size();for(int k=1;k<=n;k++)if(n%k==0&&shift(s,k)==s)return k;return n;}
bool refl(const S&s){S r=rev(s);for(int k=0;k<(int)s.size();k++)if(shift(r,k)==s)return true;return false;}
void report(const char*name,S s,bool rc){int n=s.size();int rs=n/period(s);int orbit=2*n/(rs*(refl(s)?2:1));
 int best=n,kind=0,bk=1;for(int k=1;k<n;k++){int d=ham(s,shift(s,k));if(d<best){best=d;kind=0;bk=k;}}
 S r=rev(s);for(int k=0;k<n;k++){int d=ham(s,shift(r,k));if(d<best){best=d;kind=1;bk=k;}}
 if(rc){S c=comp(r);for(int k=0;k<n;k++){int d=ham(s,shift(c,k));if(d<best){best=d;kind=2;bk=k;}}}
 printf("%s: len %d period %d rotsym %d reflective %d orbit %d orbit_ratio %.17g palindrome %d rcfixed %d dmin(rc=%d) %d kind %d k %d\n",
  name,n,period(s),rs,refl(s),orbit,orbit/(2.0*n),rev(s)==s,comp(rev(s))==s,rc,best,kind,bk);}
int main(){report("ACGTACGT",{0,1,2,3,0,1,2,3},true);report("ACGTACGT",{0,1,2,3,0,1,2,3},false);
 report("AACGTT",{0,0,1,2,3,3},false);report("AAGC",{0,0,2,1},true);report("AAGC",{0,0,2,1},false);}
