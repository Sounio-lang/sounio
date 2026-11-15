// Second-order forward jet in NV variables, long double. Independent of Sounio.
#pragma once
#include <array>
#include <cmath>
template<int NV> struct Jet {
  long double v; std::array<long double,NV> g{}; std::array<std::array<long double,NV>,NV> h{};
  Jet(long double x=0):v(x){}
  static Jet var(long double x,int i){Jet j(x); j.g[i]=1; return j;}
};
template<int N> Jet<N> operator+(const Jet<N>&a,const Jet<N>&b){Jet<N> r(a.v+b.v);for(int i=0;i<N;i++){r.g[i]=a.g[i]+b.g[i];for(int k=0;k<N;k++)r.h[i][k]=a.h[i][k]+b.h[i][k];}return r;}
template<int N> Jet<N> operator-(const Jet<N>&a,const Jet<N>&b){Jet<N> r(a.v-b.v);for(int i=0;i<N;i++){r.g[i]=a.g[i]-b.g[i];for(int k=0;k<N;k++)r.h[i][k]=a.h[i][k]-b.h[i][k];}return r;}
template<int N> Jet<N> operator*(const Jet<N>&a,const Jet<N>&b){Jet<N> r(a.v*b.v);for(int i=0;i<N;i++){r.g[i]=a.g[i]*b.v+a.v*b.g[i];for(int k=0;k<N;k++)r.h[i][k]=a.h[i][k]*b.v+a.v*b.h[i][k]+a.g[i]*b.g[k]+a.g[k]*b.g[i];}return r;}
// unary chain rule: f(a) with f0,f1,f2
template<int N> Jet<N> chain(const Jet<N>&a,long double f0,long double f1,long double f2){Jet<N> r(f0);for(int i=0;i<N;i++){r.g[i]=f1*a.g[i];for(int k=0;k<N;k++)r.h[i][k]=f1*a.h[i][k]+f2*a.g[i]*a.g[k];}return r;}
template<int N> Jet<N> inv(const Jet<N>&a){long double x=a.v;return chain(a,1/x,-1/(x*x),2/(x*x*x));}
template<int N> Jet<N> operator/(const Jet<N>&a,const Jet<N>&b){return a*inv(b);}
template<int N> Jet<N> jexp(const Jet<N>&a){long double e=expl(a.v);return chain(a,e,e,e);}
template<int N> Jet<N> jlog(const Jet<N>&a){long double x=a.v;return chain(a,logl(x),1/x,-1/(x*x));}
template<int N> Jet<N> jsqrt(const Jet<N>&a){long double s=sqrtl(a.v);return chain(a,s,0.5L/s,-0.25L/(s*a.v));}
template<int N> Jet<N> jpow(const Jet<N>&a,long double p){long double x=a.v;return chain(a,powl(x,p),p*powl(x,p-1),p*(p-1)*powl(x,p-2));}
template<int N> Jet<N> operator*(long double c,const Jet<N>&a){return Jet<N>(c)*a;}
template<int N> Jet<N> operator/(long double c,const Jet<N>&a){return Jet<N>(c)*inv(a);}
