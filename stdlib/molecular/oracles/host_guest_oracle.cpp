// Oracle for tests/stdlib/molecular/test_host_guest_oracle.sio (molecular::host_guest).
// Independent C++ construction of the sII lattice (sII.hpp: Fd-3m closed from the same
// generators, Falenty 2014 coordinates) and of the guest-host field w with its gradients,
// evaluated in long double at three fixed points; the guest-guest pair at one distance.
// Build: g++ -std=c++23 -O2 host_guest_oracle.cpp -o host_guest_oracle
#include "sII.hpp"
#include <cstdio>
using LD = long double;
int main(){
  Model M; M.T=250; M.eO=78.20; M.sO=3.166; M.init();
  std::printf("group %zu  oxygens(home) %zu  sites %zu  small %zu  large %zu\n",group().size(),M.O.size()/27,M.O.size(),M.small_c.size(),M.large_c.size());
  std::printf("lamO %.17g  rho %.17g\n",M.lamO,136.0/(M.A*M.A*M.A));
  V pts[3]={{0.3,0.2,-0.1},{6.4,6.6,6.3},{1.0,8.0,15.5}};
  for(auto p:pts){ LD eps=std::sqrt((LD)M.eO*M.eH), sig=0.5L*((LD)M.sO+M.sH), s6=std::pow(sig,6), s12=s6*s6, lam=M.lamO, c2=(LD)M.cut*M.cut;
    LD U=0,L=0,Us=0; for(auto& s:M.O){ LD a=p[0]-(LD)s[0], b=p[1]-(LD)s[1], e=p[2]-(LD)s[2], r2=a*a+b*b+e*e; if(r2>c2) continue; LD i2=1/r2,i6=i2*i2*i2,A=s12*i6*i6,B=s6*i6;
      U+=4*eps*(A-B); L+=4*eps*(132*A-30*B)*i2; Us+=4*eps*(12*A-6*B)/sig+lam*4*eps*(1584*A-180*B)*i2/sig; }
    LD rho=136.0L/((LD)M.A*M.A*M.A), c3=(LD)M.cut*M.cut*M.cut, c9=c3*c3*c3, k=16*M_PI*rho*eps;
    LD t=k*(s12/(9*c9)-s6/(3*c3)), ts=k*(12*s12/(9*c9)-6*s6/(3*c3))/sig, w=U+t+lam*L;
    std::printf("point (%g,%g,%g): w %.17Lg  dw/deH %.17Lg  dw/dsH %.17Lg\n",p[0],p[1],p[2],w,w/(2*(LD)M.eH),0.5L*(Us+ts)); }
  { LD r2=3.3L*3.3L, sH=M.sH, eH=M.eH, lam2=M.lamHH, s6=std::pow(sH,6), s12=s6*s6, i2=1/r2, i6=i2*i2*i2, A=s12*i6*i6, B=s6*i6;
    LD u=4*eH*(A-B)+lam2*4*eH*(132*A-30*B)*i2; std::printf("pair r=3.3: u %.17Lg du/deH %.17Lg du/dsH %.17Lg  lamHH %.17g\n",u,u/eH,(4*eH*(12*A-6*B)+lam2*4*eH*(1584*A-180*B)*i2)/sH,M.lamHH); }
}
