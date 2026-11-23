// Langmuir constant of H2 in the 5^12 cage of the sII lattice (model 4b) with THF in every 5^12 6^4
// cage, and its gradient in the H2 site parameters (eps_H, sig_H); the oracle of molecular::promoter.
// THF: TraPPE-UA united atoms, rigid planar regular pentagon (declared simplification), centre at
// 8b, orientations uniform and independent of the H2: the orientation average factorises per THF,
//   C = int e^{-b w_O(x)} prod_k <e^{-b u_k(x,R)}>_R  dx  (times the uniform-density tail factor).
// Usage: promoter_oracle T eO sO nOrient np nr nth mode   mode 0 = pseudo-random orientations (mt19937),
// 1..3 = one fixed orientation, 4 = Halton (bases 2, 3, 5), the set molecular::promoter uses.
// Predictions fixed before this file existed: promoter_predictions.txt.
static double halton(long i, int b){ double f = 1, r = 0; while(i > 0){ f /= b; r += f*(i % b); i /= b; } return r; }
#include "sII.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
using M3 = std::array<double,9>;
static M3 quat_rot(double a, double b, double c, double d){ // unit quaternion -> rotation
  return {1-2*(c*c+d*d), 2*(b*c-a*d), 2*(b*d+a*c), 2*(b*c+a*d), 1-2*(b*b+d*d), 2*(c*d-a*b), 2*(b*d-a*c), 2*(c*d+a*b), 1-2*(b*b+c*c)}; }
struct Site { V r; double e, s; };
int main(int argc, char** argv){
  Model M; M.T = std::atof(argv[1]); M.eO = std::atof(argv[2]); M.sO = std::atof(argv[3]); if(argc>10){ M.eH = std::atof(argv[9]); M.sH = std::atof(argv[10]); } M.init();
  int nOr = argc>4 ? std::atoi(argv[4]) : 64, np = argc>5 ? std::atoi(argv[5]) : 16, nr = argc>6 ? std::atoi(argv[6]) : 8,
      nth = argc>7 ? std::atoi(argv[7]) : 12, nph = 2*nth, mode = argc>8 ? std::atoi(argv[8]) : 0;
  const double b = 1.0/M.T, lam = M.lamO, eH = M.eH, sH = M.sH, cut = M.cut;
  // THF body frame: O then four CH2 on a regular pentagon of circumradius 1.276 A in the xy plane
  auto env = [](const char* k, double d){ const char* v = std::getenv(k); return v ? std::atof(v) : d; };
  const double xi = env("XI",1.0), eOt = env("E_OT",55.0), sOt = env("S_OT",2.80), eC = env("E_CH2",46.0), sC = env("S_CH2",3.95);
  const double Rp = 1.276; std::vector<Site> body;
  for(int i=0;i<5;++i){ double t = 2*M_PI*i/5; body.push_back({V{Rp*std::cos(t), Rp*std::sin(t), 0}, i==0 ? eOt : eC, i==0 ? sOt : sC}); }
  // orientations
  std::vector<M3> R;
  if(mode == 0){ std::mt19937_64 g(20261122); std::uniform_real_distribution<double> U(0,1);
    for(int k=0;k<nOr;++k){ double u1=U(g),u2=U(g),u3=U(g); R.push_back(quat_rot(std::sqrt(1-u1)*std::sin(2*M_PI*u2), std::sqrt(1-u1)*std::cos(2*M_PI*u2), std::sqrt(u1)*std::sin(2*M_PI*u3), std::sqrt(u1)*std::cos(2*M_PI*u3))); } }
  else if(mode == 4){ for(int k=1;k<=nOr;++k){ double u1=halton(k,2),u2=halton(k,3),u3=halton(k,5); R.push_back(quat_rot(std::sqrt(1-u1)*std::sin(2*M_PI*u2), std::sqrt(1-u1)*std::cos(2*M_PI*u2), std::sqrt(u1)*std::sin(2*M_PI*u3), std::sqrt(u1)*std::cos(2*M_PI*u3))); } }
  else if(mode == 1) R.push_back(quat_rot(1,0,0,0));
  else if(mode == 2) R.push_back(quat_rot(std::cos(M_PI/4),std::sin(M_PI/4),0,0));       // ring plane xz
  else               R.push_back(quat_rot(std::cos(M_PI/4),0,std::sin(M_PI/4),0));       // ring plane yz
  // small cage centre and the THF centres that can reach it
  V c = M.small_c[0]; for(auto& v:c) v *= M.A; const double rmax = 3.2;
  std::vector<V> thf; for(auto& q:M.large_c) for(int a=-1;a<=1;++a) for(int bb=-1;bb<=1;++bb) for(int d=-1;d<=1;++d){
    V p{M.A*(q[0]+a), M.A*(q[1]+bb), M.A*(q[2]+d)}; double r2=0; for(int i=0;i<3;++i) r2 += (p[i]-c[i])*(p[i]-c[i]);
    if(std::sqrt(r2) < cut + Rp + rmax + 0.1) thf.push_back(p); }
  // uniform-density tail of the THF sites beyond the cutoff, and its gradient
  double rho_cell = 8.0/(M.A*M.A*M.A), tail = 0, tail_e = 0, tail_s = 0;
  for(auto& s:body){ double ex = xi*std::sqrt(s.e*eH), sx = 0.5*(s.s+sH), c3 = cut*cut*cut, c9 = c3*c3*c3;
    double t = 16*M_PI*rho_cell*ex*(std::pow(sx,12)/(9*c9) - std::pow(sx,6)/(3*c3));
    tail += t; tail_e += t/(2*eH); tail_s += 0.5*16*M_PI*rho_cell*ex*(12*std::pow(sx,11)/(9*c9) - 6*std::pow(sx,5)/(3*c3)); }
  std::vector<double> rx,rw,tx,tw; gl(nr,rx,rw); gl(nth,tx,tw);
  double I0=0, I1=0, g0e=0, g0s=0, g1e=0, g1s=0;  // 0 = empty large cages, 1 = THF
  for(int p=0;p<np;++p) for(int i=0;i<nr;++i){ double r=(p+0.5+0.5*rx[i])*rmax/np, wr=0.5*rw[i]*rmax/np;
    for(int j=0;j<nth;++j){ double ct=tx[j], st=std::sqrt(1-ct*ct); for(int k=0;k<nph;++k){ double ph=2*M_PI*(k+0.5)/nph;
      V x{c[0]+r*st*std::cos(ph), c[1]+r*st*std::sin(ph), c[2]+r*ct}; V h=x; for(auto& v:h) v -= M.A*std::floor(v/M.A);
      if(M.cage(h) != 0) continue;
      double best=r*r; bool own=true; for(auto& q:M.small_c){ for(int a=-1;a<=1&&own;++a) for(int bb=-1;bb<=1&&own;++bb) for(int d=-1;d<=1&&own;++d){
        double dx=x[0]-M.A*(q[0]+a), dy=x[1]-M.A*(q[1]+bb), dz=x[2]-M.A*(q[2]+d); if(dx*dx+dy*dy+dz*dz < best-1e-9) own=false; } }
      if(!own) continue;
      Grad gO; double wO = wO_grad(M, h, gO); double wt = wr*tw[j]*(2*M_PI/nph)*r*r, f0 = std::exp(-b*wO);
      double lnF = 0, ge = 0, gs = 0;
      for(auto& P:thf){ double S=0, Se=0, Ss=0;
        for(auto& Rm:R){ double u=0, ue=0, us=0;
          for(auto& s:body){ V y{P[0]+Rm[0]*s.r[0]+Rm[1]*s.r[1]+Rm[2]*s.r[2], P[1]+Rm[3]*s.r[0]+Rm[4]*s.r[1]+Rm[5]*s.r[2], P[2]+Rm[6]*s.r[0]+Rm[7]*s.r[1]+Rm[8]*s.r[2]};
            double r2=0; for(int q=0;q<3;++q) r2 += (x[q]-y[q])*(x[q]-y[q]); if(r2 > cut*cut) continue;
            double ex=xi*std::sqrt(s.e*eH), sx=0.5*(s.s+sH), s6=std::pow(sx,6), s12=s6*s6, i2=1/r2, i6=i2*i2*i2, A=s12*i6*i6, B=s6*i6;
            double uu = 4*ex*(A-B) + lam*4*ex*(132*A-30*B)*i2;
            u += uu; ue += uu/(2*eH); us += 0.5*(4*ex*(12*A-6*B)/sx + lam*4*ex*(132*12*A-30*6*B)*i2/sx); }
          double e = std::exp(-b*u); S += e; Se += e*(-b*ue); Ss += e*(-b*us); }
        lnF += std::log(S/R.size()); ge += Se/S; gs += Ss/S; }
      double f1 = f0*std::exp(lnF);
      I0 += wt*f0; g0e += wt*f0*(-b*gO.e); g0s += wt*f0*(-b*gO.s);
      I1 += wt*f1; g1e += wt*f1*(-b*gO.e + ge); g1s += wt*f1*(-b*gO.s + gs); } } }
  double C0 = I0*M.toMPa(), C1 = I1*std::exp(-b*tail)*M.toMPa();
  std::printf("T=%g eO=%g sO=%g nOr=%zu np=%d nr=%d nth=%d mode=%d nTHF=%zu  C_empty=%.17g C_THF=%.17g d=%.17g  "
              "dlnC0/de=%.17g dlnC0/ds=%.17g dlnC1/de=%.17g dlnC1/ds=%.17g  tail=%.4f K\n",
              M.T, M.eO, M.sO, R.size(), np, nr, nth, mode, thf.size(), C0, C1, std::log(C1/C0),
              g0e/I0, g0s/I0, g1e/I1 - b*tail_e, g1s/I1 - b*tail_s, tail);
}
