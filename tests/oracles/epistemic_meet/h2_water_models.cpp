// Oracle for examples/molecular/h2_sii_water_contest.sio, the refusal fixtures
// tests/epistemic-meet/refuse_peelle*.sio and the witness in
// formal/EpistemicCorr.lean: ADR-016 meet/contest with correlated sources.
// Data, fixed below: H2 content of sII per cell at 200 K, 300 MPa from the C++ GCMC
// whose N are the predictions in examples/molecular/h2_sii_epistemic.sio (4e6 trials),
// with its Monte Carlo sd and fluctuation sensitivities dN/deps_H, dN/dsig_H, for
// SPC/E, TIP4P/Ice, TIP5P. The H2 covariance is two unit latent
// channels (Cholesky a, b, d); each model has its own MC channel.
// Prints: covariance, pairwise E_n (k = 2) correlated and independent, GLS
// weights / value / u for each pair and for all three, chi2, Birge ratio,
// n_eff = 1^T R^-1 1, and the decisions under En(1) and Birge.
#include <cstdio>
#include <cmath>
#include <array>
using R = long double;
using M3 = std::array<std::array<R,3>,3>;
static M3 inv3(const M3& A) {
  M3 B; R det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1]) - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
  B[0][0]=(A[1][1]*A[2][2]-A[1][2]*A[2][1])/det; B[0][1]=(A[0][2]*A[2][1]-A[0][1]*A[2][2])/det; B[0][2]=(A[0][1]*A[1][2]-A[0][2]*A[1][1])/det;
  B[1][0]=(A[1][2]*A[2][0]-A[1][0]*A[2][2])/det; B[1][1]=(A[0][0]*A[2][2]-A[0][2]*A[2][0])/det; B[1][2]=(A[0][2]*A[1][0]-A[0][0]*A[1][2])/det;
  B[2][0]=(A[1][0]*A[2][1]-A[1][1]*A[2][0])/det; B[2][1]=(A[0][1]*A[2][0]-A[0][0]*A[2][1])/det; B[2][2]=(A[0][0]*A[1][1]-A[0][1]*A[1][0])/det;
  return B;
}
int main() {
  const R a = 2.23364425696364341L, b = -2.52912550826065453e-2L, d = 1.12653584638716957e-3L;
  const char* nm[3] = {"SPC/E", "TIP4P/Ice", "TIP5P"};
  R x[3] = {31.820L, 33.080L, 32.345L}, e[3] = {0.038L, 0.058L, 0.061L};
  R ge[3] = {0.0950L, 0.0813L, 0.0907L}, gs[3] = {-19.059L, -20.585L, -19.881L};
  R s1[3], s2[3]; M3 V, Vind{};
  for (int i = 0; i < 3; ++i) { s1[i] = ge[i]*a + gs[i]*b; s2[i] = gs[i]*d; }
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) { V[i][j] = s1[i]*s1[j] + s2[i]*s2[j] + (i==j ? e[i]*e[i] : 0); }
  for (int i = 0; i < 3; ++i) Vind[i][i] = V[i][i];
  std::printf("u: %.12Lg %.12Lg %.12Lg\n", sqrtl(V[0][0]), sqrtl(V[1][1]), sqrtl(V[2][2]));
  std::printf("rho01 %.12Lg rho02 %.12Lg rho12 %.12Lg\n", V[0][1]/sqrtl(V[0][0]*V[1][1]), V[0][2]/sqrtl(V[0][0]*V[2][2]), V[1][2]/sqrtl(V[1][1]*V[2][2]));
  int pr[3][2] = {{1,0},{2,0},{1,2}};
  for (auto& p : pr) {
    int i = p[0], j = p[1];
    R dv = V[i][i] + V[j][j] - 2*V[i][j], di = V[i][i] + V[j][j];
    R En = fabsl(x[i]-x[j]) / (2*sqrtl(dv)), Eni = fabsl(x[i]-x[j]) / (2*sqrtl(di));
    R w = (V[j][j] - V[i][j]) / dv, xm = w*x[i] + (1-w)*x[j], vm = (V[i][i]*V[j][j] - V[i][j]*V[i][j]) / dv;
    std::printf("pair %s-%s: diff %.12Lg u(diff) %.12Lg En %.12Lg | indep u %.12Lg En %.12Lg | GLS w_%s %.12Lg x %.12Lg u %.12Lg\n",
      nm[i], nm[j], x[i]-x[j], sqrtl(dv), En, sqrtl(di), Eni, nm[i], w, xm, sqrtl(vm));
  }
  for (int pass = 0; pass < 2; ++pass) {
    const M3& C = pass ? Vind : V;
    M3 W = inv3(C); R s = 0, wv[3];
    for (int i = 0; i < 3; ++i) { wv[i] = 0; for (int j = 0; j < 3; ++j) wv[i] += W[i][j]; s += wv[i]; }
    R xm = 0; for (int i = 0; i < 3; ++i) { wv[i] /= s; xm += wv[i]*x[i]; }
    R chi2 = 0; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) chi2 += (x[i]-xm)*W[i][j]*(x[j]-xm);
    R RB = sqrtl(chi2/2);
    std::printf("%s GLS3: w %.12Lg %.12Lg %.12Lg x %.12Lg u %.12Lg chi2 %.12Lg Birge %.12Lg u_Birge %.12Lg\n",
      pass ? "indep" : "corr ", wv[0], wv[1], wv[2], xm, sqrtl(1/s), chi2, RB, sqrtl(1/s)*(RB > 1 ? RB : 1));
  }
  M3 Rm; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Rm[i][j] = V[i][j]/sqrtl(V[i][i]*V[j][j]);
  M3 Ri = inv3(Rm); R ne = 0; for (auto& r : Ri) for (R v : r) ne += v;
  std::printf("n_eff %.12Lg\n", ne);
  return 0;
}
