#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "capd/capdlib.h"

using namespace capd;
using namespace std;

int main(int argc, char** argv) {
  cout.precision(17);
  try {
    // Replay TCB: CAPD 5.3.0 tarball SHA-256 e4100959a5409d330f8907d050f101a
    // 0485489075b4ce0d5eb2e349a2f8bf228, compiled with capd-config flags.
    const double radius = argc > 1 ? std::atof(argv[1]) : 1e-8;
    const int order = argc > 2 ? std::atoi(argv[2]) : 30;
    const int period = 6;

    // Work in w = z-zsec so the Poincare section is the coordinate plane w=0.
    IMap vf("par:zs;var:x,y,w;fun:2*y*y-x*y,x*y-y*(w+zs)/2,x*y-w-zs;");
    const interval zs = interval(223274637391.) / interval(10000000000.);
    vf.setParameter("zs", zs);

    IOdeSolver solver(vf, order);
    ICoordinateSection section(3, 2);
    IPoincareMap pm(solver, section, poincare::MinusPlus);

    IVector center(2), box(2);
    const double centerX =
        argc > 3 ? std::atof(argv[3]) : 15.186446520640786;
    const double centerY =
        argc > 4 ? std::atof(argv[4]) : 10.908543194765466;
    center[0] = centerX;
    center[1] = centerY;
    box[0] = radius * interval(-1., 1.);
    box[1] = radius * interval(-1., 1.);
    const IVector X = center + box;
    const IVector embeddedCenter{center[0], center[1], interval(0.)};
    const IVector embeddedBox{X[0], X[1], interval(0.)};

    interval centerReturnTime;
    C0HOTripletonSet centerSet(embeddedCenter);
    const IVector centerImage3 = pm(centerSet, centerReturnTime, period);
    const IVector centerImage(2, centerImage3.begin());

    interval returnTime;
    C1HORect2Set derivativeSet(embeddedBox);
    IMatrix flowDerivative(3, 3);
    const IVector image3 = pm(derivativeSet, flowDerivative, returnTime, period);
    const IMatrix fullDP = pm.computeDP(image3, flowDerivative);

    IMatrix DF(2, 2);
    DF[0][0] = fullDP[0][0] - 1.;
    DF[0][1] = fullDP[0][1];
    DF[1][0] = fullDP[1][0];
    DF[1][1] = fullDP[1][1] - 1.;
    const IVector residual = centerImage - center;
    const IVector N = center - capd::matrixAlgorithms::gauss(DF, residual);

    const interval tr = fullDP[0][0] + fullDP[1][1];
    const interval det = fullDP[0][0] * fullDP[1][1]
                       - fullDP[0][1] * fullDP[1][0];
    const interval normalVelocity = image3[0] * image3[1] - image3[2] - zs;

    bool primePeriodSix = true;
    cout << "INTERMEDIATE_RETURNS_BEGIN\n";
    for (int k = 1; k < period; ++k) {
      C0HOTripletonSet kthSet(embeddedBox);
      interval kthTime;
      const IVector kthImage = pm(kthSet, kthTime, k);
      const interval kthNormal = kthImage[0] * kthImage[1] - kthImage[2] - zs;
      const bool disjoint = kthImage[0].rightBound() < X[0].leftBound()
          || kthImage[0].leftBound() > X[0].rightBound()
          || kthImage[1].rightBound() < X[1].leftBound()
          || kthImage[1].leftBound() > X[1].rightBound();
      primePeriodSix = primePeriodSix && disjoint;
      cout << "K=" << k << " TIME=" << kthTime << " IMAGE=" << kthImage
           << " NORMAL_VELOCITY=" << kthNormal
           << " DISJOINT_FROM_X=" << (disjoint ? "true" : "false") << "\n";
    }
    cout << "INTERMEDIATE_RETURNS_END\n";

    // Liouville gives det(DP^6)=exp(int div f dt) at a fixed point because
    // initial and final section-normal velocities are then identical.
    IMap vf4("par:zs;var:x,y,w,d;fun:2*y*y-x*y,x*y-y*(w+zs)/2,"
             "x*y-w-zs,x-y-(w+zs)/2-1;");
    vf4.setParameter("zs", zs);
    IOdeSolver solver4(vf4, order);
    ICoordinateSection section4(4, 2);
    IPoincareMap pm4(solver4, section4, poincare::MinusPlus);
    C0HOTripletonSet divergenceSet(
        {X[0], X[1], interval(0.), interval(0.)});
    interval divergenceReturnTime;
    const IVector divergenceImage =
        pm4(divergenceSet, divergenceReturnTime, period);
    const interval integralDivergence = divergenceImage[3];
    const interval determinantLiouville = exp(integralDivergence);
    const interval discriminant = sqr(tr) - 4. * determinantLiouville;
    const interval expandingMultiplier =
        (tr - sqrt(discriminant)) / 2.;
    const interval contractingMultiplier =
        determinantLiouville / expandingMultiplier;
    const bool hyperbolicitySeparated =
        expandingMultiplier.rightBound() < -1.
        && contractingMultiplier.leftBound() > -1.
        && contractingMultiplier.rightBound() < 0.;

    cout << "CAPD_VERSION=5.3.0\n";
    cout << "ORDER=" << order << " PERIOD=" << period
         << " RADIUS=" << radius << "\n";
    cout << "ZSEC=" << zs << "\n";
    cout << "CENTER=" << center << "\n";
    cout << "X=" << X << "\n";
    cout << "CENTER_IMAGE=" << centerImage << "\n";
    cout << "RESIDUAL=" << residual << "\n";
    cout << "CENTER_RETURN_TIME=" << centerReturnTime << "\n";
    cout << "RETURN_TIME=" << returnTime << "\n";
    cout << "IMAGE3=" << image3 << "\n";
    cout << "NORMAL_VELOCITY=" << normalVelocity << "\n";
    cout << "FLOW_DERIVATIVE=" << flowDerivative << "\n";
    cout << "POINCARE_DP=" << fullDP << "\n";
    cout << "TRACE_2D=" << tr << "\n";
    cout << "DET_2D=" << det << "\n";
    cout << "DIVERGENCE_RETURN_TIME=" << divergenceReturnTime << "\n";
    cout << "DIVERGENCE_IMAGE=" << divergenceImage << "\n";
    cout << "INTEGRAL_DIVERGENCE=" << integralDivergence << "\n";
    cout << "DET_LIOUVILLE_AT_FIXED_POINT=" << determinantLiouville << "\n";
    cout << "DISCRIMINANT_LIOUVILLE=" << discriminant << "\n";
    cout << "EXPANDING_MULTIPLIER=" << expandingMultiplier << "\n";
    cout << "CONTRACTING_MULTIPLIER=" << contractingMultiplier << "\n";
    cout << "HYPERBOLICITY_SEPARATED="
         << (hyperbolicitySeparated ? "true" : "false") << "\n";
    cout << "PRIME_PERIOD_SIX=" << (primePeriodSix ? "true" : "false")
         << "\n";
    cout << "DF=" << DF << "\n";
    cout << "NEWTON_N=" << N << "\n";
    cout << "DIAM_X=" << diam(X) << "\n";
    cout << "DIAM_N=" << diam(N) << "\n";
    cout << "NEWTON_INTERIOR="
         << (subsetInterior(N, X) ? "true" : "false") << "\n";
    return subsetInterior(N, X) && hyperbolicitySeparated && primePeriodSix
        ? EXIT_SUCCESS
        : 2;
  } catch (const std::exception& error) {
    cerr << "CAPD_EXCEPTION=" << error.what() << "\n";
    return 3;
  }
}
