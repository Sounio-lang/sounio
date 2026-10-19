# Checked Lorentz force and first-order uncertainty

Implementation: `physics::lorentz`. The force law is `F=q*(E+v cross B)`.
Inputs use SI: charge C, electric field N/C, velocity m/s, magnetic field T;
force components are N, standard uncertainties N, and covariance entries N².
The present f64 APIs document these units; they do not enforce a unit type.

## Checked vector API

```sounio
lorentz_force_independent(
 q:f64,uq:f64,
 e:&[f64;3],ue:&[f64;3],
 v:&[f64;3],uv:&[f64;3],
 b:&[f64;3],ub:&[f64;3]
)->LorentzResult
```

Array order is x,y,z. Every standard uncertainty is nonnegative. This model
declares all ten scalar inputs independent: charge and the three components
of each E, v, B. The implementation does not assess that assumption. It
propagates the nominal Jacobian, retaining covariance between output force
components induced by shared input quantities.

`LorentzResult` contains `status:i64`, `force:[f64;3]`,
`covariance:[f64;9]` (row-major stride3),
`component_standard_uncertainties:[f64;3]`,
`vector_standard_uncertainty:f64`, `uncertainty_model:i64`, and
`first_order:bool`. Model1 and first_order=true are retained on errors.

For each input j, let h_j be its derivative vector times its standard
uncertainty. Then `Cov(F)=sum_j h_j h_j^T`. With Cartesian basis e_j:

```text
h_q   = (E+v cross B)*uq
h_Ej  = q*e_j*ue[j]
h_vj  = q*(e_j cross B)*uv[j]
h_Bj  = q*(v cross e_j)*ub[j]
```

The reported vector standard uncertainty is
`sqrt(trace(Cov(F))) = sqrt(sum_j ||h_j||²)`. It measures the first-order
root mean square vector deviation. It is NOT the uncertainty of the force
magnitude, a confidence radius, a simultaneous interval or a coverage region.
A charge with zero nominal value can still contribute uncertainty through uq.

This uses the first-order law of uncertainty propagation
([JCGM 100:2008, sections 5.1–5.2](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf)).
Large relative uncertainties, nonlinear effects and correlated inputs can
invalidate this particular independent linearization. No higher-order or
Monte Carlo accuracy is claimed.

## Checked scalar proxy

`lorentz_force_scalar_independent(q,uq,e,ue,v,uv,b,ub)` returns
`LorentzScalarResult{status,force,standard_uncertainty,uncertainty_model,first_order}`.
It represents a signed collinear proxy `F=q*(E+v*B)`, rather than a generic
three-dimensional magnitude. Its independent sensitivities are
`E+v*B, q, q*B, q*v`. Charge is shared between the electric and magnetic
terms; it contributes ONCE with derivative `E+v*B`. In particular, exact
electric/magnetic cancellation also cancels that charge sensitivity.

## Status and representability

| Status | Meaning |
|---|---|
| 0 | Available under the stated independent first-order model |
| 1 | Any nonfinite input |
| 3 | A negative input standard uncertainty |
| 4 | Arithmetic or conservative representability failure |

All inputs are checked for finiteness before checking uncertainty signs.
Negative charge, signed fields and signed velocity components are valid.
Every failure clears the entire numerical payload; model/first-order tags
remain. There is no physical-domain restriction on nominal finite inputs in
this classical force law, and no relativistic velocity validation.

Private binary mantissa/exponent products avoid intermediate overflow and
underflow in individual three-factor products. Three-term force and charge
sensitivity sums combine larger exponents before smaller ones. This preserves
small remaining terms after exact cancellation of ALREADY ROUNDED large
products. Products themselves have binary64 rounding; this is not a guarantee
of exact real cancellation or uniformly small relative error for a nearly
cancelled force. Force final rounding may legitimately yield zero. A nonzero
sensitivity contribution rounding to zero returns status4.

Component and vector standard uncertainties use scaled norms. Covariance
terms and final covariance entries must be representable: an individually
nonzero product rounding to zero or overflowing returns status4 even if some
final off-diagonal cancellation would be finite or negligible. This deliberately
conservative availability policy can reject a case whose scalar standard
uncertainties alone could be returned. It distinguishes unavailable covariance
from an invented exact zero. Diagonal covariance requirements also prevent a
successful vector result when variance is unrepresentable, even if its square
root would fit.

Signed off-diagonal terms are accumulated in binary64 scaled sums. Their
relative error can be large under strong cancellation; no uniform relative
precision for such entries is claimed. Tests compare ordinary covariance to
an independent quad reference, check symmetry, shared-charge signs and moderate
principal-minor PSD bounds. Floating output need not be exactly PSD at the last
bit, and the implementation does not project or repair a matrix.

## Legacy adapters and migration

`physics::em::lorentz_force_scalar` keeps its argument/tuple shape and signed
nominal meaning, and delegates to the checked scalar API. Its standard
uncertainty now uses the four independent derivatives above.

`physics::em::lorentz_force_3d` keeps its Vec3 argument/tuple shape. Each scalar
ue,uv,ub denotes an isotropic PER-COMPONENT standard uncertainty; components
and the three groups are independent. Its scalar output uncertainty is
sqrt(trace(Cov(F))) as defined above. This replaces the former dimensionally
incompatible sum of E/v/B uncertainties and includes charge uncertainty.

The legacy Vec3 type, its x/y/z fields and vec3_new constructor are public,
matching the existing public vector adapter signature. Both adapters now
explicitly declare Mut, Div, Panic effects; callers must propagate those effects.

Both adapters return `(zero nominal,-1)` on checked failure. The negative
uncertainty is an unavailable sentinel, not a standard uncertainty. Migrate
to the checked APIs to distinguish statuses and obtain directional covariance.
Legacy callers that treated the scalar uncertainty as uncertainty of |F|
must adopt the stated vector interpretation or compute an appropriate
Jacobian for their actual scalar measurand.

Other electromagnetism routines retain their separate uncertainty limitations;
this repair does not certify the whole em module.

## Independent evidence

`tests/reference/lorentz_reference.cpp` uses C++23 __float128, raw products,
an independently assembled Cartesian Jacobian and Gram covariance. It emits
26 vector and12 scalar cases, including shared-charge cancellation, signed
forces, anisotropy, off-diagonal signs, moderate PSD checks, extreme balanced
products, tiny variance, conservative underflow/overflow and zero nominal
charge with nonzero charge uncertainty. Scalar tiny uncertainty remains
available where no covariance payload is requested. Subnormal constants use
normal/2^52 expressions for the pinned Sounio literal parser.

`physics_lorentz_status.sio` checks metadata and validation;
`physics_lorentz_legacy.sio` uses the same legacy call signatures to
discriminate the old uncertainty formulas. C++ execution occurs on t560;
native Sounio acceptance requires the separate Forge/Slurm receipt.
