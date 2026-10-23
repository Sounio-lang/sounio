# Checked ideal-gas isothermal entropy

Implementation: `physics::entropy::isothermal_entropy_independent`.

## Physical model and units

The nominal model is

```text
L = ln(Vf/Vi)
DeltaS = n R L
```

n is amount of substance in mol; Vi and Vf are positive volumes expressed
in the same unit. They need not be cubic metres because only their ratio
enters the model. un, uvi and uvf are nonnegative input **standard**
uncertainties in the same units as their respective inputs. Outputs DeltaS
and standard_uncertainty are in J/K. log_volume_ratio is dimensionless.

R is the exact SI molar gas constant 8.31446261815324 J/(mol K), represented
as a rounded binary64 coefficient by constants::physical::gas_constant_approx.
The physical constant has no uncertainty contribution in the revised SI.
This source value was checked against the shared stdlib constant and
[NIST molar gas constant](https://physics.nist.gov/cgi-bin/cuu/Value?r%7Csearch_for=R).

This is the entropy difference of the ideal-gas equilibrium states at a
common temperature and fixed amount, as represented by that thermodynamic
model. It does not certify an experimental process as isothermal, estimate
irreversible entropy production, model a real-gas equation of state, or
infer input uncertainties from observations.

## Independent first-order standard uncertainty

The API assumes mutually independent input estimates and propagates their
standard uncertainties using first-order sensitivities:

```text
dS/dn  = R L
dS/dVf = n R / Vf
dS/dVi = -n R / Vi
uS = hypot(R L un, n R uvf/Vf, n R uvi/Vi)
```

This is the independent-input specialization of the
[JCGM 100:2008 law of propagation of uncertainty, section 5.1](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf).
The result metadata uncertainty_model=1 and first_order=true always identify
that model, including errors. No covariance is inferred. If both volumes
come from the same measurement or calibration, their covariance can matter;
this API cannot represent it. Large relative uncertainty may invalidate
first-order propagation even when arithmetic succeeds. The API supplies
neither a confidence level nor a coverage interval.

Equal nominal volumes give DeltaS=0, while volume uncertainty contributions
remain when n>0. n=0 is allowed, including un>0; the amount sensitivity can
then give nonzero uncertainty even though the nominal entropy is zero.

## API and status

```sounio
pub struct EntropyResult {
    pub status: i64,
    pub delta_s: f64,
    pub standard_uncertainty: f64,
    pub log_volume_ratio: f64,
    pub uncertainty_model: i64,
    pub first_order: bool,
}
pub fn isothermal_entropy_independent(
    n:f64, un:f64, vf:f64, uvf:f64, vi:f64, uvi:f64
) -> EntropyResult with Mut, Div, Panic
```

Read status before the three numerical payload fields.

| Status | Meaning |
|---|---|
| 0 | Finite represented nominal entropy and independent first-order uncertainty |
| 1 | At least one input is NaN or infinity |
| 2 | n<0, Vi<=0, or Vf<=0 |
| 3 | At least one input standard uncertainty is negative |
| 4 | Arithmetic failure or unrepresentable nonzero nominal/sensitivity contribution or uncertainty |

Validation precedence is nonfinite, physical domain, then negative uncertainty.
For every error, delta_s, standard_uncertainty and log_volume_ratio are zero;
uncertainty_model=1 and first_order=true are retained. Exact zero factors are
valid. A nonzero contribution rounding entirely to zero is unavailable with
status4, even if another uncertainty contribution would dominate the norm.
Representable subnormal results are accepted. Normalized squared ratios inside
the norm can round to zero without constituting an unavailable contribution.

## Numerical implementation

Close volumes, within a factor of two, use d=(Vf-Vi)/Vi and
ln(1+d)=2 sum t^(2j+1)/(2j+1), t=d/(2+d), with 32 terms. This avoids
subtracting nearly equal logarithms. The factor-two branch uses halved
positive volumes rather than a possibly overflowing doubled threshold.
Subnormal halving may round at the threshold; the neighboring ratio remains
within this series convergence region. Exactly equal volumes return L=0.

Other volumes use ln_checked(Vf)-ln_checked(Vi), without forming Vf/Vi.
Each entropy or uncertainty contribution independently normalizes its three
numerator factors and denominator to binary mantissas and integer exponents.
This avoids premature n*R overflow and uv/V overflow, as well as premature
product underflow. A final subnormal is formed with one multiplication into
the subnormal range, rather than repeated subnormal halvings. The final
uncertainty uses a scaled norm to avoid overflow in the sum of squares.

The implementation uses scalar state only, no heap and no sample buffers.
Binary normalization takes bounded loops over the binary64 exponent range.
This is a bounded-domain numerical implementation, not a formal rounding proof.
It is not guaranteed correctly rounded in every binary64 case.

Independent C++23 long-double log1p/log and hypot references and native Forge
execution are the validation route. Source completion alone does not establish
native execution or a measured performance improvement.

## Legacy adapter and scientific visualization

The legacy thermo::isothermal_entropy tuple delegates to this checked kernel.
An unavailable result returns (0,-1); a negative standard uncertainty is the
legacy failure sentinel. Other thermodynamics uncertainty formulas retain their
legacy status and are outside this contract.

viz::ir::viz_line_set_standard_uncertainty attaches interval kind 4 to an
existing two-point vertical line on linear scales. Its endpoints must equal
center-u and center+u. Validation is atomic. SVG emits
`data-interval-kind="standard-uncertainty"` and `data-interval-level="0"`.
These bars represent one standard uncertainty under the independent first-order
model. They are not confidence, credible or prediction intervals. Existing
forest builders retain their separate interval-kind contract.

The pinned Madaros literal parser rounds direct minimum-subnormal decimal
literals to zero. The implementation and generated reference fixture construct
subnormal values using normal numerators divided by 2^52. This preserves the
numerical contract; it does not fix or certify the compiler literal parser.
The legacy module's unused unresolved units import was removed and existing
mutation/division effects were declared without changing those legacy formulas.
Legacy gas-law/Boltzmann/heat-capacity consumers now load the same shared
constant functions locally instead of dynamic imported global initializers.
This restores their constant values on the pinned backend; the remaining
legacy uncertainty formulas are not upgraded by this repair. A combined
thermo/SR consumer is selected, without claiming support for every compiler
or resolution mode.

## Correlated volume measurements

`isothermal_entropy_correlated_volumes(n,un,vf,uvf,vi,uvi,rho)` extends the
same ideal-gas isothermal model with a supplied correlation between the final
and initial volume measurements. Amount of substance remains independent of
both volumes. This is a first-order measurement uncertainty model; the API
does not estimate or assess the supplied correlation.

The result keeps the same layout, sets `uncertainty_model=2` and
`first_order=true`, including errors. SI output is J/K; both volumes and
their standard uncertainties must use one common volume unit. The correlation
coefficient is dimensionless. No coverage probability or confidence interval
is implied.

With nonnegative sensitivity contributions
`cn=R*abs(log(vf/vi))*un`, `cf=n*R*uvf/vf`, and `ci=n*R*uvi/vi`,
the propagated variance is

```text
u(DeltaS)^2 = cn^2 + cf^2 + ci^2 - 2*rho*cf*ci.
```

The minus sign follows from the opposing volume derivatives of
`n*R*log(vf/vi)`. The supplied two-volume correlation matrix is positive
semidefinite exactly when `-1<=rho<=1`. Perfect positive correlation and equal
relative volume uncertainties cancel the volume contribution, as expected for
a common multiplicative calibration. An independent amount uncertainty can
still produce a nonzero result. The existing independent API continues to use
model 1. For exactly zero correlation the new API calls that independent
implementation and retags the result; numeric outputs follow the same bit path.

Validation precedence is all seven inputs finite (status 1), physical domain
(status 2), nonnegative standard uncertainties (status 3), correlation range
(status 5), then arithmetic (status 4). Error payloads are zero, with model 2
and first-order metadata retained. All existing individual nominal and
sensitivity representability requirements remain in force even if correlated
contributions would cancel.

The correlated calculation subtracts `cf-ci` before scaling and computes
the remaining component as `sqrt(2*(1-rho))*sqrt(cf)*sqrt(ci)` with balanced
products. A scaled norm of that component, the unscaled difference, and
`cn` avoids overflow in squares and retains a tiny amount contribution when
large volume contributions cancel. A mathematically nonzero cross component
that cannot be represented also returns status 4, even if another component
would dominate. Legitimate exact cancellation returns zero successfully.
This is a conservative representability boundary, not a statement that the
physical uncertainty is undefined.

The sensitivities are rounded binary64 values before cancellation. Near
perfect correlation and almost equal sensitivities, their rounding errors can
dominate the remaining difference; no uniform relative precision is claimed
there. The independent C++23 quad reference tests exact cancellation,
near-perfect correlation with equal sensitivities, very different scales,
unit rescaling, swapping volumes, subnormal values and validation precedence.

The legacy Lorentz-force uncertainty routines in `physics::em` retain their
separate limitations. This extension does not certify their dimensional
uncertainty propagation or provide a general covariance-matrix engine.
