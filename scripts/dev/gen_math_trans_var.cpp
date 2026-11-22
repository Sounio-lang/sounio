#include <quadmath.h>
#include <cstdio>
#include <cmath>
#include <string>
typedef __float128 Q;
static Q dq(double d){ return (Q)d; }
static void num(const char*n, Q v){ printf("%.17g", (double)v); }
int main(){
  printf("//@ run-pass\n//@ expect-stdout: MATH_TRANS_VARIANCE_OK\n//@ requires: madaros\n");
  printf("// First-order GUM variance (variance_of) and Hessian (hessian_of) through the\n// builtins tan, atan, atan2, asin, acos, tanh and ln on Madaros. The reference\n// values come from closed-form derivatives evaluated in __float128 by\n// scripts/dev/gen_math_trans_var.cpp (g++ -std=c++23 gen_math_trans_var.cpp -lquadmath).\n//   Var(f(x)) = f'(x)^2 sigma^2,  H = f''(x)\n// Before this change tan/atan/tanh/asin/acos were refused with E221 and ln/atan2\n// were unbound (E137), so none of these programs compiled.\n\n");
  printf("fn near(a: f64, b: f64) -> bool {\n    var d = a - b\n    if d < 0.0 { d = 0.0 - d }\n    var m = b\n    if m < 0.0 { m = 0.0 - m }\n    d <= 1.0e-9 * m + 1.0e-14\n}\n\n");
  struct F{const char*n; Q x; Q s;};
  F fs[]={{"tan",dq(0.7),dq(0.05)},{"atan",dq(0.7),dq(0.05)},{"tanh",dq(0.4),dq(0.05)},{"asin",dq(0.6),dq(0.02)},{"acos",dq(0.6),dq(0.02)},{"ln",dq(2.5),dq(0.1)}};
  for(auto&f:fs){
    Q x=f.x,s=f.s,d1,d2; std::string n=f.n;
    if(n=="tan"){Q t=tanq(x); d1=1+t*t; d2=2*t*(1+t*t);}
    if(n=="atan"){d1=1/(1+x*x); d2=-2*x/((1+x*x)*(1+x*x));}
    if(n=="tanh"){Q t=tanhq(x); d1=1-t*t; d2=-2*t*(1-t*t);}
    if(n=="asin"){Q w=1-x*x; d1=1/sqrtq(w); d2=x/(w*sqrtq(w));}
    if(n=="acos"){Q w=1-x*x; d1=-1/sqrtq(w); d2=-x/(w*sqrtq(w));}
    if(n=="ln"){d1=1/x; d2=-1/(x*x);}
    printf("fn t_%s() -> bool with IO, Epistemic {\n    let k: Knowledge<f64> = measure(%.17g, uncertainty: %.17g)\n    let x = k.value\n    let v = ",f.n,(double)x,(double)s);
    printf("%.17g\n    let h = %.17g\n    near(variance_of(%s(x)), v) && near(hessian_of(%s(x), 0, 0), h)\n}\n\n",(double)(d1*d1*s*s),(double)d2,f.n,f.n);
  }
  // compound inner: atan(x*x)
  { Q x=dq(0.7),s=dq(0.05); Q g=2*x; Q fp=1/(1+x*x*x*x); Q d1=fp*g; Q d2=-2*(x*x)/((1+x*x*x*x)*(1+x*x*x*x))*g*g+fp*2;
    printf("fn t_atan_compound() -> bool with IO, Epistemic {\n    let k: Knowledge<f64> = measure(0.7, uncertainty: 0.05)\n    let x = k.value\n    let v = %.17g\n    let h = %.17g\n    near(variance_of(atan(x * x)), v) && near(hessian_of(atan(x * x), 0, 0), h)\n}\n\n",(double)(d1*d1*s*s),(double)d2); }
  // atan2(y, x), y = 0.6 (sy), x = 0.8 (sx)
  { Q y=dq(0.6),x=dq(0.8),sy=dq(0.03),sx=dq(0.04); Q r2=x*x+y*y; Q r4=r2*r2;
    Q dy=x/r2, dx=-y/r2; Q hyy=-2*x*y/r4, hxx=2*x*y/r4, hyx=(y*y-x*x)/r4;
    printf("fn t_atan2() -> bool with IO, Epistemic {\n    let ky: Knowledge<f64> = measure(0.6, uncertainty: 0.03)\n    let kx: Knowledge<f64> = measure(0.8, uncertainty: 0.04)\n    let y = ky.value\n    let x = kx.value\n    let v = %.17g\n    near(variance_of(atan2(y, x)), v)\n        && near(hessian_of(atan2(y, x), 0, 0), %.17g)\n        && near(hessian_of(atan2(y, x), 0, 1), %.17g)\n        && near(hessian_of(atan2(y, x), 1, 1), %.17g)\n}\n\n",(double)(dy*dy*sy*sy+dx*dx*sx*sx),(double)hyy,(double)hyx,(double)hxx);
    // constant y
    Q yc=dq(1.0); Q r2c=x*x+yc*yc; Q dxc=-yc/r2c;
    printf("fn t_atan2_const_y() -> bool with IO, Epistemic {\n    let kx: Knowledge<f64> = measure(0.8, uncertainty: 0.04)\n    let x = kx.value\n    near(variance_of(atan2(1.0, x)), %.17g)\n}\n\n",(double)(dxc*dxc*sx*sx));
    Q xc=dq(0.8), ycc=dq(0.6); (void)xc; (void)ycc;
    // constant x
    Q dyc=x/(x*x+y*y);
    printf("fn t_atan2_const_x() -> bool with IO, Epistemic {\n    let ky: Knowledge<f64> = measure(0.6, uncertainty: 0.03)\n    let y = ky.value\n    near(variance_of(atan2(y, 0.8)), %.17g)\n}\n\n",(double)(dyc*dyc*sy*sy));
  }
  printf("fn main() -> i64 with IO, Mut, Div, Panic, Epistemic {\n    var fails: i64 = 0\n");
  const char* names[]={"tan","atan","tanh","asin","acos","ln","atan_compound","atan2","atan2_const_y","atan2_const_x"};
  for(auto n:names) printf("    if !t_%s() { print(\"FAIL %s\\n\")\n        fails = fails + 1 }\n",n,n);
  printf("    if fails == 0 {\n        print(\"MATH_TRANS_VARIANCE_OK\\n\")\n        return 0\n    }\n    1\n}\n");
}
