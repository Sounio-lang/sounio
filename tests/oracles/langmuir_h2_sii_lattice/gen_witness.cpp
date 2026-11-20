// Writes tests/run-pass/langmuir_h2_sii_lattice_uq.sio: the site and cage-centre tables come from
// sII.hpp (the same lattice the oracle sums over), so the witness and the oracle read one geometry.
#include "sII.hpp"
#include <cstdio>
#include <cstring>
#include <string>
// a float literal: %.17g, with ".0" when it would otherwise read as an integer
static std::string fl(double v){ char b[64]; std::snprintf(b,64,"%.17g",v); std::string s=b; if(s.find_first_of(".eEn")==std::string::npos) s+=".0"; return s; }
static std::vector<V> sites_near(const Model& M, const V& c, double R){ std::vector<V> s; for(auto& o: M.O){ V d{o[0]-c[0],o[1]-c[1],o[2]-c[2]}; if(std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2])<R) s.push_back(d);} return s; }
static std::vector<V> centres_rel(const Model& M, const V& c){ std::vector<V> s; for(int t=0;t<2;++t) for(auto& q:(t?M.large_c:M.small_c)) for(int a=-1;a<=1;++a)for(int b=-1;b<=1;++b)for(int e=-1;e<=1;++e){ V d{M.A*(q[0]+a)-c[0],M.A*(q[1]+b)-c[1],M.A*(q[2]+e)-c[2]}; double r2=d[0]*d[0]+d[1]*d[1]+d[2]*d[2]; if(r2>1e-9 && r2<14.0*14.0) s.push_back(d);} return s; }
int main(){ Model M; M.init();
  std::vector<V> S[2], C[2]; for(int cg=0;cg<2;++cg){ V c=cg?M.large_c[0]:M.small_c[0]; for(auto& v:c) v*=M.A; S[cg]=sites_near(M,c,M.cut+(cg?4.6:3.2)+0.1); C[cg]=centres_rel(M,c); }
  auto P=[](const char* s){ std::fputs(s,stdout); };
  P("//@ run-pass\n//@ requires: madaros\n");
  P("//@ description: GUM of the Langmuir constant of H2 in the EXPLICIT sII lattice (Fd-3m origin 2, oxygen coordinates of Falenty et al. 2014 via GenIce), one-site H2 (Buch), Lorentz-Berthelot, quadratic Feynman-Hibbs, point-centred 12 A cutoff with a uniform-density tail, quadrature over the Voronoi domain of one cage. The H2 parameters carry the covariance of the four one-site sets (two unit latent channels, Cholesky), shared by both water models, so variance_of the difference between SPC/E and TIP4P/Ice keeps only what does not cancel. 12 values at 200 K. Expected values and their finite-difference check: tests/oracles/langmuir_h2_sii_lattice/lattice_oracle.cpp (long double, the same discrete sum), 1e-10 relative.\n");
  P("//@ expect-stdout: LANGMUIR_H2_SII_LATTICE_UQ_OK\n\n");
  for(int cg=0;cg<2;++cg){ std::printf("var SITE_%c: [f64; %zu] = [0.0; %zu]\nvar CEN_%c: [f64; %zu] = [0.0; %zu]\n",cg?'L':'S',3*S[cg].size(),3*S[cg].size(),cg?'L':'S',3*C[cg].size(),3*C[cg].size()); }
  P("\n");
  for(int cg=0;cg<2;++cg){ char t=cg?'L':'S';
    std::printf("fn fill_%c() with Mut, Div {\n    let s = [",cg?'l':'s'); for(size_t i=0;i<S[cg].size();++i) for(int a=0;a<3;++a) std::printf("%s%s",(i||a)?", ":"",fl(S[cg][i][a]).c_str()); P("]\n");
    std::printf("    var i: i64 = 0\n    while i < %zu {\n        SITE_%c[i as usize] = s[i as usize]\n        i = i + 1\n    }\n",3*S[cg].size(),t);
    P("    let c = ["); for(size_t i=0;i<C[cg].size();++i) for(int a=0;a<3;++a) std::printf("%s%s",(i||a)?", ":"",fl(C[cg][i][a]).c_str()); P("]\n");
    std::printf("    var j: i64 = 0\n    while j < %zu {\n        CEN_%c[j as usize] = c[j as usize]\n        j = j + 1\n    }\n}\n\n",3*C[cg].size(),t); }
  P(R"SIO(fn own_point(x0: f64, x1: f64, x2: f64, large: bool) -> bool with Mut, Div {
    let d0 = x0 * x0 + x1 * x1 + x2 * x2
    var n: i64 = 56
    if large { n = 44 }
    var k: i64 = 0
    while k < n {
        var a = 0.0
        var b = 0.0
        var e = 0.0
        if large {
            a = x0 - CEN_L[(3 * k) as usize]
            b = x1 - CEN_L[(3 * k + 1) as usize]
            e = x2 - CEN_L[(3 * k + 2) as usize]
        } else {
            a = x0 - CEN_S[(3 * k) as usize]
            b = x1 - CEN_S[(3 * k + 1) as usize]
            e = x2 - CEN_S[(3 * k + 2) as usize]
        }
        if a * a + b * b + e * e < d0 { return false }
        k = k + 1
    }
    true
}

// Effective guest-framework potential (K) at a point relative to the cage centre.
fn w_point(x0: f64, x1: f64, x2: f64, eps: f64, sig: f64, lam: f64, large: bool) -> f64 with Mut, Div {
    let s2 = sig * sig
    let s6 = s2 * s2 * s2
    let s12 = s6 * s6
    var n: i64 = 388
    if large { n = 482 }
    var u = 0.0
    var l = 0.0
    var k: i64 = 0
    while k < n {
        var a = 0.0
        var b = 0.0
        var e = 0.0
        if large {
            a = x0 - SITE_L[(3 * k) as usize]
            b = x1 - SITE_L[(3 * k + 1) as usize]
            e = x2 - SITE_L[(3 * k + 2) as usize]
        } else {
            a = x0 - SITE_S[(3 * k) as usize]
            b = x1 - SITE_S[(3 * k + 1) as usize]
            e = x2 - SITE_S[(3 * k + 2) as usize]
        }
        let r2 = a * a + b * b + e * e
        if r2 <= 144.0 {
            let i2 = 1.0 / r2
            let i6 = i2 * i2 * i2
            let aa = s12 * i6 * i6
            let bb = s6 * i6
            u = u + 4.0 * eps * (aa - bb)
            l = l + 4.0 * eps * (132.0 * aa - 30.0 * bb) * i2
        }
        k = k + 1
    }
    // uniform-density tail beyond 12 A: 16 pi rho eps (sig^12/(9 c^9) - sig^6/(3 c^3))
    let tail = 16.0 * 3.14159265358979323846 * 0.02622090667229109 * eps * (s12 / (9.0 * 5159780352.0) - s6 / (3.0 * 1728.0))
    u + tail + lam * l
}

fn ln_langmuir(eps_h: f64, sig_h: f64, eps_o: f64, sig_o: f64, t: f64, lam: f64, to_mpa: f64, large: bool) -> f64 with Mut, Div {
    let eps = sqrt(eps_h * eps_o)
    let sig = 0.5 * (sig_h + sig_o)
    var rmax = 3.2
    if large { rmax = 4.6 }
    let rx = [0.86113631159405257, 0.33998104358485631, -0.33998104358485631, -0.86113631159405257]
    let rw = [0.34785484513745374, 0.65214515486254643, 0.65214515486254643, 0.34785484513745374]
    let tx = [0.96028985649753629, 0.79666647741362684, 0.52553240991632899, 0.18343464249564978, -0.18343464249564978, -0.52553240991632899, -0.79666647741362684, -0.96028985649753629]
    let tw = [0.10122853629037679, 0.22238103445337445, 0.31370664587788738, 0.36268378337836199, 0.36268378337836199, 0.31370664587788738, 0.22238103445337445, 0.10122853629037679]
)SIO");
  P("    let cph = ["); for(int k=0;k<16;++k) std::printf("%s%s",k?", ":"",fl(std::cos(2*M_PI*(k+0.5)/16)).c_str()); P("]\n");
  P("    let sph = ["); for(int k=0;k<16;++k) std::printf("%s%s",k?", ":"",fl(std::sin(2*M_PI*(k+0.5)/16)).c_str()); P("]\n");
  P(R"SIO(    var acc = 0.0
    var p: i64 = 0
    while p < 10 {
        var i: i64 = 0
        while i < 4 {
            let r = ((p as f64) + 0.5 + 0.5 * rx[i as usize]) * rmax / 10.0
            let wr = 0.5 * rw[i as usize] * rmax / 10.0
            var j: i64 = 0
            while j < 8 {
                let ct = tx[j as usize]
                let st = sqrt(1.0 - ct * ct)
                var k: i64 = 0
                while k < 16 {
                    let x0 = r * st * cph[k as usize]
                    let x1 = r * st * sph[k as usize]
                    let x2 = r * ct
                    if own_point(x0, x1, x2, large) {
                        let w = w_point(x0, x1, x2, eps, sig, lam, large)
                        acc = acc + wr * tw[j as usize] * (2.0 * 3.14159265358979323846 / 16.0) * r * r * exp(0.0 - w / t)
                    }
                    k = k + 1
                }
                j = j + 1
            }
            i = i + 1
        }
        p = p + 1
    }
    log(acc * to_mpa)
}

fn close_rel(got: f64, want: f64) -> bool with Mut, Div {
    var d = got - want
    if d < 0.0 { d = 0.0 - d }
    var w = want
    if w < 0.0 { w = 0.0 - w }
    d <= 1.0e-14 + 1.0e-10 * w
}

fn report(name: string, got: f64, want: f64) -> i64 with IO, Mut, Div {
    print(name)
    print(" ")
    print_f64(got)
    if close_rel(got, want) {
        print(" ok\n")
        return 0
    }
    print(" FAIL\n")
    1
}

fn uq_cage(large: bool, want: [f64; 6]) -> i64 with IO, Epistemic, Mut, Div {
    var bad: i64 = 0
    let k1 = measure(0.0, uncertainty: 1.0)
    let k2 = measure(0.0, uncertainty: 1.0)
    let z1 = k1.value
    let z2 = k2.value
    let eh = 34.2 + 2.23364425696364341 * z1
    let sh = 2.96 + (0.0 - 2.52912550826065453e-02) * z1 + 1.12653584638716957e-03 * z2
    let lam = 0.0050131884215877977
    let tm = 0.00036214852580199598
    let a = ln_langmuir(eh, sh, 78.20, 3.166, 200.0, lam, tm, large)
    let b = ln_langmuir(eh, sh, 106.1, 3.1668, 200.0, lam, tm, large)
    bad = bad + report("lnC_spce", a, want[0])
    bad = bad + report("lnC_tip4pice", b, want[1])
    bad = bad + report("var_spce", variance_of(a), want[2])
    bad = bad + report("var_tip4pice", variance_of(b), want[3])
    let d = b - a
    bad = bad + report("diff", d, want[4])
    bad = bad + report("var_diff", variance_of(d), want[5])
    bad
}

fn main() -> i32 with IO, Epistemic, Mut, Div {
    fill_s()
    fill_l()
)SIO");
  return 0; }
