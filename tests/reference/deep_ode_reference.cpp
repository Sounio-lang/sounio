// C++23 independent analytic fixture emitter. Execute only on authorized cluster lane.
// No production ODE coefficients are copied here: reference solutions use long-double exp/sin/cos.
#include <cmath>
#include <iomanip>
#include <iostream>
int main() {
 std::cout << R"SIO(//@ run-pass
//@ expect-stdout: STDLIB_DEEP_ODE_REFERENCE_PASS
use ode::checked64::{solve_dp54_64,solve_verlet64}
use ode::solver::{default_options}
fn decay_rhs(t:f64,y:&[f64;64],dy:&![f64;64]) with Mut,Div,Panic {
 var i:i64=0
 while i<64 {dy[i as usize]=-((i+1) as f64)*y[i as usize]/64.0;i=i+1}
}
fn oscillator(t:f64,y:&[f64;64],dy:&![f64;64]) with Mut,Div,Panic {
 dy[0]=y[1];dy[1]=-y[0]
}
fn reference_acceleration(t:f64,y:&[f64;64],a:&![f64;64]) with Mut,Div,Panic {a[0]=-y[0]}
fn close(actual:f64,expected:f64,relative:f64)->bool with Div {
 let d=if actual>expected {actual-expected} else {expected-actual}
 let scale=if expected<0.0 {-expected} else {expected}
 d<=relative*scale
}
fn main()->i32 with IO,Mut,Div,Panic {
 var initial:[f64;64]=[1.0;64]
 var expected:[f64;64]=[0.0;64]
)SIO";
 std::cout<<std::setprecision(17);
 for(int i=0;i<64;++i) std::cout<<" expected["<<i<<"]="<<static_cast<double>(std::exp(-static_cast<long double>(i+1)/64.L))<<"\n";
 std::cout<<R"SIO( var options=default_options()
 options.rtol=1.0e-10;options.atol=1.0e-12
 let full=solve_dp54_64(decay_rhs,&initial,64,0.0,1.0,options)
 assert(full.status==0 && full.n==64 && full.t_final==1.0 && full.nfeval==1+6*full.attempts)
 var i:i64=0
 while i<64 {assert(close(full.state[i as usize],expected[i as usize],1.0e-8));i=i+1}
 initial[0]=1.0;initial[1]=0.0
 let harmonic=solve_dp54_64(oscillator,&initial,2,0.0,1.0,options)
 assert(harmonic.status==0)
)SIO";
 std::cout<<" assert(close(harmonic.state[0],"<<static_cast<double>(std::cos(1.L))<<",1.0e-8))\n";
 std::cout<<" assert(close(harmonic.state[1],"<<static_cast<double>(-std::sin(1.L))<<",1.0e-8))\n";
 std::cout<<R"SIO( var velocity:[f64;64]=[0.0;64]
 let symplectic=solve_verlet64(reference_acceleration,&initial,&velocity,1,0.0,1.0,1000)
 assert(symplectic.status==0 && symplectic.steps==1000 && symplectic.nfeval==1001)
)SIO";
 std::cout<<" assert(close(symplectic.position[0],"<<static_cast<double>(std::cos(1.L))<<",1.0e-6))\n";
 std::cout<<" assert(close(symplectic.velocity[0],"<<static_cast<double>(-std::sin(1.L))<<",1.0e-6))\n";
 std::cout<<R"SIO( println("STDLIB_DEEP_ODE_REFERENCE_PASS")
 0
}
)SIO";
 return std::cout.good()?0:1;
}
