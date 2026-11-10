// Independent C++23 analytic reference emitter: 128-component dynamic ODE path.
// No production DP/Verlet coefficients are used in expected values.
#include <cmath>
#include <iomanip>
#include <iostream>
int main(){
 std::cout<<std::setprecision(17)<<R"SIO(//@ run-pass
//@ expect-stdout: STDLIB_DEEP_ODE_DYNAMIC_REFERENCE_PASS
use collections::numeric_buffer::{NumericBuffer,numeric_buffer_new,numeric_buffer_status,numeric_buffer_len,numeric_buffer_at,numeric_buffer_set,numeric_buffer_close}
use ode::solver::{default_options}
use ode::checked_dynamic::{solve_dp54_dynamic,solve_verlet_dynamic}
fn decay(t:f64,y:&NumericBuffer,dy:&!NumericBuffer)->i64 with Mut,Div,Panic {
 let n=numeric_buffer_len(y)
 var i:i64=0
 while i<n {
  let value= -((i+1) as f64)*numeric_buffer_at(y,i)/(n as f64)
  if numeric_buffer_set(dy,i,value)!=0 {return 1}
  i=i+1
 }
 0
}
fn reference_dynamic_acceleration(t:f64,y:&NumericBuffer,dy:&!NumericBuffer)->i64 with Mut,Div,Panic {
 var i:i64=0
 while i<numeric_buffer_len(y) {numeric_buffer_set(dy,i,-numeric_buffer_at(y,i));i=i+1}
 0
}
fn close(a:f64,b:f64,tol:f64)->bool with Div {
 let scale=if b<0.0 {-b} else {b}
 let gap=if a>b {a-b} else {b-a}
 gap<=tol*scale
}
fn main()->i32 with IO,Mut,Div,Panic,Alloc {
 var state=numeric_buffer_new(128,1024)
 var velocity=numeric_buffer_new(128,1024)
 var trial=numeric_buffer_new(128,1024)
 var derivative=numeric_buffer_new(128,1024)
 var work=numeric_buffer_new(896,7168)
 assert(numeric_buffer_status(&state)==0 && numeric_buffer_status(&velocity)==0 && numeric_buffer_status(&trial)==0 && numeric_buffer_status(&derivative)==0 && numeric_buffer_status(&work)==0)
 var expected:[f64;128]=[0.0;128]
 var expected_v:[f64;128]=[0.0;128]
)SIO";
 for(int i=0;i<128;++i)
  std::cout<<" expected["<<i<<"]="<<static_cast<double>(std::exp(-static_cast<long double>(i+1)/128.L))<<"\n";
 std::cout<<R"SIO( var i:i64=0
 while i<128 {numeric_buffer_set(&!state,i,1.0);numeric_buffer_set(&!velocity,i,0.0);i=i+1}
 var opts=default_options()
 opts.rtol=1.0e-10;opts.atol=1.0e-12
 let result=solve_dp54_dynamic(decay,&!state,&!trial,&!derivative,&!work,0.0,1.0,opts)
 assert(result.status==0 && result.n==128 && result.nfeval==7*result.attempts)
 i=0
 while i<128 {assert(close(numeric_buffer_at(&state,i),expected[i as usize],1.0e-8));i=i+1}
)SIO";
 for(int i=0;i<128;++i) {
  const long double factor=static_cast<long double>(i+1)/128.L;
  std::cout<<" expected["<<i<<"]="<<static_cast<double>(factor*std::cos(1.L))<<"\n";
  std::cout<<" expected_v["<<i<<"]="<<static_cast<double>(-factor*std::sin(1.L))<<"\n";
 }
 std::cout<<R"SIO( i=0
 while i<128 {numeric_buffer_set(&!state,i,((i+1) as f64)/128.0);i=i+1}
 let verlet=solve_verlet_dynamic(reference_dynamic_acceleration,&!state,&!velocity,&!trial,&!derivative,&!work,0.0,1.0,1000)
 assert(verlet.status==0 && verlet.n==128 && verlet.steps==1000 && verlet.nfeval==1001)
 i=0
 while i<128 {
  assert(close(numeric_buffer_at(&state,i),expected[i as usize],1.0e-6))
  assert(close(numeric_buffer_at(&velocity,i),expected_v[i as usize],1.0e-6))
  i=i+1
 }
 numeric_buffer_close(state)
 numeric_buffer_close(velocity)
 numeric_buffer_close(trial)
 numeric_buffer_close(derivative)
 numeric_buffer_close(work)
 println("STDLIB_DEEP_ODE_DYNAMIC_REFERENCE_PASS")
 0
}
)SIO";
 return std::cout.good()?0:1;
}
