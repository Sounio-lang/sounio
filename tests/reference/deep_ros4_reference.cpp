// C++23 reference: Boost.Odeint1.83 engine with one explicitly corrected
// d4 coefficient from WRF/KPP RODAS4. Official positive-d4 is a separate control.
// WRF commit06d4240ae989cc3e50af412bb472df3d9048783c, rosenbrock.f90:1056.
// Source SHA25644eb88bcbd91ab2f12f37ba71d314038d6a771f1a8750f5f8b8c2b317940fa2a.
// Single-step concordance,
// independent analytic exponential/quadratic solutions for adaptive integration.
// Compile on the authorized worker, recording installed header hash and Boost version.
#include <boost/numeric/odeint/stepper/rosenbrock4.hpp>
#include <boost/version.hpp>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <utility>
namespace on=boost::numeric::odeint;
using OfficialBoostStep=on::rosenbrock4<double>;
struct ReferenceRodasCoefficients: on::default_rosenbrock_coefficients<double> {
 using Base=on::default_rosenbrock_coefficients<double>;
 const double d4;
 ReferenceRodasCoefficients():Base(),d4(-0.3620000000000023e-01){}
};
using Step=on::rosenbrock4<double,ReferenceRodasCoefficients>;
using State=Step::state_type;
using Matrix=Step::matrix_type;
struct Rhs {
 int kind;
 void operator()(const State& y,State& dy,double t)const {
  for(std::size_t i=0;i<y.size();++i){
   const double lambda=100.0*static_cast<double>(i+1);
   if(kind==3&&i==2){dy[i]=1;continue;}
   const double clock=kind==3?y[2]:t;
   dy[i]=kind==0 ? -lambda*y[i] : -lambda*(y[i]-clock*clock)+2*clock;
  }
 }
};
struct Jac {
 int kind;
 void operator()(const State& y,Matrix& j,double t,State& dfdt)const {
  for(std::size_t i=0;i<y.size();++i){
   const double lambda=100.0*static_cast<double>(i+1);
   for(std::size_t k=0;k<y.size();++k)j(i,k)=i==k?-lambda:0.0;
   dfdt[i]=kind==0||kind==3?0.0:2*lambda*t+2;
   if(kind==3){if(i==2){for(std::size_t k=0;k<y.size();++k)j(i,k)=0;}else j(i,2)=2*lambda*y[2]+2;}
  }
 }
};
void emit_case(int n,int kind,double t,double h){
 State y(n),trial(n),error(n);
 for(int i=0;i<n;++i)y[i]=kind==0 ? 1.0+static_cast<double>(i)/128.0 : (kind==2||kind==3) ? 0.0 : (i==0?0.09:-1.0);
 Step step;
 step.do_step(std::make_pair(Rhs{kind},Jac{kind}),y,t,trial,h,error);
 std::cout<<" {\n var expected:[f64;128]=[0.0;128]\n var expected_error:[f64;128]=[0.0;128]\n";
 for(int i=0;i<n;++i){
  if(!std::isfinite(trial[i])||!std::isfinite(error[i]))std::terminate();
  std::cout<<" expected["<<i<<"]="<<trial[i]<<";expected_error["<<i<<"]="<<error[i]<<"\n";
 }
 std::cout<<" run_case("<<n<<","<<kind<<","<<t<<","<<h<<",&expected,&expected_error)\n }\n";
}
int main(){
 std::cout<<std::setprecision(17)<<std::showpoint;
 std::cout<<"// Boost.Odeint engine with WRF/KPP negative-d4; version "<<BOOST_LIB_VERSION<<"\n";
 std::cout<<R"SIO(//@ run-pass
//@ expect-stdout: STDLIB_DEEP_ROS4_REFERENCE_PASS
use collections::numeric_buffer::{NumericBuffer,numeric_buffer_new,numeric_buffer_status,numeric_buffer_len,numeric_buffer_at,numeric_buffer_set,numeric_buffer_close}
use collections::integer_buffer::{integer_buffer_new,integer_buffer_status,integer_buffer_close}
use ode::solver::{default_options}
use ode::rosenbrock_dynamic::{ros4_workspace,ros4_step_dynamic,solve_rosenbrock4_dynamic}
fn autonomous_rhs(t:f64,y:&NumericBuffer,dy:&!NumericBuffer)->i64 with Mut,Div,Panic {
 var i:i64=0
 while i<numeric_buffer_len(y) {numeric_buffer_set(dy,i,-100.0*((i+1) as f64)*numeric_buffer_at(y,i));i=i+1}
 0
}
fn autonomous_jac(t:f64,y:&NumericBuffer,j:&!NumericBuffer,dfdt:&!NumericBuffer)->i64 with Mut,Div,Panic {
 let n=numeric_buffer_len(y)
 var i:i64=0
 while i<n {
  var k:i64=0
  while k<n {numeric_buffer_set(j,i*n+k,if i==k {-100.0*((i+1) as f64)} else {0.0});k=k+1}
  numeric_buffer_set(dfdt,i,0.0);i=i+1
 }
 0
}
fn timed_rhs(t:f64,y:&NumericBuffer,dy:&!NumericBuffer)->i64 with Mut,Div,Panic {
 var i:i64=0
 while i<numeric_buffer_len(y) {
  let lambda=100.0*((i+1) as f64)
  numeric_buffer_set(dy,i,-lambda*(numeric_buffer_at(y,i)-t*t)+2.0*t);i=i+1
 }
 0
}
fn timed_jac(t:f64,y:&NumericBuffer,j:&!NumericBuffer,dfdt:&!NumericBuffer)->i64 with Mut,Div,Panic {
 let n=numeric_buffer_len(y)
 var i:i64=0
 while i<n {
  let lambda=100.0*((i+1) as f64)
  var k:i64=0
  while k<n {numeric_buffer_set(j,i*n+k,if i==k {-lambda} else {0.0});k=k+1}
  numeric_buffer_set(dfdt,i,2.0*lambda*t+2.0);i=i+1
 }
 0
}
fn augmented_rhs(t:f64,y:&NumericBuffer,dy:&!NumericBuffer)->i64 with Mut,Div,Panic {
 let clock=numeric_buffer_at(y,2)
 var i:i64=0
 while i<2 {
  let lambda=100.0*((i+1) as f64)
  numeric_buffer_set(dy,i,-lambda*(numeric_buffer_at(y,i)-clock*clock)+2.0*clock)
  i=i+1
 }
 numeric_buffer_set(dy,2,1.0)
 0
}
fn augmented_jac(t:f64,y:&NumericBuffer,j:&!NumericBuffer,dfdt:&!NumericBuffer)->i64 with Mut,Div,Panic {
 var i:i64=0
 while i<9 {numeric_buffer_set(j,i,0.0);i=i+1}
 i=0
 while i<3 {numeric_buffer_set(dfdt,i,0.0);i=i+1}
 let clock=numeric_buffer_at(y,2)
 numeric_buffer_set(j,0,-100.0);numeric_buffer_set(j,4,-200.0)
 numeric_buffer_set(j,2,200.0*clock+2.0);numeric_buffer_set(j,5,400.0*clock+2.0)
 0
}
fn ab(x:f64)->f64 {if x<0.0 {-x} else {x}}
fn run_case(n:i64,kind:i64,t:f64,h:f64,expected:&[f64;128],expected_error:&[f64;128]) with IO,Mut,Div,Panic,Alloc {
 let q=ros4_workspace(n)
 assert(q.status==0)
 var state=numeric_buffer_new(n,n*8)
 var trial=numeric_buffer_new(n,n*8)
 var error=numeric_buffer_new(n,n*8)
 var derivative=numeric_buffer_new(n,n*8)
 var stages=numeric_buffer_new(q.stages_len,q.stages_len*8)
 var matrix=numeric_buffer_new(q.matrix_len,q.matrix_len*8)
 var factor=numeric_buffer_new(q.factor_len,q.factor_len*8)
 var work=numeric_buffer_new(q.factor_work_len,q.factor_work_len*8)
 var pivots=integer_buffer_new(n,n*8)
 var pivotwork=integer_buffer_new(n,n*8)
 assert(numeric_buffer_status(&state)==0 && numeric_buffer_status(&trial)==0 && numeric_buffer_status(&error)==0 && numeric_buffer_status(&derivative)==0 && numeric_buffer_status(&stages)==0 && numeric_buffer_status(&matrix)==0 && numeric_buffer_status(&factor)==0 && numeric_buffer_status(&work)==0 && integer_buffer_status(&pivots)==0 && integer_buffer_status(&pivotwork)==0)
 var i:i64=0
 while i<n {
  let initial=if kind==0 {1.0+(i as f64)/128.0} else if kind==2 || kind==3 {0.0} else if i==0 {0.09} else {-1.0}
  numeric_buffer_set(&!state,i,initial);i=i+1
 }
 if kind==0 {
  let step=ros4_step_dynamic(autonomous_rhs,autonomous_jac,&state,&!trial,&!error,&!derivative,&!stages,&!matrix,&!factor,&!work,&!pivots,&!pivotwork,t,h,0.0)
  assert(step.status==0 && step.nlinear==6 && step.nfeval==6 && step.njev==1 && step.nfactorizations==1)
 } else if kind==3 {
  let step=ros4_step_dynamic(augmented_rhs,augmented_jac,&state,&!trial,&!error,&!derivative,&!stages,&!matrix,&!factor,&!work,&!pivots,&!pivotwork,t,h,0.0)
  assert(step.status==0)
 } else {
  let step=ros4_step_dynamic(timed_rhs,timed_jac,&state,&!trial,&!error,&!derivative,&!stages,&!matrix,&!factor,&!work,&!pivots,&!pivotwork,t,h,0.0)
  assert(step.status==0 && step.nlinear==6)
 }
 i=0
 while i<n {
  let value=numeric_buffer_at(&trial,i);let defect=numeric_buffer_at(&error,i)
  assert(ab(value-expected[i as usize])<=1.0e-9*ab(expected[i as usize]))
  // Error correction may approach a cancellation zero: bound against initial scale.
  assert(ab(defect-expected_error[i as usize])<=1.0e-11)
  i=i+1
 }
 var opts=default_options()
 opts.rtol=1.0e-11;opts.atol=1.0e-14;opts.dt_init=0.001;opts.dt_max=0.01
 if n==1 {
  numeric_buffer_set(&!state,0,1.0)
  let solution=solve_rosenbrock4_dynamic(autonomous_rhs,autonomous_jac,&!state,&!trial,&!error,&!derivative,&!stages,&!matrix,&!factor,&!work,&!pivots,&!pivotwork,0.0,0.1,opts,0.0)
  assert(solution.status==0 && solution.t_final==0.1)
)SIO";
 std::cout<<"  assert(ab(numeric_buffer_at(&state,0)-"<<static_cast<double>(std::exp(-10.L))<<")<=1.0e-7*"<<static_cast<double>(std::exp(-10.L))<<")\n";
 std::cout<<R"SIO( }
 if kind==1 {
  i=0
  while i<n {numeric_buffer_set(&!state,i,0.0);i=i+1}
  let solution=solve_rosenbrock4_dynamic(timed_rhs,timed_jac,&!state,&!trial,&!error,&!derivative,&!stages,&!matrix,&!factor,&!work,&!pivots,&!pivotwork,0.0,1.0,opts,0.0)
  assert(solution.status==0 && solution.t_final==1.0 && solution.njev==solution.attempts)
  i=0
  while i<n {assert(ab(numeric_buffer_at(&state,i)-1.0)<=1.0e-8);i=i+1}
 }
 if kind==3 {
  i=0
  while i<n {numeric_buffer_set(&!state,i,0.0);i=i+1}
  let solution=solve_rosenbrock4_dynamic(augmented_rhs,augmented_jac,&!state,&!trial,&!error,&!derivative,&!stages,&!matrix,&!factor,&!work,&!pivots,&!pivotwork,0.0,1.0,opts,0.0)
  assert(solution.status==0 && solution.t_final==1.0)
  i=0
  while i<n {assert(ab(numeric_buffer_at(&state,i)-1.0)<=1.0e-8);i=i+1}
 }
 numeric_buffer_close(state);numeric_buffer_close(trial);numeric_buffer_close(error);numeric_buffer_close(derivative)
 numeric_buffer_close(stages);numeric_buffer_close(matrix);numeric_buffer_close(factor);numeric_buffer_close(work)
 integer_buffer_close(pivots);integer_buffer_close(pivotwork)
}
fn main()->i32 with IO,Mut,Div,Panic,Alloc {
)SIO";
 emit_case(1,0,0.0,0.0013);
 emit_case(2,2,0.0,0.001);
 emit_case(3,3,0.0,0.001);
 emit_case(2,1,0.3,0.02);
 emit_case(128,0,0.0,0.0013);
 std::cout<<R"SIO( println("STDLIB_DEEP_ROS4_REFERENCE_PASS")
 0
}
)SIO";
 return std::cout.good()?0:1;
}
