// C++23/quadmath fixture emitter: known-solution systems, not production elimination.
// Matrix formulas define inputs; RHS is computed independently in quad precision.
#include <quadmath.h>
#include <iomanip>
#include <iostream>
#include <limits>
using Q=__float128;
int main(){
 std::cout<<std::setprecision(17)<<std::showpoint<<R"SIO(//@ run-pass
//@ expect-stdout: STDLIB_DEEP_FACTOR_REFERENCE_PASS
use collections::numeric_buffer::{NumericBuffer,numeric_buffer_new,numeric_buffer_status,numeric_buffer_at,numeric_buffer_set,numeric_buffer_close,numeric_matrix}
use collections::integer_buffer::{IntegerBuffer,integer_buffer_new,integer_buffer_status,integer_buffer_at,integer_buffer_close}
use linalg::dynamic_factorize::{dynamic_factor_workspace,dynamic_lu,dynamic_cholesky,dynamic_lu_solve,dynamic_cholesky_solve}
fn ab(x:f64)->f64 {if x<0.0 {-x} else {x}}
fn close(a:f64,b:f64)->bool with Div {ab(a-b)<=1.0e-9*ab(b)}
fn matrix_value(row:i64,col:i64,n:i64,kind:i64)->f64 with Div {
 if kind==1 {
  if row==0 && col==1 {return 2.0}
  if row==1 && col==0 {return 1.0}
  if row==1 && col==1 {return 3.0}
  return 0.0
 }
 if kind==4 {if row==col {return 1.0e-200};return 0.0}
 if row==col {return 4.0}
 if row+1==col || col+1==row {return -1.0}
 0.0
}
fn verify_factor(a:&NumericBuffer,factor:&NumericBuffer,pivots:&IntegerBuffer,n:i64,stride:i64,scale:f64,method:i64) with Mut,Div,Panic {
 var i:i64=0
 while i<n {
  var j:i64=0
  while j<n {
   var sum:f64=0.0
   var correction:f64=0.0
   var k:i64=0
   while k<n {
    var left:f64=0.0;var right:f64=0.0
    if method==0 {
     if i==k {left=1.0} else if i>k {left=numeric_buffer_at(factor,i*n+k)}
     if k<=j {right=numeric_buffer_at(factor,k*n+j)}
    } else {
     if k<=i {left=numeric_buffer_at(factor,i*n+k)}
     if k<=j {right=numeric_buffer_at(factor,j*n+k)}
    }
    let term=left*right-correction
    let next=sum+term;correction=(next-sum)-term;sum=next;k=k+1
   }
   let original_row=if method==0 {integer_buffer_at(pivots,i)} else {i}
   let expected=numeric_buffer_at(a,original_row*stride+j)/scale
   assert(ab(sum-expected)<=1.0e-10)
   j=j+1
  }
  i=i+1
 }
}
fn run_case(n:i64,kind:i64,method:i64,rhs:&[f64;128],expected:&[f64;128]) with IO,Mut,Div,Panic,Alloc {
 let query=dynamic_factor_workspace(n)
 assert(query.status==0)
 let stride=if kind==3 {n+1} else {n}
 var a=numeric_buffer_new(n*stride,n*stride*8)
 var factor=numeric_buffer_new(n*n,n*n*8)
 var work=numeric_buffer_new(query.numeric_work,query.numeric_work*8)
 var pivots=integer_buffer_new(n,n*8)
 var pivot_work=integer_buffer_new(n,n*8)
 var b=numeric_buffer_new(n,n*8)
 var x=numeric_buffer_new(n,n*8)
 var solve_work=numeric_buffer_new(query.lu_solve_work,query.lu_solve_work*8)
 assert(numeric_buffer_status(&a)==0 && numeric_buffer_status(&factor)==0 && numeric_buffer_status(&work)==0 && integer_buffer_status(&pivots)==0 && integer_buffer_status(&pivot_work)==0 && numeric_buffer_status(&b)==0 && numeric_buffer_status(&x)==0 && numeric_buffer_status(&solve_work)==0)
 var i:i64=0
 while i<n*stride {numeric_buffer_set(&!a,i,0.0/0.0);i=i+1}
 i=0
 while i<n {
  var j:i64=0
  while j<n {numeric_buffer_set(&!a,i*stride+j,matrix_value(i,j,n,kind));j=j+1}
  numeric_buffer_set(&!b,i,rhs[i as usize]);i=i+1
 }
 let layout=numeric_matrix(n*stride,n,n,stride,1,0)
 if method==0 {
  let fit=dynamic_lu(&a,&layout,&!factor,&!pivots,&!work,&!pivot_work,0.0)
  assert(fit.status==0 && fit.completed_pivots==n)
  verify_factor(&a,&factor,&pivots,n,stride,fit.scale,0)
  let result=dynamic_lu_solve(&factor,&pivots,&fit,&b,&!x,&!solve_work)
  assert(result.status==0)
 } else {
  let fit=dynamic_cholesky(&a,&layout,&!factor,&!work,0.0,0.0)
  assert(fit.status==0 && fit.completed_pivots==n)
  verify_factor(&a,&factor,&pivots,n,stride,fit.scale,1)
  let result=dynamic_cholesky_solve(&factor,&fit,&b,&!x,&!solve_work)
  assert(result.status==0)
 }
 i=0
 while i<n {assert(close(numeric_buffer_at(&x,i),expected[i as usize]));i=i+1}
 numeric_buffer_close(a);numeric_buffer_close(factor);numeric_buffer_close(work)
 integer_buffer_close(pivots);integer_buffer_close(pivot_work)
 numeric_buffer_close(b);numeric_buffer_close(x);numeric_buffer_close(solve_work)
}
fn main()->i32 with IO,Mut,Div,Panic,Alloc {
 var rhs:[f64;128]=[0.0;128]
 var expected:[f64;128]=[0.0;128]
)SIO";
 std::cout<<" rhs[0]=4.0;rhs[1]=7.0;expected[0]=1.0;expected[1]=2.0\n run_case(2,1,0,&rhs,&expected)\n";
 for(int i=0;i<128;++i){
  Q total=0;
  for(int j=0;j<128;++j){
   const Q entry=i==j?Q(4):((i+1==j||j+1==i)?Q(-1):Q(0));
   total+=entry*Q(j+1)/128;
  }
  std::cout<<" rhs["<<i<<"]="<<static_cast<double>(total)<<";expected["<<i<<"]="<<static_cast<double>(Q(i+1)/128)<<"\n";
 }
 std::cout<<" run_case(128,3,0,&rhs,&expected)\n run_case(128,3,1,&rhs,&expected)\n";
 const double tiny=std::numeric_limits<double>::denorm_min();
 const double large=1e-308,scale=1e-200;
 std::cout<<" rhs[0]="<<tiny<<";rhs[1]="<<large<<"\n";
 std::cout<<" expected[0]="<<static_cast<double>(Q(tiny)/Q(scale))<<";expected[1]="<<static_cast<double>(Q(large)/Q(scale))<<"\n";
 std::cout<<R"SIO( run_case(2,4,0,&rhs,&expected)
 run_case(2,4,1,&rhs,&expected)
 println("STDLIB_DEEP_FACTOR_REFERENCE_PASS")
 0
}
)SIO";
 return std::cout.good()?0:1;
}
