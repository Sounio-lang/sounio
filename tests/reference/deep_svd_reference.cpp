// C++23 analytic SVD fixture emitter. No production Jacobi rotations mirrored.
// Expected 2x2 singular values use closed-form trace/discriminant and determinant;
// maximum-shape disjoint-support columns have analytically known equal singular values.
#include <cmath>
#include <iomanip>
#include <iostream>
int main(){
 std::cout<<std::setprecision(17)<<std::showpoint<<R"SIO(//@ run-pass
//@ expect-stdout: STDLIB_DEEP_SVD_REFERENCE_PASS
use collections::numeric_buffer::{NumericBuffer,NumericMatrix,numeric_buffer_new,numeric_buffer_status,numeric_buffer_at,numeric_buffer_set,numeric_buffer_close,numeric_matrix,numeric_matrix_get}
use linalg::dynamic_svd::{dynamic_svd_workspace,dynamic_svd}
fn ab(x:f64)->f64 {if x<0.0 {-x} else {x}}
fn verify(a:&NumericBuffer,layout:&NumericMatrix,u:&NumericBuffer,v:&NumericBuffer,sigma:&NumericBuffer,rows:i64,cols:i64,k:i64,scale:f64) with Mut,Div,Panic {
 var row:i64=0
 while row<rows {
  var col:i64=0
  while col<cols {
   var sum:f64=0.0
   var correction:f64=0.0
   var j:i64=0
   while j<k {
    let term=numeric_buffer_at(u,row*k+j)*numeric_buffer_at(sigma,j)*numeric_buffer_at(v,col*k+j)
    let adjusted=term-correction
    let next=sum+adjusted
    correction=(next-sum)-adjusted
    sum=next;j=j+1
   }
   let expected=numeric_matrix_get(a,layout,row,col)
   assert(expected.status==0 && ab(sum-expected.value/scale)<1.0e-9)
   col=col+1
  }
  row=row+1
 }
 var p:i64=0
 while p<k {
  var q=p
  while q<k {
   var su:f64=0.0
   var sv:f64=0.0
   row=0
   while row<rows {su=su+numeric_buffer_at(u,row*k+p)*numeric_buffer_at(u,row*k+q);row=row+1}
   row=0
   while row<cols {sv=sv+numeric_buffer_at(v,row*k+p)*numeric_buffer_at(v,row*k+q);row=row+1}
   let expected=if p==q {1.0} else {0.0}
   assert(ab(su-expected)<1.0e-9 && ab(sv-expected)<1.0e-9)
   q=q+1
  }
  p=p+1
 }
}
fn run_case(rows:i64,cols:i64,kind:i64,largest:f64,smallest:f64,expected_rank:i64) with IO,Mut,Div,Panic,Alloc {
 let k=if rows<cols {rows} else {cols}
 let shape=dynamic_svd_workspace(rows,cols)
 assert(shape.status==0)
 let stride=if kind==4 {cols+1} else {cols}
 var a=numeric_buffer_new(rows*stride,rows*stride*8)
 var u=numeric_buffer_new(shape.u_len,shape.u_len*8)
 var v=numeric_buffer_new(shape.v_len,shape.v_len*8)
 var sigma=numeric_buffer_new(k,k*8)
 var work=numeric_buffer_new(shape.work_len,shape.work_len*8)
 assert(numeric_buffer_status(&a)==0 && numeric_buffer_status(&u)==0 && numeric_buffer_status(&v)==0 && numeric_buffer_status(&sigma)==0 && numeric_buffer_status(&work)==0)
 var i:i64=0
 while i<rows*stride {numeric_buffer_set(&!a,i,0.0/0.0);i=i+1}
 var row:i64=0
 while row<rows {
  var col:i64=0
  while col<cols {
   var value:f64=0.0
   if kind==1 || kind==2 {
    if row==0 && col==0 {value=3.0}
    if row==0 && col==1 {value=1.0}
    if row==1 && col==1 {value=2.0}
   }
   if kind==3 {value=(row+1) as f64}
   if kind==4 {if row%2==col {value=1.0}}
   if kind==5 {if row%32==col {value=1.0}}
   numeric_buffer_set(&!a,row*stride+col,value)
   col=col+1
  }
  row=row+1
 }
 let layout=numeric_matrix(rows*stride,rows,cols,stride,1,0)
 let result=dynamic_svd(&a,&layout,&!u,&!v,&!sigma,&!work,100,1.0e-11,0.0)
 assert(result.status==0 && result.rank==expected_rank)
 assert(result.transposed==(rows<cols) && result.sigma_scale>0.0)
 let first=numeric_buffer_at(&sigma,0)*result.sigma_scale
 let last=numeric_buffer_at(&sigma,k-1)*result.sigma_scale
 if largest==0.0 {assert(first==0.0)} else {assert(ab(first-largest)<=1.0e-9*largest)}
 if smallest==0.0 {assert(last==0.0)} else {assert(ab(last-smallest)<=1.0e-9*smallest)}
 verify(&a,&layout,&u,&v,&sigma,rows,cols,k,result.sigma_scale)
 numeric_buffer_close(a);numeric_buffer_close(u);numeric_buffer_close(v);numeric_buffer_close(sigma);numeric_buffer_close(work)
}
fn budget_case() with Mut,Div,Panic,Alloc {
 var a=numeric_buffer_new(9,72)
 var u=numeric_buffer_new(9,72)
 var v=numeric_buffer_new(9,72)
 var sigma=numeric_buffer_new(3,24)
 var work=numeric_buffer_new(24,192)
 assert(numeric_buffer_status(&a)==0 && numeric_buffer_status(&u)==0 && numeric_buffer_status(&v)==0 && numeric_buffer_status(&sigma)==0 && numeric_buffer_status(&work)==0)
 var i:i64=0
 while i<9 {numeric_buffer_set(&!u,i,42.0);numeric_buffer_set(&!v,i,42.0);i=i+1}
 i=0
 while i<3 {numeric_buffer_set(&!sigma,i,42.0);i=i+1}
 numeric_buffer_set(&!a,0,1.0);numeric_buffer_set(&!a,1,2.0);numeric_buffer_set(&!a,2,3.0)
 numeric_buffer_set(&!a,3,4.0);numeric_buffer_set(&!a,4,1.0);numeric_buffer_set(&!a,5,2.0)
 numeric_buffer_set(&!a,6,2.0);numeric_buffer_set(&!a,7,5.0);numeric_buffer_set(&!a,8,1.0)
 let layout=numeric_matrix(9,3,3,3,1,0)
 let result=dynamic_svd(&a,&layout,&!u,&!v,&!sigma,&!work,1,1.0e-13,0.0)
 assert(result.status==3 && result.sweeps==1)
 assert(numeric_buffer_at(&u,0)==42.0 && numeric_buffer_at(&v,0)==42.0 && numeric_buffer_at(&sigma,0)==42.0)
 numeric_buffer_close(a);numeric_buffer_close(u);numeric_buffer_close(v);numeric_buffer_close(sigma);numeric_buffer_close(work)
}
fn main()->i32 with IO,Mut,Div,Panic,Alloc {
)SIO";
 const long double large=std::sqrt((14.L+std::sqrt(52.L))/2.L);
 const long double small=6.L/large;
 std::cout<<" run_case(2,2,1,"<<static_cast<double>(large)<<","<<static_cast<double>(small)<<",2)\n";
 std::cout<<" run_case(2,3,2,"<<static_cast<double>(large)<<","<<static_cast<double>(small)<<",2)\n";
 std::cout<<" run_case(3,2,3,"<<static_cast<double>(std::sqrt(28.L))<<",0.0,1)\n";
 std::cout<<" run_case(4,2,4,"<<static_cast<double>(std::sqrt(2.L))<<","<<static_cast<double>(std::sqrt(2.L))<<",2)\n";
 std::cout<<" run_case(1024,32,5,"<<static_cast<double>(std::sqrt(32.L))<<","<<static_cast<double>(std::sqrt(32.L))<<",32)\n";
 std::cout<<R"SIO( run_case(4,3,0,0.0,0.0,0)
 budget_case()
 println("STDLIB_DEEP_SVD_REFERENCE_PASS")
 0
}
)SIO";
 return std::cout.good()?0:1;
}
