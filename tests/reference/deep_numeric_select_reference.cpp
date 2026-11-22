// Independent C++23 std::sort rank oracle; no production partition algorithm.
// Stdout emits compact borrowed-loop Sounio checks.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
static double value(int kind,int i,int n){
 switch(kind){
 case 0:return double(i-n/2);
 case 1:return double(n-1-i-n/2);
 case 2:return double(i%5-2);
 case 3:return double(i<=n/2?i:n-1-i);
 case 4:return .125;
 case 5:{
  const double tiny=std::numeric_limits<double>::denorm_min(),max=std::numeric_limits<double>::max();
  const double a[8]={-max,-1,-tiny,-0.0,0.0,tiny,1,max};return a[i%8];
 }
 default:return double((i*7919+17)%20011-10005);
 }
}
static void literal(double x){
 if(x==std::numeric_limits<double>::denorm_min()){std::cout<<"(2.2250738585072014e-308/4503599627370496.0)";return;}
 if(x==-std::numeric_limits<double>::denorm_min()){std::cout<<"(-2.2250738585072014e-308/4503599627370496.0)";return;}
 std::cout<<x;
}
int main(){
 try{
 std::cout<<std::setprecision(17)<<std::showpoint;
 std::cout<<R"SIO(//@ run-pass
//@ expect-stdout: STDLIB_DEEP_NUMERIC_SELECT_REFERENCE_PASS
use collections::numeric_buffer::{NumericBuffer,numeric_buffer_new,numeric_buffer_status,numeric_buffer_set,numeric_buffer_at,numeric_buffer_close}
use stats::dynamic_order::numeric_select
use math::pure::is_finite
fn sample(kind:i64,i:i64,n:i64)->f64 with Mut,Div {
 if kind==0 {return (i-n/2) as f64}
 if kind==1 {return (n-1-i-n/2) as f64}
 if kind==2 {return (i%5-2) as f64}
 if kind==3 {return if i<=n/2 {i as f64} else {(n-1-i) as f64}}
 if kind==4 {return 0.125}
 if kind==5 {
  let tiny=2.2250738585072014e-308/4503599627370496.0
  let mode=i%8
  if mode==0 {return -1.7976931348623157e308}
  if mode==1 {return -1.0}
  if mode==2 {return -tiny}
  if mode==3 {return -0.0}
  if mode==4 {return 0.0}
  if mode==5 {return tiny}
  if mode==6 {return 1.0}
  return 1.7976931348623157e308
 }
 ((i*7919+17)%20011-10005) as f64
}
fn check(a:&!NumericBuffer,kind:i64,n:i64,k:i64,budget:i64,expected:f64) with Mut,Div,Panic {
 var i:i64=0
 while i<n {
  let v=sample(kind,i,n)
  numeric_buffer_set(a,i,v);i=i+1
 }
 numeric_buffer_set(a,n,0.0/0.0)
 let result=numeric_select(a,n,k,budget)
 assert(result.status==0 && result.value==expected)
 assert(numeric_buffer_at(a,k)==expected)
 assert(result.partition_work>=0 && result.partition_work<=budget && result.partitions<=result.partition_work)
 if n>1 {
  if budget<2 {assert(result.fallback && result.partition_work==0 && result.partitions==0)}
 }
 if kind==4 {
  if budget>=n {
   if n>1 {assert(!result.fallback && result.partitions==1 && result.partition_work==n)}
  }
 }
 i=0
 while i<k {assert(numeric_buffer_at(a,i)<=result.value);i=i+1}
 i=k+1
 while i<n {assert(numeric_buffer_at(a,i)>=result.value);i=i+1}
 if result.fallback {
  i=1
  while i<n {assert(numeric_buffer_at(a,i-1)<=numeric_buffer_at(a,i));i=i+1}
 }
 assert(!is_finite(numeric_buffer_at(a,n)))
}
fn small(kind:i64,n:i64,expected:&[f64;32]) with Alloc,Mut,Div,Panic {
 var a=numeric_buffer_new(n+1,(n+1)*8)
 assert(numeric_buffer_status(&a)==0)
 var k:i64=0
 while k<n {
  let target=expected[k as usize]
  check(&!a,kind,n,k,0,target)
  check(&!a,kind,n,k,1,target)
  check(&!a,kind,n,k,n*12,target)
  k=k+1
 }
 numeric_buffer_close(a)
}
fn large(kind:i64,expected:&[f64;6]) with Alloc,Mut,Div,Panic {
 let n:i64=20000
 var a=numeric_buffer_new(n+1,(n+1)*8)
 assert(numeric_buffer_status(&a)==0)
 var index:i64=0
 while index<6 {
  var k:i64=0
  if index==1 {k=1}
  if index==2 {k=n/2-1}
  if index==3 {k=n/2}
  if index==4 {k=n-2}
  if index==5 {k=n-1}
  let target=expected[index as usize]
  check(&!a,kind,n,k,0,target)
  check(&!a,kind,n,k,1,target)
  check(&!a,kind,n,k,n*12,target)
  index=index+1
 }
 numeric_buffer_close(a)
}
)SIO";
 std::vector<std::string>calls;
 for(int kind=0;kind<7;++kind){
  for(int n:std::vector<int>{1,2,3,4,5,7,8,9,16,25,31}){
   std::vector<double>a;for(int i=0;i<n;++i)a.push_back(value(kind,i,n));
   std::sort(a.begin(),a.end());
   auto name="case_"+std::to_string(kind)+"_"+std::to_string(n);calls.push_back(name);
   std::cout<<"fn "<<name<<"() with Alloc,Mut,Div,Panic {\n var expected:[f64;32]=[0.0;32]\n";
   for(int i=0;i<n;++i){std::cout<<" expected["<<i<<"]=";literal(a[i]);std::cout<<"\n";}
   std::cout<<" small("<<kind<<","<<n<<",&expected)\n}\n";
  }
  const int n=20000;std::vector<double>a;for(int i=0;i<n;++i)a.push_back(value(kind,i,n));
  std::sort(a.begin(),a.end());
  auto name="case_large_"+std::to_string(kind);calls.push_back(name);
  std::cout<<"fn "<<name<<"() with Alloc,Mut,Div,Panic {\n var expected:[f64;6]=[0.0;6]\n";
  const int ranks[6]={0,1,n/2-1,n/2,n-2,n-1};
  for(int i=0;i<6;++i){std::cout<<" expected["<<i<<"]=";literal(a[ranks[i]]);std::cout<<"\n";}
  std::cout<<" large("<<kind<<",&expected)\n}\n";
 }
 std::cout<<"fn main()->i32 with IO,Alloc,Mut,Div,Panic {\n";
 for(const auto&name:calls)std::cout<<" "<<name<<"()\n";
 std::cout<<" println(\"STDLIB_DEEP_NUMERIC_SELECT_REFERENCE_PASS\")\n 0\n}\n";
 return std::cout.good()?0:1;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
