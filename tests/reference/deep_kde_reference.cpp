// Independent binary128 raw Gaussian sums; does not call production logsumexp.
// Emit Sounio fixture. Build on authorized worker: C++23 -O2 -lquadmath.
#include <quadmath.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using Q=__float128;
struct Case {int kind,n,g;double h;};
static double observation(int kind,int i){
 if(kind==10)return std::numeric_limits<double>::denorm_min();
 if(kind==11)return 1;
 if(kind==0)return double(i-1);
 if(kind==2)return -std::numeric_limits<double>::max();
 if(kind==5)return double(i%11-5)/3;
 if(kind==6)return 1;
 if(kind==7)return i==0?-std::numeric_limits<double>::max():std::numeric_limits<double>::max();
 return 0;
}
static double point(int kind,int i,double h){
 if(kind==0)return double(i-2);
 if(kind==1)return i==0?0:double(i==1?40:50)*h;
 if(kind==2||kind==4)return std::numeric_limits<double>::max();
 if(kind==5)return double(i-32)/4;
 if(kind==6)return double(i+1);
 if(kind==9)return 40;
 return 0;
}
int main(){
 try{
 std::cout<<std::setprecision(17)<<std::showpoint;
 std::cout<<R"SIO(//@ run-pass
//@ expect-stdout: STDLIB_DEEP_KDE_REFERENCE_PASS
use collections::numeric_buffer::{numeric_buffer_new,numeric_buffer_status,numeric_buffer_at,numeric_buffer_set,numeric_buffer_close}
use collections::integer_buffer::{integer_buffer_new,integer_buffer_status,integer_buffer_at,integer_buffer_set,integer_buffer_close}
use stats::dynamic_kde::gaussian_kde_dynamic
fn ab(x:f64)->f64 {if x<0.0 {-x} else {x}}
fn density_near(a:f64,b:f64)->bool with Div {
 if b==0.0 {return a==0.0}
 if a<=0.0 {return false}
 let quantum=2.2250738585072014e-308/4503599627370496.0
 ab(a-b)<=ab(b)*1.0e-10+2.0*quantum
}
fn run_case(kind:i64,n:i64,g:i64,h:f64,expected:&[f64;64],logs:&[f64;64],bits:&[i64;64]) with IO,Alloc,Mut,Div,Panic {
 var observations=numeric_buffer_new(n,n*8)
 var mask=integer_buffer_new(n,n*8)
 var grid=numeric_buffer_new(g,g*8)
 var density=numeric_buffer_new(g,g*8)
 var log_density=numeric_buffer_new(g,g*8)
 var flags=integer_buffer_new(g,g*8)
 var work=numeric_buffer_new(2*g,2*g*8)
 var flag_work=integer_buffer_new(g,g*8)
 assert(numeric_buffer_status(&observations)==0 && integer_buffer_status(&mask)==0 && numeric_buffer_status(&grid)==0 && numeric_buffer_status(&density)==0 && numeric_buffer_status(&log_density)==0 && integer_buffer_status(&flags)==0 && numeric_buffer_status(&work)==0 && integer_buffer_status(&flag_work)==0)
 var i:i64=0
 while i<n {
  var v:f64=0.0
  if kind==10 {v=2.2250738585072014e-308/4503599627370496.0}
  if kind==11 {v=1.0}
  if kind==0 {v=(i-1) as f64}
  if kind==2 {v=-1.7976931348623157e308}
  if kind==5 {v=((i%11-5) as f64)/3.0}
  if kind==6 {v=1.0}
  if kind==7 {v=if i==0 {-1.7976931348623157e308} else {1.7976931348623157e308}}
  numeric_buffer_set(&!observations,i,v)
  integer_buffer_set(&!mask,i,1)
  i=i+1
 }
 if kind==8 {numeric_buffer_set(&!observations,1,0.0/0.0);integer_buffer_set(&!mask,1,0)}
 i=0
 while i<g {
  var v:f64=0.0
  if kind==0 {v=(i-2) as f64}
  if kind==1 {if i>0 {v=if i==1 {40.0*h} else {50.0*h}}}
  if kind==2 || kind==4 {v=1.7976931348623157e308}
  if kind==5 {v=((i-32) as f64)/4.0}
  if kind==6 {v=(i+1) as f64}
  if kind==9 {v=40.0}
  numeric_buffer_set(&!grid,i,v);i=i+1
 }
 let result=gaussian_kde_dynamic(&observations,&mask,&grid,h,n*g,&!density,&!log_density,&!flags,&!work,&!flag_work)
 assert(result.status==0)
 let selected=if kind==8 {n-1} else {n}
 assert(result.selected==selected && result.pairs==selected*g)
 i=0
 while i<g {
  let f=integer_buffer_at(&flags,i)
  assert(f==bits[i as usize])
  let pdf=numeric_buffer_at(&density,i)
  let logpdf=numeric_buffer_at(&log_density,i)
  if (f & 1)!=0 {assert(density_near(pdf,expected[i as usize]))}
  else {assert(pdf==0.0)}
  if (f & 2)!=0 {
   let target=logs[i as usize]
   assert(ab(logpdf-target)<=ab(target)*1.0e-11)
  } else {assert(logpdf==0.0)}
  i=i+1
 }
 numeric_buffer_close(observations);integer_buffer_close(mask);numeric_buffer_close(grid)
 numeric_buffer_close(density);numeric_buffer_close(log_density);integer_buffer_close(flags)
 numeric_buffer_close(work);integer_buffer_close(flag_work)
}
fn main()->i32 with IO,Alloc,Mut,Div,Panic {
)SIO";
 const double max=std::numeric_limits<double>::max(),tiny=std::numeric_limits<double>::denorm_min();
 std::vector<Case> cases={{0,3,5,1},{1,1,3,1e-300},{2,1,1,max},{3,1,1,tiny},{4,1,1,tiny},{5,1024,64,.3},{6,3,3,2},{7,2,1,max},{8,2,1,1},{9,1,1,1},{10,1,1,max},{11,1,1,tiny}};
 for(const auto&c:cases){
  std::cout<<" {\n var expected:[f64;64]=[0.0;64]\n var logs:[f64;64]=[0.0;64]\n var bits:[i64;64]=[0;64]\n";
  for(int j=0;j<c.g;++j){
   Q sum=0;int selected=c.kind==8?1:c.n;
   Q x=Q(point(c.kind,j,c.h));
   for(int i=0;i<selected;++i){Q z=(Q(observation(c.kind,i))-x)/Q(c.h);sum+=expq(-z*z/2);}
   Q value=sum/(Q(selected)*Q(c.h)*sqrtq(2*acosq(Q(-1))));
   Q log_value=logq(value);
   double pdf=double(value),logpdf=double(log_value);
   int flags=0;
   if(std::isfinite(logpdf)){flags=2;if(std::isfinite(pdf)&&pdf>0)flags=3;}
   if(flags&1)std::cout<<" expected["<<j<<"]="<<pdf<<"\n";
   if(flags&2)std::cout<<" logs["<<j<<"]="<<logpdf<<"\n";
   std::cout<<" bits["<<j<<"]="<<flags<<"\n";
  }
  std::string h=c.kind==3||c.kind==4||c.kind==11?"(2.2250738585072014e-308/4503599627370496.0)":"";
  std::cout<<" run_case("<<c.kind<<","<<c.n<<","<<c.g<<",";
  if(h.empty())std::cout<<c.h;else std::cout<<h;
  std::cout<<",&expected,&logs,&bits)\n }\n";
 }
 std::cout<<" println(\"STDLIB_DEEP_KDE_REFERENCE_PASS\")\n 0\n}\n";
 return std::cout.good()?0:1;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
