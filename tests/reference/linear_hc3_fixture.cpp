#define main hc3_raw_report_main
#include "linear_hc3_reference.cpp"
#undef main
#include <algorithm>
#include <bit>
void val(double x){if(std::isnan(x))std::cout<<"(0.0/0.0)";else if(std::isinf(x))std::cout<<(x<0?"(-1.0e308*10.0)":"(1.0e308*10.0)");else std::cout<<std::scientific<<std::setprecision(17)<<x;}
int main(){
 std::cout<<R"SIO(//@ run-pass
//@ expect-stdout: STATS_LINEAR_HC3_REFERENCE_PASS
// Independent C++23 quad reference. No absolute1+abs floor.
// Off-diagonal tolerance uses SE_i*SE_j as its covariance unit scale.
use stats::regression::robust_covariance::{LinearHc3Result,multiple_linear_hc3}
use math::pure::{is_finite,sqrt}
fn magnitude(x:f64)->f64 {if x<0.0 {-x}else{x}}
fn relative(a:f64,b:f64,tolerance:f64) with Mut,Div,Panic {
 assert(is_finite(a) && is_finite(b))
 if b==0.0 {assert(a==0.0)} else {
  assert(a!=0.0)
  assert(magnitude((a-b)/b)<=tolerance)
 }
}
fn matrix_close(a:f64,b:f64,unit:f64) with Mut,Div,Panic {
 assert(is_finite(a) && is_finite(b))
 if unit==0.0 {assert(a==0.0 && b==0.0)} else {assert(magnitude((a-b)/unit)<=1.0e-7)}
}
fn verify(r:&LinearHc3Result,beta:&[f64;8],cov:&[f64;64],se:&[f64;8],h:&[f64;256],n:i64,p:i64,fs:i64,cs:i64,perfect:bool) with Mut,Div,Panic {
 assert(r.fit.status==fs)
 assert(r.covariance_status==cs)
 assert(r.method==3 && !r.assumptions_assessed)
 if fs==0 {
  var j:i64=0
  while j<p {let actual:f64=r.fit.coefficients[j as usize] let wanted:f64=beta[j as usize] relative(actual,wanted,1.0e-9) j=j+1}
 }
 var i:i64=0
 while i<8 {
  if cs==0 && i<p && !perfect {
   let actual:f64=r.standard_errors[i as usize]
   let wanted:f64=se[i as usize]
   relative(actual,wanted,1.0e-7)
  } else if cs!=0 || i>=p {assert(r.standard_errors[i as usize]==0.0)}
  var j:i64=0
  while j<8 {
   let actual:f64=r.covariance[(i*8+j) as usize]
   if cs==0 && i<p && j<p && !perfect {
    let wanted:f64=cov[(i*8+j) as usize]
    let unit:f64=se[i as usize]*se[j as usize]
    matrix_close(actual,wanted,unit)
    assert(actual==r.covariance[(j*8+i) as usize])
    if i==j && wanted>0.0 {relative(actual,wanted,1.0e-7)}
   } else if cs!=0 || i>=p || j>=p {assert(actual==0.0)}
   j=j+1
  }
  i=i+1
 }
 i=0
 while i<256 {
  if cs==0 && i<n {let actual:f64=r.leverage[i as usize] let wanted:f64=h[i as usize] relative(actual,wanted,1.0e-9)}
  else {assert(r.leverage[i as usize]==0.0)}
  i=i+1
 }
}
// Closed form conditional on the actual reported residuals. Coefficients
// are independently checked against quad above; no perfect-data residual cutoff.
fn perfect_reported(r:&LinearHc3Result) with Mut,Div,Panic {
 var v00:f64=0.0
 var v01:f64=0.0
 var v11:f64=0.0
 var i:i64=0
 while i<5 {
  let centered=(i as f64)-2.0
  let h=0.2+centered*centered/10.0
  let residual:f64=r.fit.residuals[i as usize]
  let a0=(0.2-2.0*centered/10.0)*residual/(1.0-h)
  let a1=(centered/10.0)*residual/(1.0-h)
  v00=v00+a0*a0
  v01=v01+a0*a1
  v11=v11+a1*a1
  i=i+1
 }
 relative(r.covariance[0],v00,1.0e-7)
 relative(r.covariance[9],v11,1.0e-7)
 matrix_close(r.covariance[1],v01,sqrt(v00)*sqrt(v11))
 relative(r.standard_errors[0],sqrt(v00),1.0e-7)
 relative(r.standard_errors[1],sqrt(v11),1.0e-7)
}
)SIO";
 auto all=cases();
 for(std::size_t id=0;id<all.size();id++){
  const auto&c=all[id];const auto r=evaluate(c);bool perfect=!c.analytic_beta.empty();
  std::cerr<<id<<" "<<c.name<<" fit="<<r.fit_status<<" cov="<<r.covariance_status<<"\n";
  std::cout<<"// "<<c.name<<"\nfn case_"<<id<<"() with Mut,Div,Panic {\n var a:[f64;2048]=[0.0;2048]\n var b:[f64;256]=[0.0;256]\n";
  if(c.name=="max_256_by_8"){
   std::cout<<R"SIO( var row:i64=0
 while row<256 {
  var response=((row*7)%13-6) as f64
  response=response*0.125
  var j:i64=0
  while j<8 {
   var ti=row
   var tj=j
   var parity:i64=0
   while tj>0 {
    if ti%2==1 && tj%2==1 {parity=1-parity}
    ti=ti/2
    tj=tj/2
   }
   let x=if parity==1 {-1.0}else{1.0}
   a[(row*8+j) as usize]=x
   response=response+x*((j+1) as f64)
   j=j+1
  }
  b[row as usize]=response
  row=row+1
 }
)SIO";
  }else{
   for(int i=0;i<c.n;i++){for(int j=0;j<c.p;j++){std::cout<<" a["<<i*8+j<<"]=";val(c.a[i*8+j]);std::cout<<"\n";}std::cout<<" b["<<i<<"]=";val(c.b[i]);std::cout<<"\n";}
   if(c.name=="inactive_nonfinite")std::cout<<" a[2047]=0.0/0.0\n b[255]=1.0e308*10.0\n";
  }
  std::cout<<" var beta:[f64;8]=[0.0;8]\n var cov:[f64;64]=[0.0;64]\n var se:[f64;8]=[0.0;8]\n var h:[f64;256]=[0.0;256]\n";
  if(r.fit_status==0)for(int j=0;j<c.p;j++){std::cout<<" beta["<<j<<"]=";val(double(r.beta[j]));std::cout<<"\n";}
  if(r.covariance_status==0){
   if(c.name=="max_256_by_8"){
    for(Q value:r.leverage)if(std::bit_cast<std::uint64_t>(double(value))!=std::bit_cast<std::uint64_t>(double(r.leverage[0])))return 5;
    std::cout<<" var i:i64=0\n while i<256 {h[i as usize]=";val(double(r.leverage[0]));std::cout<<" i=i+1}\n";
   }else for(int i=0;i<c.n;i++){std::cout<<" h["<<i<<"]=";val(double(r.leverage[i]));std::cout<<"\n";}
   if(!perfect)for(int j=0;j<c.p;j++){std::cout<<" se["<<j<<"]=";val(double(r.standard_errors[j]));std::cout<<"\n";for(int k=0;k<c.p;k++){std::cout<<" cov["<<j*8+k<<"]=";val(double(r.covariance[j][k]));std::cout<<"\n";}}
  }
  std::cout<<" let r=multiple_linear_hc3(&a,&b,"<<c.n<<","<<c.p<<",";val(c.rcond);std::cout<<")\n verify(&r,&beta,&cov,&se,&h,"<<c.n<<","<<c.p<<","<<r.fit_status<<","<<r.covariance_status<<","<<(perfect?"true":"false")<<")\n";
  if(perfect)std::cout<<" perfect_reported(&r)\n";
  std::cout<<"}\n";
 }
 std::cout<<"fn main()->i32 with IO,Mut,Div,Panic {\n";for(std::size_t i=0;i<all.size();i++)std::cout<<" case_"<<i<<"()\n";std::cout<<" println(\"STATS_LINEAR_HC3_REFERENCE_PASS\")\n 0\n}\n";
 std::cerr<<"cases="<<all.size()<<"\n";
}
