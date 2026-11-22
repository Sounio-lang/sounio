#include <array>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
struct C{int mode;double x,c;};
int main(int argc,char**argv){
 if(argc!=2)return 2;
 std::ofstream o(argv[1]);o<<std::scientific<<std::setprecision(17);
 o<<"//@ run-pass\n//@ expect-stdout: SCALE_REFERENCE_PASS\n"
    "use viz::scale::{ScaleSpec,scale_forward,scale_inverse,axis_scaled_ticks}\n"
    "use math::pure::{ln_checked}\n"
    "fn near(a:f64,b:f64,s:f64)->bool with Div { let d=if a>b {a-b}else{b-a} d/s < 2.0e-11 }\n"
    ;
 std::vector<C> cases;
 for(double x:{-1e300,0.0,1e-300,1.0,1e300})cases.push_back({0,x,1});
 for(double x:{1e-300,1e-16,1.0,std::nextafter(1.0,2.0),10.0,1e16,1e300})cases.push_back({1,x,1});
 for(double c:{.1,1.0,1e-300,1e300})
  for(double x:{-1e300,-1.0,-1e-16,-1e-300,0.0,1e-300,1e-16,1.0,1e300})cases.push_back({2,x,c});
 cases.push_back({2,1e308,1e-308});cases.push_back({2,-1e308,1e-308});
 cases.push_back({2,1e308,1e308});
 int k=0;
 for(auto c:cases){
  long double x=c.x,constant=c.c;
  long double t=c.mode==0?x:c.mode==1?std::log10(x):std::copysign(std::log1p(std::abs(x)/constant),x);
  double expected=(double)t;
  long double quantum=std::numeric_limits<double>::denorm_min();
  long double forward_unit=std::max(std::abs(t),2*quantum/2e-11L);
  long double inverse_unit=std::abs(x);
  if(c.mode==2)inverse_unit=std::max(inverse_unit,2*quantum*(constant+std::abs(x))/2e-11L);
  if(x==0)inverse_unit=1;
  if(std::fpclassify(expected)==FP_SUBNORMAL){
    long double restored=std::copysign(std::expm1(std::abs((long double)expected))*constant,x);
    std::cout<<std::scientific<<std::setprecision(17)<<"subnormal_case="<<k
     <<" coordinate="<<expected<<" rounded_coordinate_roundtrip_error="<<(double)std::abs(restored-x)
     <<" propagated_two_ulp_bound="<<(double)(2*quantum*(constant+std::abs(x)))<<"\n";
  }
  o<<"fn scale_case_"<<k<<"() with IO, Mut, Div, Panic {\n"
   <<"    println(\"SCALE_CASE_"<<k<<"\")\n"
   <<"    let s"<<k<<"=ScaleSpec{mode:"<<c.mode<<",constant:"<<c.c<<"}\n"
   <<"    let f"<<k<<"=scale_forward("<<c.x<<",&s"<<k<<")\n";
  if(expected==0&&x!=0&&c.mode!=1)o<<"    assert(f"<<k<<".status == 4)\n";
  else{
   o<<"    assert(f"<<k<<".status == 0)\n";
   o<<"    assert(near(f"<<k<<".value,"<<expected<<","<<(double)(expected==0?1.0L:forward_unit)<<"))\n";
   o<<"    let r"<<k<<"=scale_inverse(f"<<k<<".value,&s"<<k<<")\n"
    <<"    assert(r"<<k<<".status == 0)\n"
    <<"    assert(near(r"<<k<<".value,"<<c.x<<","<<(double)inverse_unit<<"))\n";
  }
  o<<"}\n";k++;
 }
 o<<"fn main() with IO, Mut, Div, Panic {\n";
 for(int i=0;i<k;i++)o<<"    scale_case_"<<i<<"()\n";
 o<<"    let lin_spec=ScaleSpec{mode:0,constant:1.0}\n"
    "    let log_spec=ScaleSpec{mode:1,constant:1.0}\n"
    "    let sym_spec=ScaleSpec{mode:2,constant:1.0}\n"
    "    let bad=ScaleSpec{mode:2,constant:0.0}\n"
    "    assert(scale_forward(0.0,&log_spec).status == 3)\n"
    "    assert(scale_forward(-1.0,&log_spec).status == 3)\n"
    "    assert(scale_forward(1.0,&bad).status == 1)\n"
    "    assert(scale_inverse(1.0e308,&log_spec).status == 4)\n"
    "    assert(scale_inverse(1.0e308,&sym_spec).status == 4)\n"
    "    assert(ln_checked(0.0).status == 1 && ln_checked(-1.0).status == 1)\n"
    "    assert(ln_checked(1.0).status == 0 && ln_checked(1.0).value == 0.0)\n"
    "    var huge=1.0e308\n    let inf=huge*10.0\n"
    "    assert(scale_forward(inf,&lin_spec).status == 2)\n"
    "    assert(scale_forward(inf-inf,&sym_spec).status == 2)\n"
    "    assert(ln_checked(inf).status == 2)\n"
    "    let lt=axis_scaled_ticks(1.0e-6,1.0e6,6,&log_spec)\n"
    "    assert(lt.status == 0 && lt.n == 7)\n";
 for(int i=0;i<=6;i++){long double v=std::pow(10.0L,-6+2*i);o<<"    assert(near(lt.values["<<i<<"],"<<(double)v<<","<<(double)v<<"))\n";}
 o<<"    let st=axis_scaled_ticks(-99.0,99.0,4,&sym_spec)\n"
    "    assert(st.status == 0 && st.n == 5)\n";
 for(int i=0;i<=4;i++){
  long double t=(i-2)*std::log(100.0L)/2;
  long double v=std::copysign(std::expm1(std::abs(t)),t);
  o<<"    assert(near(st.values["<<i<<"],"<<(double)v<<","<<(v==0?1.0:(double)std::abs(v))<<"))\n";
 }
 o<<"    assert(axis_scaled_ticks(0.0,100.0,4,&log_spec).status != 0)\n"
    "    println(\"SCALE_REFERENCE_PASS\")\n    0\n}\n";
 std::cout<<"C++23 log10l/log1pl: "<<cases.size()<<" transforms/round trips plus log and symlog raw-unit ticks.\n";
 return !o;
}
