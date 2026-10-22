// C++23 quad independent raw Jacobian/Gram-matrix Lorentz reference.
// Compile on t560: g++ -std=c++23 -O2 lorentz_reference.cpp -lquadmath
#include <quadmath.h>
#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
using Q=__float128;using V=std::array<double,3>;using W=std::array<Q,3>;
struct Case{std::string name;double q,uq;V e,ue,v,uv,b,ub;};
struct Result{int status=0;W f{},u{};std::array<Q,9>cov{};Q vector=0;};
bool finite(Q x){return std::isfinite((double)x);}
bool strict(Q x){return finite(x)&&(x==0||(double)x!=0);}
W cross(W a,W b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
W quad(V a){return{(Q)a[0],(Q)a[1],(Q)a[2]};}
Result evaluate(const Case&c){
 auto error=[](int s){Result r;r.status=s;return r;};
 if(!std::isfinite(c.q)||!std::isfinite(c.uq))return error(1);
 for(auto a:{c.e,c.ue,c.v,c.uv,c.b,c.ub})for(double x:a)if(!std::isfinite(x))return error(1);
 if(c.uq<0)return error(3);
 for(auto a:{c.ue,c.uv,c.ub})for(double x:a)if(x<0)return error(3);
 auto e=quad(c.e),v=quad(c.v),b=quad(c.b),vb=cross(v,b);
 Result r;std::array<W,10> h{};
 for(int i=0;i<3;++i){r.f[i]=(Q)c.q*(e[i]+vb[i]);h[0][i]=(Q)c.uq*(e[i]+vb[i]);if(!finite(r.f[i]))return error(4);}
 for(int j=0;j<3;++j){
  W basis{};basis[j]=1;
  auto dv=cross(basis,b),db=cross(v,basis);
  h[1+j][j]=(Q)c.q*(Q)c.ue[j];
  for(int i=0;i<3;++i){h[4+j][i]=(Q)c.q*dv[i]*(Q)c.uv[j];h[7+j][i]=(Q)c.q*db[i]*(Q)c.ub[j];}
 }
 for(auto a:h)for(Q x:a)if(!strict(x))return error(4);
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){
  Q sum=0;for(auto a:h){Q term=a[i]*a[j];if(!strict(term))return error(4);sum+=term;}
  if(!strict(sum))return error(4);
  r.cov[i*3+j]=sum;
 }
 for(int i=0;i<3;++i)r.u[i]=sqrtq(r.cov[i*3+i]);
 r.vector=sqrtq(r.cov[0]+r.cov[4]+r.cov[8]);if(!strict(r.vector))return error(4);
 return r;
}
struct Scalar{std::string name;double q,uq,e,ue,v,uv,b,ub;};
struct SR{int status=0;Q f=0,u=0;};
SR evals(Scalar c){
 for(double x:{c.q,c.uq,c.e,c.ue,c.v,c.uv,c.b,c.ub})if(!std::isfinite(x))return{1,0,0};
 for(double x:{c.uq,c.ue,c.uv,c.ub})if(x<0)return{3,0,0};
 Q g=(Q)c.e+(Q)c.v*(Q)c.b;
 Q f=(Q)c.q*g;std::array<Q,4>h={(Q)c.uq*g,(Q)c.q*c.ue,(Q)c.q*c.b*c.uv,(Q)c.q*c.v*c.ub};
 if(!finite(f))return{4,0,0};
 Q sum=0;for(Q x:h){if(!strict(x))return{4,0,0};sum+=x*x;}
 Q u=sqrtq(sum);if(!strict(u))return{4,0,0};return{0,f,u};
}
std::string lit(double x){std::ostringstream o;o<<std::scientific<<std::setprecision(17);if(std::fpclassify(x)==FP_SUBNORMAL)o<<"("<<std::ldexp(x,52)<<" / 4503599627370496.0)";else o<<x;return o.str();}
void arr(std::string name,V a){std::cout<<" let "<<name<<":[f64;3]=[";for(int i=0;i<3;++i)std::cout<<(i?",":"")<<lit(a[i]);std::cout<<"]\n";}
int main(int argc,char**argv){
 bool fixture=argc>1&&std::string(argv[1])=="--fixture";V z{};
 Case a{"ordinary",2,.1,{1,-2,3},{.2,.3,.4},{2,-1,3},{.01,.02,.03},{-.5,1,2},{.04,.05,.06}};
 std::vector<Case> cases={a};
 auto add=[&](std::string n,double q,double uq,V e,V ue,V v,V uv,V b,V ub){cases.push_back({n,q,uq,e,ue,v,uv,b,ub});};
 add("exact",2,0,{1,2,3},z,z,z,z,z);
 add("zero_q_live_uq",0,1,{1,-2,3},z,z,z,z,z);
 add("isotropic_e",2,0,{1,2,3},{1,1,1},z,z,z,z);
 add("velocity_only",2,0,z,z,{1,2,3},{.1,.2,.3},{2,-3,4},z);
 add("magnetic_only",-2,0,z,z,{1,2,3},z,{2,-3,4},{.1,.2,.3});
 add("parallel_nominal",1,1,z,z,{1,2,3},z,{2,4,6},z);
 add("zero_all",0,0,z,z,z,z,z,z);
 add("large_intermediate_finite",1e-200,0,z,z,{0,1e200,0},z,{0,0,1e200},z);
 add("huge_cross_cancellation",1,0,{1,0,0},z,{0,1e200,1e200},z,{0,1e200,1e200},z);
 add("huge_cross_charge_cancellation",1,1e-150,{1,0,0},z,{0,1e200,1e200},z,{0,1e200,1e200},z);
 add("force_final_underflow",1e-100,0,{1e-300,0,0},z,z,z,z,z);
 add("tiny_covariance",1,0,z,{1e-150,2e-150,3e-150},z,z,z,z);
 add("covariance_underflow",1,0,z,{1e-200,0,0},z,z,z,z);
 add("covariance_overflow",1,0,z,{1e200,0,0},z,z,z,z);
 add("force_overflow",1e200,0,{1e200,0,0},z,z,z,z,z);
 add("negative_standard_u",1,0,z,{-1,0,0},z,z,z,z);
 add("signed_charge",-2,.1,{1,-2,3},{.2,.3,.4},{2,-1,3},{.01,.02,.03},{-.5,1,2},{.04,.05,.06});
 for(int k=0;k<8;++k)add("integer_gram_"+std::to_string(k),k-3,.25,{double(k+1),-2,3},{.5,.25,.125},{1,double(k-2),-3},{.125,.25,.5},{-1,2,double(k+1)},{.25,.125,.5});
 std::vector<Scalar> sc={
 {"scalar_ordinary",2,.1,1,.2,2,.3,3,.4},
 {"shared_q_once",1,1,1,0,2,0,3,0},
 {"shared_q_cancel",1,1,-6,0,2,0,3,0},
 {"zero_q_uncertain",0,1,1,0,2,0,3,0},
 {"signed", -2,0,1,0,2,0,3,0},
 {"negative_inputs",2,.1,-1,.2,-2,.3,-3,.4},
 {"only_e_zero_nominal",2,0,0,.5,0,0,0,0},
 {"tiny_scalar_u",1,0,0,1e-300,0,0,0,0},
 {"large_intermediate",1e-200,0,0,0,1e200,0,1e200,0},
 {"tiny_force_rounds0",1e-100,0,1e-300,0,0,0,0,0},
 {"scalar_overflow",1e200,0,1e200,0,0,0,0,0},
 {"scalar_underflow_h",1e-200,0,0,1e-200,0,0,0,0}
 };
 if(fixture)std::cout<<"//@ run-pass\n//@ expect-stdout: PHYSICS_LORENTZ_REFERENCE_PASS\n// Independent C++23 raw quad Jacobian/Gram reference.\nuse physics::lorentz::{lorentz_force_independent,lorentz_force_scalar_independent}\nfn close(a:f64,b:f64,at:f64)->bool {let d=if a>b{a-b}else{b-a} let v=if b<0.0{-b}else{b} return d<=at+v*3.0e-11}\n";
 for(size_t n=0;n<cases.size();++n){auto c=cases[n];auto r=evaluate(c);
  if(!fixture){std::cout<<n<<"\t"<<c.name<<"\t"<<r.status<<"\t"<<lit((double)r.vector)<<"\n";continue;}
  std::cout<<"fn vector_"<<n<<"() with Mut,Div,Panic {\n";arr("e",c.e);arr("ue",c.ue);arr("v",c.v);arr("uv",c.uv);arr("b",c.b);arr("ub",c.ub);
  std::cout<<" let r=lorentz_force_independent("<<lit(c.q)<<","<<lit(c.uq)<<",&e,&ue,&v,&uv,&b,&ub)\n assert(r.status=="<<r.status<<")\n assert(r.uncertainty_model==1 && r.first_order)\n";
  auto emit=[&](std::string f,Q x,Q at=0){if(x==0)std::cout<<" assert(r."<<f<<"==0.0)\n";else std::cout<<" assert(close(r."<<f<<","<<lit((double)x)<<","<<lit((double)at)<<"))\n";};
  for(int i=0;i<3;++i){emit("force["+std::to_string(i)+"]",r.f[i]);emit("component_standard_uncertainties["+std::to_string(i)+"]",r.u[i],2*std::numeric_limits<double>::denorm_min());}
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)emit("covariance["+std::to_string(i*3+j)+"]",r.cov[i*3+j],2*std::numeric_limits<double>::denorm_min());
  emit("vector_standard_uncertainty",r.vector,2*std::numeric_limits<double>::denorm_min());
  // Moderate cases independently check each principal 2x2 minor's PSD bound.
  if(r.status==0&&n<8)for(int i=0;i<3;++i)for(int j=i+1;j<3;++j)std::cout<<" assert(r.covariance["<<i*3+i<<"]*r.covariance["<<j*3+j<<"]>=r.covariance["<<i*3+j<<"]*r.covariance["<<i*3+j<<"]*(1.0-1.0e-12))\n";
  std::cout<<"}\n";
 }
 for(size_t n=0;n<sc.size();++n){auto c=sc[n];auto r=evals(c);
  if(!fixture){std::cout<<"scalar"<<n<<"\t"<<c.name<<"\t"<<r.status<<"\t"<<lit((double)r.f)<<"\t"<<lit((double)r.u)<<"\n";continue;}
  std::cout<<"fn scalar_"<<n<<"() with Mut,Div,Panic {\n let r=lorentz_force_scalar_independent(";
  bool first=true;for(double x:{c.q,c.uq,c.e,c.ue,c.v,c.uv,c.b,c.ub}){std::cout<<(first?"":",")<<lit(x);first=false;}
  std::cout<<")\n assert(r.status=="<<r.status<<" && r.uncertainty_model==1 && r.first_order)\n";
  for(auto [name,x]:{std::pair{"force",r.f},std::pair{"standard_uncertainty",r.u}})if(x==0)std::cout<<" assert(r."<<name<<"==0.0)\n";else std::cout<<" assert(close(r."<<name<<","<<lit((double)x)<<","<<lit(2*std::numeric_limits<double>::denorm_min())<<"))\n";
  std::cout<<"}\n";
 }
 if(fixture){std::cout<<"fn main() with IO,Mut,Div,Panic {\n";for(size_t n=0;n<cases.size();++n)std::cout<<" vector_"<<n<<"()\n";for(size_t n=0;n<sc.size();++n)std::cout<<" scalar_"<<n<<"()\n";std::cout<<" println(\"PHYSICS_LORENTZ_REFERENCE_PASS\")\n 0\n}\n";}
}
