// C++23 independent correlated-volume entropy oracle; compile with -lquadmath.
// Raw __float128 sensitivities, analytic covariance; no production scaling reuse.
#include <quadmath.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
using Q=__float128;
struct Case{std::string name;double n,un,vf,uf,vi,ui,rho;};
struct Result{int status=0;Q delta=0,u=0,l=0;};
bool representable(Q x){return std::isfinite((double)x)&&(x==0||(double)x!=0);}
Result evaluate(const Case&c){
 Result r;
 for(double v:{c.n,c.un,c.vf,c.uf,c.vi,c.ui,c.rho})if(!std::isfinite(v)){r.status=1;return r;}
 if(c.n<0||c.vf<=0||c.vi<=0){r.status=2;return r;}
 if(c.un<0||c.uf<0||c.ui<0){r.status=3;return r;}
 if(c.rho < -1||c.rho>1){r.status=5;return r;}
 Q R=(Q)8.31446261815324, vf=c.vf,vi=c.vi;
 Q rel=(vf-vi)/vi;
 Q l=fabsq(rel)<(Q).5?log1pq(rel):logq(vf)-logq(vi);
 Q d=(Q)c.n*R*l,a=R*fabsq(l)*(Q)c.un,b=(Q)c.n*R*(Q)c.uf/vf,e=(Q)c.n*R*(Q)c.ui/vi;
 // Equivalent positive decomposition of the covariance quadratic form.
 // In quad this resolves near rho=1 without cancellation of squared terms.
 Q cross=sqrtq(2*(1-(Q)c.rho)*b*e);
 Q u=sqrtq(a*a+(b-e)*(b-e)+cross*cross);
 for(Q v:{d,a,b,e,cross,u})if(!representable(v)){r.status=4;return r;}
 r.delta=d;r.u=u;r.l=l;return r;
}
std::string literal(double v){std::ostringstream o;o<<std::scientific<<std::setprecision(17);if(std::fpclassify(v)==FP_SUBNORMAL)o<<"("<<std::ldexp(v,52)<<" / 4503599627370496.0)";else o<<v;return o.str();}
std::vector<Case> cases(){
 double tiny=std::numeric_limits<double>::denorm_min(),next=std::nextafter(1.,2.),near=std::nextafter(1.,0.);
 return{
 {"independent",2,.1,2,.02,1,.01,0},
 {"positive",2,.1,2,.02,1,.01,.75},
 {"negative",2,.1,2,.02,1,.01,-.75},
 {"perfect_shared_scale",2,0,2,.02,1,.01,1},
 {"perfect_anticorrelation",2,0,2,.02,1,.01,-1},
 {"near_shared_scale",2,0,2,.02,1,.01,near},
 {"shared_with_amount",2,.1,2,.02,1,.01,1},
 {"unequal_sensitivity",2,0,2,.04,1,.01,1},
 {"swapped",2,.1,1,.01,2,.02,.75},
 {"unit_large",2,.1,2e200,2e198,1e200,1e198,.75},
 {"unit_small",2,.1,2e-200,2e-202,1e-200,1e-202,.75},
 {"uncertainty_scaled",2,.2,2,.04,1,.02,.75},
 {"retain_tiny_amount",1e150,1e-300,2,2e150,1,1e150,1},
 {"equal_nominal_shared",2,.1,1,.01,1,.01,1},
 {"zero_amount_uncertain",0,.1,4,.2,1,.1,.8},
 {"zero_known",0,0,4,.2,1,.1,-1},
 {"finite_nominal_nR_overflow",1e308,0,next,0,1,0,1},
 {"subnormal_volume",2,.1,2e-320,1e-321,1e-320,5e-322,.5},
 {"subnormal_amount",tiny,0,2,0,1,0,1},
 {"subnormal_amount_u",0,tiny,2,0,1,0,1},
 {"huge_contributions_cancel",1e150,0,2,2e150,1,1e150,1},
 {"huge_contributions_finite_norm",1e150,0,2,2e157,1,1e157,-1},
 {"huge_contributions_cross_overflow",1e150,0,2,4e157,1,2e157,-1},
 {"cross_underflow",tiny,0,1,1,1,1,near},
 {"small_vol_large_u",1e-310,0,1e-308,1e308,1e-308,0,-1},
 {"underflow_nominal",tiny,0,next,0,1,0,1},
 {"underflow_sensitivity",tiny,0,1,tiny,1,0,1},
 {"rho_high",1,0,2,0,1,0,1.01},
 {"rho_low",1,0,2,0,1,0,-1.01},
 {"physical_precedes_rho",-1,0,2,0,1,0,2},
 {"uncertainty_precedes_rho",1,-1,2,0,1,0,2},
 {"ordinary_compression",1,.01,1e-308,1e-310,1e308,1e306,-.5}
 };
}
int main(int argc,char**argv){
 bool fixture=argc>1&&std::string(argv[1])=="--fixture";
 auto all=cases();std::cout<<std::scientific<<std::setprecision(17);
 // Independent metamorphic properties on the raw quad oracle.
 auto a=evaluate(all[1]),b=evaluate(all[8]),s=evaluate(all[11]);
 if(a.status||b.status||s.status||fabsq(a.delta+b.delta)>fabsq(a.delta)*(Q)1e-29||fabsq(a.u-b.u)>a.u*(Q)1e-29||fabsq(s.u-2*a.u)>a.u*(Q)1e-29)return 2;
 if(evaluate(all[3]).u!=0||evaluate(all[12]).u<=0||evaluate(all[20]).u!=0)return 3;
 if(fixture)std::cout<<"//@ run-pass\n//@ expect-stdout: PHYSICS_ENTROPY_COVARIANCE_REFERENCE_PASS\n// Generated independent C++23 quad covariance oracle.\nuse physics::entropy::{isothermal_entropy_correlated_volumes,isothermal_entropy_independent}\nfn close(a:f64,b:f64)->bool {let d=if a>b{a-b}else{b-a} let v=if b<0.0{-b}else{b} return d<=v*2.0e-11+"<<literal(2*std::numeric_limits<double>::denorm_min())<<"}\n";
 for(size_t i=0;i<all.size();++i){
  auto c=all[i];auto r=evaluate(c);
  if(!fixture){std::cout<<i<<"\t"<<c.name<<"\t"<<r.status<<"\t"<<(double)r.delta<<"\t"<<(double)r.u<<"\t"<<(double)r.l<<"\n";continue;}
  std::cout<<"// "<<c.name<<"\nfn case_"<<i<<"() with Mut,Div,Panic {\n let r=isothermal_entropy_correlated_volumes("<<literal(c.n)<<","<<literal(c.un)<<","<<literal(c.vf)<<","<<literal(c.uf)<<","<<literal(c.vi)<<","<<literal(c.ui)<<","<<literal(c.rho)<<")\n assert(r.status=="<<r.status<<")\n assert(r.uncertainty_model==2 && r.first_order)\n";
  for(auto [name,v]:{std::pair{"delta_s",r.delta},std::pair{"standard_uncertainty",r.u},std::pair{"log_volume_ratio",r.l}}){
   if(v==0)std::cout<<" assert(r."<<name<<"==0.0)\n";
   else std::cout<<" assert(close(r."<<name<<","<<literal((double)v)<<"))\n";
  }
  std::cout<<"}\n";
 }
 if(fixture){
 std::cout<<"fn independent_identity() with Mut,Div,Panic {\n let a=isothermal_entropy_independent(2.0,0.1,2.0,0.02,1.0,0.01)\n let b=isothermal_entropy_correlated_volumes(2.0,0.1,2.0,0.02,1.0,0.01,0.0)\n assert(a.status==b.status && a.delta_s==b.delta_s && a.standard_uncertainty==b.standard_uncertainty && a.log_volume_ratio==b.log_volume_ratio)\n}\nfn main() with IO,Mut,Div,Panic {\n";
 for(size_t i=0;i<all.size();++i)std::cout<<" case_"<<i<<"()\n";
 std::cout<<" independent_identity()\n println(\"PHYSICS_ENTROPY_COVARIANCE_REFERENCE_PASS\")\n 0\n}\n";
 }
}
