// C++23 independent ideal-gas isothermal entropy/first-order oracle.
// All input literals are binary64; intermediate oracle arithmetic is long double.
// R is the exact binary64 value shipped by the API. Correlations excluded.
// Usage: entropy_reference --fixture > physics_entropy_reference.sio
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <utility>
#include <sstream>
struct Case {std::string name;double n,un,vf,uvf,vi,uvi;};
struct Result {int status=0;long double delta=0,u=0,logratio=0;};
Result eval(const Case& c) {
 Result r;
 for(double v:{c.n,c.un,c.vf,c.uvf,c.vi,c.uvi})if(!std::isfinite(v)){r.status=1;return r;}
 if(c.n<0||c.vf<=0||c.vi<=0){r.status=2;return r;}
 if(c.un<0||c.uvf<0||c.uvi<0){r.status=3;return r;}
 const long double R=(long double)8.31446261815324;
 long double vf=c.vf,vi=c.vi;
 long double rel=(vf-vi)/vi;
 long double L=std::abs(rel)<.5L?std::log1p(rel):std::log(vf)-std::log(vi);
 long double delta=(long double)c.n*R*L;
 long double a=R*L*(long double)c.un;
 long double b=(long double)c.n*R*(long double)c.uvf/vf;
 long double d=(long double)c.n*R*(long double)c.uvi/vi;
 long double u=std::hypot(a,b,d);
 for(long double v:{delta,a,b,d,u}) {
  double rounded=(double)v;
  if(!std::isfinite(rounded)||(v!=0&&rounded==0)){r.status=4;return r;}
 }
 r.delta=delta;r.u=u;r.logratio=L;return r;
}
std::vector<Case> cases() {
 const double next=std::nextafter(1.,2.);
 const double tiny=std::numeric_limits<double>::denorm_min();
 const double max=std::numeric_limits<double>::max();
 return {
 {"ordinary_expansion",2,.1,2,.02,1,.01},
 {"ordinary_contraction",2,.1,1,.01,2,.02},
 {"equal_volumes_nonzero_u",2,.1,1,.02,1,.01},
 {"zero_amount_uncertain",0,.1,4,.02,1,.01},
 {"zero_amount_known",0,0,4,.02,1,.01},
 {"overflowing_nR_finite_delta",1e308,0,next,0,1,0},
 {"overflowing_nR_finite_u",1e308,1e307,next,0,1,0},
 {"overflowing_uv_over_v_finite_u",1e-310,0,1e-308,1e308,1e-308,0},
 {"subnormal_volumes",2,.1,2e-320,1e-321,1e-320,5e-322},
 {"opposite_volume_extremes",1,.01,1e308,1e306,1e-308,1e-310},
 {"tiny_representable_delta",1e-308,0,next,0,1,0},
 {"subnormal_amount",tiny,0,2,0,1,0},
 {"overflow_delta",max,0,2,0,1,0},
 {"overflow_amount_uncertainty",0,max,4,0,1,0},
 {"overflow_volume_uncertainty",1,0,tiny,max,tiny,0},
 {"uncertainty_times_three",2,.3,2,.06,1,.03},
 {"volume_units_large",2,.1,2e200,2e198,1e200,1e198},
 {"volume_units_small",2,.1,2e-200,2e-202,1e-200,1e-202},
 {"large_amount_equal_volumes",1e308,0,1,0,1,0},
 {"large_amount_equal_finite_u",1e308,0,1,1e-308,1,0},
 {"underflow_delta",tiny,0,next,0,1,0},
 {"underflow_volume_contribution",tiny,0,1,tiny,1,0},
 {"underflow_amount_contribution",0,tiny,next,0,1,0},
 {"subnormal_uncertainty",0,tiny,2,0,1,0},
 {"zero_volume",1,0,0,0,1,0},
 {"negative_amount",-1,0,2,0,1,0},
 {"negative_uncertainty",1,-.1,2,0,1,0},
 {"zero_equal_volumes",0,0,1,0,1,0},
 {"tiny_volume_ratio_compression",1,.01,1e-308,1e-310,1e308,1e306},
 {"underflow_volume_u_with_valid_delta",1e-308,0,next,1e-308,1,0},
 {"hypot_prevents_square_overflow",1e150,0,1,.1,1,.1}
 };
}
// The pinned compiler's decimal parser flushes subnormal literals to zero.
// Express each finite subnormal as an exact normal numerator divided by 2^52.
std::string literal(double value) {
 std::ostringstream out;out<<std::scientific<<std::setprecision(17);
 if(std::fpclassify(value)==FP_SUBNORMAL)out<<"("<<std::ldexp(value,52)<<" / 4503599627370496.0)";
 else out<<value;
 return out.str();
}
int main(int argc,char**argv) {
 bool fixture=argc>1&&std::string(argv[1])=="--fixture";
 auto all=cases();std::cout<<std::scientific<<std::setprecision(17);
 // Analytic invariances independently check the selected oracle cases.
 auto base=eval(all[0]),reverse=eval(all[1]),scale=eval(all[15]);
 if(base.status||reverse.status||scale.status)return 2;
 if(std::abs(base.delta+reverse.delta)>1e-18L*std::abs(base.delta))return 3;
 if(std::abs(base.u-reverse.u)>1e-18L*base.u)return 4;
 if(std::abs(scale.u/base.u-3)>1e-14L)return 5;
 if(fixture)std::cout<<"//@ run-pass\n//@ expect-stdout: PHYSICS_ENTROPY_REFERENCE_PASS\n// Generated C++23 long-double oracle; independent first-order uncertainty.\nuse physics::entropy::{isothermal_entropy_independent}\nfn close(a:f64,b:f64)->bool { let d=if a>b{a-b}else{b-a} let v=if b<0.0{-b}else{b} return d<=v*1.0e-11+"<<literal(std::ldexp(std::numeric_limits<double>::denorm_min(),1))<<" }\n";
 for(size_t i=0;i<all.size();++i) {
  const auto& c=all[i];auto r=eval(c);
  if(!fixture){std::cout<<i<<"\t"<<c.name<<"\t"<<r.status<<"\t"<<(double)r.delta<<"\t"<<(double)r.u<<"\t"<<(double)r.logratio<<"\n";continue;}
  std::cout<<"// "<<c.name<<"\nfn case_"<<i<<"() with Mut,Div,Panic {\n let r=isothermal_entropy_independent("<<literal(c.n)<<","<<literal(c.un)<<","<<literal(c.vf)<<","<<literal(c.uvf)<<","<<literal(c.vi)<<","<<literal(c.uvi)<<")\n assert(r.status=="<<r.status<<")\n assert(r.uncertainty_model==1)\n assert(r.first_order)\n";
  for(auto [name,value]:{std::pair{"delta_s",r.delta},std::pair{"standard_uncertainty",r.u},std::pair{"log_volume_ratio",r.logratio}}) {
   if(value==0)std::cout<<" assert(r."<<name<<"==0.0)\n";
   else std::cout<<" assert(close(r."<<name<<","<<literal((double)value)<<"))\n";
  }
  std::cout<<"}\n";
 }
 if(fixture) {
 std::cout<<"fn identities() with Mut,Div,Panic {\n let a=isothermal_entropy_independent(2.0,0.1,2.0,0.02,1.0,0.01)\n let b=isothermal_entropy_independent(2.0,0.1,1.0,0.01,2.0,0.02)\n let c=isothermal_entropy_independent(2.0,0.3,2.0,0.06,1.0,0.03)\n assert(a.status==0 && b.status==0 && c.status==0)\n assert(close(a.delta_s,0.0-b.delta_s))\n assert(close(a.standard_uncertainty,b.standard_uncertainty))\n assert(close(c.standard_uncertainty,3.0*a.standard_uncertainty))\n let d=isothermal_entropy_independent(2.0,0.1,2.0e200,2.0e198,1.0e200,1.0e198)\n let e=isothermal_entropy_independent(2.0,0.1,2.0e-200,2.0e-202,1.0e-200,1.0e-202)\n assert(d.status==0 && e.status==0)\n assert(close(a.delta_s,d.delta_s) && close(a.delta_s,e.delta_s))\n assert(close(a.standard_uncertainty,d.standard_uncertainty) && close(a.standard_uncertainty,e.standard_uncertainty))\n}\nfn main() with IO,Mut,Div,Panic {\n";
 for(size_t i=0;i<all.size();++i)std::cout<<" case_"<<i<<"()\n";
 std::cout<<" identities()\n println(\"PHYSICS_ENTROPY_REFERENCE_PASS\")\n}\n";
 }
}
