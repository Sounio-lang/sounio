// C++23 independent conditional weighted linear inference; -lquadmath.
// Quad raw weighted MGS (included independent oracle) plus inverse upper R.
// Student two-tail I_{nu/(nu+t^2)}(nu/2,1/2), independent quad fraction.
// Primary formulas: https://dlmf.nist.gov/8.17 (8.17.22) and
// https://itl.nist.gov/div898/handbook/eda/section3/eda3664.htm
#include <quadmath.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
// Place the existing reference CLI in a namespace; its main is never called.
namespace independent_weighted_reference {
#include "weighted_least_squares_reference.cpp"
}
using independent_weighted_reference::Q;
using independent_weighted_reference::V;
using independent_weighted_reference::M;
using independent_weighted_reference::WeightedCase;
using independent_weighted_reference::WeightedResult;
using independent_weighted_reference::weighted_eval;
using independent_weighted_reference::weighted_cases;
using independent_weighted_reference::factor;
using independent_weighted_reference::dense;
using independent_weighted_reference::lit;
using independent_weighted_reference::input;
Q wi_betacf(Q a,Q b,Q x) {
 Q tiny=scalbnq((Q)1,-16000),qab=a+b,qap=a+1,qam=a-1;
 Q c=1,d=1-qab*x/qap;if(fabsq(d)<tiny)d=tiny;d=1/d;Q h=d;
 for(int k=1;k<10000;++k){
  Q m=k,mm=2*m,aa=m*(b-m)*x/((qam+mm)*(a+mm));
  d=1+aa*d;if(fabsq(d)<tiny)d=tiny;c=1+aa/c;if(fabsq(c)<tiny)c=tiny;d=1/d;h*=d*c;
  aa=-(a+m)*(qab+m)*x/((a+mm)*(qap+mm));
  d=1+aa*d;if(fabsq(d)<tiny)d=tiny;c=1+aa/c;if(fabsq(c)<tiny)c=tiny;d=1/d;Q delta=d*c;h*=delta;
  if(fabsq(delta-1)<(Q)1e-31)return h;
 }
 std::cerr<<"BETA_CONVERGENCE_FAIL\n";std::exit(20);
}
Q wi_ibeta(Q a,Q b,Q x) {
 if(x<=0)return 0;
 if(x>=1)return 1;
 Q front=expq(lgammaq(a+b)-lgammaq(a)-lgammaq(b)+a*logq(x)+b*log1pq(-x));
 if(x<(a+1)/(a+b+2))return front*wi_betacf(a,b,x)/a;
 return 1-front*wi_betacf(b,a,1-x)/b;
}
Q wi_student_p(Q t,int df) {if(t==0)return 1;Q v=df;return wi_ibeta(v/2,(Q).5,v/(v+t*t));}
Q wi_critical(int df,double conf) {
 Q target=1-(Q)conf,lo=0,hi=1;
 while(wi_student_p(hi,df)>target)hi*=2;
 for(int j=0;j<128;++j){Q mid=(lo+hi)/2;if(wi_student_p(mid,df)>target)lo=mid;else hi=mid;}
 return (lo+hi)/2;
}
struct InferenceCase{WeightedCase fit;double confidence=.95;bool assume=true;};
struct InferenceReference {
 WeightedResult fit;int status=0,cov_status=0,inf_status=0,calibration=0;
 double level=0;Q critical=0;
 std::array<Q,64>cov{};std::array<Q,8>se{},t{},p{},lower{},upper{};
};
InferenceReference infer_eval(const InferenceCase&c) {
 InferenceReference out;out.fit=weighted_eval(c.fit);
 // Metadata/status constants finalized against the public contract below.
 if(out.fit.status){out.status=out.fit.status;out.cov_status=4;out.inf_status=4;return out;}
 if(!c.assume){out.cov_status=1;out.inf_status=1;return out;}
 out.calibration=1;
 if(out.fit.df<1){out.cov_status=2;out.inf_status=2;return out;}
 const auto&d=c.fit.data;M design(d.p,std::vector<Q>(d.m));
 for(int j=0;j<d.p;++j)for(int i=0;i<d.m;++i)design[j][i]=(Q)d.a[i*8+j]*sqrtq((Q)c.fit.weights[i]/(Q)out.fit.weight_scale);
 auto qr=factor(design,d.m,d.p,false,0);
 M inverse(d.p,std::vector<Q>(d.p));
 for(int col=0;col<d.p;++col)for(int row=d.p-1;row>=0;--row){Q v=row==col?1:0;for(int k=row+1;k<d.p;++k)v-=qr.r[row][k]*inverse[k][col];inverse[row][col]=v/qr.r[row][row];}
 Q variance=out.fit.weighted_norm*out.fit.weighted_norm/out.fit.df;
 for(int i=0;i<d.p;++i)for(int j=0;j<d.p;++j){Q dot=0;for(int k=0;k<d.p;++k)dot+=inverse[i][k]*inverse[j][k];out.cov[i*8+j]=variance*dot;}
 for(int i=0;i<d.p;++i)out.se[i]=sqrtq(out.cov[i*8+i]);
 if(!std::isfinite(c.confidence)||c.confidence<.5||c.confidence>.999999){out.inf_status=5;return out;}
 for(int i=0;i<d.p;++i)if(out.se[i]==0){out.inf_status=3;return out;}
 out.critical=wi_critical(out.fit.df,c.confidence);out.calibration=1;out.level=c.confidence;
 for(int i=0;i<d.p;++i){out.t[i]=out.fit.coef[i]/out.se[i];out.p[i]=wi_student_p(out.t[i],out.fit.df);out.lower[i]=out.fit.coef[i]-out.critical*out.se[i];out.upper[i]=out.fit.coef[i]+out.critical*out.se[i];}
 // Public binary64 availability rejects collapsed intervals collectively.
 for(int i=0;i<d.p;++i)if((double)out.lower[i]>=(double)out.upper[i]){out.inf_status=4;out.level=0;out.t={};out.p={};out.lower={};out.upper={};break;}
 return out;
}
std::vector<InferenceCase> inference_cases() {
 auto w=weighted_cases();std::vector<InferenceCase> out;
 auto add=[&](WeightedCase c,double conf=.95,bool assume=true){out.push_back({c,conf,assume});};
 add(w[0]);add(w[38]);add(w[39]);
 for(auto [name,scale]:{std::pair{"inference_weight_tiny",1e-150},std::pair{"inference_weight_huge",1e150}}){auto c=w[38];c.data.name=name;for(auto&v:c.weights)v*=scale;add(c);}
 for(auto [name,scale]:{std::pair{"inference_design_tiny",1e-150},std::pair{"inference_design_huge",1e150}}){auto c=w[38];c.data.name=name;for(auto&v:c.data.a)v*=scale;add(c);}
 for(auto [name,scale]:{std::pair{"inference_response_tiny",1e-150},std::pair{"inference_response_huge",1e150}}){auto c=w[38];c.data.name=name;for(auto&v:c.data.b)v*=scale;add(c);}
 auto c=w[38];c.data.name="inference_mixed_columns";for(int i=0;i<c.data.m;++i){c.data.a[i*8]*=1e-150;c.data.a[i*8+1]*=1e150;}add(c);
 c=w[38];c.data.name="inference_columns_permuted";for(int i=0;i<c.data.m;++i)std::swap(c.data.a[i*8],c.data.a[i*8+1]);add(c);
 add(w[6]);add(w[19]);add(w[38],.95,false);add(w[13]);add(w[14]);
 add({dense("inference_perfect_nonzero",{{1},{1},{1},{1}},{2,2,2,2}),{1,1,1,1}});
 c=w[38];c.data.name="inference_conf90";add(c,.90);c.data.name="inference_conf99";add(c,.99);
 add({dense("inference_mean_df3",{{1},{1},{1},{1}},{1,2,4,5}),{1,1,1,1}});
 add(w[15]);add(w[34]);add(w[27]);
 c=w[38];c.data.name="inference_conf_lower";add(c,.5);c.data.name="inference_conf_upper";add(c,.999999);
 c.data.name="inference_conf_invalid_low";add(c,.49);c.data.name="inference_conf_invalid_high";add(c,1.0);c.data.name="inference_conf_invalid_nan";add(c,std::numeric_limits<double>::quiet_NaN());
 add({dense("inference_extreme_df1",{{1},{0}},{1,1e-155}),{1,1}});
 add({dense("inference_extreme_df2",{{1},{0},{0}},{1,1e-155,1e-155}),{1,1,1}});
 return out;
}
int main(int argc,char**argv) {
 bool fixture=argc>1&&std::string(argv[1])=="--fixture";auto all=inference_cases();
 std::cout<<std::scientific<<std::setprecision(17);
 if(fixture)std::cout<<"//@ run-pass\n//@ expect-stdout: STATS_WEIGHTED_INFERENCE_REFERENCE_PASS\n// C++23 quad raw-design MGS/covariance/Student reference; conditional model.\nuse stats::regression::weighted_inference::{weighted_linear_inference}\nfn close(a:f64,b:f64,at:f64,rt:f64)->bool {let d=if a>b{a-b}else{b-a} let v=if b<0.0{-b}else{b} return d<=at+rt*v}\n";
 for(size_t n=0;n<all.size();++n){auto r=infer_eval(all[n]);const auto&c=all[n].fit.data;
  if(!fixture){std::cout<<n<<"\t"<<c.name<<"\t"<<r.status<<"\t"<<r.cov_status<<"\t"<<r.inf_status<<"\t"<<r.fit.df<<"\t"<<(double)r.critical<<"\n";
   if(r.fit.status==0){for(int j=0;j<c.p;++j)std::cout<<"coefficient\t"<<n<<"\t"<<j<<"\t"<<(double)r.fit.coef[j]<<"\t"<<(double)r.se[j]<<"\t"<<(double)r.t[j]<<"\t"<<(double)r.p[j]<<"\t"<<(double)r.lower[j]<<"\t"<<(double)r.upper[j]<<"\n";for(int i=0;i<c.p;++i)for(int j=0;j<c.p;++j)std::cout<<"covariance\t"<<n<<"\t"<<i<<"\t"<<j<<"\t"<<(double)r.cov[i*8+j]<<"\n";}
  } else {
   const auto&ic=all[n];const auto&wc=ic.fit;
   std::cout<<"// "<<c.name<<"\nfn case_"<<n<<"() with Mut,Div,Panic {\n";input("a",2048,c.a);input("b",256,c.b);input("w",256,wc.weights);
   std::cout<<" let r=weighted_linear_inference(&a,&b,&w,"<<c.m<<","<<c.p<<","<<lit(c.rcond)<<","<<lit(ic.confidence)<<","<<(ic.assume?"true":"false")<<")\n assert(r.status=="<<r.status<<")\n assert(r.m=="<<c.m<<" && r.p=="<<c.p<<")\n assert(r.rank=="<<r.fit.rank<<" && r.df=="<<r.fit.df<<")\n assert(r.covariance_status=="<<r.cov_status<<" && r.inference_status=="<<r.inf_status<<")\n assert(r.calibration=="<<r.calibration<<")\n assert(r.level=="<<lit(r.level)<<")\n assert(!r.variance_model_assessed)\n assert(r.weight_scale=="<<lit(r.fit.weight_scale)<<")\n assert(close(r.rcond_used,"<<lit(r.fit.used)<<",0.0,1.0e-14))\n";
   for(int j=0;j<8;++j){Q value=r.fit.coef[j];if(value==0)std::cout<<" assert(r.coefficients["<<j<<"]==0.0)\n";else std::cout<<" assert(close(r.coefficients["<<j<<"],"<<lit((double)value)<<",0.0,1.0e-9))\n";}
   if(r.cov_status!=0)std::cout<<" var i=0\n while i<64 {assert(r.covariance[i as usize]==0.0) i=i+1}\n i=0\n while i<8 {assert(r.standard_errors[i as usize]==0.0) i=i+1}\n";
   else {
    for(int i=0;i<c.p;++i) {
     if(r.se[i]==0)std::cout<<" assert(r.standard_errors["<<i<<"]==0.0)\n";
     else std::cout<<" assert(close(r.standard_errors["<<i<<"],"<<lit((double)r.se[i])<<",0.0,2.0e-8))\n";
     for(int j=0;j<c.p;++j){Q value=r.cov[i*8+j];Q at=value==0?sqrtq(r.cov[i*8+i]*r.cov[j*8+j])*(Q)2e-10:2*(Q)std::numeric_limits<double>::denorm_min();if(i==j&&value>0)std::cout<<" assert(r.covariance["<<i*8+j<<"]>0.0)\n";std::cout<<" assert(close(r.covariance["<<i*8+j<<"],"<<lit((double)value)<<","<<lit((double)at)<<",2.0e-8))\n";}
    }
   }
   if(r.inf_status!=0)std::cout<<" var z=0\n while z<8 {assert(r.t_statistics[z as usize]==0.0 && r.p_values[z as usize]==0.0 && r.ci_lower[z as usize]==0.0 && r.ci_upper[z as usize]==0.0) z=z+1}\n";
   else for(int j=0;j<c.p;++j) {
    if(r.t[j]==0)std::cout<<" assert(r.t_statistics["<<j<<"]==0.0)\n";else std::cout<<" assert(close(r.t_statistics["<<j<<"],"<<lit((double)r.t[j])<<",0.0,3.0e-8))\n";
    if((double)r.p[j]==0)std::cout<<" assert(r.p_values["<<j<<"]==0.0)\n";else std::cout<<" assert(r.p_values["<<j<<"]>0.0)\n assert(close(r.p_values["<<j<<"],"<<lit((double)r.p[j])<<","<<lit(2*std::numeric_limits<double>::denorm_min())<<",3.0e-8))\n";
    Q at=r.se[j]*(Q)3e-8;
    std::cout<<" assert(close(r.ci_lower["<<j<<"],"<<lit((double)r.lower[j])<<","<<lit((double)at)<<",1.0e-10))\n assert(close(r.ci_upper["<<j<<"],"<<lit((double)r.upper[j])<<","<<lit((double)at)<<",1.0e-10))\n";
   }
   Q bound=r.fit.bnorm*(Q)3e-11;
   std::cout<<" assert(close(r.residual_norm,"<<lit((double)r.fit.norm)<<","<<lit((double)bound)<<",0.0))\n assert(close(r.normalized_weighted_residual_norm,"<<lit((double)r.fit.weighted_norm)<<","<<lit((double)bound)<<",0.0))\n}\n";
  }
 }
 if(fixture){std::cout<<"fn main() with IO,Mut,Div,Panic {\n";for(size_t j=0;j<all.size();++j)std::cout<<" case_"<<j<<"()\n";std::cout<<" println(\"STATS_WEIGHTED_INFERENCE_REFERENCE_PASS\")\n}\n";}
 // Analytic Student-tail crosschecks across independent df1/df2/df3 forms.
 Q pi=acosq((Q)-1);
 for(double value:{.1,1.,10.}){Q t=value;
  if(fabsq(wi_student_p(t,1)-2/pi*atanq(1/t))>(Q)1e-29)return 3;
  Q df2=2/(sqrtq(t*t+2)*(sqrtq(t*t+2)+t));
  if(fabsq(wi_student_p(t,2)-df2)>(Q)1e-29)return 4;
  Q theta=atanq(sqrtq((Q)3)/t),df3=2/pi*(theta-sinq(theta)*cosq(theta));
  if(fabsq(wi_student_p(t,3)-df3)>(Q)1e-29)return 5;
  Q u=t/sqrtq(6+t*t),df6=1-(Q)15/8*(u-2*u*u*u/3+u*u*u*u*u/5);
  if(fabsq(wi_student_p(t,6)-df6)>(Q)1e-29)return 6;
 }
 return 0;
}
