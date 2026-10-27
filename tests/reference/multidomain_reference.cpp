// Independent C++23 reference: integrate Student's density, then invert its CDF.
// Uses long double quadrature rather than the Sounio incomplete-beta path.
#include <array>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <functional>

long double student_pdf(long double x, long double df) {
    const auto pi = std::acos(-1.0L);
    return std::exp(std::lgamma((df+1)/2) - std::lgamma(df/2)
        - std::log(df*pi)/2 - (df+1)/2 * std::log1p(x*x/df));
}
long double integrate(long double x, long double df) {
    auto f = [df](long double t) { return student_pdf(t,df); };
    std::function<long double(long double,long double,long double,long double,
                              long double,long double,long double,int)> adapt;
    adapt = [&](auto a, auto b, auto fa, auto fm, auto fb, auto whole,
                auto eps, int depth) -> long double {
        const auto m=(a+b)/2, lm=(a+m)/2, rm=(m+b)/2;
        const auto fl=f(lm), fr=f(rm);
        const auto left=(m-a)*(fa+4*fl+fm)/6;
        const auto right=(b-m)*(fm+4*fr+fb)/6;
        const auto delta=left+right-whole;
        if (std::abs(delta)<=15*eps) return left+right+delta/15;
        if (depth==0) throw std::runtime_error("quadrature did not converge");
        return adapt(a,m,fa,fl,fm,left,eps/2,depth-1)
             + adapt(m,b,fm,fr,fb,right,eps/2,depth-1);
    };
    const auto fa=f(0), fm=f(x/2), fb=f(x);
    return adapt(0,x,fa,fm,fb,x*(fa+4*fm+fb)/6,4e-17L,28);
}
long double critical(long double p,long double df) {
    if (!(p>0.5L && p<1 && df>0))
        throw std::runtime_error("unsupported reference input");
    long double lo=0, hi=1;
    while (0.5L+integrate(hi,df)<p) hi*=2;
    for(int i=0;i<100;++i) {
        const auto mid=(lo+hi)/2;
        if (0.5L+integrate(mid,df)<p) lo=mid; else hi=mid;
    }
    return (lo+hi)/2;
}
long double mad_reference(std::vector<long double> data, long double median) {
    for(auto& x:data) x=std::abs(x-median);
    std::sort(data.begin(),data.end());
    const auto mid=data.size()/2;
    return data.size()%2 ? data[mid] : data[mid-1]/2+data[mid]/2;
}

#include <fstream>
#include <string>
using R=long double;
R mean_ref(const std::vector<R>& x) { R s=0; for(R v:x) s+=v; return s/x.size(); }
R var_ref(const std::vector<R>& x) { R m=mean_ref(x),s=0; for(R v:x) s+=(v-m)*(v-m); return s/(x.size()-1); }
void head(std::ofstream& s,const char* imports,const char* marker,const char* tolerance="2.0e-7") {
 s<<std::setprecision(18)<<std::showpoint<<"//@ run-pass\n//@ expect-stdout: "<<marker<<"\n"<<imports<<"\n";
 s<<"fn close(g: f64, e: f64) -> bool {\n let scale = if e < 0.0 { 0.0-e } else { e }\n"
   <<" let delta = if g > e { g-e } else { e-g }\n delta <= "<<tolerance<<" * scale + 1.0e-290\n}\n";
 s<<"fn main() with IO, Mut, Div, Panic {\n";
}
void check(std::ofstream& s,const std::string& expr,R value) { s<<" assert(close("<<expr<<", "<<value<<"))\n"; }
void array(std::ofstream& s,const char* name,const std::vector<R>& x) {
 s<<" var "<<name<<": [f64; 100] = [0.0; 100]\n";
 for(std::size_t i=0;i<x.size();++i) s<<" "<<name<<"["<<i<<"] = "<<x[i]<<"\n";
}
int main(int argc,char**argv) {
 if(argc!=6) return 2;
 std::ofstream stats(argv[1]),tpm(argv[2]),phys(argv[3]),math(argv[4]);
 head(stats,"use stats::inferential::{t_test_one_sample, t_test_independent, t_test_welch, t_test_paired}","STATS_T_TEST_CI_PASS");
 std::cout<<std::setprecision(18)<<"method\tdifference\tse\tdf\tt\tp\tlower\tupper\n";
 const std::vector<R> x={1,2,4,5,8},y={2,5,9,10,14,20};
 auto emit=[&](int mode,std::vector<R> a,std::vector<R> b,R mu,int id) {
  std::string r="r"+std::to_string(id);
  stats<<" println(\"t_case_"<<id<<"\")\n";
  const std::string ax="x"+std::to_string(id),by="y"+std::to_string(id);
  array(stats,ax.c_str(),a); if(mode!=0) array(stats,by.c_str(),b);
  R d=0,se=0,df=0,sd=0;
  if(mode==0) {
   d=mean_ref(a)-mu; sd=std::sqrt(var_ref(a)); se=sd/std::sqrt(R(a.size())); df=a.size()-1;
  } else {
   d=mean_ref(a)-mean_ref(b);
   R va=var_ref(a),vb=var_ref(b);
   if(mode==1) {
    df=a.size()+b.size()-2;
    sd=std::sqrt(((a.size()-1)*va+(b.size()-1)*vb)/df);
    se=sd*std::sqrt(1/R(a.size())+1/R(b.size()));
   } else {
    R aa=va/a.size(),bb=vb/b.size();
    se=std::sqrt(aa+bb);
    df=(aa+bb)*(aa+bb)/(aa*aa/(a.size()-1)+bb*bb/(b.size()-1));
    sd=std::sqrt((va+vb)/2);
   }
  }
  R t=d/se,p=1-2*integrate(std::abs(t),df),k=critical(0.975L,df),lo=d-k*se,hi=d+k*se;
  const char* names[]={"t_test_one_sample","t_test_independent","t_test_welch","t_test_paired"};
  stats<<" let "<<r<<" = "<<names[mode]<<"("<<ax<<", "<<a.size();
  if(mode==0) stats<<", "<<mu;
  else stats<<", "<<by<<", "<<b.size();
  stats<<")\n";
  for(const auto& [field,value]:std::vector<std::pair<std::string,R>>{{"mean_diff",d},{"se_diff",se},{"df",df},{"t_stat",t},{"p_value",p},{"ci_lower",lo},{"ci_upper",hi},{"cohens_d",d/sd}})
   check(stats,r+"."+field,value);

  std::cout<<names[mode]<<'\t'<<d<<'\t'<<se<<'\t'<<df<<'\t'<<t<<'\t'<<p<<'\t'<<lo<<'\t'<<hi<<'\n';
 };
 emit(0,x,{},3,1);
 emit(1,x,y,0,2);
 emit(2,x,y,0,3);
 emit(0,{1,3},{},0,4);
 auto tinyx=x,tinyy=y; for(auto&v:tinyx)v*=1e-12L;for(auto&v:tinyy)v*=1e-12L;
 emit(2,tinyx,tinyy,0,5);
 auto shiftedx=x,shiftedy=y; for(auto&v:shiftedx)v+=1e6L;for(auto&v:shiftedy)v+=1e6L;
 emit(2,shiftedx,shiftedy,0,6);
 auto bigx=x,bigy=y;for(auto&v:bigx)v*=1e50L;for(auto&v:bigy)v*=1e50L;
 emit(2,bigx,bigy,0,7);
 // Paired reference computed independently from differences.
 array(stats,"paired_x",x); const std::vector<R> py={0,1,2,3,4};array(stats,"paired_y",py);
 auto dif=x;for(std::size_t i=0;i<dif.size();++i)dif[i]-=py[i];
 R d=mean_ref(dif),sd=std::sqrt(var_ref(dif)),se=sd/std::sqrt(R(dif.size())),df=dif.size()-1;
 stats<<" let paired = t_test_paired(paired_x, paired_y, 5)\n";
 check(stats,"paired.mean_diff",d);check(stats,"paired.se_diff",se);check(stats,"paired.df",df);
 check(stats,"paired.ci_lower",d-critical(.975L,df)*se);check(stats,"paired.ci_upper",d+critical(.975L,df)*se);
 stats<<" println(\"STATS_T_TEST_CI_PASS\")\n 0\n}\n";

 head(tpm,"use genomics::expression::{tpm_new, tpm_accumulate, tpm_value, EXPRESSION_OK, EXPRESSION_INVALID_INPUT, EXPRESSION_EMPTY_TOTAL}","GENOMICS_TPM_PASS","1.0e-12");
 tpm<<" var state = tpm_new()\n";
 std::vector<R> counts={20,30,50},lengths={1000,2000,500};
 R rates=0; for(std::size_t i=0;i<counts.size();++i)rates+=counts[i]/lengths[i];
 for(std::size_t i=0;i<counts.size();++i)tpm<<" assert(tpm_accumulate(&!state, "<<counts[i]<<", "<<lengths[i]<<"))\n";
 tpm<<" assert(state.features == 3)\n assert(state.status == EXPRESSION_OK)\n var total: f64 = 0.0\n";
 for(std::size_t i=0;i<counts.size();++i) {
  R v=counts[i]/lengths[i]/rates*1e6L;
  tpm<<" let g"<<i<<" = tpm_value(&state, "<<counts[i]<<", "<<lengths[i]<<")\n assert(g"<<i<<".status == EXPRESSION_OK)\n";
  check(tpm,"g"+std::to_string(i)+".value",v);tpm<<" total = total + g"<<i<<".value\n";
  std::cout<<"TPM\t"<<i<<'\t'<<v<<'\n';
 }
 check(tpm,"total",1e6L);
 tpm<<" var empty = tpm_new()\n let e = tpm_value(&empty, 0.0, 100.0)\n assert(e.status == EXPRESSION_EMPTY_TOTAL)\n"
    <<" assert(!tpm_accumulate(&!empty, 1.0, 0.0))\n assert(empty.status == EXPRESSION_INVALID_INPUT)\n"
    <<" assert(!tpm_accumulate(&!empty, 1.0, 1.0))\n assert(empty.features == 0)\n"
    <<" var streamed = tpm_new()\n var i: i64 = 0\n while i < 5000 {\n assert(tpm_accumulate(&!streamed, 1.0, 1000.0))\n i = i + 1\n }\n"
    <<" assert(streamed.features == 5000)\n let uniform = tpm_value(&streamed, 1.0, 1000.0)\n";
 check(tpm,"uniform.value",200);
 tpm<<" var compensated = tpm_new()\n assert(tpm_accumulate(&!compensated, 1.0e16, 1.0))\n i = 0\n while i < 10000 {\n assert(tpm_accumulate(&!compensated, 1.0, 1.0))\n i = i + 1\n }\n";
 R compensated_ref=1e16L;for(int j=0;j<10000;++j)compensated_ref+=1;
 tpm<<" assert(compensated.rate_sum == "<<compensated_ref<<")\n";
 tpm<<" var scaled = tpm_new()\n";
 for(std::size_t j=0;j<counts.size();++j)tpm<<" assert(tpm_accumulate(&!scaled, "<<counts[j]*10<<", "<<lengths[j]<<"))\n";
 tpm<<" let scaled_first = tpm_value(&scaled, 200.0, 1000.0)\n";
 check(tpm,"scaled_first.value",counts[0]/lengths[0]/rates*1e6L);
 tpm<<" let zero = tpm_value(&state, 0.0, 1000.0)\n assert(zero.value == 0.0)\n assert(zero.status == EXPRESSION_OK)\n";
 tpm<<" println(\"GENOMICS_TPM_PASS\")\n 0\n}\n";

 head(phys,"use physics::scattering::{electron_compton_shift, COMPTON_OK, COMPTON_INVALID_INPUT}","PHYSICS_COMPTON_PASS","1.0e-12");
 R pi=std::acos(-1.0L),lambda=6.62607015e-34L/(9.1093837139e-31L*299792458.0L);
 int i=0;for(R theta:{0.0L,1e-12L,1e-8L,pi/6,pi/2,pi}) {
  phys<<" let p"<<i<<" = electron_compton_shift("<<theta<<", 0.01)\n assert(p"<<i<<".status == COMPTON_OK)\n";
  R shift=2*lambda*std::sin(theta/2)*std::sin(theta/2),u=lambda*std::abs(std::sin(theta))*.01L;
  check(phys,"p"+std::to_string(i)+".shift_m",shift);
  if(theta==pi) phys<<" assert(p"<<i<<".angular_standard_uncertainty_m < 1.0e-26)\n";
  else check(phys,"p"+std::to_string(i)+".angular_standard_uncertainty_m",u);
  std::cout<<"COMPTON\t"<<theta<<'\t'<<shift<<'\t'<<u<<'\n';++i;
 }
 phys<<" let bad = electron_compton_shift(-1.0, 0.01)\n assert(bad.status == COMPTON_INVALID_INPUT)\n"
     <<" let bad_u = electron_compton_shift(1.0, -0.1)\n assert(bad_u.status == COMPTON_INVALID_INPUT)\n"
     <<" println(\"PHYSICS_COMPTON_PASS\")\n 0\n}\n";

 head(math,"use math::pure::{tan}\nuse stats::student_t::{t_quantile}","MATH_CAUCHY_SMALL_ANGLE_PASS","1.0e-12");
 for(R p:{.01L,.025L,.25L,.75L,.975L,.99L}) {
  R q=std::tan(pi*(p-.5L));
  math<<" println(\"Cauchy quantile\")\n";check(math,"t_quantile("+std::to_string(double(p))+", 1.0)",q);
 }
 for(R v:{1e-12L,-1e-12L,1e-8L,.1L}) {
  math<<" println(\"small tangent\")\n";
  math<<" assert(close(tan("<<v<<"), "<<std::tan(v)<<"))\n";
 }
 math<<" println(\"MATH_CAUCHY_SMALL_ANGLE_PASS\")\n 0\n}\n";
 std::ofstream roots(argv[5]);
 head(roots,"use math::pure::{sqrt}","MATH_SQRT_RANGE_PASS","1.0e-12");
 for(R v:{1e-300L,1e-100L,2.2250738585072014e-308L,1.0L,2.0L,1e12L,1e100L,1e300L}) {
  roots<<" println(\"sqrt range\")\n assert(close(sqrt("<<v<<"), "<<std::sqrt(v)<<"))\n";
 }
 roots<<" assert(sqrt(0.0) == 0.0)\n println(\"MATH_SQRT_RANGE_PASS\")\n 0\n}\n";
 std::cout<<"CPP23_MULTIDOMAIN_REFERENCE_PASS\n";
}
