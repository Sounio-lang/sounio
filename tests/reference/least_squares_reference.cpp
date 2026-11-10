// C++23 independent bounded least-squares oracle; compile with -lquadmath.
// Solve: raw-input modified Gram-Schmidt, twice reorthogonalized, __float128.
// NO normal equations, NO Householder reflection, NO binary64 input scaling.
// Rank validation separately applies declared column equilibration/rcond policy.
#include <quadmath.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
using Q=__float128;using V=std::vector<double>;using M=std::vector<std::vector<Q>>;
struct Case{std::string name;V a,b;int m,p;double rcond=0,forward=1e-10;bool poison=false,tiny_exact=false;int first_pivot=-1;};
struct QR{int rank=0;M q,r;std::vector<int> pivot;Q first=0,last=0;};
struct Result{int status=0,rank=0,df=0;double used=0;Q diagonal_ratio=0,norm=0,bnorm=0;std::vector<Q>coef,fitted,residual;};
Q dot(const std::vector<Q>&a,const std::vector<Q>&b){Q v=0;for(size_t i=0;i<a.size();++i)v+=a[i]*b[i];return v;}
QR factor(M w,int m,int p,bool pivot,Q cutoff) {
 QR f;f.q=M(p,std::vector<Q>(m));f.r=M(p,std::vector<Q>(p));f.pivot.resize(p);std::iota(f.pivot.begin(),f.pivot.end(),0);
 for(int j=0;j<p;++j){
  if(pivot){int best=j;Q norm=dot(w[j],w[j]);for(int k=j+1;k<p;++k){Q v=dot(w[k],w[k]);if(v>norm){best=k;norm=v;}}if(best!=j){std::swap(w[j],w[best]);std::swap(f.pivot[j],f.pivot[best]);for(int k=0;k<j;++k)std::swap(f.r[k][j],f.r[k][best]);}}
  Q norm=sqrtq(dot(w[j],w[j]));if(j==0)f.first=norm;
  if(norm==0||norm<=cutoff*f.first)break;
  f.r[j][j]=norm;f.last=norm;++f.rank;
  for(int i=0;i<m;++i)f.q[j][i]=w[j][i]/norm;
  for(int k=j+1;k<p;++k){Q projection=dot(f.q[j],w[k]);for(int i=0;i<m;++i)w[k][i]-=projection*f.q[j][i];Q again=dot(f.q[j],w[k]);for(int i=0;i<m;++i)w[k][i]-=again*f.q[j][i];f.r[j][k]=projection+again;}
 }
 return f;
}
Result eval(const Case&c) {
 Result out;out.coef.resize(8);out.fitted.resize(256);out.residual.resize(256);
 if(c.m<1||c.m>256||c.p<1||c.p>8||c.p>c.m){out.status=1;return out;}
 if(!std::isfinite(c.rcond)||c.rcond<0||c.rcond>=1){out.status=5;return out;}
 for(int i=0;i<c.m;++i){if(!std::isfinite(c.b[i])){out.status=2;return out;}for(int j=0;j<c.p;++j)if(!std::isfinite(c.a[i*8+j])){out.status=2;return out;}}
 out.used=c.rcond==0?c.m*std::numeric_limits<double>::epsilon():c.rcond;
 M raw(c.p,std::vector<Q>(c.m)),equilibrated=raw;
 std::vector<Q>b(c.m);
 for(int i=0;i<c.m;++i)b[i]=c.b[i];
 for(int j=0;j<c.p;++j){Q scale=0;for(int i=0;i<c.m;++i){raw[j][i]=c.a[i*8+j];scale=fmaxq(scale,fabsq(raw[j][i]));}for(int i=0;i<c.m;++i)equilibrated[j][i]=scale==0?0:raw[j][i]/scale;}
 auto rank=factor(equilibrated,c.m,c.p,true,(Q)out.used);out.rank=rank.rank;
 if(rank.first>0&&rank.rank>0)out.diagonal_ratio=rank.last/rank.first;
 if(out.rank<c.p){out.status=3;return out;}
 auto solve=factor(raw,c.m,c.p,false,0);
 std::vector<Q>z(c.p);
 for(int j=0;j<c.p;++j)z[j]=dot(solve.q[j],b);
 for(int j=c.p-1;j>=0;--j){Q v=z[j];for(int k=j+1;k<c.p;++k)v-=solve.r[j][k]*out.coef[k];out.coef[j]=v/solve.r[j][j];}
 for(int j=0;j<c.p;++j)if(!std::isfinite((double)out.coef[j])||(out.coef[j]!=0&&(double)out.coef[j]==0)){out=Result{};out.status=4;out.coef.resize(8);out.fitted.resize(256);out.residual.resize(256);return out;}
 Q rr=0,bb=0;
 for(int i=0;i<c.m;++i){Q fit=0;for(int j=0;j<c.p;++j)fit+=(Q)c.a[i*8+j]*out.coef[j];out.fitted[i]=fit;out.residual[i]=b[i]-fit;rr+=out.residual[i]*out.residual[i];bb+=b[i]*b[i];}
 out.norm=sqrtq(rr);out.bnorm=sqrtq(bb);out.df=c.m-c.p;
 return out;
}
Case dense(std::string name,const std::vector<V>&rows,V b,double rcond=0){int m=rows.size(),p=rows.empty()?1:rows[0].size();Case c{name,V(m*8),b,m,p,rcond};for(int i=0;i<m;++i)for(int j=0;j<p;++j)c.a[i*8+j]=rows[i][j];return c;}
std::vector<Case> cases(){
 std::vector<Case>c;std::vector<V>rows;V b;
 for(int i=-2;i<=3;++i){rows.push_back({1.,(double)i});double noise[]={.2,-.1,.1,-.2,.05,-.05};b.push_back(2+3*i+noise[i+2]);}
 c.push_back(dense("rectangular_noisy",rows,b));
 c.push_back(dense("forced_pivot",{{1,1},{0,1},{0,1},{0,1}},{3,2,3,2}));c.back().first_pivot=1;
 c.push_back(dense("column_permuted",{{1,1},{1,0},{1,0},{1,0}},{3,2,3,2}));c.back().first_pivot=0;
 rows.clear();b.clear();for(int i=-2;i<=3;++i){rows.push_back({1.,(double)i,(double)(i*i)});b.push_back(1-2*i+3*i*i+((i%2)?-.1:.1));}c.push_back(dense("polynomial",rows,b));
 rows.clear();b.clear();for(int i=0;i<8;++i){double u=i%2?1.:-1.;rows.push_back({1.,(double)i,i+1e-8*u});b.push_back(1-2*rows.back()[1]+3*rows.back()[2]);}
 c.push_back(dense("near_dependent",rows,b));c.back().forward=2e-6;
 auto base=c[0];
 for(auto [name,scale]:{std::pair{"design_tiny",1e-200},std::pair{"design_huge",1e200}}){auto d=base;d.name=name;for(double&v:d.a)v*=scale;c.push_back(d);}
 for(auto [name,scale]:{std::pair{"response_tiny",1e-200},std::pair{"response_huge",1e200}}){auto d=base;d.name=name;for(double&v:d.b)v*=scale;c.push_back(d);}
 auto mixed=base;mixed.name="mixed_column_units";for(int i=0;i<mixed.m;++i){mixed.a[i*8]*=1e-200;mixed.a[i*8+1]*=1e200;}c.push_back(mixed);
 c.push_back(dense("square",{{1,0},{0,1}},{2,-3}));
 auto zero=base;zero.name="zero_response";std::fill(zero.b.begin(),zero.b.end(),0);c.push_back(zero);
 c.push_back(dense("duplicate_columns",{{1,2},{2,4},{3,6},{4,8}},{1,2,3,4}));
 c.push_back(dense("zero_column",{{0,1},{0,2},{0,3},{0,4}},{1,2,3,4}));
 c.push_back(dense("custom_rank_reject",{{1,1},{0,1e-6},{0,0}},{5,3e-6,0},1e-4));
 c.push_back(dense("custom_rank_accept",{{1,1},{0,1e-6},{0,0}},{5,3e-6,0},1e-8));
 Case large{"max256p8",V(2048),V(256),256,8};for(int i=0;i<256;++i){large.a[i*8+i%8]=1;large.b[i]=1+i%8+((i/8)%2?.125:-.125);}c.push_back(large);
 auto inactive=base;inactive.name="inactive_poison";inactive.poison=true;c.push_back(inactive);
 double tiny=std::numeric_limits<double>::denorm_min();
 c.push_back(dense("subnormal_design",{{tiny}},{tiny}));c.back().tiny_exact=true;
 c.push_back(dense("subnormal_response",{{1},{2}},{tiny,2*tiny}));c.back().tiny_exact=true;
 c.push_back(dense("coefficient_overflow",{{1e-300}},{1e300}));
 c.push_back(dense("coefficient_underflow",{{1e300}},{tiny}));
 auto bad=base;bad.name="invalid_size";bad.p=9;c.push_back(bad);
 bad=base;bad.name="invalid_rcond";bad.rcond=1;c.push_back(bad);
 bad=base;bad.name="nonfinite_matrix";bad.a[0]=std::numeric_limits<double>::infinity();c.push_back(bad);
 bad=base;bad.name="nonfinite_response";bad.b[1]=std::numeric_limits<double>::quiet_NaN();c.push_back(bad);
 c.push_back(dense("visual_two_sensor",{{1,-1,-1.5},{1,-1,-.5},{1,-1,.5},{1,-1,1.5},{1,1,-1.5},{1,1,-.5},{1,1,.5},{1,1,1.5}},{3.7,6.4,9.6,12.3,7.4,10.7,13.3,16.6}));
 c.push_back(dense("restore_intermediate_underflow",{{1e-200,0},{1e-200,0},{1e-200,0},{1e-200,0},{0,1e-200}},{tiny,0,0,0,1e-308}));
 return c;
}
std::string lit(double v){if(std::isnan(v))return "(0.0/0.0)";if(std::isinf(v))return "(1.0e308*2.0)";std::ostringstream o;o<<std::scientific<<std::setprecision(17);if(std::fpclassify(v)==FP_SUBNORMAL)o<<"("<<std::ldexp(v,52)<<" / 4503599627370496.0)";else o<<v;return o.str();}
void input(const std::string&name,int size,const V&a){std::cout<<" var "<<name<<":[f64;"<<size<<"]=[0.0;"<<size<<"]\n";for(size_t j=0;j<a.size();++j)if(a[j]!=0)std::cout<<" "<<name<<"["<<j<<"]="<<lit(a[j])<<"\n";}
int main(int argc,char**argv) {
 bool f=argc>1&&std::string(argv[1])=="--fixture";auto all=cases();std::cout<<std::scientific<<std::setprecision(17);
 if(f)std::cout<<"//@ run-pass\n//@ expect-stdout: LINALG_LEAST_SQUARES_REFERENCE_PASS\n// Independent C++23 quadmath reorthogonalized-MGS reference, stride8.\nuse linalg::least_squares::{qr_least_squares}\nfn close(a:f64,b:f64,at:f64,rt:f64)->bool {let d=if a>b{a-b}else{b-a} let v=if b<0.0{-b}else{b} return d<=at+rt*v}\n";
 for(size_t n=0;n<all.size();++n){const auto&c=all[n];auto r=eval(c);
  if(!f){std::cout<<n<<"\t"<<c.name<<"\t"<<r.status<<"\t"<<r.rank<<"\t"<<r.df<<"\t"<<(double)r.norm<<"\t"<<(double)r.bnorm;for(int j=0;j<c.p&&j<8;++j)std::cout<<"\t"<<(double)r.coef[j];std::cout<<"\n";if(c.name=="visual_two_sensor")for(int i=0;i<c.m;++i)std::cout<<"visual_row\t"<<i<<"\t"<<(double)r.fitted[i]<<"\t"<<(double)r.residual[i]<<"\n";continue;}
  std::cout<<"// "<<c.name<<"\nfn case_"<<n<<"() with Mut,Div,Panic {\n";input("a",2048,c.a);input("b",256,c.b);
  if(c.poison)std::cout<<" a[7]=(0.0/0.0)\n a[2047]=(1.0e308*2.0)\n b[255]=(0.0/0.0)\n";
  std::cout<<" let r=qr_least_squares(&a,&b,"<<c.m<<","<<c.p<<","<<lit(c.rcond)<<")\n assert(r.status=="<<r.status<<")\n assert(r.m=="<<c.m<<" && r.p=="<<c.p<<")\n";
  if(r.status!=0){if(r.status==3)std::cout<<" assert(r.rank=="<<r.rank<<")\n assert(close(r.rcond_used,"<<lit(r.used)<<",0.0,1.0e-14))\n";
   std::cout<<" assert(r.df==0)\n assert(r.residual_norm==0.0)\n var j=0\n while j<8 {assert(r.coefficients[j as usize]==0.0) j=j+1}\n j=0\n while j<256 {assert(r.fitted[j as usize]==0.0 && r.residuals[j as usize]==0.0) j=j+1}\n}\n";continue;}
  std::cout<<" assert(r.rank=="<<c.p<<" && r.df=="<<c.m-c.p<<")\n assert(close(r.rcond_used,"<<lit(r.used)<<",0.0,1.0e-14))\n assert(r.r_diagonal_ratio>r.rcond_used && r.r_diagonal_ratio<=1.000000000001)\n var seen:[i64;8]=[0;8]\n var j=0\n while j<"<<c.p<<" {let v=r.pivot[j as usize] assert(v>=0 && v<"<<c.p<<") assert(seen[v as usize]==0) seen[v as usize]=1 j=j+1}\n";
  if(c.first_pivot>=0)std::cout<<" assert(r.pivot[0]=="<<c.first_pivot<<")\n";
  for(int j=0;j<c.p;++j){if(r.coef[j]==0)std::cout<<" assert(r.coefficients["<<j<<"]==0.0)\n";else std::cout<<" assert(close(r.coefficients["<<j<<"],"<<lit((double)r.coef[j])<<",0.0,"<<lit(c.forward)<<"))\n";}
  // Output error is measured in response units; never a fixed absolute 1.
  Q budget=r.bnorm*(Q)2e-11;
  if(c.tiny_exact)budget=0;
  std::cout<<" var ef:[f64;256]=[0.0;256]\n var er:[f64;256]=[0.0;256]\n";
  for(int i=0;i<c.m;++i){if(r.fitted[i]!=0)std::cout<<" ef["<<i<<"]="<<lit((double)r.fitted[i])<<"\n";if(r.residual[i]!=0)std::cout<<" er["<<i<<"]="<<lit((double)r.residual[i])<<"\n";}
  std::cout<<" j=0\n while j<"<<c.m<<" {assert(close(r.fitted[j as usize],ef[j as usize],"<<lit((double)budget)<<",0.0)) assert(close(r.residuals[j as usize],er[j as usize],"<<lit((double)budget)<<",0.0)) j=j+1}\n assert(close(r.residual_norm,"<<lit((double)r.norm)<<","<<lit((double)budget)<<",0.0))\n}\n";
 }
 if(f){std::cout<<"fn main() with IO,Mut,Div,Panic {\n";for(size_t j=0;j<all.size();++j)std::cout<<" case_"<<j<<"()\n";std::cout<<" println(\"LINALG_LEAST_SQUARES_REFERENCE_PASS\")\n}\n";}
 auto square=eval(all[10]);if(square.status||square.coef[0]!=(Q)2||square.coef[1]!=(Q)-3)return 2;
 auto duplicate=eval(all[12]);if(duplicate.status!=3||duplicate.rank!=1)return 3;
}
