// Independent C++23 quad CR0 / CR1S cluster-sandwich reference + fixture emitter.
// g++ -std=c++23 -O2 -Wall -Wextra -pedantic linear_cluster_reference.cpp -lquadmath -o linear_cluster_reference
// Reuses only the independent C++ raw-design point-fit reference. Production
// Sounio factorization, grouping, and covariance helpers are not imported.
// Group RAW scores X_g' e_g with std::map, then solve R' z=score and R t=z.
// This differs from grouping production's restored observation influences.
#define main linear_hc3_report_main
#include "linear_hc3_reference.cpp"
#undef main
#include <map>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <utility>
using Label=std::int64_t;
struct ClusterCase {Case data;std::array<Label,256> labels{};int method=0;bool max_input=false;};
struct ClusterReference {
 Result point;int status=0,groups=0;Q correction=0;
 Matrix covariance;Vec se;
};
ClusterReference cluster_reference(const ClusterCase&c){
 ClusterReference r;r.point=evaluate(c.data);
 const int n=c.data.n,p=c.data.p;
 auto error=[&](int status){r.status=status;r.correction=0;r.covariance.assign(p>0?p:0,Vec(p>0?p:0));r.se.assign(p>0?p:0,0);return r;};
 if(r.point.fit_status!=0)return error(1);
 if(c.method!=0&&c.method!=1)return error(5);
 if(n<=p)return error(2);
 std::map<Label,Vec> scores;
 for(int i=0;i<n;++i){
  auto [it,inserted]=scores.try_emplace(c.labels[i],Vec(p));
  (void)inserted;
  for(int j=0;j<p;++j)it->second[j]+=static_cast<Q>(c.data.a[i*8+j])*r.point.reported_residual[i];
 }
 r.groups=static_cast<int>(scores.size());
 if(r.groups<2)return error(3);
 Matrix q(p,Vec(n)),upper(p,Vec(p));
 for(int j=0;j<p;++j){
  Vec v(n);for(int i=0;i<n;++i)v[i]=static_cast<Q>(c.data.a[i*8+j]);
  for(int pass=0;pass<2;++pass)for(int k=0;k<j;++k){
   Q a=dot(q[k],v);upper[k][j]+=a;
   for(int i=0;i<n;++i)v[i]-=a*q[k][i];
  }
  upper[j][j]=sqrtq(dot(v,v));
  if(!finiteq(upper[j][j])||upper[j][j]<=0)return error(4);
  for(int i=0;i<n;++i)q[j][i]=v[i]/upper[j][j];
 }
 const Q correction=c.method==0?static_cast<Q>(1):
  (static_cast<Q>(r.groups)/(r.groups-1))*(static_cast<Q>(n-1)/(n-p));
 const Q scale=sqrtq(correction);
 r.covariance.assign(p,Vec(p));r.se.assign(p,0);
 for(const auto&[label,score]:scores){
  (void)label;Vec z(p),influence(p);
  for(int j=0;j<p;++j){
   Q value=score[j];for(int k=0;k<j;++k)value-=upper[k][j]*z[k];
   z[j]=value/upper[j][j];
   if(!finiteq(z[j]))return error(4);
  }
  for(int j=p-1;j>=0;--j){
   Q value=z[j];for(int k=j+1;k<p;++k)value-=upper[j][k]*influence[k];
   influence[j]=value/upper[j][j];
   if(!finiteq(influence[j]))return error(4);
  }
  for(Q&v:influence){v*=scale;if(!strict_double(v))return error(4);}
  for(int j=0;j<p;++j)for(int k=0;k<p;++k)r.covariance[j][k]+=influence[j]*influence[k];
 }
 for(int j=0;j<p;++j){
  for(int k=0;k<p;++k)if(!finite_double(r.covariance[j][k]))return error(4);
  r.se[j]=sqrtq(r.covariance[j][j]);
  if(!strict_double(r.covariance[j][j])||!strict_double(r.se[j]))return error(4);
 }
 r.correction=correction;return r;
}
std::vector<ClusterCase> cluster_cases(){
 const auto hc=cases();
 auto named=[&](const std::string&name){for(const auto&c:hc)if(c.name==name)return c;std::cerr<<"Missing case "<<name<<"\n";std::exit(2);};
 auto labels=[](std::initializer_list<Label>x){std::array<Label,256>a{};std::copy(x.begin(),x.end(),a.begin());return a;};
 Case visual=dense("visual_CR0",{{1,-1},{1,-1},{1,1},{1,1},{1,-1},{1,-1},{1,1},{1,1}},{8,10,12,14,9,11,13,15});
 std::vector<ClusterCase> all;
 auto add=[&](Case data,std::array<Label,256>ids,int method,std::string name){data.name=std::move(name);all.push_back({data,ids,method,false});};
 auto pair=labels({0,0,1,1,2,2,3,3});
 add(visual,pair,0,"visual_CR0");add(visual,pair,1,"visual_CR1S");
 auto uneven=labels({0,0,0,7,7,99,99,99});
 auto ordinary=named("heteroscedastic");
 add(ordinary,uneven,0,"unequal_clusters_CR0");add(ordinary,uneven,1,"unequal_clusters_CR1S");
 add(ordinary,labels({0,1,0,1,2,3,2,3}),0,"noncontiguous");
 add(visual,labels({std::numeric_limits<Label>::min(),std::numeric_limits<Label>::min(),-17,-17,std::numeric_limits<Label>::max(),std::numeric_limits<Label>::max(),0,0}),1,"signed_relabel");
 Case row=ordinary;auto ids=uneven;
 for(int i=0;i<row.n/2;++i){int k=row.n-1-i;std::swap(row.b[i],row.b[k]);std::swap(ids[i],ids[k]);for(int j=0;j<row.p;++j)std::swap(row.a[i*8+j],row.a[k*8+j]);}
 add(row,ids,1,"row_permutation");
 add(named("column_permutation"),uneven,1,"column_permutation");
 for(const char*name:{"response_tiny","response_huge","design_tiny","design_huge","mixed_column_units"})
  add(named(name),uneven,1,name);
 std::array<Label,256> singleton{};for(int i=0;i<256;++i)singleton[i]=i;
 add(ordinary,singleton,0,"singleton_CR0_HC0");add(ordinary,singleton,1,"singleton_CR1S_HC1");
 auto mean=dense("mean",{{1},{1},{1},{1}},{1,2,3,6});
 add(mean,singleton,0,"singleton_intercept_CR0");add(mean,singleton,1,"singleton_intercept_CR1S");
 add(ordinary,{},0,"one_cluster");
 add(named("zero_response"),labels({0,0,1,1,1}),0,"zero_response_CR0");
 add(named("zero_response"),labels({0,0,1,1,1}),1,"zero_response_CR1S");
 add(dense("group_cancel",{{1},{1},{1},{1}},{1,-1,2,-2}),labels({0,0,1,1}),0,"exact_group_score_zero");
 add(named("duplicate_columns"),labels({0,0,1,1}),0,"rank_failure");
 add(named("active_nonfinite"),uneven,0,"nonfinite_failure");
 add(named("active_nonfinite"),uneven,7,"fit_before_method");
 add(ordinary,uneven,7,"invalid_method");
 add(named("square_no_df"),labels({0,1}),0,"no_df");
 add(named("covariance_underflow"),uneven,0,"covariance_underflow");
 add(named("covariance_overflow"),uneven,1,"covariance_overflow");
 auto maximum=named("max_256_by_8");
 add(maximum,singleton,0,"max_G256_CR0");all.back().max_input=true;
 add(maximum,singleton,1,"max_G256_CR1S");all.back().max_input=true;
 std::array<Label,256> halves{};for(int i=0;i<256;++i)halves[i]=i/128;
 Case contrast=maximum;
 for(int i=0;i<256;++i)for(int j=0;j<8;++j)
  contrast.b[i]+=(i<128?1.0:-1.0)*contrast.a[i*8+j]*(j+1)*0.0625;
 add(contrast,halves,1,"max_G2_CR1S");all.back().max_input=true;
 return all;
}
std::string literal(double value){
 if(std::isnan(value))return "(0.0/0.0)";
 if(std::isinf(value))return value<0?"(-1.0e308*10.0)":"(1.0e308*10.0)";
 std::ostringstream s;s<<std::scientific<<std::setprecision(17);
 if(std::fpclassify(value)==FP_SUBNORMAL)s<<"("<<std::ldexp(value,52)<<" / 4503599627370496.0)";
 else s<<value;
 return s.str();
}
std::string label_literal(Label value){
 if(value==std::numeric_limits<Label>::min())return "(0-9223372036854775807-1)";
 return std::to_string(value);
}
const char* fixture_helpers=R"SIO(
// Independent raw-score-map / quad triangular-solve oracle.
// CR1S means G/(G-1)*(n-1)/(n-p), not clubSandwich's bare CR1.
use stats::regression::robust_covariance::{LinearClusterResult,multiple_linear_cluster}
use math::pure::{is_finite,sqrt}
fn magnitude(x:f64)->f64 {if x<0.0 {-x}else{x}}
fn relative(a:f64,b:f64,tolerance:f64) with Mut,Div,Panic {
 assert(is_finite(a) && is_finite(b))
 if b==0.0 {assert(a==0.0)} else {assert(a!=0.0) assert(magnitude((a-b)/b)<=tolerance)}
}
fn covariance_close(a:f64,b:f64,unit:f64) with Mut,Div,Panic {
 assert(is_finite(a) && is_finite(b) && is_finite(unit))
 if unit==0.0 {assert(a==0.0 && b==0.0)} else {assert(magnitude((a-b)/unit)<=2.0e-7)}
}
// Conditional on actual reported residuals: this analytically cancelling
// dataset does not force a rounded production fit to publish invented zeros.
fn group_zero_reported(r:&LinearClusterResult) with Mut,Div,Panic {
 let g0=(r.fit.residuals[0]+r.fit.residuals[1])/4.0
 let g1=(r.fit.residuals[2]+r.fit.residuals[3])/4.0
 let covariance=g0*g0+g1*g1
 relative(r.covariance[0],covariance,2.0e-7)
 relative(r.standard_errors[0],sqrt(covariance),2.0e-7)
 var i:i64=1
 while i<64 {assert(r.covariance[i as usize]==0.0) i=i+1}
 i=1
 while i<8 {assert(r.standard_errors[i as usize]==0.0) i=i+1}
}
)SIO";
int main(int argc,char**argv){
 const bool fixture=argc==2&&std::string(argv[1])=="--fixture";
 if(argc>1&&!fixture)return 2;
 const auto all=cluster_cases();
 // Exact rational visual design: independent closed-form anchors supplement QR.
 const auto v0=cluster_reference(all[0]),v1=cluster_reference(all[1]);
 if(v0.status||v1.status||fabsq(v0.covariance[0][0]-static_cast<Q>(1)/16)>static_cast<Q>(1e-28)||
    fabsq(v0.covariance[1][1]-static_cast<Q>(1)/16)>static_cast<Q>(1e-28)||
    fabsq(v1.correction-static_cast<Q>(14)/9)>static_cast<Q>(1e-28))return 3;
 if(fixture)std::cout<<"//@ run-pass\n//@ expect-stdout: STATS_LINEAR_CLUSTER_REFERENCE_PASS\n"<<fixture_helpers;
 for(std::size_t id=0;id<all.size();++id){
  const auto&c=all[id];const auto r=cluster_reference(c);const auto&d=c.data;
  if(!fixture){
   std::cout<<"case\t"<<d.name<<"\t"<<d.n<<"\t"<<d.p<<"\t"<<c.method<<"\t"<<r.point.fit_status<<"\t"<<r.status<<"\t"<<r.groups<<"\t";number(r.correction);std::cout<<"\n";
   if(r.point.fit_status==0)for(int j=0;j<d.p;++j){entry("coefficient",d,j,-1,r.point.beta[j]);if(r.status==0){entry("standard_error",d,j,-1,r.se[j]);for(int k=0;k<d.p;++k)entry("covariance",d,j,k,r.covariance[j][k]);}}
   continue;
  }
  std::cout<<"// "<<d.name<<"\nfn case_"<<id<<"() with Mut,Div,Panic {\n var a:[f64;2048]=[0.0;2048]\n var b:[f64;256]=[0.0;256]\n var ids:[i64;256]=[0;256]\n";
  if(c.max_input){
   std::cout<<" let grouped="<<(d.name=="max_G2_CR1S"?"true":"false")<<"\n";
   std::cout<<R"SIO( var row:i64=0
 while row<256 {
  var response=(((row*7)%13-6) as f64)*0.125
  var j:i64=0
  while j<8 {
   var ti=row
   var tj=j
   var parity:i64=0
   while tj>0 {if ti%2==1 && tj%2==1 {parity=1-parity} ti=ti/2 tj=tj/2}
   let x=if parity==1 {-1.0}else{1.0}
   a[(row*8+j) as usize]=x
   response=response+x*((j+1) as f64)
   if grouped {let sign=if row<128 {1.0}else{-1.0} response=response+sign*x*((j+1) as f64)*0.0625}
   j=j+1
  }
  b[row as usize]=response
)SIO";
   std::cout<<"  ids[row as usize]="<<(d.name=="max_G2_CR1S"?"row/128":"row")<<"\n  row=row+1\n }\n";
  }else{
   for(int i=0;i<d.n;++i){for(int j=0;j<d.p;++j)std::cout<<" a["<<i*8+j<<"]="<<literal(d.a[i*8+j])<<"\n";std::cout<<" b["<<i<<"]="<<literal(d.b[i])<<"\n ids["<<i<<"]="<<label_literal(c.labels[i])<<"\n";}
  }
  std::cout<<" let r=multiple_linear_cluster(&a,&b,&ids,"<<d.n<<","<<d.p<<","<<literal(d.rcond)<<","<<c.method<<")\n assert(r.fit.status=="<<r.point.fit_status<<" && r.covariance_status=="<<r.status<<")\n assert(r.clusters=="<<r.groups<<" && r.method=="<<c.method<<" && !r.assumptions_assessed)\n";
  if(r.status==0)std::cout<<" relative(r.correction_factor,"<<literal(static_cast<double>(r.correction))<<",1.0e-13)\n";
  else std::cout<<" assert(r.correction_factor==0.0)\n";
  if(r.point.fit_status==0)for(int j=0;j<d.p;++j){if(d.name=="exact_group_score_zero")std::cout<<" assert(magnitude(r.fit.coefficients["<<j<<"])<=2.0e-12)\n";else std::cout<<" relative(r.fit.coefficients["<<j<<"],"<<literal(static_cast<double>(r.point.beta[j]))<<",1.0e-9)\n";}
  if(r.status==0){
   if(d.name=="exact_group_score_zero")std::cout<<" group_zero_reported(&r)\n";
   else {
   for(int j=0;j<d.p;++j){
    std::cout<<" relative(r.standard_errors["<<j<<"],"<<literal(static_cast<double>(r.se[j]))<<",2.0e-7)\n";
    for(int k=0;k<d.p;++k){
     const Q unit=r.se[j]*r.se[k];
     std::cout<<" covariance_close(r.covariance["<<j*8+k<<"],"<<literal(static_cast<double>(r.covariance[j][k]))<<","<<literal(static_cast<double>(unit))<<")\n assert(r.covariance["<<j*8+k<<"]==r.covariance["<<k*8+j<<"])\n";
     if(j==k)std::cout<<" relative(r.covariance["<<j*8+k<<"],"<<literal(static_cast<double>(r.covariance[j][k]))<<",2.0e-7)\n";
    }
   }
   std::cout<<" var i:i64=0\n while i<8 {if i>="<<d.p<<" {assert(r.standard_errors[i as usize]==0.0)} var j:i64=0 while j<8 {if i>="<<d.p<<" || j>="<<d.p<<" {assert(r.covariance[(i*8+j) as usize]==0.0)} j=j+1} i=i+1}\n";
  }
  }else std::cout<<" var i:i64=0\n while i<64 {assert(r.covariance[i as usize]==0.0) i=i+1}\n i=0\n while i<8 {assert(r.standard_errors[i as usize]==0.0) i=i+1}\n";
  std::cout<<"}\n";
 }
 if(fixture){std::cout<<"fn main()->i32 with IO,Mut,Div,Panic {\n";for(std::size_t id=0;id<all.size();++id)std::cout<<" case_"<<id<<"()\n";std::cout<<" println(\"STATS_LINEAR_CLUSTER_REFERENCE_PASS\")\n 0\n}\n";}
}
