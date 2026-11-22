// Independent C++23 weighted least-squares reference, linked with -lquadmath.
// Direct quadmath weighted raw design/response; not production binary64
// sqrt(w)/sqrt(maxw), outer normalization, or Householder implementation.
// The included independent MGS oracle provides only high precision primitives.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#define main unweighted_reference_unused_cli
#include "least_squares_reference.cpp"
#undef main
#pragma GCC diagnostic pop
struct WeightedCase{Case data;V weights;bool strict_norm=false;};
struct WeightedResult:Result{double weight_scale=0;Q weighted_norm=0;};
WeightedResult weighted_eval(const WeightedCase&wc) {
 const auto&c=wc.data;WeightedResult out;out.coef.resize(8);out.fitted.resize(256);out.residual.resize(256);
 if(c.m<1||c.m>256||c.p<1||c.p>8||c.p>c.m){out.status=1;return out;}
 if(!std::isfinite(c.rcond)||c.rcond<0||c.rcond>=1){out.status=5;return out;}
 for(int i=0;i<c.m;++i){if(!std::isfinite(c.b[i])){out.status=2;return out;}for(int j=0;j<c.p;++j)if(!std::isfinite(c.a[i*8+j])){out.status=2;return out;}}
 for(int i=0;i<c.m;++i)if(!std::isfinite(wc.weights[i])||wc.weights[i]<=0){out.status=6;return out;}
 for(double w:wc.weights)out.weight_scale=std::max(out.weight_scale,w);
 out.used=c.rcond==0?c.m*std::numeric_limits<double>::epsilon():c.rcond;
 M weighted(c.p,std::vector<Q>(c.m)),equilibrated=weighted;
 std::vector<Q>b(c.m),roots(c.m);
 for(int i=0;i<c.m;++i){roots[i]=sqrtq((Q)wc.weights[i]/(Q)out.weight_scale);b[i]=(Q)c.b[i]*roots[i];}
 for(int j=0;j<c.p;++j){Q scale=0;for(int i=0;i<c.m;++i){weighted[j][i]=(Q)c.a[i*8+j]*roots[i];scale=fmaxq(scale,fabsq(weighted[j][i]));}for(int i=0;i<c.m;++i)equilibrated[j][i]=scale==0?0:weighted[j][i]/scale;}
 auto rank=factor(equilibrated,c.m,c.p,true,(Q)out.used);out.rank=rank.rank;
 if(rank.rank>0)out.diagonal_ratio=rank.last/rank.first;
 if(out.rank<c.p){out.status=3;return out;}
 auto solve=factor(weighted,c.m,c.p,false,0);
 std::vector<Q>z(c.p);for(int j=0;j<c.p;++j)z[j]=dot(solve.q[j],b);
 for(int j=c.p-1;j>=0;--j){Q v=z[j];for(int k=j+1;k<c.p;++k)v-=solve.r[j][k]*out.coef[k];out.coef[j]=v/solve.r[j][j];}
 for(int j=0;j<c.p;++j)if(!std::isfinite((double)out.coef[j])||(out.coef[j]!=0&&(double)out.coef[j]==0)){out=WeightedResult{};out.status=4;out.coef.resize(8);out.fitted.resize(256);out.residual.resize(256);return out;}
 Q rr=0,ww=0,bb=0;
 for(int i=0;i<c.m;++i){
  Q fit=0;for(int j=0;j<c.p;++j)fit+=(Q)c.a[i*8+j]*out.coef[j];
  out.fitted[i]=fit;
  // Compare the API's raw residual convention using rounded reported fits.
  double raw=(double)c.b[i]-(double)fit;out.residual[i]=(Q)raw;
  rr+=out.residual[i]*out.residual[i];
  ww+=((Q)wc.weights[i]/(Q)out.weight_scale)*out.residual[i]*out.residual[i];
  bb+=(Q)c.b[i]*(Q)c.b[i];
 }
 out.norm=sqrtq(rr);out.weighted_norm=sqrtq(ww);out.bnorm=sqrtq(bb);out.df=c.m-c.p;
 return out;
}
std::vector<WeightedCase> weighted_cases() {
 auto base=cases();std::vector<WeightedCase> out;
 auto add=[&](Case c,V w,bool strict=false){out.push_back({c,w,strict});};
 auto ordinary=base[0];ordinary.name="weighted_noisy";V weights={1,4,1,4,1,4};add(ordinary,weights);
 ordinary.name="equal_weights_ols";add(ordinary,V(6,1));
 for(auto [name,scale]:{std::pair{"weights_global_tiny",1e-200},std::pair{"weights_global_huge",1e200}}){ordinary=base[0];ordinary.name=name;V w=weights;for(double&v:w)v*=scale;add(ordinary,w);}
 for(int index:{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27}) {
  auto c=base[index];c.name="weighted_"+c.name;V w(c.m,1);
  if(index==0||index==3||(index>=5&&index<=9)||index==16)for(int i=0;i<c.m;++i)w[i]=1+i%4;
  add(c,w);
 }
 double tiny=std::numeric_limits<double>::denorm_min(),max=std::numeric_limits<double>::max();
 add(dense("critical_extreme_weight_root",{{0},{1}},{1,2}),{max,tiny});
 add(dense("tiny_orthogonal_response",{{0},{1}},{1,2e-300}),{1,1});
 add(dense("opposed_residual_scales",{{0},{0},{1}},{1e308,1e-7,0}),{tiny,max,max},true);
 auto leverage=base[0];leverage.name="high_leverage_weight";V lw(6,1);lw.back()=1e8;add(leverage,lw);
 for(auto [name,w]:{std::pair{"invalid_zero_weight",0.},std::pair{"invalid_negative_weight",-1.},std::pair{"invalid_nan_weight",std::numeric_limits<double>::quiet_NaN()},std::pair{"invalid_inf_weight",std::numeric_limits<double>::infinity()}}){auto c=base[0];c.name=name;V ws(6,1);ws[1]=w;add(c,ws);}
 auto visual=dense("visual_wls",{{1,0},{1,1},{1,2},{1,3},{1,4},{1,5},{1,6},{1,7}},{1.2,2.7,5.4,6.6,9.6,10.4,13.5,14.5});
 add(visual,{1,4,1,4,1,4,1,4});visual.name="visual_ols";add(visual,V(8,1));
 return out;
}
int main(int argc,char**argv) {
 bool fixture=argc>1&&std::string(argv[1])=="--fixture";auto all=weighted_cases();
 std::cout<<std::scientific<<std::setprecision(17);
 if(fixture)std::cout<<"//@ run-pass\n//@ expect-stdout: LINALG_WEIGHTED_LEAST_SQUARES_REFERENCE_PASS\n// C++23 quadmath weighted raw-design MGS reference; no inferential claim.\nuse linalg::weighted_least_squares::{weighted_qr_least_squares}\nfn close(a:f64,b:f64,at:f64,rt:f64)->bool {let d=if a>b{a-b}else{b-a} let v=if b<0.0{-b}else{b} return d<=at+rt*v}\n";
 for(size_t n=0;n<all.size();++n){const auto&wc=all[n];const auto&c=wc.data;auto r=weighted_eval(wc);
  if(!fixture){std::cout<<n<<"\t"<<c.name<<"\t"<<r.status<<"\t"<<r.rank<<"\t"<<r.df<<"\t"<<r.weight_scale<<"\t"<<(double)r.norm<<"\t"<<(double)r.weighted_norm;for(int j=0;j<c.p&&j<8;++j)std::cout<<"\t"<<(double)r.coef[j];std::cout<<"\n";if(c.name=="visual_wls"||c.name=="visual_ols")for(int i=0;i<c.m;++i)std::cout<<c.name<<"_row\t"<<i<<"\t"<<(double)r.fitted[i]<<"\t"<<(double)r.residual[i]<<"\n";continue;}
  std::cout<<"// "<<c.name<<"\nfn case_"<<n<<"() with Mut,Div,Panic {\n";input("a",2048,c.a);input("b",256,c.b);input("w",256,wc.weights);
  if(c.poison)std::cout<<" a[7]=(0.0/0.0)\n a[2047]=(1.0e308*2.0)\n b[255]=(0.0/0.0)\n w[255]=(0.0/0.0)\n";
  std::cout<<" let r=weighted_qr_least_squares(&a,&b,&w,"<<c.m<<","<<c.p<<","<<lit(c.rcond)<<")\n assert(r.status=="<<r.status<<")\n assert(r.m=="<<c.m<<" && r.p=="<<c.p<<")\n assert(!r.variance_model_assessed)\n";
  if(r.status!=0){if(r.status==3)std::cout<<" assert(r.rank=="<<r.rank<<")\n assert(close(r.rcond_used,"<<lit(r.used)<<",0.0,1.0e-14))\n assert(r.weight_scale=="<<lit(r.weight_scale)<<")\n";else std::cout<<" assert(r.weight_scale==0.0)\n";
   std::cout<<" assert(r.df==0)\n assert(r.residual_norm==0.0 && r.normalized_weighted_residual_norm==0.0)\n var j=0\n while j<8 {assert(r.coefficients[j as usize]==0.0) j=j+1}\n j=0\n while j<256 {assert(r.fitted[j as usize]==0.0 && r.residuals[j as usize]==0.0) j=j+1}\n}\n";continue;}
  std::cout<<" assert(r.rank=="<<c.p<<" && r.df=="<<c.m-c.p<<")\n assert(r.weight_scale=="<<lit(r.weight_scale)<<")\n assert(close(r.rcond_used,"<<lit(r.used)<<",0.0,1.0e-14))\n assert(r.r_diagonal_ratio>r.rcond_used && r.r_diagonal_ratio<=1.000000000001)\n var seen:[i64;8]=[0;8]\n var j=0\n while j<"<<c.p<<" {let v=r.pivot[j as usize] assert(v>=0 && v<"<<c.p<<") assert(seen[v as usize]==0) seen[v as usize]=1 j=j+1}\n";
  if(c.first_pivot>=0)std::cout<<" assert(r.pivot[0]=="<<c.first_pivot<<")\n";
  for(int j=0;j<c.p;++j){if(r.coef[j]==0)std::cout<<" assert(r.coefficients["<<j<<"]==0.0)\n";else std::cout<<" assert(close(r.coefficients["<<j<<"],"<<lit((double)r.coef[j])<<",0.0,"<<lit(c.forward)<<"))\n";}
  Q budget=r.bnorm*(Q)3e-11;if(c.tiny_exact)budget=0;
  std::cout<<" var ef:[f64;256]=[0.0;256]\n var er:[f64;256]=[0.0;256]\n";
  for(int i=0;i<c.m;++i){if(r.fitted[i]!=0)std::cout<<" ef["<<i<<"]="<<lit((double)r.fitted[i])<<"\n";if(r.residual[i]!=0)std::cout<<" er["<<i<<"]="<<lit((double)r.residual[i])<<"\n";}
  std::cout<<" j=0\n while j<"<<c.m<<" {assert(close(r.fitted[j as usize],ef[j as usize],"<<lit((double)budget)<<",0.0)) assert(close(r.residuals[j as usize],er[j as usize],"<<lit((double)budget)<<",0.0)) j=j+1}\n assert(close(r.residual_norm,"<<lit((double)r.norm)<<","<<lit((double)budget)<<",0.0))\n";
  // Distinguish tiny weighted objective norms from huge raw response norms.
  // Allow a stated binary64 root-factor quantum budget (not a blanket 1).
  Q factor_budget_sq=0;
  for(int i=0;i<c.m;++i){Q root=sqrtq((Q)wc.weights[i]/(Q)r.weight_scale);Q uncertainty=4*(Q)std::numeric_limits<double>::epsilon()*root+2*(Q)std::numeric_limits<double>::denorm_min();Q term=fabsq(r.residual[i])*uncertainty;factor_budget_sq+=term*term;}
  bool meaningful=wc.strict_norm||r.weighted_norm>r.bnorm*(Q)1e-12;
  Q wat=meaningful?sqrtq(factor_budget_sq):budget;Q wrt=meaningful?(Q)1e-9:0;
  std::cout<<" assert(close(r.normalized_weighted_residual_norm,"<<lit((double)r.weighted_norm)<<","<<lit((double)wat)<<","<<lit((double)wrt)<<"))\n}\n";
 }
 if(fixture){std::cout<<"fn main() with IO,Mut,Div,Panic {\n";for(size_t j=0;j<all.size();++j)std::cout<<" case_"<<j<<"()\n";std::cout<<" println(\"LINALG_WEIGHTED_LEAST_SQUARES_REFERENCE_PASS\")\n}\n";}
 // Reference invariances and the max/min weight root regression.
 auto a=weighted_eval(all[0]),s=weighted_eval(all[2]),l=weighted_eval(all[3]);
 for(int j=0;j<2;++j)if(fabsq(a.coef[j]-s.coef[j])>fabsq(a.coef[j])*(Q)1e-14||fabsq(a.coef[j]-l.coef[j])>fabsq(a.coef[j])*(Q)1e-14)return 2;
 for(const auto&c:all)if(c.data.name=="critical_extreme_weight_root"){auto r=weighted_eval(c);if(r.status||r.rank!=1||r.coef[0]!=(Q)2||r.norm!=(Q)1||r.weighted_norm!=(Q)1)return 3;}
 return 0;
}
