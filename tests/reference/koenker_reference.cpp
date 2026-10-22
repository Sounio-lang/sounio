#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <utility>
struct R { long double r2,lm,p; };
R reference(const std::vector<double>& x,const std::vector<double>& e) {
 long double mx=0,my=0;
 for(size_t i=0;i<x.size();i++){mx+=x[i];my+=(long double)e[i]*e[i];}
 mx/=x.size();my/=x.size();
 long double xx=0,xy=0,yy=0;
 for(size_t i=0;i<x.size();i++){auto dx=(long double)x[i]-mx;auto dy=(long double)e[i]*e[i]-my;xx+=dx*dx;xy+=dx*dy;yy+=dy*dy;}
 // Explicit fitted auxiliary residual SSE, unlike candidate correlation formula.
 auto b=xy/xx;long double sse=0;
 for(size_t i=0;i<x.size();i++){auto residual=(long double)e[i]*e[i]-my-b*((long double)x[i]-mx);sse+=residual*residual;}
 auto r2=1-sse/yy;
 if(r2<0 && r2>-1e-17L)r2=0;
 auto lm=x.size()*r2;
 return {r2,lm,std::erfc(std::sqrt(lm/2))};
}
int main(){
 static_assert(std::numeric_limits<long double>::max_exponent>=16384);
 static_assert(-0.5*(1.0-0.25)+1.5*0.25==0.0); // exact zero tick for the scientific figure
 std::vector<std::pair<std::vector<double>,std::vector<double>>> cases;
 std::vector<double> x{0,1,2,3,4,5,6,7},e{.1,-.3,.2,-.6,.8,-.2,1.1,-1.8};
 cases.push_back({x,e});
 for(double factor:{1e-200,1e200,-1e-200,-1e200}){auto u=x,v=e;for(auto&a:u)a*=factor;for(auto&a:v)a*=factor;cases.push_back({u,v});}
 auto off=x;for(auto&a:off)a+=1e15;cases.push_back({off,e});
 cases.push_back({{-1.7e308,1.7e308,0,1e308,-1e308,5e307,-7e307},{1,2,-3,4,-5,6,-7}});
 cases.push_back({{0,1,2,3,4},{1e-320,-2e-320,3e-320,-5e-320,8e-320}});
 cases.push_back({{0,1,2,3,4},{1,1+0x1p-52,1+0x1p-51,1+0x1.8p-51,1+0x1p-50}});
 cases.push_back({{-2,-1,0,1,2},{2,1,0,-1,-2}});
 cases.push_back({{1,2,3},{1,-2,4}});
 std::vector<double> hundred_x,hundred_e;
 for(int i=0;i<100;i++){hundred_x.push_back(i);hundred_e.push_back((i%2?-1:1)*std::sqrt(i+1.));}
 cases.push_back({hundred_x,hundred_e});
 std::cout<<std::scientific<<std::setprecision(17);
 std::cout<<"//@ run-pass\n//@ expect-stdout: STATS_KOENKER_REFERENCE_PASS\nuse stats::heteroscedasticity::{koenker_bp_simple}\nuse stats::regression::assessments::{assess_simple_linear}\nuse special::igamma::{igamma_upper}\n";
 std::cout<<"fn close(a:f64,b:f64,at:f64,rt:f64)->bool {let d=if a>b{a-b}else{b-a} let v=if b<0.0{-b}else{b} return d<=at+rt*v}\nfn main() with IO,Mut,Div,Panic {\n";
 for(size_t c=0;c<cases.size();c++){
 auto [u,v]=cases[c];auto r=reference(u,v);
 std::cerr<<std::setprecision(21)<<"case="<<c<<" n="<<u.size()<<" R2="<<r.r2<<" LM="<<r.lm<<" p="<<r.p<<"\n";
 std::cout<<" var x"<<c<<":[f64;100]=[0.0;100]\n var e"<<c<<":[f64;100]=[0.0;100]\n";
 for(size_t j=0;j<u.size();j++)std::cout<<" x"<<c<<"["<<j<<"]="<<u[j]<<"\n e"<<c<<"["<<j<<"]="<<v[j]<<"\n";
 std::cout<<" let r"<<c<<"=koenker_bp_simple(&e"<<c<<",&x"<<c<<","<<u.size()<<")\n assert(r"<<c<<".status==0)\n";
 std::cout<<" assert(close(r"<<c<<".lm,"<<(double)r.lm<<",3.0e-11,3.0e-11))\n";
 std::cout<<" assert(close(r"<<c<<".r_squared,"<<(double)r.r2<<",3.0e-12,3.0e-11))\n";
 std::cout<<" assert(close(r"<<c<<".p,"<<(double)r.p<<",1.0e-300,8.0e-11))\n";
 }
 for(double lm:{0.,1e-12,.01,.1,1.,2.999,3.,3.001,10.,20.,50.,100.}){
 auto p=std::erfc(std::sqrt((long double)lm/2));
 std::cerr<<"tail LM="<<lm<<" p="<<p<<"\n";
 std::cout<<" assert(close(igamma_upper(0.5,"<<lm/2<<"),"<<(double)p<<",1.0e-300,8.0e-11))\n";
 }
 std::vector<double> mx,my;
 for(int i=0;i<20;i++){mx.push_back(i);my.push_back(2+3*i+(i%2?-1:1)*(.2+.1*i));}
 // Independent long-double intercept/slope and raw residuals.
 long double ax=0,ay=0;for(int i=0;i<20;i++){ax+=mx[i];ay+=my[i];}ax/=20;ay/=20;
 long double xx=0,xy=0;for(int i=0;i<20;i++){xx+=(mx[i]-ax)*(mx[i]-ax);xy+=(mx[i]-ax)*(my[i]-ay);}
 auto slope=xy/xx;auto intercept=ay-slope*ax;
 std::vector<double> residuals;
 long double m2=0,m3=0,m4=0;
 for(int i=0;i<20;i++){auto d=(long double)my[i]-intercept-slope*mx[i];residuals.push_back((double)d);m2+=d*d;m3+=d*d*d;m4+=d*d*d*d;}
 m2/=20;m3/=20;m4/=20;auto s=m3/std::pow(m2,1.5L),k=m4/(m2*m2);
 auto jb=20.L/6*(s*s+(k-3)*(k-3)/4);auto kb=reference(mx,residuals);
 std::cerr<<"model slope="<<slope<<" intercept="<<intercept<<" JB="<<jb<<" pJB="<<std::exp(-jb/2)<<" LM="<<kb.lm<<" pKB="<<kb.p<<"\n";
 std::cout<<" var mx:[f64;100]=[0.0;100]\n var my:[f64;100]=[0.0;100]\n";
 for(int i=0;i<20;i++)std::cout<<" mx["<<i<<"]="<<mx[i]<<"\n my["<<i<<"]="<<my[i]<<"\n";
 std::cout<<" let a=assess_simple_linear(&mx,&my,20)\n assert(a.fit.status==0 && a.normality.status==0 && a.heteroscedasticity.status==0)\n";
 std::cout<<" assert(close(a.fit.slope,"<<(double)slope<<",1.0e-12,1.0e-12))\n";
 std::cout<<" assert(close(a.normality.jb,"<<(double)jb<<",3.0e-11,3.0e-11))\n";
 std::cout<<" assert(close(a.heteroscedasticity.lm,"<<(double)kb.lm<<",3.0e-11,3.0e-11))\n";
 std::cout<<" assert(close(a.heteroscedasticity.p,"<<(double)kb.p<<",1.0e-300,8.0e-11))\n";
 std::cout<<" println(\"STATS_KOENKER_REFERENCE_PASS\")\n 0\n}\n";
}
