#include <array>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>
struct Result {long double s,k,j,p;};
Result reference(const std::vector<double>& x) {
 // Independent long-double raw central moments, no normalized candidate kernel.
 long double mean=0; for(auto v:x) mean+=v; mean/=x.size();
 long double a=0,b=0,c=0;
 for(auto v:x){long double d=(long double)v-mean; a+=d*d;b+=d*d*d;c+=d*d*d*d;}
 a/=x.size();b/=x.size();c/=x.size();
 long double s=b/std::pow(a,1.5L),k=c/(a*a);
 long double j=x.size()/6.L*(s*s+(k-3)*(k-3)/4);
 return {s,k,j,std::exp(-j/2)};
}
int main(){
 static_assert(std::numeric_limits<long double>::max_exponent>=16384);
 std::vector<std::vector<double>> cases;
 std::vector<double> base{2,4,4,4,5,5,7,9};
 cases.push_back(base);
 for(double scale:{1e-200,1e200,-1e-200,-1e200}){auto x=base;for(auto&v:x)v*=scale;cases.push_back(x);}
 auto offset=base;for(auto&v:offset)v+=1e15;cases.push_back(offset);
 cases.push_back({-1.7e308,1.7e308,0,1e308,-1e308,5e307,-7e307});
 cases.push_back({-1e-320,2e-320,3e-320,-4e-320,5e-320});
 cases.push_back({1,2});cases.push_back({1,2,8});
 std::vector<double> outlier(256,0);outlier[255]=1;cases.push_back(outlier);
 std::vector<double> symmetric;for(int i=0;i<256;i++)symmetric.push_back(i-127.5);cases.push_back(symmetric);
 std::cout<<std::setprecision(17);
 std::cout<<"//@ run-pass\n//@ expect-stdout: STATS_NORMALITY_REFERENCE_PASS\nuse stats::normality::{normality}\n";
 std::cout<<"fn close(a:f64,b:f64,at:f64,rt:f64)->bool { let d=if a>b {a-b}else{b-a} let v=if b<0.0 {-b}else{b} return d<=at+rt*v }\n";
 std::cout<<"fn main() with IO,Mut,Div,Panic {\n";
 for(size_t i=0;i<cases.size();i++){
 auto r=reference(cases[i]);
 std::cerr<<std::setprecision(21)<<"case="<<i<<" n="<<cases[i].size()<<" skew="<<r.s<<" kurt="<<r.k<<" jb="<<r.j<<" p="<<r.p<<"\n";
 std::cout<<" var x"<<i<<":[f64;256]=[0.0;256]\n";
 for(size_t j=0;j<cases[i].size();j++)if(cases[i][j]!=0)std::cout<<" x"<<i<<"["<<j<<"]="<<std::scientific<<cases[i][j]<<"\n";
 std::cout<<" let r"<<i<<"=normality(&x"<<i<<","<<cases[i].size()<<")\n assert(r"<<i<<".status==0)\n";
 auto emit=[&](const char*field,long double val,double at,double rt){std::cout<<" assert(close(r"<<i<<"."<<field<<","<<(double)val<<","<<at<<","<<rt<<"))\n";};
 emit("skewness",r.s,3e-12,3e-12);emit("kurtosis",r.k,3e-12,3e-12);emit("jb",r.j,3e-12,3e-12);
 // Underflow is expected for the 256-observation single outlier.
 emit("p",r.p,r.p==0?0:1e-300,3e-11);
 }
 std::cout<<" println(\"STATS_NORMALITY_REFERENCE_PASS\")\n 0\n}\n";
}
