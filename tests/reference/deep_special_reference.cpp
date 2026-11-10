// Independent closed-form C++23/quadmath oracle and Sounio fixture emitter.
// Compile on authorized cluster: g++ -std=c++23 -O2 -Wall -Wextra -pedantic FILE -lquadmath -o EMITTER
#include <quadmath.h>
#include <iomanip>
#include <iostream>
#include <string>
using Q=__float128;
struct Pair {Q p,q,lp,lq;};
Q number(const char* s){return strtoflt128(s,nullptr);}
Pair lower(Q p){return {p,1-p,logq(p),log1pq(-p)};}
Pair upper(Q q){return {1-q,q,log1pq(-q),logq(q)};}
void checkscalar(const std::string& name,Q v){
 double expected=static_cast<double>(v);
 if(expected==0) std::cout<<" assert(!r."<<name<<"_available && r."<<name<<"==0.0)\n";
 else std::cout<<" assert(r."<<name<<"_available && close(r."<<name<<","<<expected<<",2.0e-11))\n";
}
void emit(const std::string& call,Pair v){
 std::cout<<" {\n let r="<<call<<"\n assert(r.status==0)\n";
 checkscalar("lower",v.p);checkscalar("upper",v.q);
 checkscalar("log_lower",v.lp);checkscalar("log_upper",v.lq);
 std::cout<<" }\n";
}
int main(){
 std::cout<<std::setprecision(17)<<std::showpoint<<R"SIO(//@ run-pass
//@ expect-stdout: STDLIB_DEEP_SPECIAL_REFERENCE_PASS
use special::checked_gamma_beta::{gamma_pq_checked,beta_pq_checked,gamma_inverse_checked,beta_inverse_checked}
fn close(a:f64,b:f64,tol:f64)->bool with Div {
 let scale=if b<0.0 {-b} else {b}
 let gap=if a>b {a-b} else {b-a}
 gap<=tol*scale
}
fn main()->i32 with IO,Mut,Div,Panic {
)SIO";
 emit("gamma_pq_checked(1.0,1.0,10000,1.0e-12)",upper(expq(-Q(1))));
 emit("gamma_pq_checked(1.0,1000.0,10000,1.0e-12)",upper(expq(-Q(1000))));
 emit("gamma_pq_checked(1.0,1.0e-300,10000,1.0e-12)",lower(-expm1q(-Q(static_cast<double>(1e-300)))));
 emit("gamma_pq_checked(2.0,3.0,10000,1.0e-12)",upper(Q(4)*expq(-Q(3))));
 emit("gamma_pq_checked(0.5,2.0,10000,1.0e-12)",upper(erfcq(sqrtq(Q(2)))));
 emit("beta_pq_checked(1.0,1.0,0.25,10000,1.0e-12)",lower(Q(1)/4));
 Q x=Q(static_cast<double>(1e-300));
 emit("beta_pq_checked(1.0,1.0,1.0e-300,10000,1.0e-12)",lower(x));
 x=Q(static_cast<double>(1e-200));
 emit("beta_pq_checked(2.0,2.0,1.0e-200,10000,1.0e-12)",lower(x*x*(3-2*x)));
 x=Q(static_cast<double>(0.2));
 emit("beta_pq_checked(2.0,2.0,0.2,10000,1.0e-12)",lower(x*x*(3-2*x)));
 emit("beta_pq_checked(1.0,3.0,0.5,10000,1.0e-12)",upper(Q(1)/8));
 const Q pi=acosq(-Q(1));
 emit("beta_pq_checked(0.5,0.5,0.25,10000,1.0e-12)",lower(2*asinq(Q(1)/2)/pi));
 std::cout<<R"SIO( {
 let r=gamma_inverse_checked(1.0,0.25,true,10000,1.0e-10)
 assert(r.status==0 && r.forward_log_available && r.lower<=r.value && r.upper>=r.value)
)SIO";
 std::cout<<" assert(close(r.value,"<<static_cast<double>(logq(Q(4)))<<",2.0e-9))\n }\n";
 std::cout<<R"SIO( {
 let r=gamma_inverse_checked(1.0,1.0e-100,true,10000,1.0e-10)
 assert(r.status==0 && r.forward_log_available)
)SIO";
 std::cout<<" assert(close(r.value,"<<static_cast<double>(-logq(Q(static_cast<double>(1e-100))))<<",2.0e-9))\n }\n";
 std::cout<<R"SIO( {
 let r=beta_inverse_checked(1.0,1.0,0.25,false,10000,1.0e-10)
 assert(r.status==0 && close(r.value,0.25,2.0e-9))
 }
 {
 let r=beta_inverse_checked(1.0,2.0,0.25,true,10000,1.0e-10)
 assert(r.status==0 && close(r.value,0.5,2.0e-9))
 }
 println("STDLIB_DEEP_SPECIAL_REFERENCE_PASS")
 0
}
)SIO";
 return std::cout.good()?0:1;
}
