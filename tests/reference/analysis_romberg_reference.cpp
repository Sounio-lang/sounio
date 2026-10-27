// Independent C++23 incremental Romberg table and analytic-function witness generator.
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
struct Result { long double value, error; int iterations, converged; };
long double f(int id, long double x) {
    switch (id) { case 0: return x*x; case 1: return std::sin(x); case 2: return std::exp(x); case 3: return 1/x; }
    throw std::runtime_error("unsupported reference function");
}
Result reference(int id, long double a, long double b, int requested) {
    const int limit = std::max(2, std::min(8, requested));
    std::array<std::array<long double,8>,8> table{};
    const long double width=b-a;
    table[0][0]=width*(f(id,a)+f(id,b))/2;
    long double delta=0;
    for(int k=1;k<limit;k++) {
        const int count=1<<k;
        const long double h=width/count;
        long double added=0;
        for(int j=1;j<count;j+=2) added += f(id,a+j*h);
        table[k][0]=table[k-1][0]/2+h*added;
        long double factor=4;
        for(int col=1;col<=k;col++) {
            table[k][col]=table[k][col-1]+(table[k][col-1]-table[k-1][col-1])/(factor-1);
            factor*=4;
        }
        delta=std::abs(table[k][k]-table[k-1][k-1]);
        if(delta<1e-9L) return {table[k][k],delta,k+1,1};
    }
    return {table[limit-1][limit-1],delta,limit,0};
}
long double analytic(int id,long double a,long double b) {
    switch(id) {case 0:return (b*b*b-a*a*a)/3;case 1:return std::cos(a)-std::cos(b);case 2:return std::exp(b)-std::exp(a);case 3:return std::log(b)-std::log(a);}
    throw std::runtime_error("unsupported integral");
}
int main(int argc,char**argv) try {
    if(argc!=2) return 64;
    const std::string mode=argv[1];
    if(mode!="--generate" && mode!="--table") return 64;
    if(mode=="--generate") {
        std::cout << "//@ run-pass\n//@ expect-stdout: ANALYSIS_CPP23_ROMBERG_CONCORDANCE_OK\n"
                  << "use analysis::lib::{romberg_integration}\n"
                  << "fn abs_probe(x: f64) -> f64 { if x < 0.0 { return 0.0 - x } return x }\n"
                  << "fn main() -> i32 with IO, Mut, Div, Panic {\n";
    } else std::cout<<"function,budget,value,last_change,iterations,converged,analytic,actual_error\n";
    std::cout << std::scientific << std::setprecision(17);
    int index=0;
    for(const int id:{0,1,2,3}) for(const int budget:{2,3,6,8}) {
        const long double a=id==3?1.L:0.L;
        const long double b=id==1?std::acos(-1.L):(id==3?2.L:1.L);
        const auto r=reference(id,a,b,budget);
        if(!std::isfinite(r.value)||!std::isfinite(r.error)||r.error<0) throw std::runtime_error("invalid reference");
        if(r.converged && std::abs(r.value-analytic(id,a,b))>1e-7L) throw std::runtime_error("reference/analytic discordance");
        if(mode=="--table") {
            std::cout<<id<<','<<budget<<','<<r.value<<','<<r.error<<','<<r.iterations<<','<<r.converged<<','<<analytic(id,a,b)<<','<<std::abs(r.value-analytic(id,a,b))<<'\n';
            continue;
        }
        const std::string n="r"+std::to_string(++index);
        std::cout << " let " << n << " = romberg_integration(" << id << ", " << static_cast<double>(a) << ", " << static_cast<double>(b) << ", " << budget << ")\n"
                  << " if abs_probe(" << n << ".value - (" << static_cast<double>(r.value) << ")) > 1.0e-10 { return " << 100+index << " }\n"
                  << " if abs_probe(" << n << ".error_estimate - (" << static_cast<double>(r.error) << ")) > 1.0e-10 { return " << 200+index << " }\n"
                  << " if " << n << ".iterations != " << r.iterations << " || " << n << ".converged != " << r.converged << " { return " << 300+index << " }\n";
    }
    if(mode=="--generate") std::cout << " print(\"ANALYSIS_CPP23_ROMBERG_CONCORDANCE_OK\\n\")\n return 0\n}\n";
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
