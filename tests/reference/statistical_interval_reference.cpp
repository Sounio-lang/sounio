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
int main() {
    if(std::abs(critical(0.75L,1)-1)>1e-12L)
        throw std::runtime_error("Cauchy reference identity failed");
    if(std::abs(critical(0.975L,2)-std::sqrt(2*0.95L*0.95L/(1-0.95L*0.95L)))>1e-12L)
        throw std::runtime_error("df=2 reference failed");
    struct Input {long double mean,sd; int n;};
    constexpr std::array<Input,3> inputs{{{10,2,10},{15,3,12},{8,1.5L,9}}};
    std::cout<<std::setprecision(18)<<"mean\tsd\tn\tcritical\tlower\tupper\n";
    for(const auto& v:inputs) {
        const auto t=critical(0.975L,v.n-1);
        const auto half=t*v.sd/std::sqrt(static_cast<long double>(v.n));
        std::cout<<v.mean<<'\t'<<v.sd<<'\t'<<v.n<<'\t'<<t<<'\t'
                 <<v.mean-half<<'\t'<<v.mean+half<<'\n';
    }
    const auto odd=mad_reference({1,2,3,4,100},3);
    const auto even=mad_reference({0,1,2,100},1.5L);
    const auto singleton=mad_reference({7},7);
    const auto affine=mad_reference({10,14,18,22,406},18);
    std::vector<long double> full(100,0);
    std::fill(full.begin()+50,full.end(),2);
    const auto capacity=mad_reference(full,1);
    if(odd!=1 || even!=1 || singleton!=0 || affine!=4 || capacity!=1)
        throw std::runtime_error("MAD reference failed");
    std::cout<<"MAD_RAW\t"<<odd<<'\t'<<even<<'\t'<<singleton<<'\t'<<affine<<'\t'<<capacity<<'\n';
    std::cout<<"CPP23_STAT_INTERVAL_REFERENCE_PASS\n";
}
