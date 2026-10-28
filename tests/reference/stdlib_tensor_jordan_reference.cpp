#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
template<size_t N> using V=std::array<double,N>;
template<size_t N> V<N> conjugate(V<N> a){for(size_t i=1;i<N;++i)a[i]=-a[i];return a;}
template<size_t N> V<N> multiply(const V<N>&a,const V<N>&b){
 if constexpr(N==1)return {a[0]*b[0]};
 else {constexpr auto H=N/2;V<H>x{},y{},u{},v{};for(size_t i=0;i<H;++i){x[i]=a[i];y[i]=a[i+H];u[i]=b[i];v[i]=b[i+H];}
 auto xu=multiply(x,u),vy=multiply(conjugate(v),y),vx=multiply(v,x),yu=multiply(y,conjugate(u));V<N>r{};for(size_t i=0;i<H;++i){r[i]=xu[i]-vy[i];r[i+H]=vx[i]+yu[i];}return r;}
}
int main(int argc,char**argv){if(argc<2||argc>3)return 64;std::ifstream input(argv[1]);if(!input)return 65;size_t count=0;for(int s=0;s<3;++s){std::array<V<16>,16>cols{};for(int i=0;i<16;++i)for(int j=0;j<16;++j){if(s==1&&i==j)cols[j][i]=1;if(s==2)cols[j][i]=double((i*7+j*3)%11-5)/8;}
 for(int i=0;i<16;++i)for(int j=0;j<16;++j)for(int k=0;k<16;++k){const auto jk=multiply(cols[j],cols[k]),kj=multiply(cols[k],cols[j]);double expected=jk[i]+kj[i];if(i==0&&j==1&&k==2)expected+=0.5;if(i==3&&j==4&&k==5)expected-=0.25;if(argc==3&&count==0)expected+=1;double actual=0;if(!(input>>actual)||!std::isfinite(actual)||actual!=expected){std::cerr<<"mismatch scenario="<<s<<" i="<<i<<" j="<<j<<" k="<<k<<" got="<<actual<<" expected="<<expected<<'\n';return 1;}++count;}}
 std::string extra;if(input>>extra)return 2;std::cout<<"TENSOR_JORDAN_CPP23_OK coefficients="<<count<<'\n';}
