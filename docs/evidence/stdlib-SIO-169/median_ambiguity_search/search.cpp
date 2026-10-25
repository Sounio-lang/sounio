#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <array>
#include <iostream>
#include <iomanip>
using I=boost::multiprecision::cpp_int;using R=boost::multiprecision::cpp_rational;using F=boost::multiprecision::cpp_bin_float_100;
R best(R x,int limit){I n=numerator(x),d=denominator(x),p0=0,q0=1,p1=1,q1=0;while(d!=0){I a=n/d,q2=q0+a*q1;if(q2>limit)break;I p2=p0+a*p1;p0=p1;p1=p2;q0=q1;q1=q2;I next=n-a*d;n=d;d=next;}if(d==0)return R(p1)/q1;I k=(limit-q0)/q1;R a=R(p0+k*p1)/(q0+k*q1),b=R(p1)/q1;return abs(a-x)<abs(b-x)?a:b;}
int main(){std::array<int,4>risks{1000000,900001,800011,700001};bool above=false,below=false;unsigned state=1729;for(int i=0;i<1000000&&(!above||!below);i++){R h=1;std::array<int,4>deaths;for(int k=0;k<4;k++){state=1664525*state+1013904223;deaths[k]=1000+state%4000;h*=R(risks[k]-deaths[k])/risks[k];}if(numerator(h)<I(9223372036854775807LL)&&denominator(h)<I(9223372036854775807LL))continue;R ratio=best(R(1)/(2*h),risks[3]-deaths[3]);int r=numerator(ratio).convert_to<int>(),s=denominator(ratio).convert_to<int>();R diff=h*ratio-R(1)/2;if(diff==0||abs(diff)>R(1)/I("1000000000000000"))continue;if((diff>0&&above)||(diff<0&&below))continue;std::cout<<"MEDIAN_AMBIGUITY_CONTROL ";for(int k=0;k<4;k++)std::cout<<risks[k]<<' '<<deaths[k]<<' ';std::cout<<s<<' '<<s-r<<" difference="<<std::setprecision(40)<<F(diff)<<" exact="<<diff<<'\n';if(diff>0)above=true;else below=true;}return above&&below?0:1;}
