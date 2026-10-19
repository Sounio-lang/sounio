// Independent long-double raw-KL reference for checked embedding JSD.
// C++23; no shared Sounio distance identity.
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <cstdlib>
using Profile=std::array<long double,4>;
long double distance(const Profile& x,const Profile& y) {
 long double sx=0,sy=0;
 for(unsigned i=0;i<4;++i){sx+=x[i];sy+=y[i];}
 if(!(sx>0&&sy>0)) std::abort();
 long double js=0;
 for(unsigned i=0;i<4;++i){
  const long double p=x[i]/sx,q=y[i]/sy,m=(p+q)/2;
  if(p>0) js+=p*std::log(p/m)/2;
  if(q>0) js+=q*std::log(q/m)/2;
 }
 return std::sqrt(js/std::log(2.0L));
}
int main(){
 const Profile a{1024,1024,1024,1024},b{1025,1023,1024,1024};
 const Profile left{1,0,0,0},right{0,1,0,0};
 const auto d=distance(a,b);
 if(distance(a,a)!=0||std::fabs(distance(left,right)-1)>1e-18L||
    std::fabs(d-distance(b,a))>1e-18L||!(d>0&&d<1))return 1;
 std::cout<<std::setprecision(21)<<"NEAR_EQUAL_JSD_DISTANCE "<<d<<"\n";
 std::cout<<"KMER_EMBEDDING_CPP_PASS\n";
}
