#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
int main(){
 const long double tiny=std::numeric_limits<double>::min(),two52=4503599627370496.0L;
 const double minimum=static_cast<double>(tiny/two52);
 const double half=static_cast<double>(tiny/(two52*2));
 const double odd_half=static_cast<double>((3*tiny)/(two52*2));
 const double balanced=static_cast<double>((1e300L*1e300L)/(1e300L*1e300L));
 const double small_balanced=static_cast<double>((1e-300L*1e-300L)/(1e-300L*1e-300L));
 if(minimum!=std::numeric_limits<double>::denorm_min()||half!=0||
    odd_half!=2*minimum||balanced!=1||small_balanced!=1)return 1;
 std::cout<<std::setprecision(17)<<"minimum="<<minimum<<" half="<<half<<" odd_half="<<odd_half
          <<" balanced="<<balanced<<" small_balanced="<<small_balanced<<"\nSCALED_PRODUCT_CPP_PASS\n";
}
