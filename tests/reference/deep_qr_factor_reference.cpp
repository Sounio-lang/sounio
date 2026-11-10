#include <fstream>
#include <iostream>
#include <array>
#include <string>
#include <stdexcept>
#include <boost/multiprecision/cpp_dec_float.hpp>
using Q=boost::multiprecision::cpp_dec_float_100;
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("need fixture");std::ifstream f(argv[1]);std::string s;
 while(std::getline(f,s)&&s!="QR_FACTOR"){}
 if(s!="QR_FACTOR")throw std::runtime_error("missing QR");
 std::array<Q,144> R;std::array<Q,12> scales;std::array<int,12> perm;std::array<bool,12> seen{};
 for(auto&x:R){std::getline(f,s);x=Q(s);}for(auto&x:scales){std::getline(f,s);x=Q(s);if(x<=0)throw std::runtime_error("scale");}
 for(auto&x:perm){f>>x;if(x<0||x>=12||seen[x])throw std::runtime_error("permutation");seen[x]=true;}
 for(int j=0;j<12;++j)for(int k=0;k<12;++k){
 Q expected=0,actual=0;
 for(int i=0;i<512;++i){const auto a=(1+(i%12==perm[j]?perm[j]+1:0))*(perm[j]+2);const auto b=(1+(i%12==perm[k]?perm[k]+1:0))*(perm[k]+2);expected+=Q(a)*b;}
 for(int i=0;i<12;++i)actual+=R[i*12+j]*R[i*12+k];
 actual*=scales[perm[j]]*scales[perm[k]];
 if(abs(actual-expected)>Q("1e-11")*expected)throw std::runtime_error("R scale permutation Gram identity");
 if(k<j&&R[j*12+k]!=0)throw std::runtime_error("triangular");
 }
 std::cout<<"QR_FACTOR_REFERENCE_PASS\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}
