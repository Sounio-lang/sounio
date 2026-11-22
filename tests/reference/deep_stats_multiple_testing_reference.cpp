// Independent C++23 exact-rational multiple-testing reference.
// Central Slurm compile:
// g++ -std=c++23 -O2 tests/reference/deep_stats_multiple_testing_reference.cpp -o deep_stats_multiple_testing_reference
// Optional sole argument: actual native fixture stdout for concordance.
// Rank candidates are enumerated all-pairs; no production sort/backward pass.
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/rational.hpp>
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
using I=boost::multiprecision::cpp_int;
using R=boost::rational<I>;
using D=boost::multiprecision::cpp_dec_float_100;
static void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
static R harmonic(size_t n){R h(0);for(size_t j=1;j<=n;++j)h+=R(I(1),I(j));return h;}
static D decimal(const R& r){return D(r.numerator().convert_to<std::string>())/D(r.denominator().convert_to<std::string>());}
static std::vector<R> adjust(const std::vector<R>& p,const std::vector<int>& available,int method){
 const size_t n=p.size();require(n==available.size(),"shape");
 require(method>=0&&method<=2,"method");
 std::vector<R> effective(n,R(1)),out(n,R(1));
 for(size_t i=0;i<n;++i){
  require(available[i]==0||available[i]==1,"availability");
  if(available[i]){require(p[i]>=R(0)&&p[i]<=R(1),"p domain");effective[i]=p[i];}
 }
 const R h=method==2?harmonic(n):R(1);
 for(size_t i=0;i<n;++i){
  if(!available[i])continue;
  if(method==0){R q=R(I(n))*effective[i];out[i]=q<R(1)?q:R(1);continue;}
  R best(1);
  for(size_t j=0;j<n;++j){
   if(effective[j]<effective[i])continue;
   size_t rank=0;
   for(size_t k=0;k<n;++k)if(effective[k]<=effective[j])++rank;
   require(rank>0,"positive rank");
   R candidate=R(I(n))*h*effective[j]/R(I(rank));
   if(candidate<best)best=candidate;
  }
  out[i]=best;
 }
 return out;
}
int main(int argc,char** argv){
 try{
  const std::vector<R> p{R(1,100),R(4,100),R(3,100),R(2,100),R(9,10),R(0),R(-123),R(1)};
  const std::vector<int> available{1,1,1,1,1,1,0,1};
  const std::array<R,8> expected_bh{R(1,25),R(8,125),R(3,50),R(4,75),R(1),R(0),R(1),R(1)};
  require(harmonic(8)==R(761,280),"H8 exact fixture");
  std::array<std::vector<R>,3> results;
  std::cout<<"method\trow\texact_fraction\tdecimal\n"<<std::setprecision(60);
  for(int method=0;method<3;++method){
   results[method]=adjust(p,available,method);
   for(size_t i=0;i<p.size();++i){
    if(method==1)require(results[method][i]==expected_bh[i],"BH exact fixture");
    require(results[method][i]>=R(0)&&results[method][i]<=R(1),"adjusted domain");
    std::cout<<method<<'\t'<<i<<'\t'<<results[method][i].numerator()<<'/'<<results[method][i].denominator()
     <<'\t'<<decimal(results[method][i])<<'\n';
   }
  }
  I denominator=I(1)<<1074;
  const R quantum(I(1),denominator);
  auto tiny_bh=adjust({quantum,R(1)},{1,1},1);
  auto tiny_by=adjust({quantum,R(1)},{1,1},2);
  require(tiny_bh[0]==R(2)*quantum&&tiny_by[0]==R(3)*quantum,"exact subnormal fixture");
  require(harmonic(513)>R(4),"513 BY clipping");
  // An unavailable member must increase neither availability nor shorten family.
  require(results[1][0]==R(1,25)&&results[1][6]==R(1),"complete family policy");
  if(argc==2){
   std::ifstream input(argv[1]);require(bool(input),"native report open");
   std::string line;int method=-1;size_t row=0,total=0;
   while(std::getline(input,line)){
    if(line.starts_with("adjusted method: ")){
     if(method>=0)require(row==8,"previous native block");
     method=std::stoi(line.substr(17));require(method>=0&&method<3,"native method");
     require(size_t(method)==total/8,"native method order");row=0;
    }else if(line.starts_with("adjusted value: ")){
     require(method>=0&&row<8,"native row");
     D actual(line.substr(16)),expected=decimal(results[method][row]);
     require(actual==actual,"native NaN");
     if(results[method][row]==R(0))require(actual==0,"native exact zero");
     else require(abs(actual-expected)<=abs(expected)*D("1e-12")+D("1e-30"),"native rational concordance");
     ++row;++total;
    }
   }
   require(total==24&&row==8,"complete native report");
  }
  std::cout<<"DEEP_MULTIPLE_TESTING_CPP_REFERENCE_PASS\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
