// Independent analytic oracle for actual NB2 slot stdout.
// alpha=mean=1 gives the geometric PMF P(Y=k)=2^(-k-1).
// No production helper or shared numerical preprocessing is used.
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
int main(int argc,char** argv) {
 if(argc!=2){std::cerr<<"usage: deep_nb2_slots_reference actual.stdout\n";return 2;}
 std::ifstream in(argv[1]); if(!in){std::cerr<<"cannot open stdout\n";return 2;}
 std::string token;int rows=0;bool poisson_seen=false;
 while(in>>token){
  if(token=="NB2_SLOT_POISSON"){
   long double actual=0;
   if(!(in>>actual)||poisson_seen||!std::isfinite(actual)||std::fabs(actual+1.L)>2e-11L*(1+std::fabs(-1.L))){
    std::cerr<<"Poisson zero-count analytic mismatch or duplicate\n";return 1;
   }
   poisson_seen=true;continue;
  }
  if(token!="NB2_SLOT_ROW") continue;
  long long k=-1;long double logp=0,residual=0,deviance=0;
  if(!(in>>k>>logp>>residual>>deviance)){std::cerr<<"incomplete row\n";return 1;}
  if(k!=rows || k<0 || k>7){std::cerr<<"row order/count\n";return 1;}
  const long double y=static_cast<long double>(k);
  const long double expected_logp=-(y+1)*std::log(2.L);
  const long double first=k==0?0.L:y*std::log(y);
  const long double expected_deviance=2*(first-(y+1)*std::log((y+1)/2));
  const long double expected_residual=k==1?0.L:(k<1?-1.L:1.L)*std::sqrt(expected_deviance);
  auto close=[](long double a,long double b){return std::isfinite(a) && std::fabs(a-b)<=2e-11L*(1+std::fabs(b));};
  if(!close(logp,expected_logp)||!close(residual,expected_residual)||!close(deviance,expected_deviance)){
   std::cerr<<"analytic mismatch row "<<k<<"\n";return 1;
  }
  ++rows;
 }
 if(rows!=8){std::cerr<<"missing rows "<<rows<<"\n";return 1;}
 if(!poisson_seen){std::cerr<<"missing Poisson snapshot value\n";return 1;}
 std::cout<<"NB2_SLOT_ANALYTIC_ORACLE_PASS rows=8 geometric-PMF signed-deviance poisson-zero-count=1\n";
 return 0;
}
