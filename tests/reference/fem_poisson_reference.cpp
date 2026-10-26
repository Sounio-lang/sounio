// C++23 binary128 analytic oracle for actual Sounio FEM node rows.
#include <quadmath.h>
#include <bit>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
using Q=__float128;
Q absq(Q x){return x<0?-x:x;}
void printq(const char* key,Q x){char b[100];quadmath_snprintf(b,sizeof b,"%.20Qg",x);std::cout<<key<<b<<"\n";}
int poisson_check(const char* path,bool refinement=false){
 std::ifstream f(path);if(!f)return 3;
 std::string line,tag,extra;Q maximum=0,maxresidual=0;std::int64_t rows=0;
 Q previous_continuum=0;
 const std::vector<int> grids=refinement?std::vector<int>{8,16,32,64}:std::vector<int>{2,8,16,32,256};
 for(int n:grids){
  Q continuum=0;
  const std::int64_t side=n+1,area=side*side;std::vector<Q> u(area);
  for(std::int64_t i=0;i<area;++i){
   if(!std::getline(f,line))return 9;
   std::istringstream s(line);int nn;std::int64_t j,ub;
   if(!(s>>tag>>nn>>j>>ub)||tag!=(refinement?"MGS":"MG")||nn!=n||j!=i||s>>extra)return 4;
   const double ud=std::bit_cast<double>(ub);if(!std::isfinite(ud))return 5;u[i]=Q(ud);
   const Q x=Q(i%side)/n,y=Q(i/side)/n,pi=acosq(Q(-1));
   const Q sine=(i%side==0||i%side==n||i/side==0||i/side==n)?Q(0):sinq(pi*x)*sinq(pi*y);
   const Q eigen=8*Q(n)*n*sinq(pi/(2*n))*sinq(pi/(2*n));
   const Q expected=refinement?(2*pi*pi/eigen)*sine:x*(1-x)*y*(1-y);
   if(refinement && absq(u[i]-sine)>continuum)continuum=absq(u[i]-sine);
   const Q error=absq(u[i]-expected);if(error>maximum)maximum=error;
   if(error>Q(2.0e-11))return 8;
   if((i%side==0||i%side==n||i/side==0||i/side==n)&&ud!=0.0)return 12;
   ++rows;
  }
  if(!std::getline(f,line))return 9;
  std::istringstream s(line);int nn;std::int64_t cycles,work,updates,residuals,restricted,prolonged,coarse,setup,rb,packed,bytes;
  if(!(s>>tag>>nn>>cycles>>work>>updates>>residuals>>restricted>>prolonged>>coarse>>setup>>rb>>packed>>bytes)||s>>extra||tag!=(refinement?"MGS_WORK":"MG_WORK")||nn!=n||cycles<1||cycles>80)return 6;
  const double residual=std::bit_cast<double>(rb);if(!std::isfinite(residual)||residual<0||residual>1.0e-10)return 7;
  std::int64_t expectedpacked=0,nonbottom=0,restrictsum=0;
  for(int m=n;m>=2;m/=2){expectedpacked+=std::int64_t(m+1)*(m+1);if(m>2){nonbottom+=std::int64_t(m-1)*(m-1);restrictsum+=std::int64_t(m/2-1)*(m/2-1);}}
  const auto finestinterior=std::int64_t(n-1)*(n-1);
  if(packed!=expectedpacked||bytes!=8*(3*packed+area)||coarse!=cycles||updates!=cycles*(6*nonbottom+1)||restricted!=cycles*restrictsum||prolonged!=cycles*nonbottom||residuals!=finestinterior+cycles*(finestinterior+nonbottom)||work!=updates+residuals+restricted+prolonged||setup!=packed+cycles*(packed-area))return 13;
  Q physical=0;
  for(int y=1;y<n;++y)for(int x=1;x<n;++x){
   const auto i=std::int64_t(y)*side+x;const Q xx=Q(x)/n,yy=Q(y)/n;
   const Q pi=acosq(Q(-1));
   const Q rhs=refinement?2*pi*pi*sinq(pi*xx)*sinq(pi*yy):2*(xx*(1-xx)+yy*(1-yy));
   const Q r=absq(rhs-Q(n)*n*(4*u[i]-u[i-1]-u[i+1]-u[i-side]-u[i+side]));
   if(r>physical)physical=r;
  }
  if(physical>maxresidual)maxresidual=physical;
  if(physical>Q(1.1e-10)||absq(physical-Q(residual))>Q(1.0e-11))return 14;
  if(!std::getline(f,line))return 9;
  {std::istringstream late(line);int nn;std::int64_t work,updates;
   if(!(late>>tag>>nn>>work>>updates)||late>>extra||tag!=(refinement?"MGS_LATE":"MG_LATE")||nn!=n||updates!=(n==2?0:7)||work!=finestinterior+updates)return 16;}
  if(!std::getline(f,line)||line!=(refinement?"MGS_REFUSALS ":"MG_REFUSALS ")+std::to_string(n)+" 3 3 2")return 11;
  if(refinement){
   const std::string key="continuum_error_n"+std::to_string(n)+"=";printq(key.c_str(),continuum);
   if(previous_continuum>0){const Q ratio=previous_continuum/continuum;printq("observed_refinement_ratio=",ratio);if(ratio<Q(3.8)||ratio>Q(4.2))return 15;}
   previous_continuum=continuum;
  }
 }
 if(!std::getline(f,line)||line!=(refinement?"POISSON_REFINEMENT_PASS":"POISSON_CHECKED_PASS")||std::getline(f,line))return 9;
 std::cout<<"POISSON_REFERENCE_PASS rows="<<rows<<"\n";printq("max_absolute_grid_error=",maximum);printq("max_physical_residual=",maxresidual);return 0;
}
int main(int argc,char**argv){
 if(argc==3 && std::string(argv[1])=="--legacy-cg"){
  std::ifstream f(argv[2]);std::string line,tag,extra;
  for(int mode=0;mode<3;++mode)for(int i=0;i<3;++i){
   if(!std::getline(f,line))return 9;
   std::istringstream row(line);int m,j,available,iterations;std::int64_t kb,fb,ub,resultbits;
   if(!(row>>tag>>m>>j>>available>>kb>>fb>>ub>>resultbits>>iterations)||row>>extra||tag!="FEM_CG"||m!=mode||j!=i)return 4;
   const Q k=Q(std::bit_cast<double>(kb)),rhs=Q(std::bit_cast<double>(fb)),value=Q(std::bit_cast<double>(ub)),expected=rhs/k;
   if(!finiteq(k)||!finiteq(rhs)||!finiteq(value)||k<=0)return 5;
   if(mode<2){if(available!=1||resultbits!=ub||iterations!=1||absq(value-expected)>Q(2e-14)*absq(expected))return 8;}
   else if(available!=0||value!=7||absq(expected)>=Q(std::numeric_limits<double>::denorm_min()))return 17;
  }
  if(!std::getline(f,line)||line!="FEM_LEGACY_CG_PASS"||std::getline(f,line))return 9;
  std::cout<<"FEM_CG_REFERENCE_PASS rows=9\n";return 0;
 }
 if(argc==3 && std::string(argv[1])=="--singular-before"){
  std::ifstream f(argv[2]);std::string tag;double value;
  if(!(f>>tag>>value)||tag!="FEM_SINGULAR"||!std::isfinite(value))return 4;
  const Q residual=absq(Q(0)*Q(value)-1);
  printq("singular_equation_residual=",residual);
  return residual>Q(1e-12)?8:0;
 }
 if(argc==3 && std::string(argv[1])=="--scaled-load"){
  std::ifstream f(argv[2]);std::string line,tag,extra;
  if(!std::getline(f,line))return 9;
  std::istringstream row(line);int status;std::int64_t qb,eab,ub,rb;
  if(!(row>>tag>>status>>qb>>eab>>ub>>rb)||row>>extra||tag!="FEM_LOAD"||status!=0||qb!=1)return 4;
  const Q q=Q(std::bit_cast<double>(qb)),ea=Q(std::bit_cast<double>(eab)),value=Q(std::bit_cast<double>(ub));
  const Q expected=q*ea/8;
  if(!finiteq(expected)||!finiteq(value)||expected<=0||value<=0||absq(value-expected)>Q(1e-14)*expected)return 8;
  const Q reported=Q(std::bit_cast<double>(rb)),physical=absq(4*value-q*ea/2);
  if(!finiteq(reported)||reported<0||absq(reported-physical)>32*Q(std::numeric_limits<double>::epsilon())*absq(q*ea/2))return 14;
  if(!std::getline(f,line)||line!="FEM_SCALED_LOAD_PASS"||std::getline(f,line))return 9;
  std::cout<<"FEM_SCALED_LOAD_REFERENCE_PASS\n";printq("expected=",expected);printq("actual=",value);return 0;
 }
 if(argc==3 && std::string(argv[1])=="--poisson")return poisson_check(argv[2]);
 if(argc==3 && std::string(argv[1])=="--refinement")return poisson_check(argv[2],true);
 if(argc!=2)return 2;
 std::ifstream f(argv[1]);if(!f)return 3;
 std::string line,tag,extra;Q maximum=0,maxresidual=0;int rows=0;
 const int counts[3]={3,17,257};
 for(int mode=0;mode<3;++mode){
  const int n=counts[mode];std::vector<Q> xs(n),us(n);
  for(int i=0;i<n;++i){
   if(!std::getline(f,line))return 9;
   std::istringstream s(line);int m,j;std::int64_t xb,ub;
   if(!(s>>tag>>m>>j>>xb>>ub)||tag!="FEM"||m!=mode||j!=i||s>>extra)return 4;
   const double xd=std::bit_cast<double>(xb),ud=std::bit_cast<double>(ub);
   double expectedx=double(i)/double(n-1);if(mode==1)expectedx*=expectedx;
   if(!std::isfinite(xd)||!std::isfinite(ud)||xd!=expectedx)return 5;
   xs[i]=Q(xd);us[i]=Q(ud);
   const Q exact=mode==2?Q(-1)+3*xs[i]+xs[i]*(1-xs[i])/2:Q(1)+2*xs[i]+xs[i]*(1-xs[i]);
   const Q err=absq(us[i]-exact);if(err>maximum)maximum=err;
   if(err>Q(2.0e-12)*(1+absq(exact))){std::cerr<<"fem-node "<<mode<<" "<<i<<"\n";return 8;}
   ++rows;
  }
  if(!std::getline(f,line))return 9;
  std::istringstream s(line);int m,nn,assembled,eliminated;std::int64_t rb;
  if(!(s>>tag>>m>>nn>>assembled>>eliminated>>rb)||tag!="FEM_WORK"||m!=mode||nn!=n||assembled!=n-1||eliminated!=n-1||s>>extra)return 6;
  const double rd=std::bit_cast<double>(rb);if(!std::isfinite(rd)||rd<0)return 7;
  // EA and q share exactly the same binary64 input in tiny-scale case.
  const Q ea=mode==2?Q(double(1.0e-150)):Q(2),q=mode==2?ea:Q(4);
  Q physical=0;
  for(int i=1;i<n-1;++i){
   const Q hl=xs[i]-xs[i-1],hr=xs[i+1]-xs[i];
   const Q r=absq(ea*(us[i]-us[i-1])/hl-ea*(us[i+1]-us[i])/hr-q*(hl+hr)/2);
   if(r>physical)physical=r;
  }
  if(physical/ea>maxresidual)maxresidual=physical/ea;
  if(physical/ea>Q(2.0e-10)||Q(rd)/ea>Q(2.0e-10))return 10;
  if(!std::getline(f,line)||line!="FEM_REFUSALS "+std::to_string(mode)+" 2 2 3 2")return 11;
 }
 if(!std::getline(f,line)||line!="FEM_CHECKED_PASS"||std::getline(f,line))return 9;
 std::cout<<"FEM_REFERENCE_PASS rows="<<rows<<"\n";printq("max_absolute_node_error=",maximum);printq("max_physical_residual_div_EA=",maxresidual);
}
