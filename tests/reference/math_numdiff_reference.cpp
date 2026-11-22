// ADR009 C++23 independent analytic differentiation / represented-input roots.
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
long double number(std::ifstream& f){std::string s;if(!std::getline(f,s))throw 1;const auto v=std::stold(s);if(!std::isfinite(v))throw 2;return v;}
bool close(long double a,long double b,long double atol,long double rtol){return std::isfinite(a)&&std::abs(a-b)<=atol+rtol*std::abs(b);}
int roots(const char*path){
 std::ifstream f(path);std::string tag;if(!std::getline(f,tag)||tag!="ROOTS")return 2;
 const long double q=std::numeric_limits<double>::denorm_min();
 const long double expected[]={std::sqrt(static_cast<long double>(std::numeric_limits<double>::epsilon())),std::cbrt(static_cast<long double>(std::numeric_limits<double>::epsilon())),std::cbrt(static_cast<long double>(1e-300)),std::cbrt(static_cast<long double>(1e300)),std::cbrt(static_cast<long double>(-1e-300)),0,std::sqrt(q)};
 for(auto v:expected){auto a=number(f);if(v==0?a!=0:!close(a,v,0,1e-12L))return 11;}
 if(!std::getline(f,tag)||tag!="NUMDIFF_LEGACY_ROOTS_PASS")return 13;
 std::cout<<"NUMDIFF_ROOT_CPP_PASS values=7 relative_tolerance=1e-12\n";return 0;
}
int nonconvergence(const char* path){
 const long double x=1,h=.125L;
 const auto f=[](long double v){return v*v*v*v;};
 const auto first=[&](long double q){return (f(x+q)-f(x-q))/(2*q);};
 const auto second=[&](long double q){return (f(x+q)-2*f(x)+f(x-q))/(q*q);};
 if(first(h)==first(h/2)||second(h)==second(h/2))return 15;
 std::ifstream input(path);std::string line;int expected=0;bool done=false;
 while(std::getline(input,line)){
  if(line=="NUMDIFF_DYNAMIC_NONCONVERGENCE_PASS"){done=true;continue;}
  std::istringstream header(line);std::string tag;int kind,status,calls;
  if(!(header>>tag>>kind>>status>>calls)||tag!="NDC"||kind!=expected||status!=7||calls!=5)return 16;
  ++expected;
 }
 if(expected!=3||!done)return 13;
 std::cout<<"NUMDIFF_NONCONVERGENCE_CPP_PASS quartic_paired_discrepancies_positive methods=3 actual_callbacks_each=5\n";return 0;
}
int main(int argc,char**argv){try{
 if(argc==3 && std::string(argv[1])=="--nonconvergence")return nonconvergence(argv[2]);
 if(argc==3 && std::string(argv[1])=="--roots")return roots(argv[2]);
 if(argc!=2)return 64;
 std::ifstream f(argv[1]);if(!f)return 66;
 std::string line;int scalar_count=0;std::set<int> kinds;std::size_t cells_checked=0;
 bool complete=false,edges=false,hot=false;
 while(std::getline(f,line)){
  if(line=="NUMDIFF_CHECKED_PASS"){complete=true;continue;}
  if(line=="DIFF_EDGE_CASES_PASS"){edges=true;continue;}
  if(line=="DIFF_HOT_CHECKSUM"){if(number(f)!=1000000)return 14;hot=true;continue;}
  if(line.starts_with("DIFF ")){
   std::istringstream h(line);std::string tag;int id;long long calls;if(!(h>>tag>>id>>calls)||id!=scalar_count||id>=10)return 3;
   if(calls<5||calls>256||(calls-1)%2)return 4;
   long double actual=number(f),error=number(f),step=number(f),left=number(f),right=number(f);
   long double truth=0;
   if(id==0)truth=2;else if(id==1)truth=5;else if(id==2)truth=1;else if(id==3)truth=std::exp(3.L);else if(id==4)truth=std::cos(static_cast<long double>(0.3));else if(id==5)truth=std::cos(-2.L);else if(id<9)truth=1;
   if(!close(actual,truth,2e-8L,5e-8L)){std::cerr<<"scalar disagreement id="<<id<<" actual="<<actual<<" analytic="<<truth<<'\n';return 11;}
   // This checks reported convergence, not that the estimate bounds true error.
   if(error<0||error>(1e-11L+1e-8L*std::abs(actual))*(1+1e-12L))return 5;
   if(step<=0||left<=0||right<=0||!close(step,std::max(left,right),0,2e-13L))return 6;
   ++scalar_count;continue;
  }
  if(line.starts_with("VECTOR ")){
   std::istringstream h(line);std::string tag;int kind,n,m;long long calls;
   if(!(h>>tag>>kind>>n>>m>>calls)||!kinds.insert(kind).second)return 7;
   if(kind==0){if(n!=257||m!=1)return 7;}else if(kind==1){if(n!=257||m!=3)return 7;}else if(kind==2){if(n!=65||m!=1)return 7;}else return 7;
   const long long required=kind==2?1+4LL*n+16LL*n*(n-1)/2:1+4LL*n;
   if(calls!=required)return 8;
   const int cells=kind==2?n*n:n*m;
   for(int at=0;at<cells;++at){
    const auto actual=number(f),error=number(f);long double truth;
    if(kind==2){const int row=at/n,col=at%n;truth=row==col?2.L*(row%7+1)+.25L:.25L;}
    else {const int row=at/n,col=at%n;const long double x=static_cast<long double>(col%5-2)/4;truth=2.L*(col%7+1)*x+(row+1);}
    if(!close(actual,truth,1e-12L,1e-12L)||error!=0){std::cerr<<"matrix disagreement kind="<<kind<<" at="<<at<<'\n';return 11;}
    ++cells_checked;
   }
   continue;
  }
  if(!line.empty()){std::cerr<<"unexpected output: "<<line<<'\n';return 9;}
 }
 if(!complete||!edges||!hot||scalar_count!=10||kinds.size()!=3)return 13;
 std::cout<<"NUMDIFF_CPP_CONCORDANCE_PASS scalar_cases="<<scalar_count<<" matrix_cells="<<cells_checked<<" dimensions=257_gradient,3x257_Jacobian,65x65_Hessian hot_into=1000000 callback_accounting=passed error_estimates_not_bounds\n";
 return 0;
}catch(...){return 10;}}
