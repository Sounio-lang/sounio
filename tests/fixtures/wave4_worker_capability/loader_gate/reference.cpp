#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
struct Row {std::int64_t kind,input,serial,worker;};
std::vector<Row> read_rows(const std::string&path){
 std::ifstream file(path);if(!file)throw std::runtime_error("missing rows");
 std::vector<Row> rows;std::string line;
 while(std::getline(file,line)){
  if(!line.starts_with("NORMAL_BITS "))continue;
  std::istringstream input(line.substr(12));Row row{};std::string extra;
  if(!(input>>row.kind>>row.input>>row.serial>>row.worker)||(input>>extra))throw std::runtime_error("malformed row");
  rows.push_back(row);
 }
 if(rows.size()!=60)throw std::runtime_error("expected exactly 60 rows");
 return rows;
}
void write_rows(const std::string&path,const std::vector<Row>&rows){
 std::ofstream file(path);if(!file)throw std::runtime_error("output");
 for(const auto&r:rows)file<<"NORMAL_BITS "<<r.kind<<' '<<r.input<<' '<<r.serial<<' '<<r.worker<<'\n';
 if(!file)throw std::runtime_error("write");
}
int main(int argc,char**argv){
 try{
  if(argc<3)throw std::runtime_error("usage: reference verify LOG | mutate/reorder LOG OUT");
  auto rows=read_rows(argv[2]);const std::string mode=argv[1];
  if(mode=="mutate"||mode=="reorder"){
   if(argc!=4)throw std::runtime_error("mutation output");
   if(mode=="mutate")rows[0].worker=std::bit_cast<std::int64_t>(1.0);
   else std::swap(rows[0],rows[1]);
   write_rows(argv[3],rows);return 0;
  }
  if(mode!="verify"||argc!=3)throw std::runtime_error("mode");
  const double infinity=std::numeric_limits<double>::infinity();
  const std::array<double,15> inputs{-40.,-8.,-2.,-1.,-.5,-0.0,0.0,.5,1.,2.,8.,40.,infinity,-infinity,std::bit_cast<double>(std::int64_t{9221120237041090626})};
  const long double sqrt_two=std::sqrt(2.0L);
  unsigned finite=0,nans=0,infinities=0;
  for(std::size_t i=0;i<rows.size();++i){
   const auto&r=rows[i];
   if(r.kind!=static_cast<std::int64_t>(i/inputs.size())||r.input!=std::bit_cast<std::int64_t>(inputs[i%inputs.size()]))throw std::runtime_error("row order or identity");
   if(r.serial!=r.worker)throw std::runtime_error("serial/worker bit mismatch");
   const double x=std::bit_cast<double>(r.input),actual=std::bit_cast<double>(r.worker);
   const long double z=static_cast<long double>(x);
   long double expected=0;
   if(r.kind==0)expected=0.5L*std::erfc(-z/sqrt_two);
   else if(r.kind==1)expected=0.5L*std::erfc(z/sqrt_two);
   else if(r.kind==2)expected=std::log(0.5L*std::erfc(-z/sqrt_two));
   else expected=std::log(0.5L*std::erfc(z/sqrt_two));
   const double rounded=static_cast<double>(expected);
   if(std::isnan(rounded)){if(!std::isnan(actual))throw std::runtime_error("NaN support");++nans;continue;}
   if(std::isinf(rounded)){if(actual!=rounded)throw std::runtime_error("infinite support");++infinities;continue;}
   if(!std::isfinite(actual))throw std::runtime_error("unexpected nonfinite");
   const long double error=std::abs(static_cast<long double>(actual)-expected);
   const long double tolerance=r.kind<2?2.0e-12L*std::abs(expected):2.0e-12L;
   if(rounded==0.0&&r.kind<2){if(actual!=0.0)throw std::runtime_error("binary64 underflow endpoint");}
   else if(error>tolerance){std::cerr<<std::setprecision(22)<<"row="<<i<<" x="<<x<<" actual="<<actual<<" expected="<<expected<<" error="<<error<<" tolerance="<<tolerance<<"\n";throw std::runtime_error("canonical normal tolerance");}
   ++finite;
  }
  if(finite+nans+infinities!=60||nans!=4||infinities!=2)throw std::runtime_error("nonempty case accounting");
  std::cout<<"WORKER_IMPORTED_NORMAL_CXX23_OK rows=60 kernels=4 bit_concordance=exact probability_relative=2e-12 log_absolute=2e-12\n";
 }catch(const std::exception&e){std::cerr<<"WORKER_IMPORTED_NORMAL_REFUSED "<<e.what()<<'\n';return 1;}
}
