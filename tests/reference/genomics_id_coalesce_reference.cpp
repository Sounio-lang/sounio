// C++23 exact independent grouping; count accumulation uses unsigned128.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>
using I=std::int64_t;using U=unsigned __int128;
struct Reader {std::ifstream in;explicit Reader(const char*p):in(p){} bool row(const std::string&tag,const std::vector<I>&expected){std::string line,got,extra;if(!std::getline(in,line))return false;std::istringstream s(line);if(!(s>>got)||got!=tag)return false;for(I x:expected){I y;if(!(s>>y)||y!=x)return false;}return !(s>>extra);}};
int main(int argc,char**argv){
 if(argc!=2){return 2;}Reader r(argv[1]);if(!r.in)return 3;
 const std::vector<std::string> names{"b","a","b","a","c"};
 const std::vector<std::vector<I>> matrix{{2,0,3,0},{0,5,0,0},{7,0,1,4},{0,6,0,8},{0,0,0,0}};
 std::map<std::string,std::vector<U>> grouped;
 for(std::size_t i=0;i<names.size();++i){auto&v=grouped[names[i]];if(v.empty())v.resize(4);for(int j=0;j<4;++j)v[j]+=U(matrix[i][j]);}
 std::vector<std::vector<I>> expected;
 for(const auto& [name,v]:grouped){(void)name;std::vector<I> row;for(U value:v){if(value>U(std::numeric_limits<I>::max()))return 10;row.push_back(I(value));}expected.push_back(row);}
 const std::vector<std::vector<I>> headers{{0,0,5,1,74},{1,5,5,1,37},{2,3,0,0,36},{3,3,5,1,37},{4,3,5,1,37},{5,4,0,0,5},{6,0,1,1,12},{7,0,0,1,0}};
 I values_checked=0;
 for(I tag=0;tag<8;++tag){
  if(!r.row("COALESCE",headers[tag])){std::cerr<<"coalescence status/required/budget "<<tag<<"\n";return 8;}
  if(tag!=0&&tag!=6&&tag!=7)continue;
  const auto rows=tag==0?expected:(tag==6?std::vector<std::vector<I>>{{std::numeric_limits<I>::max()}}:std::vector<std::vector<I>>{});
  const I samples=tag==6?1:4;I nnz=0;for(const auto&row:rows)for(I x:row)if(x!=0)++nnz;
  if(!r.row("COALESCE_SHAPE",{tag,I(rows.size()),samples,nnz}))return 4;
  I offset=0;
  for(I i=0;i<I(rows.size());++i){
   if(!r.row("COALESCE_OFFSET",{tag,i,offset}))return 4;
   for(I j=0;j<samples;++j){if(!r.row("COALESCE_VALUE",{tag,i,j,rows[i][j]})){std::cerr<<"exact coalesced count "<<tag<<" "<<i<<" "<<j<<"\n";return 8;}if(rows[i][j]!=0)++offset;++values_checked;}
  }
  if(!r.row("COALESCE_OFFSET",{tag,I(rows.size()),offset}))return 4;
 }
 // True overflow and the exact maximum are different outcomes, not saturation.
 const U overflow=U(std::numeric_limits<I>::max())+1;
 if(overflow<=U(std::numeric_limits<I>::max()))return 10;
 std::string line;if(!std::getline(r.in,line)||line!="COUNT_IDENTITY_COALESCE_PASS"||std::getline(r.in,line)){std::cerr<<"coalescence completion\n";return 9;}
 std::cout<<"GENOMICS_ID_COALESCE_REFERENCE_PASS exact_cells="<<values_checked<<" groups=lexicographic zero_row=preserved overflow=refused max_i64=available budget=actual_two_pass\n";
}
