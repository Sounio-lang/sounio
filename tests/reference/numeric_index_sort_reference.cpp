// C++23 independent lexicographic ordering control for actual native sort rows.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
int main(int argc,char**argv){
 const std::vector<std::int64_t> sizes={0,1,2,97,1000,1000000};
 if(argc==2&&std::string(argv[1])=="--generate"){
  std::cout<<R"SIO(use collections::numeric_buffer::{numeric_buffer_new,numeric_buffer_set,numeric_buffer_at,numeric_buffer_close}
use collections::integer_buffer::{integer_buffer_new,integer_buffer_set,integer_buffer_at,integer_buffer_close}
use collections::numeric_index_sort::{numeric_index_sort_checked}
fn run(n:i64) with IO,Alloc,Mut,Div,Panic {
 var keys=numeric_buffer_new(n,n*8)
 var indices=integer_buffer_new(n,n*8)
 var seen=integer_buffer_new(n,n*8)
 var i:i64=0
 while i<n {numeric_buffer_set(&!keys,i,((n-i)%97) as f64);integer_buffer_set(&!indices,i,-99);integer_buffer_set(&!seen,i,0);i=i+1}
 let r=numeric_index_sort_checked(&keys,n,&!indices,80000000)
 assert(r.status==0 && r.validated==n && r.comparisons<=80000000)
 var digest:i64=0
 var prior_key:f64= -1.0
 var prior_index:i64= -1
 i=0
 while i<n {
  let index=integer_buffer_at(&indices,i)
  assert(index>=0 && index<n)
  assert(integer_buffer_at(&seen,index)==0)
  integer_buffer_set(&!seen,index,1)
  let key=numeric_buffer_at(&keys,index)
  assert(key>=prior_key)
  if key==prior_key {assert(index>prior_index)}
  digest=(digest*65599+index)%1000000007
  prior_key=key;prior_index=index;i=i+1
 }
 let status:i64=r.status;let comparisons:i64=r.comparisons;let swaps:i64=r.swaps
 println("SORT {} {} {} {} {}",n,status,comparisons,swaps,digest)
 numeric_buffer_close(keys);integer_buffer_close(indices);integer_buffer_close(seen)
}
fn main()->i32 with IO,Alloc,Mut,Div,Panic {
 var keys=numeric_buffer_new(4,32)
 var indices=integer_buffer_new(4,32)
 var i:i64=0
 while i<4 {numeric_buffer_set(&!keys,i,1.0);integer_buffer_set(&!indices,i,-99);i=i+1}
 numeric_buffer_set(&!keys,3,0.0/0.0)
 let bad=numeric_index_sort_checked(&keys,4,&!indices,100)
 assert(bad.status==2 && bad.validated==3 && bad.comparisons==0)
 i=0
 while i<4 {assert(integer_buffer_at(&indices,i)== -99);i=i+1}
 assert(numeric_index_sort_checked(&keys,5,&!indices,100).status==1)
 assert(numeric_index_sort_checked(&keys,3,&!indices,0).status==3)
 let prefix=numeric_index_sort_checked(&keys,3,&!indices,100)
 assert(prefix.status==0)
 i=0
 while i<3 {assert(integer_buffer_at(&indices,i)==i);i=i+1}
 numeric_buffer_set(&!keys,0,-0.0);numeric_buffer_set(&!keys,1,0.0);numeric_buffer_set(&!keys,2,-0.0)
 assert(numeric_index_sort_checked(&keys,3,&!indices,100).status==0)
 i=0
 while i<3 {assert(integer_buffer_at(&indices,i)==i);i=i+1}
 numeric_buffer_close(keys);integer_buffer_close(indices)
)SIO";
  for(auto n:sizes)std::cout<<" run("<<n<<")\n";
  std::cout<<" println(\"INDEX_SORT_COMPLETE\")\n 0\n}\n";return 0;
 }
 if(argc!=2)return 2;
 std::ifstream in(argv[1]);if(!in)return 3;
 std::string line;std::size_t row=0;bool done=false;
 while(std::getline(in,line)){
  if(line=="INDEX_SORT_COMPLETE"){if(done)return 4;done=true;continue;}
  std::istringstream s(line);std::string tag,extra;std::int64_t n,status,comparisons,swaps,digest;
  if(!(s>>tag>>n>>status>>comparisons>>swaps>>digest)||tag!="SORT"||row>=sizes.size()||n!=sizes[row++]||s>>extra||status||comparisons<0||comparisons>80000000||swaps<0)return 5;
  std::vector<std::int64_t> indices(static_cast<std::size_t>(n));std::iota(indices.begin(),indices.end(),0);
  std::sort(indices.begin(),indices.end(),[=](auto a,auto b){auto x=(n-a)%97,y=(n-b)%97;return x<y||(x==y&&a<b);});
  std::int64_t expected=0;for(auto index:indices)expected=(expected*65599+index)%1000000007;
  if(digest!=expected)return 6;
 }
 if(!done||row!=sizes.size())return 7;
 std::cout<<"NUMERIC_INDEX_SORT_REFERENCE_PASS rows="<<row<<" max_n=1000000 native_order_and_permutation_assertions\n";
}
