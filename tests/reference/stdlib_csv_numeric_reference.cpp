#include <cassert>
#include <charconv>
#include <cstdint>
#include <limits>
#include <string_view>
#include <cstdio>
using I=std::int64_t;
struct Result{I value;bool valid;};
Result integer(std::string_view s){
 if(!s.empty()&&s.front()=='+')s.remove_prefix(1);
 I v=0;auto [p,e]=std::from_chars(s.data(),s.data()+s.size(),v);
 return {e==std::errc{}&&p==s.data()+s.size()?v:0,e==std::errc{}&&p==s.data()+s.size()};
}
Result scaled(std::string_view s,int decimals){
 if(decimals<0||decimals>18||s.empty())return {0,false};
 bool negative=s.front()=='-';if(negative||s.front()=='+')s.remove_prefix(1);
 __int128 coefficient=0;int fractional=0,digits=0;bool point=false;
 for(char c:s){
  if(c=='.'&&!point){point=true;continue;}
  if(c<'0'||c>'9')return {0,false};
  if(++digits>36)return {0,false};
  coefficient=coefficient*10+(c-'0');if(point)++fractional;
 }
 if(!digits)return {0,false};
 while(fractional>decimals){coefficient/=10;--fractional;}
 while(fractional<decimals){if(coefficient>std::numeric_limits<__int128>::max()/10)return {0,false};coefficient*=10;++fractional;}
 if(negative)coefficient=-coefficient;
 if(coefficient<std::numeric_limits<I>::min()||coefficient>std::numeric_limits<I>::max())return {0,false};
 return {static_cast<I>(coefficient),true};
}
int main(){
 const I low=std::numeric_limits<I>::min(),high=std::numeric_limits<I>::max();
 assert(integer("9223372036854775807").value==high);
 assert(integer("-9223372036854775808").value==low);
 assert(!integer("9223372036854775808").valid);
 assert(!integer("-9223372036854775809").valid);
 assert(!integer("1.0").valid);assert(integer("+7").value==7);
 assert(scaled("92233720368547758.07",2).value==high);
 assert(scaled("-92233720368547758.08",2).value==low);
 assert(!scaled("92233720368547758.08",2).valid);
 assert(!scaled("-92233720368547758.09",2).valid);
 assert(scaled("1.12345",2).value==112);
 assert(scaled(".5",2).value==50);
 assert(!scaled(".",2).valid);
 assert(!scaled("1",-1).valid);assert(!scaled("1",19).valid);
 std::puts("CSV_NUMERIC_CPP_OK");
}
