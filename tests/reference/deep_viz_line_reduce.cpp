// C++23 independent integer-domain provenance oracle. No production helper reuse.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using I=long long;
struct Sample{I x,y;bool active;};
std::vector<I> reference(const std::vector<Sample>& data,I lo,I hi,I buckets){
 std::set<I> selected;
 const I width=(hi-lo)/buckets;
 if(width<=0 || width*buckets!=hi-lo) throw std::runtime_error("noninteger oracle bins");
 I i=0,n=static_cast<I>(data.size());
 while(i<n){
  if(!data[static_cast<std::size_t>(i)].active){selected.insert(i);while(i<n && !data[static_cast<std::size_t>(i)].active)++i;continue;}
  I start=i;while(i<n && data[static_cast<std::size_t>(i)].active)++i;
  I end=i-1;selected.insert(start);selected.insert(end);
  std::map<I,std::vector<I>> groups;
  I before=-1,after=-1;
  for(I j=start;j<=end;++j){const auto& p=data[static_cast<std::size_t>(j)];
   if(p.x<lo) before=j;
   else if(p.x>hi){if(after<0)after=j;}
   else {I bin=p.x==hi?buckets-1:(p.x-lo)/width;groups[bin].push_back(j);}
  }
  if(before>=0)selected.insert(before);
  if(after>=0)selected.insert(after);
  for(const auto& [key,points]:groups){
   (void)key;selected.insert(points.front());selected.insert(points.back());
   const auto less=[&](I a,I b){return data[static_cast<std::size_t>(a)].y<data[static_cast<std::size_t>(b)].y;};
   selected.insert(*std::min_element(points.begin(),points.end(),less));
   selected.insert(*std::max_element(points.begin(),points.end(),less));
  }
 }
 return {selected.begin(),selected.end()};
}
int main(int argc,char** argv){try{
 if(argc!=2)throw std::runtime_error("usage: deep_viz_line_reduce actual-native-stdout");
 std::vector<Sample> data;data.reserve(1000000);
 for(I i=0;i<1000000;++i){I y=0;if(i%100003==0)y=1000000;if(i%100019==0)y=-1000000;
  const bool gap=(i>=100000&&i<100010)||(i>=500000&&i<500020)||(i>=900000&&i<900010);
  data.push_back({i,y,!gap});
 }
 const auto expected=reference(data,1024,525312,512);
 std::ifstream input(argv[1]);if(!input)throw std::runtime_error("cannot open native stdout");
 std::string line;std::vector<I> actual;I reported=-1;bool complete=false;
 while(std::getline(input,line)){
  std::istringstream row(line);std::string tag;row>>tag;
  if(tag=="LINE_COUNT"){if(reported>=0)throw std::runtime_error("duplicate count");row>>reported;}
  if(tag=="LINE_INDEX"){I ordinal=-1,index=-1;row>>ordinal>>index;if(!row || ordinal!=static_cast<I>(actual.size()))throw std::runtime_error("ordinal");actual.push_back(index);}
  if(tag=="STDLIB_DEEP_VIZ_LINE_REDUCE_MILLION_PASS")complete=true;
 }
 if(!complete || reported!=static_cast<I>(expected.size()) || actual.size()!=expected.size())throw std::runtime_error("completion/count mismatch");
 for(std::size_t j=0;j<expected.size();++j)if(actual[j]!=expected[j])throw std::runtime_error("provenance mismatch at "+std::to_string(j));
 std::cout<<"DEEP_VIZ_LINE_REDUCE_REFERENCE_PASS n=1000000 indices="<<actual.size()<<"\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
