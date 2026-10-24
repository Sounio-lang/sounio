// Independent C++23 original-index oracle. No Sounio selection helper reused.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
static void need(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
static std::vector<std::int64_t> reference(){
 const std::array<double,17> x{-2,-1,0,0,.25,.5,.75,1,1.25,1.5,2,2.5,3,4,5,6,7};
 const std::array<double,17> y{0,1,0,10,-10,2,2,3,0,50,-50,0,1,0,1,2,3};
 const std::array<int,17> a{1,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,1};
 std::set<std::int64_t> chosen;
 std::size_t i=0;
 while(i<x.size()){
  if(!a[i]){chosen.insert(static_cast<std::int64_t>(i));while(i<x.size()&&!a[i])++i;continue;}
  std::map<int,std::vector<std::size_t>> bins;
  while(i<x.size()&&a[i]){
   // This dyadic fixture has exact integer bucket boundaries at 0,1,2,3,4.
   const int key=x[i]<0?-1:(x[i]>4?4:std::min(3,static_cast<int>(std::floor(x[i]))));
   bins[key].push_back(i);++i;
  }
  for(const auto& [key,v]:bins){
   chosen.insert(static_cast<std::int64_t>(v.front()));chosen.insert(static_cast<std::int64_t>(v.back()));
   if(key>=0&&key<4){
    auto low=v.front(),high=v.front();
    for(const auto j:v){if(y[j]<y[low])low=j;if(y[j]>y[high])high=j;}
    chosen.insert(static_cast<std::int64_t>(low));chosen.insert(static_cast<std::int64_t>(high));
   }
  }
 }
 return {chosen.begin(),chosen.end()};
}
int main(int argc,char** argv){
 try{
  need(argc==3,"usage: ORACLE stream.stdout runs.stdout");
  const auto expected=reference();std::ifstream stream(argv[1]);need(bool(stream),"open stream stdout");
  std::vector<std::int64_t> batch;std::array<std::vector<std::int64_t>,18> cuts;
  std::array<bool,18> done{};bool batch_seen=false,complete=false;std::string token;
  while(stream>>token){
   if(token=="STREAM_BATCH"){long long n=-1;need(bool(stream>>n)&&n==static_cast<long long>(expected.size())&&!batch_seen,"batch count");batch_seen=true;}
   else if(token=="STREAM_INDEX"){std::int64_t index=-1;need(bool(stream>>index),"batch index read");batch.push_back(index);}
   else if(token=="STREAM_CUT_INDEX"){int cut=-1;std::int64_t index=-1;need(bool(stream>>cut>>index)&&cut>=0&&cut<18,"cut index read");cuts[static_cast<std::size_t>(cut)].push_back(index);}
   else if(token=="STREAM_CUT_DONE"){int cut=-1;need(bool(stream>>cut)&&cut>=0&&cut<18,"cut read");need(!done[static_cast<std::size_t>(cut)],"duplicate cut");done[static_cast<std::size_t>(cut)]=true;}
   else if(token=="STDLIB_DEEP_VIZ_LINE_STREAM_PASS")complete=true;
  }
  need(batch_seen&&complete&&batch==expected,"batch provenance mismatch/incomplete");
  for(std::size_t i=0;i<cuts.size();++i)need(done[i]&&cuts[i]==expected,"chunk boundary provenance mismatch/incomplete");
  std::ifstream runs(argv[2]);need(bool(runs),"open runs stdout");bool found=false,ended=false;
  while(runs>>token){
   if(token=="REDUCE_RUNS"){
    std::int64_t n=0,actual=0;need(bool(runs>>n>>actual)&&n==2000001&&!found,"runs record");
    std::int64_t checksum=0;
    for(std::int64_t index=0;index<n;++index)checksum=(checksum+index)%1000000007;
    need(actual==checksum,"runs checksum");found=true;
   }else if(token=="STDLIB_DEEP_VIZ_REDUCE_RUNS_PASS")ended=true;
  }
  need(found&&ended,"runs incomplete");
  std::cout<<"VIZ_LINE_STREAM_REFERENCE_PASS cuts=18 original_indices="<<expected.size()<<" runs_rows=2000001\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
