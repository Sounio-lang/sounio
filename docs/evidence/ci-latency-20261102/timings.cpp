#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
struct Row{std::string name;long seconds;};
std::vector<std::string> fields(const std::string&s){std::vector<std::string>v;std::istringstream in(s);std::string x;while(std::getline(in,x,'\t'))v.push_back(x);return v;}
std::time_t utc(const std::string&s){std::tm tm{};std::istringstream in(s);in>>std::get_time(&tm,"%Y-%m-%dT%H:%M:%SZ");if(in.fail())throw std::runtime_error("bad timestamp: "+s);return timegm(&tm);}
int main(int argc,char**argv){
 if(argc!=3)return 2;
 const std::string mode=argv[1];std::ifstream f(argv[2]);if(!f)return 2;
 std::string line;std::vector<Row> rows;std::time_t first=0,last=0;long total=0;
 while(std::getline(f,line)){auto v=fields(line);if(mode=="runs"){
  if(v.size()!=6)return 3;
  auto created=utc(v[2]),started=utc(v[3]),ended=utc(v[4]);
  std::cout<<v[0]<<" "<<v[1]<<" elapsed_seconds="<<ended-started<<" initial_queue_seconds="<<started-created<<" conclusion="<<v[5]<<"\n";continue;
 }
 const bool step=mode=="steps";const std::size_t i=step?2:1;if(v.size()<i+3)return 3;
 if(v[i].empty()||v[i+1].empty()||v[i]=="null"||v[i+1]=="null")continue;
 if(v[i+2]=="skipped"){std::cerr<<"EXCLUDED_SKIPPED "<<v[0]<<"\n";continue;}
 auto a=utc(v[i]),b=utc(v[i+1]);if(b<a){std::cerr<<"NEGATIVE_TIMESTAMP "<<v[0]<<" start="<<v[i]<<" end="<<v[i+1]<<" outcome="<<v[i+2]<<"\n";return 4;}if(!first||a<first)first=a;if(b>last)last=b;
 const long seconds=b-a;total+=seconds;rows.push_back({step?v[0]+" / "+v[1]:v[0],seconds});
 }
 if(mode=="runs")return 0;
 std::sort(rows.begin(),rows.end(),[](const Row&a,const Row&b){return a.seconds>b.seconds;});
 std::cout<<"WALL_SECONDS="<<last-first<<" SUM_SECONDS="<<total<<" ROWS="<<rows.size()<<"\n";
 for(const auto&r:rows)std::cout<<r.seconds<<"s\t"<<std::fixed<<std::setprecision(2)<<r.seconds/60.0<<"min\t"<<r.name<<"\n";
}
