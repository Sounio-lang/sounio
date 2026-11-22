// C++23 metadata/prefix custody report. No fitting or simulated probabilities.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static void need(bool b,const char* w){if(!b)throw std::runtime_error(w);}
static std::vector<std::string> science(const std::string& path){std::ifstream f(path);need(bool(f),"scientific stdout open");std::vector<std::string> v;std::string s;while(std::getline(f,s))if(s.starts_with("CAL_RESULT ")||s.starts_with("CAL_P "))v.push_back(s);return v;}
int main(int argc,char** argv){try{
 need(argc==3,"pilot_batch baseline_stdout");std::string root=argv[1];auto old=science(argv[2]),fresh=science(root+"/native.stdout");
 need(!old.empty()&&old.size()%2==0&&fresh.size()==32&&old.size()<=fresh.size(),"complete prefix and 16 isolated dataset rows");
 for(std::size_t i=0;i<old.size();i++)if(old[i]!=fresh[i]){std::cerr<<"PREFIX_MISMATCH line "<<i<<"\nold "<<old[i]<<"\nnew "<<fresh[i]<<'\n';return 2;}
 std::cout<<"baseline_scientific_rows\t"<<old.size()<<"\nprefix_byte_equal\t1\ncase\tmaximum_rss_kib\telapsed_wall\n";
 std::int64_t maximum=0;
 for(int i=0;i<16;i++){
  std::string stem=root+"/case-"+std::to_string(i);std::ifstream status(stem+".status");int rc=-1;status>>rc;need(bool(status)&&rc==0,"isolated child status");
  std::ifstream resource(stem+".resource");need(bool(resource),"case resource receipt");std::string line,elapsed;std::int64_t rss=-1;
  while(std::getline(resource,line)){
   if(line.find("Maximum resident set size (kbytes):")!=std::string::npos)rss=std::stoll(line.substr(line.rfind(':')+1));
   if(line.find("Elapsed (wall clock)")!=std::string::npos){auto at=line.find("): ");need(at!=std::string::npos,"elapsed syntax");elapsed=line.substr(at+3);}
  }
  need(rss>=0&&!elapsed.empty(),"complete GNUtime fields");maximum=std::max(maximum,rss);std::cout<<i<<'\t'<<rss<<'\t'<<elapsed<<'\n';
 }
 std::cout<<"max_single_process_rss_kib\t"<<maximum<<"\nINTERPRETATION\tper-dataset process peak; not a bound for B199/n256 or an isolated kernel benchmark\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
