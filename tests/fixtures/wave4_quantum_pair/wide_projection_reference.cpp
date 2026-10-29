#define main wide_reference_main
#include "wide_counts_reference.cpp"
#undef main
int main(int argc,char**argv){
 if(argc!=2){return 64;}std::ifstream f(argv[1]);if(!f){return 2;}
 const auto all=cases();const Case* selected=nullptr;for(const auto&c:all){if(c.name=="bench0"){selected=&c;}}
 if(!selected){return 1;}const auto expected=count(*selected);std::map<std::string,U> got;std::string line;bool ended=false;
 while(std::getline(f,line)){
  if(line=="SINGLE_FILE_PROJECTION_ONLY"){if(ended){return 1;}ended=true;continue;}
  if(ended){return 1;}auto eq=line.find('=');if(eq==std::string::npos){return 1;}auto key=line.substr(0,eq),val=line.substr(eq+1);U n=0;auto p=std::from_chars(val.data(),val.data()+val.size(),n);if(p.ec!=std::errc{}||p.ptr!=val.data()+val.size()||got.contains(key)){return 1;}got[key]=n;
 }
 if(!ended||got.size()!=8){return 1;}const char* keys[]={"N00","N01","N10","N11"};for(unsigned i=0;i<4;++i){if(got["LITERAL_"+std::string(keys[i])]!=expected[i]||got["HOIST_"+std::string(keys[i])]!=expected[i]){return 1;}}
 std::cout<<"SINGLE_FILE_SOURCE_PROJECTION_CPP_OK\n";return 0;
}
