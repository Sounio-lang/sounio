#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
int main(int argc,char**argv){
 try{
  if(argc!=2)throw std::runtime_error("one log required");
  std::ifstream in(argv[1]);if(!in)throw std::runtime_error("log open");
  bool guards=false,pass=false,immutable=false;std::string line;
  while(std::getline(in,line)){
   if(line=="ENTRY_ALL_FIELDS_IMMUTABLE")immutable=true;
   if(line=="ENTRY_CELL_GUARDS_OK")guards=true;
   if(line=="BORROW_COMPONENT_OK entries=512 loans=4097 branches=24 immutable_forks=1")pass=true;
  }
  if(!guards||!pass||!immutable)throw std::runtime_error("missing component evidence");
  std::cout<<"BORROW_COMPONENT_CPP23_OK\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
