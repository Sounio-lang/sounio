#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
// Independent exact field oracle; does not implement Sounio return lowering.
int main(int argc,char**argv){
 if(argc!=2)return 64;
 std::ifstream in(argv[1]);if(!in)return 2;
 std::string marker,extra;std::int64_t id=0,value=0,tag=0;
 if(!(in>>marker>>id>>value>>tag)||marker!="PACKET_RETURN")return 3;
 if(in>>extra)return 5;
 if(id!=11||value!=22||tag!=33)return 4;
 std::cout<<"PACKET_FIELDS_CPP23_OK\n";return 0;
}
