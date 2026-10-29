#include <fstream>
#include <iostream>
#include <string>
int main(int argc,char**argv){if(argc!=2)return 64;std::ifstream f(argv[1]);std::string a,b,c,d;if(!std::getline(f,a)||!std::getline(f,b)||!std::getline(f,c)||std::getline(f,d))return 1;if(a!="SIZE0=0"||b!="SIZE1=1"||c!="SIZE4=4")return 1;std::cout<<"GRAPH_PUBLIC_SLICE_LENGTHS_OK\n";}
