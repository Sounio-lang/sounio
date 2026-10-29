#include <fstream>
#include <iterator>
#include <string>
#include <iostream>
int main(int ac,char**av){if(ac!=2)return 2;std::ifstream f(av[1]);std::string s{std::istreambuf_iterator<char>(f),{}};if(s!="111 17 7\n")return 1;std::cout<<"PASS Unit bare-comma/nested/valued return native effects\n";return 0;}

