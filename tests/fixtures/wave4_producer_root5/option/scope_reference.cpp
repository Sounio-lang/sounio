#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
int main(int ac,char**av){if(ac!=2)return 2;std::ifstream f(av[1]);std::string s{std::istreambuf_iterator<char>(f),{}};if(s!="17 1 -9 0\n")return 1;std::cout<<"PASS caller_first_arm_proof_survives_nested_closure\n";return 0;}

